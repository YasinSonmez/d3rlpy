import dataclasses
from typing import Any, Optional
from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import create_continuous_q_function, create_deterministic_policy
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from ...dataset import ReplayBufferBase
from ...torch_utility import convert_to_torch_recursively
from .base import QLearningAlgoBase
from .torch.ddpg_impl import DDPGModules
from .torch.online_il_td3_plus_bc_impl import OnlineILTD3PlusBCImpl

__all__ = ["OnlineILTD3PlusBCConfig", "OnlineILTD3PlusBC"]


@dataclasses.dataclass()
class OnlineILTD3PlusBCConfig(LearnableConfig):
    observation_scaler: Optional[str] = None
    action_scaler: Optional[str] = None
    reward_scaler: Optional[str] = None
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    target_smoothing_sigma: float = 0.2
    target_smoothing_clip: float = 0.5
    alpha: float = 2.5
    update_actor_interval: int = 2
    compile_graph: bool = False

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "OnlineILTD3PlusBC":
        return OnlineILTD3PlusBC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "online_il_td3_plus_bc"


class OnlineILTD3PlusBC(QLearningAlgoBase[OnlineILTD3PlusBCImpl, OnlineILTD3PlusBCConfig]):
    _expert_buffer: Optional[ReplayBufferBase]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expert_buffer = None

    def set_expert_buffer(self, buffer: ReplayBufferBase) -> None:
        self._expert_buffer = buffer

    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        policy = create_deterministic_policy(
            observation_shape, action_size, self._config.actor_encoder_factory, device=self._device
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape, action_size, self._config.critic_encoder_factory,
            self._config.q_func_factory, n_ensembles=self._config.n_critics, device=self._device
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape, action_size, self._config.critic_encoder_factory,
            self._config.q_func_factory, n_ensembles=self._config.n_critics, device=self._device
        )
        
        actor_optim = self._config.actor_optim_factory.create(policy.named_modules(), lr=self._config.actor_learning_rate)
        critic_optim = self._config.critic_optim_factory.create(q_funcs.named_modules(), lr=self._config.critic_learning_rate)

        modules = DDPGModules(
            policy=policy, q_funcs=q_funcs, targ_q_funcs=targ_q_funcs,
            actor_optim=actor_optim, critic_optim=critic_optim
        )

        self._impl = OnlineILTD3PlusBCImpl(
            observation_shape=observation_shape, action_size=action_size, modules=modules,
            q_func_forwarder=q_func_forwarder, targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma, tau=self._config.tau,
            target_smoothing_sigma=self._config.target_smoothing_sigma,
            target_smoothing_clip=self._config.target_smoothing_clip,
            alpha=self._config.alpha, update_actor_interval=self._config.update_actor_interval,
            compiled=self._config.compile_graph, device=self._device
        )

    def inner_update(self, batch: Any, grad_step: int) -> dict[str, float]:
        assert self._impl
        if self._expert_buffer and self._expert_buffer.transition_count >= self.batch_size:
            expert_batch_raw = self._expert_buffer.sample_transition_batch(self.batch_size)
            expert_batch = convert_to_torch_recursively(expert_batch_raw, self._impl.device)
        else:
            expert_batch = batch
        
        return self._impl.inner_update_with_expert(batch, expert_batch, grad_step)


register_learnable(OnlineILTD3PlusBCConfig)
