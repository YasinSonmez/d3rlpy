import dataclasses
from typing import Any, Optional
from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace
from ...types import Shape
from ...dataset import ReplayBufferBase
from ...torch_utility import convert_to_torch_recursively, TorchMiniBatch
from ...dataset import TransitionMiniBatch
from .td3_plus_bc import TD3PlusBCConfig, TD3PlusBC
from .torch.online_il_td3_plus_bc_impl import OnlineILTD3PlusBCImpl

__all__ = ["OnlineILTD3PlusBCConfig", "OnlineILTD3PlusBC"]


@dataclasses.dataclass()
class OnlineILTD3PlusBCConfig(TD3PlusBCConfig):
    """Online IL variant of TD3+BC that uses separate expert buffer for BC term."""
    
    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "OnlineILTD3PlusBC":
        return OnlineILTD3PlusBC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "online_il_td3_plus_bc"


class OnlineILTD3PlusBC(TD3PlusBC):
    _expert_buffer: Optional[ReplayBufferBase]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expert_buffer = None

    def set_expert_buffer(self, buffer: ReplayBufferBase) -> None:
        self._expert_buffer = buffer

    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        # Use parent's implementation to create modules
        super().inner_create_impl(observation_shape, action_size)
        
        # Replace impl with OnlineIL version
        assert self._impl is not None
        modules = self._impl._modules
        
        self._impl = OnlineILTD3PlusBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=self._impl._q_func_forwarder,
            targ_q_func_forwarder=self._impl._targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            target_smoothing_sigma=self._config.target_smoothing_sigma,
            target_smoothing_clip=self._config.target_smoothing_clip,
            alpha=self._config.alpha,
            update_actor_interval=self._config.update_actor_interval,
            compiled=self._config.compile_graph,
            device=self._device,
        )

    def update(self, batch: TransitionMiniBatch) -> dict[str, float]:
        """Override update to use separate expert batch for BC term."""
        assert self._impl
        
        # Convert online batch to torch
        torch_batch = TorchMiniBatch.from_batch(
            batch=batch,
            gamma=self._config.gamma,
            compute_returns_to_go=self.need_returns_to_go,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )
        
        # Sample expert batch if available
        if self._expert_buffer and self._expert_buffer.transition_count >= self.batch_size:
            expert_batch_raw = self._expert_buffer.sample_transition_batch(self.batch_size)
            expert_batch = TorchMiniBatch.from_batch(
                batch=expert_batch_raw,
                gamma=self._config.gamma,
                compute_returns_to_go=self.need_returns_to_go,
                device=self._device,
                observation_scaler=self._config.observation_scaler,
                action_scaler=self._config.action_scaler,
                reward_scaler=self._config.reward_scaler,
            )
        else:
            expert_batch = torch_batch
        
        metrics = self._impl.inner_update_with_expert(torch_batch, expert_batch, self._grad_step)
        self._grad_step += 1
        return metrics


register_learnable(OnlineILTD3PlusBCConfig)
