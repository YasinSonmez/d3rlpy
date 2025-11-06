import dataclasses
from typing import Any, Optional
from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace
from ...types import Shape
from ...dataset import ReplayBufferBase
from ...torch_utility import convert_to_torch_recursively
from .bc_sac import BCSACConfig, BCSAC
from .torch.online_il_bc_sac_impl import OnlineILBCSACImpl

__all__ = ["OnlineILBCSACConfig", "OnlineILBCSAC"]


@dataclasses.dataclass()
class OnlineILBCSACConfig(BCSACConfig):
    """Online IL variant of BCSAC that uses separate expert buffer for BC term."""
    
    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "OnlineILBCSAC":
        return OnlineILBCSAC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "online_il_bcsac"


class OnlineILBCSAC(BCSAC):
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
        
        self._impl = OnlineILBCSACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=self._impl._q_func_forwarder,
            targ_q_func_forwarder=self._impl._targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            bc_lambda=self._config.bc_lambda,
            compiled=self._config.compile_graph,
            device=self._device,
        )

    def inner_update(self, batch: Any, grad_step: int) -> dict[str, float]:
        assert self._impl
        if self._expert_buffer and self._expert_buffer.transition_count >= self.batch_size:
            expert_batch_raw = self._expert_buffer.sample_transition_batch(self.batch_size)
            expert_batch = convert_to_torch_recursively(expert_batch_raw, self._impl.device)
        else:
            expert_batch = batch
        
        return self._impl.inner_update_with_expert(batch, expert_batch, grad_step)


register_learnable(OnlineILBCSACConfig)
