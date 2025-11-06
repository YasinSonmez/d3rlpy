# pylint: disable=too-many-ancestors
import dataclasses
import torch
from typing import Dict
from ....models.torch import ActionOutput, ContinuousEnsembleQFunctionForwarder
from ....torch_utility import TorchMiniBatch
from ....types import Shape
from .ddpg_impl import DDPGBaseActorLoss, DDPGModules
from .td3_impl import TD3Impl

__all__ = ["OnlineILTD3PlusBCImpl", "OnlineILTD3PlusBCActorLoss"]


@dataclasses.dataclass(frozen=True)
class OnlineILTD3PlusBCActorLoss(DDPGBaseActorLoss):
    bc_loss: torch.Tensor


class OnlineILTD3PlusBCImpl(TD3Impl):
    _alpha: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DDPGModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        update_actor_interval: int,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            update_actor_interval=update_actor_interval,
            compiled=compiled,
            device=device,
        )
        self._alpha = alpha

    def compute_actor_loss_with_expert(
        self, rl_batch: TorchMiniBatch, bc_batch: TorchMiniBatch, action: ActionOutput
    ) -> OnlineILTD3PlusBCActorLoss:
        # Q-maximization on RL batch (mixed data)
        q_t = self._q_func_forwarder.compute_expected_q(
            rl_batch.observations, action.squashed_mu, "none"
        )[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        
        # BC term on expert batch only
        bc_action = self._modules.policy(bc_batch.observations).squashed_mu
        bc_loss = ((bc_batch.actions - bc_action) ** 2).mean()
        
        return OnlineILTD3PlusBCActorLoss(
            actor_loss=lam * -q_t.mean() + bc_loss, 
            bc_loss=bc_loss
        )

    def inner_update_with_expert(self, rl_batch: TorchMiniBatch, bc_batch: TorchMiniBatch, grad_step: int) -> Dict[str, float]:
        self._modules.q_funcs.train()
        self._modules.policy.train()
        
        # Critic update
        self._modules.critic_optim.zero_grad()
        q_tpn = self.compute_target(rl_batch)
        critic_loss = self.compute_critic_loss(rl_batch, q_tpn)
        critic_loss.backward()
        self._modules.critic_optim.step()
        
        metrics = {"critic_loss": float(critic_loss.cpu().detach().numpy())}
        
        # Actor update (only if interval met) with separate batches
        if grad_step % self._update_actor_interval == 0:
            self._modules.q_funcs.eval()
            self._modules.actor_optim.zero_grad()
            action = self._modules.policy(rl_batch.observations)
            actor_loss_dict = self.compute_actor_loss_with_expert(rl_batch, bc_batch, action)
            actor_loss_dict.actor_loss.backward()
            self._modules.actor_optim.step()
            
            self.update_critic_target()
            self.update_actor_target()
            
            metrics["actor_loss"] = float(actor_loss_dict.actor_loss.cpu().detach().numpy())
            metrics["bc_loss"] = float(actor_loss_dict.bc_loss.cpu().detach().numpy())
        
        return metrics
