# pylint: disable=too-many-ancestors
import dataclasses
import torch
from typing import Dict
from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    Parameter,
    get_parameter,
    build_squashed_gaussian_distribution,
)
from ....torch_utility import TorchMiniBatch
from ....types import Shape
from .sac_impl import SACActorLoss, SACImpl, SACModules

__all__ = ["OnlineILBCSACImpl", "OnlineILBCSACActorLoss"]


@dataclasses.dataclass(frozen=True)
class OnlineILBCSACActorLoss(SACActorLoss):
    sac_policy_loss: torch.Tensor
    bc_log_likelihood: torch.Tensor


class OnlineILBCSACImpl(SACImpl):
    _bc_lambda: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: SACModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        bc_lambda: float,
        compiled: bool,
        device: str,
    ):
        if not isinstance(modules.policy, NormalPolicy):
            raise ValueError("OnlineILBCSACImpl requires NormalPolicy")
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            compiled=compiled,
            device=device,
        )
        self._bc_lambda = bc_lambda

    def compute_actor_loss_with_expert(
        self, rl_batch: TorchMiniBatch, bc_batch: TorchMiniBatch, action: ActionOutput
    ) -> OnlineILBCSACActorLoss:
        # SAC term on RL batch (mixed data)
        dist = build_squashed_gaussian_distribution(action)
        sampled_action, log_prob_pi = dist.sample_with_log_prob()

        if self._modules.temp_optim:
            temp_loss = self.update_temp(log_prob_pi)
        else:
            temp_loss = torch.tensor(0.0, dtype=torch.float32, device=sampled_action.device)

        temp = get_parameter(self._modules.log_temp).exp()
        q_t = self._q_func_forwarder.compute_expected_q(rl_batch.observations, sampled_action, "min")
        sac_policy_loss = (temp * log_prob_pi - q_t).mean()

        # BC term on expert batch only
        bc_action_dist = build_squashed_gaussian_distribution(
            self._modules.policy(bc_batch.observations)
        )
        bc_log_likelihood = bc_action_dist.log_prob(bc_batch.actions).mean()

        total_actor_loss = sac_policy_loss - self._bc_lambda * bc_log_likelihood

        return OnlineILBCSACActorLoss(
            actor_loss=total_actor_loss,
            sac_policy_loss=sac_policy_loss,
            bc_log_likelihood=bc_log_likelihood,
            temp_loss=temp_loss,
            temp=temp[0][0],
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
        
        # Actor update with separate batches
        self._modules.q_funcs.eval()
        self._modules.actor_optim.zero_grad()
        action = self._modules.policy(rl_batch.observations)
        actor_loss_dict = self.compute_actor_loss_with_expert(rl_batch, bc_batch, action)
        actor_loss_dict.actor_loss.backward()
        self._modules.actor_optim.step()
        
        self.update_target()
        
        return {
            "critic_loss": float(critic_loss.cpu().detach().numpy()),
            "actor_loss": float(actor_loss_dict.actor_loss.cpu().detach().numpy()),
            "temp_loss": float(actor_loss_dict.temp_loss.cpu().detach().numpy()),
            "temp": float(actor_loss_dict.temp.cpu().detach().numpy()),
        }
