# pylint: disable=too-many-ancestors
import dataclasses

import torch

from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy, # Assuming NormalPolicy is used
    Parameter,
    get_parameter,
    build_squashed_gaussian_distribution,
)
from ....torch_utility import TorchMiniBatch
from ....types import Shape
from .sac_impl import SACActorLoss, SACImpl, SACModules # Use base SAC classes

__all__ = ["BCSACImpl", "BCSACActorLoss"]


@dataclasses.dataclass(frozen=True)
class BCSACActorLoss(SACActorLoss):
    """Dataclass for SAC+BC actor loss.

    Attributes:
        actor_loss: Total actor loss value (SAC - lambda * BC_log_likelihood).
        sac_policy_loss: The standard SAC policy loss component (alpha*log_pi - Q).
        bc_log_likelihood: The average log-likelihood of dataset actions under the current policy.
        temp_loss: Temperature loss value.
        temp: Current temperature value.
    """
    sac_policy_loss: torch.Tensor
    bc_log_likelihood: torch.Tensor


class BCSACImpl(SACImpl):
    """Soft Actor-Critic + Behavioral Cloning (Log-Likelihood objective) implementation.

    Args:
        observation_shape: Observation shape.
        action_size: Action size.
        modules: SAC modules (must contain NormalPolicy).
        q_func_forwarder: Q-function forwarder.
        targ_q_func_forwarder: Target Q-function forwarder.
        gamma: Discount factor.
        tau: Target network synchronization coefficient.
        bc_lambda: Weight factor (lambda) for the BC log-likelihood term.
        compiled: Whether to compile the computation graph.
        device: Torch device.
    """

    _bc_lambda: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: SACModules, # Requires modules.policy to be NormalPolicy for log_prob
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        bc_lambda: float, # BC specific parameter (lambda from the formula)
        compiled: bool,
        device: str,
    ):
        # Ensure the policy is appropriate for log_prob calculation
        if not isinstance(modules.policy, NormalPolicy):
             raise ValueError("BCSACImpl requires modules.policy to be d3rlpy.models.torch.policies.NormalPolicy to compute log probabilities.")

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

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput # action is output of policy(batch.observations)
    ) -> BCSACActorLoss:
        """Computes the SAC+BC (likelihood) actor loss.

        Args:
            batch: Mini-batch data (including observations and actions from dataset D).
            action: Action output from the policy network for batch.observations.

        Returns:
            Computed actor loss for SAC+BC (likelihood).
        """
        # --- Standard SAC Loss Components ---
        dist = build_squashed_gaussian_distribution(action)
        sampled_action, log_prob_pi = dist.sample_with_log_prob() # Log prob of actions sampled from policy

        if self._modules.temp_optim:
            temp_loss = self.update_temp(log_prob_pi) # Update temp based on policy entropy
        else:
            temp_loss = torch.tensor(
                0.0, dtype=torch.float32, device=sampled_action.device
            )

        temp = get_parameter(self._modules.log_temp).exp()
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, sampled_action, "min"
        ) # Q value of actions sampled from policy

        # Standard SAC policy loss: alpha * log pi(a|s) - Q(s,a) (we minimize this)
        sac_policy_loss = (temp * log_prob_pi - q_t).mean()

        # --- BC Log-Likelihood Component ---
        # Calculate log probability of dataset actions (batch.actions) under the current policy
        # The distribution 'dist' was built from policy(batch.observations)
        log_prob_dataset_actions = dist.log_prob(batch.actions)
        bc_log_likelihood = log_prob_dataset_actions.mean()

        # --- Combine Losses ---
        # We want to maximize BC log-likelihood, so minimize its negative
        # Total loss = SAC_loss - lambda * BC_log_likelihood
        total_actor_loss = sac_policy_loss - self._bc_lambda * bc_log_likelihood

        return BCSACActorLoss(
            actor_loss=total_actor_loss,
            sac_policy_loss=sac_policy_loss,
            bc_log_likelihood=bc_log_likelihood,
            temp_loss=temp_loss,
            temp=temp[0][0],
        )