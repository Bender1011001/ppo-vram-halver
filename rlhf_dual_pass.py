import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleModelPPOLoss(nn.Module):
    """
    Computes PPO policy, value, and sampled reference log-ratio losses using one model
    by dynamically enabling and disabling PEFT adapters.
    """
    def __init__(self, kl_beta=0.01, clip_range=0.2, value_clip_range=0.2):
        super().__init__()
        self.kl_beta = kl_beta
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range

    def get_log_probs(self, logits, labels):
        """
        Extracts the log probabilities of the selected action tokens.
        """
        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log-softmax over vocabulary
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log-probabilities of the actual chosen tokens
        # Ignore positions where label is padding/masked (-100)
        mask = (shift_labels != -100)
        gather_labels = shift_labels.clone()
        gather_labels[~mask] = 0  # Fill mask to prevent out-of-bounds in gather
        
        selected_log_probs = torch.gather(log_probs, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
        selected_log_probs = selected_log_probs * mask.float()
        
        return selected_log_probs, mask

    def reference_log_probs(self, model, input_ids, attention_mask=None):
        """Score actions with the frozen base model, without retaining its logits."""
        if not callable(getattr(model, "disable_adapter", None)):
            raise TypeError("model must provide a disable_adapter() context manager")

        was_training = model.training
        model.eval()  # Match a separate frozen reference, including dropout behavior.
        try:
            with model.disable_adapter(), torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
                log_probs, _ = self.get_log_probs(outputs.logits, input_ids)
                return log_probs
        finally:
            model.train(was_training)

    def forward(self, model, inputs, action_masks, old_log_probs, advantages, returns, old_values=None, value_head=None):
        """
        Executes reference and active passes on a single model and computes PPO losses.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if input_ids.ndim != 2 or input_ids.shape[-1] < 2:
            raise ValueError("input_ids must have shape [batch, sequence] with at least two tokens")
        for name, tensor in (
            ("action_masks", action_masks),
            ("old_log_probs", old_log_probs),
            ("advantages", advantages),
            ("returns", returns),
            ("attention_mask", attention_mask),
            ("old_values", old_values),
        ):
            if tensor is not None and tensor.shape != input_ids.shape:
                raise ValueError(f"{name} must have the same shape as input_ids")
        if (value_head is None) != (old_values is None):
            raise ValueError("value_head and old_values must be provided together")

        # 1. REFERENCE PASS: keep only selected-token log probabilities.
        ref_log_probs = self.reference_log_probs(model, input_ids, attention_mask)
        
        # 2. ACTOR PASS: Run with active adapters (and track gradients)
        actor_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=value_head is not None
        )
        actor_logits = actor_outputs.logits
        
        # Compute active policy log probs
        actor_log_probs, token_mask = self.get_log_probs(actor_logits, input_ids)
        active = token_mask & action_masks[..., 1:].bool()
        if attention_mask is not None:
            active = active & attention_mask[..., 1:].bool()
        mask_float = active.float()
        token_counts = mask_float.sum(dim=-1).clamp_min(1)
        
        # 3. PPO POLICY LOSS
        # Ratio of new policy log probs to old rollout policy log probs
        # old_log_probs represent log probabilities from the rollout phase (detached)
        log_ratios = torch.where(active, actor_log_probs - old_log_probs[..., 1:], 0.0)
        ratios = torch.exp(log_ratios)
        
        # Policy gradient surrogate objective
        masked_advantages = torch.where(active, advantages[..., 1:], 0.0)
        surr1 = ratios * masked_advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * masked_advantages
        
        # Sum over sequence dimension and mean over batch dimension
        policy_loss = -torch.sum(torch.min(surr1, surr2) * mask_float, dim=-1)
        # Normalize by active token count
        policy_loss = torch.mean(policy_loss / token_counts)
        
        # 4. SAMPLED REFERENCE PENALTY (not the exact distributional KL)
        kl_div = torch.where(active, actor_log_probs - ref_log_probs, 0.0)
        kl_penalty = torch.sum(kl_div * mask_float, dim=-1)
        kl_penalty = torch.mean(kl_penalty / token_counts)
        
        # Combined objective: policy loss + KL penalty
        total_loss = policy_loss + self.kl_beta * kl_penalty
        
        # 5. VALUE LOSS (If Critic Value Head is provided)
        value_loss = torch.tensor(0.0, device=input_ids.device)
        if value_head is not None:
            # We pass hidden states to the value head to get values
            # (Usually done by projecting hidden states of the transformer)
            if not getattr(actor_outputs, "hidden_states", None):
                raise ValueError("model must return hidden_states when value_head is provided")
            hidden_states = actor_outputs.hidden_states[-1]
            values = value_head(hidden_states).squeeze(-1)
            values = torch.where(active, values[..., :-1], 0.0)
            prior_values = torch.where(active, old_values[..., 1:], 0.0)
            targets = torch.where(active, returns[..., 1:], 0.0)
            
            # Value function clipping
            values_clipped = prior_values + torch.clamp(
                values - prior_values,
                -self.value_clip_range, 
                self.value_clip_range
            )
            
            vf_losses1 = (values - targets) ** 2
            vf_losses2 = (values_clipped - targets) ** 2
            vf_loss = 0.5 * torch.sum(torch.max(vf_losses1, vf_losses2) * mask_float, dim=-1)
            value_loss = torch.mean(vf_loss / token_counts)
            
            total_loss = total_loss + value_loss
            
        return total_loss, policy_loss.item(), kl_penalty.item(), value_loss.item()


def step_ppo_dual_pass(model, inputs, action_masks, old_log_probs, advantages, returns, optimizer, 
                       old_values=None, value_head=None, kl_beta=0.01, clip_range=0.2):
    """
    Performs a single-model dual-pass PPO training step.
    Disables adapters for reference logits, re-enables for actor logits,
    computes PPO policy/KL/value loss, performs backward pass and optimizer step.
    """
    model.train()
    if value_head is not None:
        value_head.train()
    optimizer.zero_grad()
    
    loss_fn = SingleModelPPOLoss(kl_beta=kl_beta, clip_range=clip_range)
    total_loss, p_loss, kl_pen, val_loss = loss_fn(
        model=model,
        inputs=inputs,
        action_masks=action_masks,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        old_values=old_values,
        value_head=value_head
    )
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), p_loss, kl_pen, val_loss
