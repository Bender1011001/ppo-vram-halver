import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel

class SingleModelPPOLoss(nn.Module):
    """
    Computes PPO policy, value, and KL-divergence losses using a single model shell
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

    def forward(self, model, inputs, action_masks, old_log_probs, advantages, returns, old_values=None, value_head=None):
        """
        Executes reference and active passes on a single model and computes PPO losses.
        """
        is_peft = isinstance(model, PeftModel)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # 1. REFERENCE PASS: Disable adapters and run under torch.no_grad()
        if is_peft:
            with model.disable_adapter():
                with torch.no_grad():
                    ref_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True
                    )
                    ref_logits = ref_outputs.logits.detach()
        else:
            with torch.no_grad():
                ref_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True
                )
                ref_logits = ref_outputs.logits.detach()
                
        # Compute reference log probs over action tokens
        ref_log_probs, token_mask = self.get_log_probs(ref_logits, input_ids)
        ref_log_probs = ref_log_probs * action_masks[..., 1:]
        
        # 2. ACTOR PASS: Run with active adapters (and track gradients)
        actor_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True
        )
        actor_logits = actor_outputs.logits
        
        # Compute active policy log probs
        actor_log_probs, _ = self.get_log_probs(actor_logits, input_ids)
        actor_log_probs = actor_log_probs * action_masks[..., 1:]
        
        # 3. PPO POLICY LOSS
        # Ratio of new policy log probs to old rollout policy log probs
        # old_log_probs represent log probabilities from the rollout phase (detached)
        ratios = torch.exp(actor_log_probs - old_log_probs[..., 1:])
        
        # Policy gradient surrogate objective
        surr1 = ratios * advantages[..., 1:]
        surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages[..., 1:]
        
        # Sum over sequence dimension and mean over batch dimension
        mask_float = (token_mask * action_masks[..., 1:]).float()
        policy_loss = -torch.sum(torch.min(surr1, surr2) * mask_float, dim=-1)
        # Normalize by active token count
        policy_loss = torch.mean(policy_loss / (torch.sum(mask_float, dim=-1) + 1e-8))
        
        # 4. KL DIVERGENCE PENALTY (Active vs Reference)
        # Compute token-wise KL: kl = log(pi_active) - log(pi_reference)
        kl_div = actor_log_probs - ref_log_probs
        kl_penalty = torch.sum(kl_div * mask_float, dim=-1)
        kl_penalty = torch.mean(kl_penalty / (torch.sum(mask_float, dim=-1) + 1e-8))
        
        # Combined objective: policy loss + KL penalty
        total_loss = policy_loss + self.kl_beta * kl_penalty
        
        # 5. VALUE LOSS (If Critic Value Head is provided)
        value_loss = torch.tensor(0.0, device=input_ids.device)
        if value_head is not None and old_values is not None:
            # We pass hidden states to the value head to get values
            # (Usually done by projecting hidden states of the transformer)
            hidden_states = actor_outputs.hidden_states[-1] if hasattr(actor_outputs, "hidden_states") and actor_outputs.hidden_states is not None else actor_logits
            values = value_head(hidden_states).squeeze(-1)
            values = values[..., :-1] * action_masks[..., 1:]
            
            # Value function clipping
            values_clipped = old_values[..., 1:] + torch.clamp(
                values - old_values[..., 1:], 
                -self.value_clip_range, 
                self.value_clip_range
            )
            
            vf_losses1 = (values - returns[..., 1:]) ** 2
            vf_losses2 = (values_clipped - returns[..., 1:]) ** 2
            vf_loss = 0.5 * torch.sum(torch.max(vf_losses1, vf_losses2) * mask_float, dim=-1)
            value_loss = torch.mean(vf_loss / (torch.sum(mask_float, dim=-1) + 1e-8))
            
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
