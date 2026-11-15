#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory Profiling Benchmark: Standard PPO RLHF vs Single-Model Dual-Pass PPO.
Measures and compares peak GPU VRAM allocation during RLHF training steps.
"""

import gc
import sys
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from rlhf_dual_pass import SingleModelPPOLoss, step_ppo_dual_pass

def get_gpu_memory_allocated():
    """Returns currently allocated GPU memory in MiB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0

def get_peak_gpu_memory():
    """Returns peak allocated GPU memory in MiB and resets stats."""
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
        return mem
    return 0.0

def run_standard_ppo_step(model_name, lora_config, inputs, ppo_inputs):
    """Simulates standard PPO step (loading separate Actor and Reference models)."""
    print("\n--- Running Standard PPO RLHF Step ---")
    gc.collect()
    torch.cuda.empty_cache()
    get_peak_gpu_memory()  # Reset peak tracker
    
    start_mem = get_gpu_memory_allocated()
    print(f"  Starting PyTorch CUDA allocation: {start_mem:.2f} MiB")
    
    # 1. Load Actor Model (Active with LoRA)
    print("  Loading Actor model with LoRA...")
    actor_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).cuda()
    actor = get_peft_model(actor_base, lora_config)
    actor.train()
    
    # 2. Load Reference Model (Frozen copy)
    print("  Loading separate Reference model (frozen)...")
    reference = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).cuda().eval()
    
    # 3. Create Value Head (Critic)
    hidden_size = getattr(actor_base.config, "hidden_size", getattr(actor_base.config, "n_embd", 2048))
    value_head = nn.Linear(hidden_size, 1).cuda().to(torch.bfloat16).train()
    
    loaded_mem = get_gpu_memory_allocated()
    print(f"  Allocation after loading models and critic: {loaded_mem:.2f} MiB (Delta: {loaded_mem - start_mem:.2f} MiB)")
    
    # Define optimizer
    optimizer = torch.optim.AdamW(list(actor.parameters()) + list(value_head.parameters()), lr=1e-5)
    optimizer.zero_grad()
    
    # 4. Reference pass (on reference model)
    loss_fn = SingleModelPPOLoss()
    with torch.no_grad():
        ref_outputs = reference(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            output_hidden_states=False
        )
        ref_log_probs, _ = loss_fn.get_log_probs(ref_outputs.logits, inputs["input_ids"])
    del ref_outputs
    
    # 5. Actor pass (on actor model)
    actor_outputs = actor(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=False,
        output_hidden_states=True
    )
    actor_logits = actor_outputs.logits
    
    # Value function hidden states
    hidden_states = actor_outputs.hidden_states[-1]
    values = value_head(hidden_states).squeeze(-1)
    
    # Get actor log probabilities over action space
    actor_log_probs, token_mask = loss_fn.get_log_probs(actor_logits, inputs["input_ids"])
    active = token_mask & ppo_inputs["action_masks"][..., 1:].bool() & inputs["attention_mask"][..., 1:].bool()
    mask_float = active.float()
    token_counts = mask_float.sum(dim=-1).clamp_min(1)
    
    # 6. Compute PPO losses
    # Ratios
    log_ratios = torch.where(active, actor_log_probs - ppo_inputs["old_log_probs"][..., 1:], 0.0)
    ratios = torch.exp(log_ratios)
    masked_advantages = torch.where(active, ppo_inputs["advantages"][..., 1:], 0.0)
    surr1 = ratios * masked_advantages
    surr2 = torch.clamp(ratios, 0.8, 1.2) * masked_advantages
    
    policy_loss = -torch.sum(torch.min(surr1, surr2) * mask_float, dim=-1)
    policy_loss = torch.mean(policy_loss / token_counts)
    
    kl_div = torch.where(active, actor_log_probs - ref_log_probs, 0.0)
    kl_penalty = torch.sum(kl_div * mask_float, dim=-1)
    kl_penalty = torch.mean(kl_penalty / token_counts)
    
    # Value loss
    values = torch.where(active, values[..., :-1], 0.0)
    prior_values = torch.where(active, ppo_inputs["old_values"][..., 1:], 0.0)
    targets = torch.where(active, ppo_inputs["returns"][..., 1:], 0.0)
    values_clipped = prior_values + torch.clamp(
        values - prior_values, -0.2, 0.2
    )
    vf_losses1 = (values - targets) ** 2
    vf_losses2 = (values_clipped - targets) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_losses1, vf_losses2) * mask_float, dim=-1)
    value_loss = torch.mean(vf_loss / token_counts)
    
    total_loss = policy_loss + 0.01 * kl_penalty + value_loss
    
    # 7. Backward pass
    total_loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    peak_mem = get_peak_gpu_memory()
    print(f"  Peak PyTorch CUDA allocation: {peak_mem:.2f} MiB")
    
    # Cleanup memory
    del reference, actor, actor_base, value_head, total_loss, policy_loss, kl_penalty, value_loss
    del actor_logits, values, ratios, surr1, surr2
    gc.collect()
    torch.cuda.empty_cache()
    
    return peak_mem - start_mem

def run_single_model_ppo_step(model_name, lora_config, inputs, ppo_inputs):
    """Simulates Single-Model Dual-Pass PPO step."""
    print("\n--- Running Single-Model PPO RLHF Step ---")
    gc.collect()
    torch.cuda.empty_cache()
    get_peak_gpu_memory()  # Reset peak tracker
    
    start_mem = get_gpu_memory_allocated()
    print(f"  Starting PyTorch CUDA allocation: {start_mem:.2f} MiB")
    
    # 1. Load Single Actor Model (Active with LoRA)
    print("  Loading single Actor model with LoRA...")
    actor_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).cuda()
    actor = get_peft_model(actor_base, lora_config)
    actor.train()
    
    # 2. Create Value Head (Critic)
    hidden_size = getattr(actor_base.config, "hidden_size", getattr(actor_base.config, "n_embd", 2048))
    value_head = nn.Linear(hidden_size, 1).cuda().to(torch.bfloat16).train()
    
    loaded_mem = get_gpu_memory_allocated()
    print(f"  Allocation after loading single model and critic: {loaded_mem:.2f} MiB (Delta: {loaded_mem - start_mem:.2f} MiB)")
    
    # Define optimizer
    optimizer = torch.optim.AdamW(list(actor.parameters()) + list(value_head.parameters()), lr=1e-5)
    
    # 3. Perform dual-pass step
    print("  Executing Dual-Pass step...")
    loss, p_loss, kl_pen, val_loss = step_ppo_dual_pass(
        model=actor,
        inputs=inputs,
        action_masks=ppo_inputs["action_masks"],
        old_log_probs=ppo_inputs["old_log_probs"],
        advantages=ppo_inputs["advantages"],
        returns=ppo_inputs["returns"],
        optimizer=optimizer,
        old_values=ppo_inputs["old_values"],
        value_head=value_head,
        kl_beta=0.01,
        clip_range=0.2
    )
    
    torch.cuda.synchronize()
    peak_mem = get_peak_gpu_memory()
    print(f"  Peak PyTorch CUDA allocation: {peak_mem:.2f} MiB")
    
    # Cleanup memory
    del actor, actor_base, value_head, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return peak_mem - start_mem

def main():
    parser = argparse.ArgumentParser(description="RLHF PPO VRAM Memory Benchmarking")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model name to benchmark"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["both", "standard", "dual_pass"],
        default="both",
        help="Which benchmark step to run"
    )
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is not available. This benchmark requires a GPU.")
        sys.exit(1)
        
    print(f"============================================================")
    print(f" RLHF PPO VRAM Memory Benchmark on {torch.cuda.get_device_name(0)}")
    print(f" Target Model: {args.model_name}")
    print(f" Benchmark Step: {args.step}")
    print(f"============================================================")
    
    # Initialize tokenizer and inputs
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Synthetic batch; max_length is a cap, not a fixed sequence length.
    torch.manual_seed(0)
    dummy_text = ["RLHF alignment tuning involves updating policy models while measuring divergence from the reference policy."] * 2
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    batch_size, seq_len = inputs["input_ids"].shape
    print(f" Tokenized batch: {batch_size} x {seq_len} tokens")
    
    # Setup dummy PPO trajectory states
    action_masks = torch.zeros((batch_size, seq_len), device="cuda")
    action_masks[:, seq_len // 2:] = 1.0
    
    ppo_inputs = {
        "action_masks": action_masks,
        "old_log_probs": torch.randn((batch_size, seq_len), device="cuda") * 0.1 - 2.0,
        "advantages": torch.randn((batch_size, seq_len), device="cuda"),
        "returns": torch.randn((batch_size, seq_len), device="cuda"),
        "old_values": torch.randn((batch_size, seq_len), device="cuda"),
    }
    
    # LoRA config
    target_modules = ["q_proj", "v_proj"]
    if "gpt2" in args.model_name.lower():
        target_modules = ["c_attn"]
        
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    standard_vram = None
    dual_pass_vram = None
    
    # Run standard PPO benchmark
    if args.step in ["both", "standard"]:
        try:
            standard_vram = run_standard_ppo_step(args.model_name, lora_config, inputs, ppo_inputs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n[-] OUT OF MEMORY: Standard PPO RLHF exceeded VRAM limits!")
                standard_vram = float("inf")
            else:
                raise e
            
    # Run Single-Model PPO benchmark
    if args.step in ["both", "dual_pass"]:
        try:
            dual_pass_vram = run_single_model_ppo_step(args.model_name, lora_config, inputs, ppo_inputs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n[-] OUT OF MEMORY: Single-Model Dual-Pass PPO exceeded VRAM limits!")
                dual_pass_vram = float("inf")
            else:
                raise e
            
    # Print results
    print(f"\n============================================================")
    print(f" BENCHMARK RESULTS")
    print(f"============================================================")
    if standard_vram is not None:
        if standard_vram == float("inf"):
            print("  Standard PPO Peak:        CUDA OUT OF MEMORY")
        else:
            print(f"  Standard PPO Peak:        {standard_vram:.2f} MiB")
        
    if dual_pass_vram is not None:
        if dual_pass_vram == float("inf"):
            print("  Single-Model PPO Peak:    CUDA OUT OF MEMORY")
        else:
            print(f"  Single-Model PPO Peak:    {dual_pass_vram:.2f} MiB")
        
    if standard_vram is not None and dual_pass_vram is not None:
        if standard_vram != float("inf") and dual_pass_vram != float("inf"):
            savings = standard_vram - dual_pass_vram
            pct_savings = (savings / standard_vram) * 100.0
            print(f"  VRAM Savings:             {savings:.2f} MiB ({pct_savings:.1f}% reduction)")
    print(f"============================================================")
    
if __name__ == "__main__":
    main()
