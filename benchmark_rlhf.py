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
from rlhf_dual_pass import step_ppo_dual_pass

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
    print(f"  Starting VRAM: {start_mem:.2f} MiB")
    
    # 1. Load Actor Model (Active with LoRA)
    print("  Loading Actor model with LoRA...")
    actor_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    actor = get_peft_model(actor_base, lora_config)
    actor.train()
    
    # 2. Load Reference Model (Frozen copy)
    print("  Loading separate Reference model (frozen)...")
    reference = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda().eval()
    
    # 3. Create Value Head (Critic)
    hidden_size = getattr(actor_base.config, "hidden_size", getattr(actor_base.config, "n_embd", 2048))
    value_head = nn.Linear(hidden_size, 1).cuda().to(torch.bfloat16).train()
    
    loaded_mem = get_gpu_memory_allocated()
    print(f"  VRAM after loading models and critic: {loaded_mem:.2f} MiB (Delta: {loaded_mem - start_mem:.2f} MiB)")
    
    # Define optimizer
    optimizer = torch.optim.AdamW(list(actor.parameters()) + list(value_head.parameters()), lr=1e-5)
    optimizer.zero_grad()
    
    # 4. Reference pass (on reference model)
    with torch.no_grad():
        ref_outputs = reference(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            output_hidden_states=True
        )
        ref_logits = ref_outputs.logits.detach()
        
    # Get reference log probabilities over action space
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    ref_log_probs_all = torch.log_softmax(shift_ref_logits, dim=-1)
    
    mask = (shift_labels != -100)
    gather_labels = shift_labels.clone()
    gather_labels[~mask] = 0
    ref_log_probs = torch.gather(ref_log_probs_all, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    ref_log_probs = ref_log_probs * mask.float() * ppo_inputs["action_masks"][..., 1:]
    
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
    values = values[..., :-1] * ppo_inputs["action_masks"][..., 1:]
    
    # Get actor log probabilities over action space
    shift_actor_logits = actor_logits[..., :-1, :].contiguous()
    actor_log_probs_all = torch.log_softmax(shift_actor_logits, dim=-1)
    actor_log_probs = torch.gather(actor_log_probs_all, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    actor_log_probs = actor_log_probs * mask.float() * ppo_inputs["action_masks"][..., 1:]
    
    # 6. Compute PPO losses
    # Ratios
    ratios = torch.exp(actor_log_probs - ppo_inputs["old_log_probs"][..., 1:])
    surr1 = ratios * ppo_inputs["advantages"][..., 1:]
    surr2 = torch.clamp(ratios, 0.8, 1.2) * ppo_inputs["advantages"][..., 1:]
    
    mask_float = (mask * ppo_inputs["action_masks"][..., 1:]).float()
    policy_loss = -torch.sum(torch.min(surr1, surr2) * mask_float, dim=-1)
    policy_loss = torch.mean(policy_loss / (torch.sum(mask_float, dim=-1) + 1e-8))
    
    kl_div = actor_log_probs - ref_log_probs
    kl_penalty = torch.sum(kl_div * mask_float, dim=-1)
    kl_penalty = torch.mean(kl_penalty / (torch.sum(mask_float, dim=-1) + 1e-8))
    
    # Value loss
    values_clipped = ppo_inputs["old_values"][..., 1:] + torch.clamp(
        values - ppo_inputs["old_values"][..., 1:], -0.2, 0.2
    )
    vf_losses1 = (values - ppo_inputs["returns"][..., 1:]) ** 2
    vf_losses2 = (values_clipped - ppo_inputs["returns"][..., 1:]) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_losses1, vf_losses2) * mask_float, dim=-1)
    value_loss = torch.mean(vf_loss / (torch.sum(mask_float, dim=-1) + 1e-8))
    
    total_loss = policy_loss + 0.01 * kl_penalty + value_loss
    
    # 7. Backward pass
    total_loss.backward()
    optimizer.step()
    
    peak_mem = get_peak_gpu_memory()
    print(f"  Peak VRAM during execution: {peak_mem:.2f} MiB")
    
    # Cleanup memory
    del reference, actor, actor_base, value_head, total_loss, policy_loss, kl_penalty, value_loss
    del ref_logits, actor_logits, values, ratios, surr1, surr2
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
    print(f"  Starting VRAM: {start_mem:.2f} MiB")
    
    # 1. Load Single Actor Model (Active with LoRA)
    print("  Loading single Actor model with LoRA...")
    actor_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    actor = get_peft_model(actor_base, lora_config)
    actor.train()
    
    # 2. Create Value Head (Critic)
    hidden_size = getattr(actor_base.config, "hidden_size", getattr(actor_base.config, "n_embd", 2048))
    value_head = nn.Linear(hidden_size, 1).cuda().to(torch.bfloat16).train()
    
    loaded_mem = get_gpu_memory_allocated()
    print(f"  VRAM after loading single model and critic: {loaded_mem:.2f} MiB (Delta: {loaded_mem - start_mem:.2f} MiB)")
    
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
    
    peak_mem = get_peak_gpu_memory()
    print(f"  Peak VRAM during execution: {peak_mem:.2f} MiB")
    
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
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Tokenizer load failed, using gpt2 tokenizer fallback: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Dummy sequence batch: batch_size=2, sequence_length=256
    dummy_text = ["RLHF alignment tuning involves updating policy models while measuring divergence from the reference policy."] * 2
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    batch_size, seq_len = inputs["input_ids"].shape
    
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
            print(f"  Standard PPO Peak:        CUDA OUT OF MEMORY (> 8,188 MiB)")
        else:
            print(f"  Standard PPO Peak:        {standard_vram:.2f} MiB")
        
    if dual_pass_vram is not None:
        if dual_pass_vram == float("inf"):
            print(f"  Single-Model PPO Peak:    CUDA OUT OF MEMORY (> 8,188 MiB)")
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
