# PPO VRAM Halver

**A small PyTorch example of memory-efficient PPO / RLHF with LoRA and PEFT.** It scores a frozen reference policy and trains the actor through the same base language model, avoiding a second copy of the base weights.

This is a standalone demonstration of a technique already used in Hugging Face TRL; see [prior work](#prior-work).

On a synthetic `Qwen/Qwen2.5-3B-Instruct` PPO step, the shared-base version used **49.3% less peak PyTorch CUDA tensor allocation** than a separate actor and reference on our RTX 4060 Ti test machine. See the [benchmark details](#benchmark) before treating that number as a hardware capacity claim.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bender1011001/ppo-vram-halver/blob/master/notebooks/RLHF_Dual_Pass_Colab.ipynb)

## How it works

The actor is the frozen base model plus a trainable LoRA adapter. The reference is that **same frozen base model** with the adapter disabled. Each PPO step runs two forward passes:

```text
One frozen base model
  ├─ adapter off · eval mode · no gradients → reference action log probabilities
  └─ adapter on  · train mode · gradients    → actor log probabilities and values
                                               ↓
                                       PPO and value losses
```

Only the selected-token reference log probabilities are kept for the actor pass. The full reference logits and hidden states are released first. This saves a copy of the base weights and some temporary memory; it does **not** remove the value head, reward model, rollout storage, or optimizer costs from a complete RLHF system.

## Benchmark

We reran the repository's two benchmark arms in separate processes on **Windows, RTX 4060 Ti 8GB, driver 610.88**, with Python 3.11, PyTorch `2.14.1+cu130`, Transformers `5.23.0`, and PEFT `0.21.3`. The model was `Qwen/Qwen2.5-3B-Instruct` in BF16 with LoRA rank 16 on `q_proj` and `v_proj`. The tokenizer produced a **2 × 16-token** batch; the script's `max_length=256` is only a cap. We used the same cached weights and library environment for both revisions.

| Code | Separate actor + reference | Shared base, dual pass | Reduction |
| --- | ---: | ---: | ---: |
| [Original `6308671`](https://github.com/Bender1011001/ppo-vram-halver/commit/6308671f0f538627e459dc658b3ce4259a819d5a) | 12,215.09 MiB | 6,185.26 MiB | 49.36% |
| Revised code | 12,174.53 MiB | 6,174.98 MiB | 49.28% |

The revised dual-pass arm used **10.28 MiB less** than the original in this setup (0.17%). The revised pair also passed `tests/verify_rlhf.py`; its repeated measurements matched the table. The main saving, about 6,000 MiB, was already present in the original design. These runs do not establish a meaningful speed or training-quality improvement.

[Full benchmark setup and reproduction commands](benchmarks/2026-11-15-rtx4060ti.md)

**What the metric means:** The script reports `torch.cuda.max_memory_allocated()` minus its starting allocation. That counts live tensors tracked by PyTorch, not physical dedicated VRAM or total process memory. The 12 GiB result on an 8GB Windows card is possible because [Windows can page graphics allocations beyond physical video memory](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/paging-video-memory-resources); [NVIDIA explains that WDDM can do this for CUDA allocations](https://forums.developer.nvidia.com/t/does-cudamalloc-increases-the-private-bytes-used-on-host/45797/8). Performance can change sharply when paging occurs. The original README described its baseline as both 12,272.98 MiB and an OOM; no raw log for that historical run is in the repository, so the fresh comparison above supersedes it.

The benchmark uses random placeholder rollout values and a short synthetic batch. It demonstrates memory allocation for one update step. It does not test real reward modeling, generation, training convergence, or maximum model size that fits fully in dedicated VRAM.

## Run it

1. Install a CUDA-enabled PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/) if you want the GPU benchmark. CPU PyTorch is enough for the unit checks.
2. Clone the repository and install the remaining dependencies:

   ```bash
   git clone https://github.com/Bender1011001/ppo-vram-halver.git
   cd ppo-vram-halver
   python -m pip install transformers peft accelerate
   ```

3. Check the implementation, then run the memory comparison:

   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   python -c "import torch; print(torch.cuda.is_available())"  # must be True for the next command
   python tests/verify_rlhf.py
   ```

The GPU benchmark downloads roughly 6GB of Qwen weights on first use. It runs the separate-reference and shared-base arms in fresh processes. A confirmed CUDA OOM in the separate-reference arm is reported as such; an unrelated crash fails validation. When both arms complete, the verifier checks for at least 35% lower peak allocation in the shared-base arm. That threshold is specific to this demonstration.

## Use in a custom PPO loop

Create a PEFT model whose base weights stay frozen and a value head with the same device and dtype as the model. Collect `old_log_probs`, `old_values`, `advantages`, and `returns` from a real rollout **before** the update; the benchmark's random tensors are not valid training data.

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from rlhf_dual_pass import step_ppo_dual_pass

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", dtype=torch.bfloat16
).cuda()
model = get_peft_model(base, LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
))
weight = next(model.parameters())
value_head = torch.nn.Linear(model.config.hidden_size, 1).to(
    device=weight.device, dtype=weight.dtype
)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad] + list(value_head.parameters()),
    lr=1e-5,
)

# Each rollout tensor has shape [batch, sequence]. action_masks is 1 only on
# generated action tokens. inputs contains input_ids and, if padded, attention_mask.
loss, policy_loss, reference_penalty, value_loss = step_ppo_dual_pass(
    model=model, inputs=inputs, action_masks=action_masks,
    old_log_probs=old_log_probs, advantages=advantages, returns=returns,
    old_values=old_values, value_head=value_head, optimizer=optimizer,
)
```

`model.disable_adapter()` must be a context manager. The reference pass temporarily uses eval mode and no gradients, then restores training mode for the actor. The reference policy must equal the frozen base model; this method is invalid if the reference has other trained weights or the base weights change.

The `reference_penalty` is the sampled action log-ratio `log π_actor(a|s) − log π_ref(a|s)`. It can be negative and is **not** the exact distributional KL when the actions came from an older rollout policy. This helper is a single-step example, not a full PPO trainer.

## Prior work

The shared-base idea predates this repository. The [LoRA paper](https://arxiv.org/abs/2106.09685) introduced frozen base weights with trainable low-rank adapters; [PEFT provides `disable_adapter()`](https://huggingface.co/docs/peft/main/package_reference/peft_model) for base-model inference. Hugging Face [TRL PPOTrainer v0.11.4](https://github.com/huggingface/trl/blob/v0.11.4/trl/trainer/ppo_trainer.py#L249-L262) already omitted a separate reference model for PEFT models and [ran reference scoring with adapters disabled](https://github.com/huggingface/trl/blob/v0.11.4/trl/trainer/ppo_trainer.py#L745-L763). This repo is a compact implementation and benchmark of that memory layout, not a new PPO algorithm.

## Repository map

| Path | Purpose |
| --- | --- |
| [`rlhf_dual_pass.py`](rlhf_dual_pass.py) | Single-model PPO loss and update step |
| [`benchmark_rlhf.py`](benchmark_rlhf.py) | Synthetic GPU memory comparison |
| [`benchmarks/2026-11-15-rtx4060ti.md`](benchmarks/2026-11-15-rtx4060ti.md) | Measured setup and original-versus-revised results |
| [`tests/test_dual_pass.py`](tests/test_dual_pass.py) | CPU correctness and PEFT integration checks |
| [`tests/verify_rlhf.py`](tests/verify_rlhf.py) | Two-process GPU benchmark verifier |
| [`notebooks/RLHF_Dual_Pass_Colab.ipynb`](notebooks/RLHF_Dual_Pass_Colab.ipynb) | Colab companion notebook |
