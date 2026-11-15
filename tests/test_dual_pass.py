import unittest
from contextlib import contextmanager
from subprocess import CompletedProcess
from types import SimpleNamespace

import torch
from torch import nn

from rlhf_dual_pass import SingleModelPPOLoss
from verify_rlhf import parse_benchmark_result


class ToyAdapterModel(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 8, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.adapter = nn.Parameter(torch.zeros(8))
        self.embedding.requires_grad_(False)
        self.projection.requires_grad_(False)
        self.adapter_enabled = True
        self.calls = []

    @contextmanager
    def disable_adapter(self):
        previous = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = previous

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False):
        self.calls.append((self.training, self.adapter_enabled, torch.is_grad_enabled(), output_hidden_states))
        hidden = self.dropout(self.embedding(input_ids))
        logits = self.projection(hidden)
        if self.adapter_enabled:
            logits = logits + self.adapter
        return SimpleNamespace(logits=logits, hidden_states=(hidden,) if output_hidden_states else None)


class DualPassTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = ToyAdapterModel()
        self.inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        }
        self.masks = torch.ones((1, 4))

    def test_reference_is_deterministic_and_restores_training_mode(self):
        loss_fn = SingleModelPPOLoss()
        self.model.train()
        first = loss_fn.reference_log_probs(self.model, **self.inputs)
        second = loss_fn.reference_log_probs(self.model, **self.inputs)
        torch.testing.assert_close(first, second)
        self.assertEqual(self.model.calls, [(False, False, False, False)] * 2)
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.adapter_enabled)

    def test_masked_rollout_values_do_not_poison_loss_or_gradients(self):
        loss_fn = SingleModelPPOLoss()
        value_head = nn.Linear(4, 1)
        loss, *_ = loss_fn(
            model=self.model,
            inputs=self.inputs,
            action_masks=self.masks,
            old_log_probs=torch.tensor([[0.0, -2.0, -2.0, -1e9]]),
            advantages=torch.tensor([[0.0, 1.0, 1.0, float("nan")]]),
            returns=torch.tensor([[0.0, 0.0, 0.0, float("nan")]]),
            old_values=torch.tensor([[0.0, 0.0, 0.0, float("nan")]]),
            value_head=value_head,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.adapter.grad).all())
        self.assertTrue(torch.isfinite(value_head.weight.grad).all())
        self.assertEqual(self.model.calls[0], (False, False, False, False))
        self.assertEqual(self.model.calls[1], (True, True, True, True))

    def test_missing_adapter_and_unpaired_value_inputs_are_rejected(self):
        loss_fn = SingleModelPPOLoss()
        with self.assertRaisesRegex(TypeError, "disable_adapter"):
            loss_fn.reference_log_probs(nn.Linear(4, 8), self.inputs["input_ids"])
        with self.assertRaisesRegex(ValueError, "provided together"):
            loss_fn(
                model=self.model,
                inputs=self.inputs,
                action_masks=self.masks,
                old_log_probs=torch.zeros((1, 4)),
                advantages=torch.zeros((1, 4)),
                returns=torch.zeros((1, 4)),
                value_head=nn.Linear(4, 1),
            )

    def test_real_peft_adapter_matches_frozen_base(self):
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            self.skipTest("peft and transformers are required for the integration test")

        config = GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=0,
            eos_token_id=1,
            resid_pdrop=0.5,
            embd_pdrop=0.5,
            attn_pdrop=0.5,
        )
        model = get_peft_model(
            GPT2LMHeadModel(config),
            LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], fan_in_fan_out=True, task_type="CAUSAL_LM"),
        )
        tokens = torch.tensor([[1, 2, 3, 4]])
        loss_fn = SingleModelPPOLoss()
        model.eval()
        with model.disable_adapter(), torch.no_grad():
            outputs = model(input_ids=tokens, use_cache=False, output_hidden_states=False)
            expected, _ = loss_fn.get_log_probs(outputs.logits, tokens)
        model.train()
        actual = loss_fn.reference_log_probs(model, tokens)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(model.training)

        loss, *_ = loss_fn(
            model=model,
            inputs={"input_ids": tokens},
            action_masks=torch.tensor([[0, 1, 1, 1]]),
            old_log_probs=torch.zeros((1, 4)),
            advantages=torch.ones((1, 4)),
            returns=torch.zeros((1, 4)),
        )
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.parameters() if parameter.requires_grad
        ))


class BenchmarkVerifierTests(unittest.TestCase):
    def test_unrelated_baseline_failure_is_not_counted_as_oom(self):
        result = CompletedProcess([], 1, "", "ModuleNotFoundError: No module named 'peft'")
        with self.assertRaises(RuntimeError):
            parse_benchmark_result(result, "Standard PPO")

    def test_cuda_oom_and_success_are_distinguished(self):
        oom = CompletedProcess([], 0, "Standard PPO Peak: CUDA OUT OF MEMORY (> 8,188 MiB)", "")
        self.assertIsNone(parse_benchmark_result(oom, "Standard PPO"))
        success = CompletedProcess([], 0, "Standard PPO Peak: 1234.50 MiB", "")
        self.assertEqual(parse_benchmark_result(success, "Standard PPO"), 1234.5)


if __name__ == "__main__":
    unittest.main()
