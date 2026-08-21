"""High-value correctness tests for the byte-level Stream inference contract."""

import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # local docs/static environments
    raise unittest.SkipTest('PyTorch is required for Stream numerical tests') from exc

from model import SSMBlock, Stream, StreamConfig


class StreamContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = StreamConfig(n_embd=16, n_layer=2, ssm_d_state=4,
                                   n_predict=4, block_size=16)
        self.model = Stream(self.config).eval()
        self.idx = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)

    def test_block_step_matches_causal_full_forward(self):
        block = SSMBlock(n_embd=16, ssm_d_state=4).eval()
        x = torch.randn(2, 6, 16)
        full, _ = block(x)
        state = None
        stepped = []
        for t in range(x.shape[1]):
            out, state = block.step(x[:, t], state)
            stepped.append(out)
        self.assertTrue(torch.allclose(full, torch.stack(stepped, dim=1), atol=2e-5, rtol=2e-5))

    def test_prefill_then_step_matches_full_model(self):
        full, _ = self.model(self.idx)
        full = full.view(1, self.idx.shape[1], self.config.n_predict, 256)
        logits, state = self.model.prefill(self.idx[:, :-1])
        self.assertTrue(torch.allclose(logits, full[:, -2, 0], atol=2e-5, rtol=2e-5))
        stepped, _ = self.model.step(self.idx[:, -1], state)
        self.assertTrue(torch.allclose(stepped, full[:, -1, 0], atol=2e-5, rtol=2e-5))

    def test_generation_is_byte_autoregressive_and_restores_mode(self):
        self.model.train()
        torch.manual_seed(3)
        out = self.model.generate(self.idx[:, :2], max_new_tokens=5, top_k=8)
        self.assertEqual(tuple(out.shape), (1, 7))
        self.assertTrue(self.model.training)

    def test_loss_has_finite_gradients(self):
        self.model.train()
        targets = torch.randint(0, 256, self.idx.shape)
        _, loss = self.model(self.idx, targets=targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))

    def test_activation_checkpointing_preserves_logits(self):
        checkpointed_cfg = StreamConfig(n_embd=16, n_layer=2, ssm_d_state=4,
                                        n_predict=4, block_size=16,
                                        activation_checkpointing=True)
        checkpointed = Stream(checkpointed_cfg).train()
        checkpointed.load_state_dict(self.model.state_dict())
        reference = self.model.train()
        logits_ref, loss_ref = reference(self.idx, targets=self.idx)
        logits_ckpt, loss_ckpt = checkpointed(self.idx, targets=self.idx)
        self.assertTrue(torch.allclose(logits_ref, logits_ckpt, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(loss_ref, loss_ckpt, atol=2e-6, rtol=2e-6))

    def test_non_ssm_stream_rejects_unverified_stateful_inference(self):
        experimental = Stream(StreamConfig(n_embd=16, n_layer=2, ssm_d_state=4,
                                           block_size=16, n_retrieval=1,
                                           n_attn_head=4))
        with self.assertRaises(NotImplementedError):
            experimental.prefill(self.idx)


if __name__ == '__main__':
    unittest.main()
