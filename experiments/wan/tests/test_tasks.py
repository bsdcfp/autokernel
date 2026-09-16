"""CPU reference tests. Skipped locally when torch is absent; run on target Python."""
import importlib.util
import unittest

from wanbench.tasks import TASK_IDS, build_reference, make_inputs, flatten_outputs


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class TaskReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t = torch

    def test_all_six_interfaces_and_output_shapes(self):
        t = self.t
        for task in TASK_IDS:
            with self.subTest(task=task):
                args = make_inputs(t, task, "debug", 17, t.bfloat16, "cpu")
                before = tuple(x.clone() for x in args)
                output = build_reference(t, task)(*args)
                self.assertTrue(all(t.equal(x, y) for x, y in zip(args, before)))
                for x in flatten_outputs(t, output):
                    self.assertTrue(t.isfinite(x).all().item())
                    self.assertEqual(x.shape[0], 1)
                    self.assertEqual(x.shape[1], 256)
                if task == "t2-qknorm-rope":
                    self.assertEqual(len(output), 2)
                    self.assertEqual(output[0].shape, (1, 256, 12, 128))
                elif task == "wan-linear-gelu":
                    self.assertEqual(output.shape, (1, 256, 8960))

    def test_constant_layernorm_returns_shift(self):
        t = self.t
        x = t.ones((1, 3, 1536), dtype=t.bfloat16)
        shift = t.arange(1536).float().reshape(1, 1, -1) / 1536
        out = build_reference(t, "t1-norm-modulation")(x, t.ones_like(shift), shift)
        t.testing.assert_close(out, shift.expand_as(out), rtol=0, atol=0)

    def test_layernorm_nonzero_mean_has_correct_variance(self):
        t = self.t
        x = t.cat([t.ones(768), t.full((768,), 3)]).to(t.bfloat16).reshape(1, 1, 1536)
        zeros = t.zeros((1, 1, 1536), dtype=t.float32)
        out = build_reference(t, "t1-norm-modulation")(x, zeros, zeros)
        expected = t.cat([-t.ones(768), t.ones(768)]).reshape_as(out)
        # FP32 layernorm values round to exactly +/-1 in the required BF16 cast.
        t.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_rope_zero_position_and_padding_are_unchanged(self):
        t = self.t
        x, grid, freqs = make_inputs(t, "wan-rope3d", "debug", 9, t.bfloat16, "cpu")
        grid = t.tensor([[1, 1, 1]])
        out = build_reference(t, "wan-rope3d")(x, grid, freqs)
        t.testing.assert_close(out, x.float(), rtol=0, atol=0)

    def test_rope_preserves_pair_norm(self):
        t = self.t
        args = make_inputs(t, "wan-rope3d", "debug", 9, t.bfloat16, "cpu")
        out = build_reference(t, "wan-rope3d")(*args)
        before = args[0].float().reshape(1, 256, 12, 64, 2).square().sum(-1)
        after = out.reshape(1, 256, 12, 64, 2).square().sum(-1)
        t.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)

    def test_rmsnorm_reduces_across_hidden_not_head(self):
        t = self.t
        x = t.ones((1, 1, 1536), dtype=t.float32)
        x[..., :128] = 2
        out = build_reference(t, "wan-rmsnorm")(x, t.ones(1536))
        # Global denominator: (128*4 + 1408*1)/1536 = 1.25.
        t.testing.assert_close(out[..., 0], t.tensor([[2 / (1.25 + 1e-6) ** .5]]))
        t.testing.assert_close(out[..., 128], t.tensor([[1 / (1.25 + 1e-6) ** .5]]))


if __name__ == "__main__":
    unittest.main()
