import copy
import tempfile
import unittest
from pathlib import Path

from wanbench.core import (build_plan, compare_smoke, digest, dump_new,
                           load_json, object_digest, safe_id, verify_sources,
                           validate_device_selection)
from wanbench.cli import ROOT, prepare


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.c = load_json(ROOT / "configs/pilot.json")
        self.fs = {x["id"]: x for x in load_json(ROOT / "configs/frameworks.lock.json")["frameworks"]}
        self.ts = {x["id"]: x for x in load_json(ROOT / "configs/tasks.json")["tasks"]}

    def test_full_matrix_and_no_fabricated_results(self):
        p = build_plan(self.c, self.fs, self.ts)
        self.assertEqual(len(p["trials"]), 15)
        self.assertFalse(p["execution_enabled"])
        self.assertEqual(len({t["id"] for t in p["trials"]}), 15)
        self.assertTrue(all(t["latency_ms"] is None and t["status"] == "planned" for t in p["trials"]))
        for repeat in (1, 2, 3):
            self.assertEqual({t["framework"] for t in p["trials"] if t["repeat"] == repeat}, set(self.fs))

    def test_order_is_reproducible(self):
        self.assertEqual(build_plan(self.c, self.fs, self.ts), build_plan(self.c, self.fs, self.ts))

    def test_unknown_duplicate_and_invalid_budgets_rejected(self):
        for patch in [{"frameworks": ["unknown"]}, {"tasks": ["t1-norm-modulation"] * 2},
                      {"wall_seconds": 0}, {"token_limit": True}, {"repeats": -1},
                      {"status": "frozen"}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                build_plan(dict(self.c, **patch), self.fs, self.ts)

    def test_workspace_empty_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            prepare(d, "autokernel", "t1-norm-modulation", "trial-one")
            self.assertEqual(list((Path(d) / "trial-one/candidate").iterdir()), [])
            with self.assertRaises(FileExistsError):
                prepare(d, "autokernel", "t1-norm-modulation", "trial-one")

    def test_path_traversal_rejected(self):
        for value in ["../escape", "/absolute", "UPPER", "x/y", "", "x;touch y"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_id(value)


class IntegrityTests(unittest.TestCase):
    def test_only_selected_physical_gpu_is_visible(self):
        selection = {"uuid": "GPU-test"}
        validate_device_selection(selection, "GPU-test", 0)
        for visible, logical in [(None, 0), ("0", 0), ("3", 0), ("GPU-other", 0),
                                 ("GPU-test,GPU-other", 0), ("GPU-test", 3)]:
            with self.subTest(visible=visible, logical=logical), self.assertRaises(ValueError):
                validate_device_selection(selection, visible, logical)

    def test_tamper_and_missing_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "file"; p.write_text("original")
            manifest = {"files": [{"path": "file", "sha256": digest(p)}]}
            self.assertTrue(verify_sources(d, manifest)["ok"])
            p.write_text("changed")
            self.assertEqual(verify_sources(d, manifest)["files"][0]["status"], "changed")
            p.unlink()
            self.assertEqual(verify_sources(d, manifest)["files"][0]["status"], "missing")

    def test_manifest_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            verify_sources(d, {"files": [{"path": "../outside", "sha256": "0" * 64}]})

    def test_artifact_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "result.json"; dump_new(p, {"x": 1})
            with self.assertRaises(FileExistsError):
                dump_new(p, {"x": 2})
            self.assertEqual(load_json(p), {"x": 1})

    def test_digest_stable_across_key_order(self):
        self.assertEqual(object_digest({"a": 1, "b": 2}), object_digest({"b": 2, "a": 1}))


class ComparisonTests(unittest.TestCase):
    # These are synthetic unit-test values, never written to experiment results.
    def setUp(self):
        self.b = {"kind": "exploratory-gpu", "profile": "none", "status": "pass",
                  "correctness": {"passed": True}, "samples_ms": [2., 2., 2.],
                  "variant": "compile-default", "task_hash": "a", "fixture_hash": "b",
                  "environment_hash": "c", "measurement_hash": "d"}
        self.c = copy.deepcopy(self.b)
        self.c.update(variant="candidate", candidate_sha256="a" * 64, samples_ms=[1., 1., 1.])

    def test_exploratory_only(self):
        r = compare_smoke(self.b, self.c)
        self.assertEqual(r["speedup"], 2.)
        self.assertEqual(r["status"], "exploratory-only")
        self.assertIsNone(r["whole_block_speedup"])

    def test_wrong_profile_synthetic_or_failed_rejected(self):
        for patch in [{"profile": "torch"}, {"kind": "synthetic"}, {"status": "fail"},
                      {"correctness": {"passed": False}}, {"correctness": {"passed": "true"}}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                compare_smoke(self.b, dict(self.c, **patch))

    def test_no_wrong_denominator(self):
        for field in ["task_hash", "fixture_hash", "environment_hash", "measurement_hash"]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_smoke(self.b, dict(self.c, **{field: "different"}))
        with self.assertRaises(ValueError):
            compare_smoke(dict(self.b, variant="eager"), self.c)

    def test_bad_timings_rejected(self):
        for samples in [[0, 1], [-1, 1], [float("nan"), 1], [float("inf"), 1],
                        [True, 1], [1], [], ["1", "2"]]:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                compare_smoke(self.b, dict(self.c, samples_ms=samples))

    def test_source_provenance_required(self):
        with self.assertRaises(ValueError):
            compare_smoke(self.b, dict(self.c, candidate_sha256=None))


if __name__ == "__main__":
    unittest.main()
