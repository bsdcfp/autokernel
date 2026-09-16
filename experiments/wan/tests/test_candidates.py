"""Prevent execution of a replaced candidate or a candidate for the wrong task."""
import json
from pathlib import Path
import tempfile
import unittest

from preflight import resolve_candidate
from wanbench.core import digest


class CandidateProvenanceTests(unittest.TestCase):
    def test_changed_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / 'kernel.py'
            candidate.write_text('def run(x): return x\n')
            (root / 'generation.json').write_text(json.dumps({'task': 'wan-rmsnorm', 'candidate_sha256': digest(candidate)}))
            mapping = {'wan-rmsnorm': 'kernel.py'}
            self.assertEqual(resolve_candidate(root, mapping, 'wan-rmsnorm'), candidate.resolve())
            candidate.write_text('def run(x): return x + 1\n')
            with self.assertRaises(ValueError):
                resolve_candidate(root, mapping, 'wan-rmsnorm')

    def test_wrong_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / 'kernel.py'
            candidate.write_text('def run(x): return x\n')
            (root / 'generation.json').write_text(json.dumps({'task': 'wan-rope3d', 'candidate_sha256': digest(candidate)}))
            with self.assertRaises(ValueError):
                resolve_candidate(root, {'wan-rmsnorm': 'kernel.py'}, 'wan-rmsnorm')


if __name__ == '__main__':
    unittest.main()
