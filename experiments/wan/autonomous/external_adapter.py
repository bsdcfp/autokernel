"""Interface-only adapter for an immutable ModelNew snapshot beside this file."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('native_autokernel_candidate', Path(__file__).with_name('kernel.py'))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_model = _module.ModelNew().eval().cuda()


def run(x, weight):
    return _model(x, weight)
