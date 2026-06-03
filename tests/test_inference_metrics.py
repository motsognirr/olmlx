"""Wiring guards for inference -> metrics instrumentation (#371).

The inference seams call ``_metrics.observe_inference(...)`` via the module
attribute so it stays monkeypatchable, and they read the surface from
``surface_var``. Full end-to-end generation is covered by the existing inference
suite; here we assert the wiring is present and references the live module.
"""

import inspect

from olmlx.engine import inference
from olmlx.utils import metrics


def test_inference_uses_live_metrics_module():
    # The seam must reference the module (inference._metrics is metrics) so that
    # monkeypatching metrics.observe_inference is observed at call time.
    assert inference._metrics is metrics


def test_seams_call_observe_inference():
    src = inspect.getsource(inference)
    # Guard that both success seams (stream done-chunk, non-stream return) and at
    # least one error seam are wired, without pinning an exact call count (which
    # is brittle as seams are added). Behavioural coverage of recording lives in
    # the inference suite; this just asserts the wiring exists and reads surface.
    assert src.count("_metrics.observe_inference(") >= 2
    assert "surface_var.get()" in src
    assert "error=True" in src
