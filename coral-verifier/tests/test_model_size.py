"""A model from model_url runs at its own input size (e.g. 416): EdgeTpuYolo(path, None) reads it from the model's input. No Coral: a stand-in
tflite_runtime module provides the interpreter."""
import sys, types
import numpy as np
import pytest

import yolo_edgetpu as Y


def fake_runtime(monkeypatch, in_shape, out_shapes):
    class Interp:
        def __init__(self, model_path=None, experimental_delegates=None, num_threads=1):
            pass

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{"shape": np.array(in_shape), "dtype": np.uint8, "quantization": (1 / 255, 0), "index": 0, "name": "images"}]

        def get_output_details(self):
            return [{"shape": np.array(s), "dtype": np.int8, "quantization": (0.05 + 0.01 * i, -10 + i), "index": 10 + i, "name": f"o{i}"}
                    for i, s in enumerate(out_shapes)]
    mod = types.ModuleType("tflite_runtime"); sub = types.ModuleType("tflite_runtime.interpreter")
    sub.Interpreter, sub.load_delegate = Interp, (lambda *a, **k: object())
    mod.interpreter = sub
    monkeypatch.setitem(sys.modules, "tflite_runtime", mod); monkeypatch.setitem(sys.modules, "tflite_runtime.interpreter", sub)


def split_shapes(size, nc=17):
    return [[1, size // s, size // s, c] for s in Y.STRIDES for c in (64, nc)]


def test_size_comes_from_the_model(monkeypatch):
    fake_runtime(monkeypatch, [1, 416, 416, 3], split_shapes(416))
    m = Y.EdgeTpuYolo("x.tflite", None)
    assert m.size == 416 and m.split is not None and m.dec.n == Y.n_anchors(416)


def test_a_given_size_must_match(monkeypatch):
    fake_runtime(monkeypatch, [1, 416, 416, 3], split_shapes(416))
    with pytest.raises(ValueError):
        Y.EdgeTpuYolo("x.tflite", 512)                      # the built-in path still checks the option against the model


@pytest.mark.parametrize("shape", [[1, 416, 384, 3], [1, 420, 420, 3]])
def test_odd_inputs_are_refused(monkeypatch, shape):
    fake_runtime(monkeypatch, shape, split_shapes(416))
    with pytest.raises(ValueError, match="square and a multiple of 32"):
        Y.EdgeTpuYolo("x.tflite", None)


def test_yolo26_style_model_loads(monkeypatch):
    fake_runtime(monkeypatch, [1, 416, 416, 3], [[1, 416 // s, 416 // s, c] for s in Y.STRIDES for c in (4, 17)])
    m = Y.EdgeTpuYolo("x.tflite", None)
    assert m.size == 416 and m.box_ch == 4 and m.split is not None
