"""TEST-ONLY stand-in for EdgeTpuYolo (no hardware): builds int8 tensors from a list of scripted items. Also simulates delay, hangs and failures.
Selected with VERIFIER_FAKE_TPU=1 or injected by the tests."""
import json, os, threading, time
import numpy as np
import yolo_edgetpu as Y

S_SC, Z_SC = 0.0625, -10
S_BX, Z_BX = 0.25, 0
CLS = {n: i for i, n in enumerate(Y.LABELS17)}


class FakeModel(Y.EdgeTpuYolo):
    def __init__(self, size, items=None, delay=0.002, ctl=None):
        self.size = size; self.load_s = 0.0; self.dec = Y.Decoder(size); self.timing = {}
        self.cd = {"quantization": (S_SC, Z_SC), "index": 1}; self.bd = {"quantization": (S_BX, Z_BX)}
        self.inp = {"quantization": (1 / 255, -128), "dtype": np.int8, "shape": [1, size, size, 3]}
        self.outs = []; self.bpos = 0; self.cpos = 1; self.lut = Y.input_lut(np.int8, 1 / 255, -128); self._qbuf = None
        self.delay = delay                   # seconds per invoke; 0 = instant (non-Coral cost benchmark)
        self.hang = threading.Event()        # when set, invoke blocks until `release` is set (simulates a stalled USB transfer)
        self.release = threading.Event()
        self.fail = False                    # when True, invoke raises CoralError
        self.calls = 0
        self.last_canvas = None
        self._cache = (None, None)
        self.ctl = ctl                       # test control directory (used when the model runs in a child process): files hang, fail, first_fail, delay, items.json, load_fail
        self._items_mtime = None
        if ctl:
            self._ctl_append("loads", b"x")
            lf = os.path.join(ctl, "load_fail")
            if os.path.exists(lf):
                raise Y.CoralLoadError(open(lf).read().strip() or "cannot load Edge TPU delegate: no Edge TPU found")
            if os.path.exists(os.path.join(ctl, "load_hang")):
                time.sleep(3600)
        # items: (cls_name, logit, stride, gx, gy, (l, t, r, b) distances in stride units), same convention as tests/test_decode.py
        self.items = items if items is not None else [("person", 3.0, 16, size // 32, size // 32, (3, 3, 3, 3))]

    def _ctl_append(self, name, data):
        with open(os.path.join(self.ctl, name), "ab") as f:
            f.write(data)

    def describe(self):
        return {"fake": True}

    def _tensors(self):
        key = repr(self.items)
        if self._cache[0] == key:
            return self._cache[1]
        n = Y.n_anchors(self.size)
        sc = np.full((n, 17), int(round(-6 / S_SC)) + Z_SC, np.int8); bx = np.zeros((n, 64), np.int8)
        for name, logit, stride, gx, gy, ltrb in self.items:
            off = 0
            for s in Y.STRIDES:
                if s == stride:
                    break
                off += (self.size // s) ** 2
            a = off + gy * (self.size // stride) + gx
            sc[a, CLS[name]] = int(round(logit / S_SC)) + Z_SC
            for side, dist in enumerate(ltrb):
                bx[a, side * 16:(side + 1) * 16] = int(-8 / S_BX); bx[a, side * 16 + int(dist)] = int(8 / S_BX)
        self._cache = (key, (sc, bx))
        return sc, bx

    def quantize(self, canvas_rgb):
        self.last_canvas = canvas_rgb
        return canvas_rgb

    def invoke_raw(self, x):
        self.calls += 1
        t = time.perf_counter()
        delay = self.delay
        if self.ctl:
            c = lambda n: os.path.join(self.ctl, n)
            self._ctl_append("calls", b"x")
            if os.path.exists(c("fail")) or (self.calls == 1 and os.path.exists(c("first_fail"))):
                raise Y.CoralError("fake failure")
            if os.path.exists(c("delay")):
                delay = float(open(c("delay")).read())
            if os.path.exists(c("items.json")):
                mt = os.stat(c("items.json")).st_mtime_ns
                if mt != self._items_mtime:
                    self._items_mtime = mt
                    self.items = [tuple(i[:5]) + (tuple(i[5]),) for i in json.load(open(c("items.json")))]
            t_start = time.time()
            if os.path.exists(c("hang")):
                time.sleep(3600)
        if self.hang.is_set():
            self.release.wait()
        if self.fail:
            raise Y.CoralError("fake failure")
        if delay:
            time.sleep(delay)
        if self.ctl:
            self._ctl_append("trace", f"{os.getpid()} {t_start:.6f} {time.time():.6f}\n".encode())
        self.timing = {"infer_ms": (time.perf_counter() - t) * 1e3}
        return self._tensors()
