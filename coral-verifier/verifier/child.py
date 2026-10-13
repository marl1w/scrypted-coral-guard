"""The child process that OWNS the Edge TPU interpreter. The parent never touches libedgetpu: a hung invoke is cured by SIGKILL of this process,
so no thread can leak and no second interpreter can ever run beside a stuck one.

Protocol over one duplex Pipe (parent -> child: raw RGB canvas bytes (S*S*3); child -> parent: a header tuple, then raw buffers):
  child -> parent at start:  ("ready", {"size", "cd_q", "bd_q", "scores_shape", "boxes_shape"})  or  ("load_error", message)
  per frame:                 ("ok", infer_ms, quant_ms) then scores_q bytes then boxes_q bytes     or     ("err", message)
"""
import ctypes, os, sys, time
import numpy as np


def _die_with_parent():
    """Linux: SIGKILL this process when the parent dies (even by os._exit), so a stuck libedgetpu call never outlives the service."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9, 0, 0, 0)       # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


def build_model(spec):
    if spec["kind"] == "fake":                                              # tests only: the module is not in the runtime image
        from fake_tpu import FakeModel
        return FakeModel(spec["size"], ctl=spec["ctl"])
    from yolo_edgetpu import EdgeTpuYolo
    return EdgeTpuYolo(spec["path"], spec["size"])


def child_main(conn, spec):
    _die_with_parent()
    try:
        model = build_model(spec)
        size = model.size
        info = {"size": size, "cd_q": tuple(model.cd["quantization"]), "bd_q": tuple(model.bd["quantization"])}
        import yolo_edgetpu as Y
        n = Y.n_anchors(size)
        info["scores_shape"] = (n, 17); info["boxes_shape"] = (n, 64)
    except BaseException as e:                                              # includes CoralLoadError: report, then exit
        try:
            conn.send(("load_error", f"{e}"))
        except Exception:
            pass
        return
    conn.send(("ready", info))
    while True:
        try:
            data = conn.recv_bytes()
        except (EOFError, OSError):
            return
        try:
            canvas = np.frombuffer(data, np.uint8).reshape(size, size, 3)
            t0 = time.perf_counter()
            x = model.quantize(canvas)
            q_ms = (time.perf_counter() - t0) * 1e3
            s, b = model.invoke_raw(x)
            conn.send(("ok", model.timing.get("infer_ms", 0.0), q_ms))
            conn.send_bytes(np.ascontiguousarray(s).reshape(-1))
            conn.send_bytes(np.ascontiguousarray(b).reshape(-1))
        except BaseException as e:
            try:
                conn.send(("err", f"invoke failed: {e}"))
            except Exception:
                return
