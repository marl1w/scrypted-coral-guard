"""Edge TPU YOLOv9-s (dbro, ReLU6, 17 COCO classes, person = 0) pre/post-processing.

Output format (from dbro/frigate-detector-edgetpu-yolo9 edgetpu_tfl.py, a 'yolo-generic' detector plugin with a multi-tensor model):
  two int8 output tensors, both shape (1, N, C), N = anchors over strides 8/16/32 (2100 @320, 5376 @512):
    boxes   C = 64 = 4 sides x 16 DFL bins (reg_max 16), raw DFL logits, order l,t,r,b
    classes C = 17 raw LOGITS (sigmoid not applied on the TPU), index 0 = person
  tensors are told apart by shape (C == 64 -> boxes), not by name/order.
  decode: dequant (q-zp)*scale -> softmax over 16 bins -> expectation sum(p*arange(16)) = distance in stride units
          anchor = (grid_x + 0.5, grid_y + 0.5) row-major per stride, strides concatenated 8,16,32
          x1y1 = (anchor - lt) * stride ; x2y2 = (anchor + rb) * stride   (pixels of the model input)
          score = sigmoid(class logit)
Adapted from the author's earlier validated benchmark decode (the Edge TPU run itself needs real hardware)."""
import math, time
import numpy as np
import cv2

cv2.setNumThreads(1)
STRIDES = (8, 16, 32)
REG_MAX = 16
THR = 0.05
NMS_IOU = 0.45
class CoralError(RuntimeError):
    """Raised when the Edge TPU (delegate / USB / invoke) fails. The service treats it as a stalled Coral."""


class CoralLoadError(CoralError):
    """The Edge TPU delegate / interpreter could not be loaded (state 'missing')."""


class CoralStall(CoralError):
    """An invoke did not finish within the stall limit."""


LABELS17 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear"]


# ---------------- preprocessing (identical to eval/yolov9_torch.py _letterbox) ----------------
def letterbox(img, size):
    h, w = img.shape[:2]; r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def input_lut(dtype, scale, zero_point):
    """uint8 pixel (0..255) -> quantised input value, i.e. float = pixel/255 fed through the tensor's own (scale, zero_point).
    For scale = 1/255 and zp = -128 (int8) this is pixel-128, the usual XOR-128 trick."""
    info = np.iinfo(dtype)
    if not scale:
        scale, zero_point = 1.0 / 255.0, 0
    q = np.round(np.arange(256, dtype=np.float64) / 255.0 / scale + zero_point)
    return np.clip(q, info.min, info.max).astype(dtype)


# ---------------- decode ----------------
def anchors_and_strides(size):
    anc, strd = [], []
    for s in STRIDES:
        fh = fw = None
        fh, fw = size // s, size // s
        gy, gx = np.meshgrid(np.arange(fh, dtype=np.float32), np.arange(fw, dtype=np.float32), indexing="ij")
        anc.append(np.stack((gx.ravel(), gy.ravel()), 1) + 0.5)
        strd.append(np.full((fh * fw, 1), s, np.float32))
    return np.concatenate(anc), np.concatenate(strd)


def n_anchors(size):
    return sum((size // s) ** 2 for s in STRIDES)


def pick_tensors(details):
    """details: list of dicts with 'shape' and 'index'. Returns (boxes_pos, classes_pos) positions in the list. Same rule as the dbro plugin."""
    bi = ci = None
    for i, d in enumerate(details):
        sh = tuple(d["shape"])
        if len(sh) == 3 and sh[2] == 64:
            bi = i
        elif len(sh) == 3 and sh[2] > 1:
            ci = i
    if bi is None or ci is None:
        raise ValueError(f"unrecognised output tensors: {[tuple(d['shape']) for d in details]}")
    return bi, ci


MAX_CAND = 100          # per class, kept after the logit pre-filter and before NMS


def _nms(boxes_xyxy, scores, thr, iou):
    if len(scores) == 0:
        return np.zeros(0, int)
    xywh = np.column_stack((boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1])).tolist()
    keep = cv2.dnn.NMSBoxes(xywh, scores.tolist(), thr, iou)
    return np.array(keep, dtype=int).ravel()


class Decoder:
    def __init__(self, size):
        self.size = size
        self.anchors, self.strides = anchors_and_strides(size)
        self.project = np.arange(REG_MAX, dtype=np.float32)
        self.n = len(self.anchors)

    def decode(self, scores_q, boxes_q, s_scale, s_zp, b_scale, b_zp, classes=(0,), thr=THR, nms_iou=NMS_IOU):
        """scores_q (N, NC) int8 logits, boxes_q (N, 64) int8 DFL. Returns list of (cls, prob, x0, y0, x1, y1) in model-input pixels,
        per-class NMS, sorted by prob descending. Only classes in `classes` are considered. thr: float or {class: float}.
        Cheap path: the threshold is applied on the int8 logits (no float work for background anchors), at most MAX_CAND candidates per
        class survive, and the DFL softmax runs only on those anchors."""
        if scores_q.shape[0] != self.n or boxes_q.shape != (self.n, 4 * REG_MAX):
            raise ValueError(f"shape mismatch: scores {scores_q.shape} boxes {boxes_q.shape} anchors {self.n}")
        per = {}
        for c in classes:
            t = max(float(thr[c] if isinstance(thr, dict) else thr), 1e-6)
            qmin = math.ceil(math.log(t / (1.0 - t)) / s_scale + s_zp)
            idx = np.flatnonzero(scores_q[:, c] >= qmin)
            if len(idx) > MAX_CAND:
                idx = idx[np.argpartition(scores_q[idx, c], -MAX_CAND)[-MAX_CAND:]]
            if len(idx):
                per[c] = (idx, t)
        if not per:
            return []
        allidx = np.unique(np.concatenate([v[0] for v in per.values()]))
        b = (boxes_q[allidx].astype(np.float32) - b_zp) * b_scale
        d = b.reshape(-1, 4, REG_MAX)
        d = np.exp(d - d.max(axis=2, keepdims=True))
        d /= d.sum(axis=2, keepdims=True)
        dist = d @ self.project                                   # (K, 4) expectation over the 16 bins
        a, st = self.anchors[allidx], self.strides[allidx]
        xyxy = np.concatenate(((a - dist[:, :2]) * st, (a + dist[:, 2:]) * st), 1)
        out = []
        for c, (idx, t) in per.items():
            pos = np.searchsorted(allidx, idx)
            logit = (scores_q[idx, c].astype(np.float32) - s_zp) * s_scale
            prob = 1.0 / (1.0 + np.exp(-logit))
            bx = xyxy[pos]
            for k in _nms(bx, prob, t, nms_iou):
                x0, y0, x1, y1 = bx[k].tolist()
                out.append((int(c), float(prob[k]), x0, y0, x1, y1))
        out.sort(key=lambda r: -r[1])
        return out


def to_normalized(det_px, r, left, top, W, H):
    """model-input pixels -> original-image fractions, exactly like yolov9_torch.detect()."""
    c, p, x0, y0, x1, y1 = det_px
    x0, x1 = (x0 - left) / r / W, (x1 - left) / r / W
    y0, y1 = (y0 - top) / r / H, (y1 - top) / r / H
    cl = lambda v: float(np.clip(v, 0, 1))
    return (p, cl(x0), cl(y0), cl(x1), cl(y1), LABELS17[c] if c < len(LABELS17) else str(c))


# ---------------- Edge TPU wrapper ----------------
class EdgeTpuYolo:
    def __init__(self, model_path, size, delegate="libedgetpu.so.1.0", device=None):
        t0 = time.perf_counter()
        from tflite_runtime.interpreter import Interpreter, load_delegate
        opts = {"device": device} if device else {}
        try:
            self.interp = Interpreter(model_path=model_path, experimental_delegates=[load_delegate(delegate, opts)], num_threads=1)
            self.interp.allocate_tensors()
        except Exception as e:                     # no Edge TPU delegate / device
            raise CoralLoadError(f"cannot load Edge TPU delegate (no Edge TPU found, or USB not accessible?): {e}") from e
        self.load_s = time.perf_counter() - t0
        self.size = size
        self.inp = self.interp.get_input_details()[0]
        self.outs = self.interp.get_output_details()
        self.dec = Decoder(size)
        self.bpos, self.cpos = pick_tensors(self.outs)
        self.bd, self.cd = self.outs[self.bpos], self.outs[self.cpos]
        if tuple(self.inp["shape"][1:3]) != (size, size):
            raise ValueError(f"model input {self.inp['shape']} != {size}")
        self.lut = input_lut(self.inp["dtype"], *self.inp["quantization"])
        self.timing = {}
        self._qbuf = None

    def describe(self):
        f = lambda d: {"name": d["name"], "shape": [int(x) for x in d["shape"]], "dtype": np.dtype(d["dtype"]).name,
                       "scale": float(d["quantization"][0]), "zero_point": int(d["quantization"][1]), "index": int(d["index"])}
        return {"input": f(self.inp), "outputs": [f(d) for d in self.outs], "boxes_tensor_pos": self.bpos, "classes_tensor_pos": self.cpos}

    def quantize(self, canvas_rgb):
        """uint8 RGB canvas (S,S,3) -> model input tensor (1,S,S,3) through the preallocated LUT buffer (cv2.LUT: one pass, no allocation)."""
        if self._qbuf is None:
            self._qbuf = np.empty((self.size, self.size, 3), np.uint8)
        lut = self.lut if self.lut.dtype == np.uint8 else self.lut.view(np.uint8)
        cv2.LUT(canvas_rgb, lut, dst=self._qbuf)
        x = self._qbuf if self.lut.dtype == np.uint8 else self._qbuf.view(self.lut.dtype)
        return x[None]

    def invoke_raw(self, x):
        """x: quantised input (1,S,S,3) from quantize(). Returns (scores_q (N,NC), boxes_q (N,64)); self.timing['infer_ms'] = set+invoke+get."""
        try:
            t0 = time.perf_counter()
            self.interp.set_tensor(self.inp["index"], x)
            self.interp.invoke()
            s = self.interp.get_tensor(self.cd["index"])[0]
            b = self.interp.get_tensor(self.bd["index"])[0]
            self.timing = {"infer_ms": (time.perf_counter() - t0) * 1e3}
        except Exception as e:                     # USB reset / firmware reload / delegate error
            raise CoralError(f"invoke failed: {e!r}") from e
        return s, b

    def detect_px(self, canvas, classes=(0,), thr=THR, nms_iou=NMS_IOU):
        """canvas: uint8 RGB (S,S,3). Returns decoded detections in model pixels; self.timing gets infer_ms and post_ms (quantize counts as preprocess)."""
        t0 = time.perf_counter()
        x = self.quantize(canvas)
        q_ms = (time.perf_counter() - t0) * 1e3
        s, b = self.invoke_raw(x)
        self.timing["quant_ms"] = q_ms
        t = time.perf_counter()
        r = self.dec.decode(s, b, *self.cd["quantization"], *self.bd["quantization"], classes=classes, thr=thr, nms_iou=nms_iou)
        self.timing["post_ms"] = (time.perf_counter() - t) * 1e3
        return r

    def detect(self, image_rgb, thr=THR, nms_iou=NMS_IOU, classes=(0,)):
        """Drop-in for yolov9_torch.detect: [(score,x0,y0,x1,y1,label)] normalised, sorted desc. Person only by default."""
        H, W = image_rgb.shape[:2]
        t = time.perf_counter()
        canvas, r, left, top = letterbox(image_rgb, self.size)
        dets = self.detect_px(canvas, classes, thr, nms_iou)
        return sorted((to_normalized(d, r, left, top, W, H) for d in dets), reverse=True)
