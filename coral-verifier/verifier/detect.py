"""JPEG decode (reduced-size when valid), allocation-light letterbox/tiling around the Edge TPU model, tile merge.
Images are BGR (cv2 native) until the model input is built: the channel swap happens on the small 512x512 canvas, not on the full frame.
Engine.detect() is NOT thread-safe: only the single Coral worker calls it."""
import os, time
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 24))      # OpenCV's own decompression-bomb guard (16 MP); must be set before cv2 loads
import numpy as np
import cv2
import yolo_edgetpu as Y
from defaults import TILING as _TL, EDGE_MARGIN, VEHICLE_NMS_IOU, VEHICLE_CONTAIN, DETECT_THR, WEAK_VEHICLE_THR, WEAK_VEHICLE_CLASSES

cv2.setNumThreads(1)
TILE_FRAC = _TL["two_tile_frac"]     # explicit tiles=2: each of the two tiles covers 55 % of the width (10 % overlap in the middle)
TILE_NMS = 0.5
ASPECT_TILE = _TL["auto_aspect"]    # tiles=auto splits when width/height > 2.2
TILE_OVERLAP = _TL["overlap"]
GROUP_THR = {}                                                            # class name -> (group, threshold); person is dynamic
for _g, _t in DETECT_THR.items():
    for _n, _v in _t.items():
        GROUP_THR[_n] = (_g, _v)
CLASSES = (0,) + tuple(sorted(Y.LABELS17.index(n) for n in GROUP_THR))   # person + vehicles (1,2,3,5,7) + animals (9..14,16)
_REDUCED = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def jpeg_size(data):
    """(w, h) from the JPEG SOF marker without decoding, or None."""
    n = min(len(data), 262144)           # the frame header sits in the first few KB; never scan megabytes of junk in Python
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m == 0x01 or 0xD0 <= m <= 0xD8:
            i += 2
            continue
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            if i + 9 > n:
                return None
            return (data[i + 7] << 8) | data[i + 8], (data[i + 5] << 8) | data[i + 6]
        i += 2 + ((data[i + 2] << 8) | data[i + 3])
    return None


def _width_for(W, n):
    """Tile width so that n evenly spread tiles with TILE_OVERLAP cover W exactly."""
    return W / (n - (n - 1) * TILE_OVERLAP)


def tile_ranges(W, H, n):
    """THE tile layout: list of n (x0, x1) column ranges, each full height. n=1 whole frame; n=2 legacy two 55 % tiles; n>=3 evenly spread tiles
    with TILE_OVERLAP overlap that exactly cover [0, W]."""
    if n == "hybrid":
        tw = int(round(W * TILE_FRAC))
        return [(0, W), (0, tw), (W - tw, W)]
    n = int(n)
    if n <= 1:
        return [(0, W)]
    if n == 2:
        tw = int(round(W * TILE_FRAC))
        return [(0, tw), (W - tw, W)]
    tw = min(W, int(round(_width_for(W, n))))
    xs = [int(round(i * (W - tw) / (n - 1))) for i in range(n)]
    return [(x, x + tw) for x in xs]


def resolve_tiles(w, h, tiles):
    """-> "1"|"2"|"3"|"hybrid": explicit "1"|"2"|"3" as given; auto: "1" when width/height <= 2.2, else "hybrid" (whole frame + two 55 % tiles)."""
    if tiles in ("1", "2", "3", "hybrid"):
        return tiles
    return "hybrid" if tiles == "auto" and w / h > ASPECT_TILE else "1"


def reduce_factor(w, h, size, tiles):
    """Largest DCT scale-down (1,2,4,8) that still leaves the image at least as large as its letterboxed model input (never upsampled)."""
    r = min(size / h, size / min(x1 - x0 for x0, x1 in tile_ranges(w, h, tiles)))      # the narrowest pass needs the most resolution
    for f in (8, 4, 2):
        if r * f <= 1.0:
            return f
    return 1


MAX_IMAGE_PIXELS = 2 ** 24      # 16 MP (4096 x 4096)
MAX_SIDE = 16384


def decode_jpeg(data, size=512, tiles="auto", reduce=False, max_pixels=MAX_IMAGE_PIXELS):
    """bytes -> (BGR uint8 image, resolved tiles "1"|"2"). JPEG ONLY: the body must start with FFD8 and carry a parsable SOF marker whose
    dimensions are checked against the pixel cap BEFORE any cv2 decode (decompression-bomb guard). Raises ValueError otherwise."""
    if not data or data[:2] != b"\xff\xd8":
        raise ValueError("body must be a JPEG")
    wh = jpeg_size(data)
    if wh is None:
        raise ValueError("JPEG without a readable frame header")
    w0, h0 = wh
    if w0 < 16 or h0 < 16 or w0 > MAX_SIDE or h0 > MAX_SIDE or w0 * h0 > max_pixels:
        raise ValueError(f"unsupported image size {w0}x{h0} (max {max_pixels} pixels)")
    tiles = resolve_tiles(w0, h0, tiles)
    f = reduce_factor(w0, h0, size, tiles) if reduce else 1
    img = cv2.imdecode(np.frombuffer(data, np.uint8), _REDUCED[f] | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError("not a decodable image")
    return img, tiles


def merge_tiles(dets, iou=TILE_NMS):
    """dets: [(score, x0, y0, x1, y1, label)] normalised to the full frame. Per-label NMS; returns sorted by score desc."""
    out = []
    for label in {d[5] for d in dets}:
        ds = [d for d in dets if d[5] == label]
        xywh = [[d[1], d[2], d[3] - d[1], d[4] - d[2]] for d in ds]
        keep = cv2.dnn.NMSBoxes(xywh, [float(d[0]) for d in ds], 0.0, iou) if ds else []
        out += [ds[int(k)] for k in np.array(keep, dtype=int).ravel()]
    return sorted(out, reverse=True)


def _group(label):
    return "person" if label == "person" else GROUP_THR[label][0]


def _shadowed_by_whole(d, whole):
    """Hybrid frames: a tile detection loses to a same-group whole-frame detection that contains most of it (> hybrid_contain of its area), or that it
    overlaps (IoU > hybrid_big_iou) when it is taller than hybrid_big_frac of the frame height (big, near objects are cut by tiles: trust the whole pass)."""
    area = (d[3] - d[1]) * (d[4] - d[2])
    for w in whole:
        if _group(w[5]) != _group(d[5]):
            continue
        iw, ih = min(d[3], w[3]) - max(d[1], w[1]), min(d[4], w[4]) - max(d[2], w[2])
        if iw <= 0 or ih <= 0:
            continue
        inter = iw * ih
        if inter / max(area, 1e-9) > _TL["hybrid_contain"]:
            return True
        if d[4] - d[2] > _TL["hybrid_big_frac"] and inter / max(area + (w[3] - w[1]) * (w[4] - w[2]) - inter, 1e-9) > _TL["hybrid_big_iou"]:
            return True
    return False


_MOTOR = ("car", "truck", "bus", "motorcycle")


def vehicle_group_nms(dets):
    """dets: [(score, x0, y0, x1, y1, label)] sorted by score desc. Cross-class NMS over car/truck/bus/motorcycle (bicycle and other groups untouched):
    a vehicle box is dropped when IoU > VEHICLE_NMS_IOU with a higher-scored kept vehicle box, or when more than VEHICLE_CONTAIN of its area lies
    inside a larger vehicle box. The survivor keeps its own class."""
    veh = [d for d in dets if d[5] in _MOTOR]
    if len(veh) < 2:
        return dets
    area = lambda d: max(0.0, d[3] - d[1]) * max(0.0, d[4] - d[2])
    drop = set()
    for i, a in enumerate(veh):
        for j, b in enumerate(veh):
            if i == j or j in drop:
                continue
            iw, ih = min(a[3], b[3]) - max(a[1], b[1]), min(a[4], b[4]) - max(a[2], b[2])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            aa, ab = area(a), area(b)
            if i > j and inter / max(aa + ab - inter, 1e-9) > VEHICLE_NMS_IOU:          # b ranks higher (earlier): a loses
                drop.add(i)
            elif ab > aa and inter / max(aa, 1e-9) > VEHICLE_CONTAIN:                  # a mostly inside the larger b
                drop.add(i)
    gone = {id(veh[i]) for i in drop}
    return [d for d in dets if id(d) not in gone]


class Engine:
    def __init__(self, model):
        self.model = model
        self.size = model.size
        self._canvas = np.full((self.size, self.size, 3), 114, np.uint8)     # reused letterbox canvas (BGR)
        self._rgb = np.empty_like(self._canvas)
        self._geom = None                                                      # (nh, nw, top, left) currently painted on the canvas

    def _letterbox(self, img):
        """Letterbox `img` (BGR) onto the reused canvas; returns (canvas_rgb, r, left, top). Gray border is repainted only when the geometry changes."""
        S = self.size
        h, w = img.shape[:2]
        r = min(S / h, S / w)
        nh, nw = round(h * r), round(w * r)
        top, left = (S - nh) // 2, (S - nw) // 2
        if self._geom != (nh, nw, top, left):
            self._canvas[:] = 114
            self._geom = (nh, nw, top, left)
        self._canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB, dst=self._rgb)
        return self._rgb, r, left, top

    @staticmethod
    def _floors(pthr, vehicle_thr, animal_thr):
        """-> (decode thresholds {class index: thr}, report floor {label: thr}, weak band active, strong vehicle floor)."""
        vthr = GROUP_THR["car"][1] if vehicle_thr is None else vehicle_thr          # strong floor of car/truck/bus/motorcycle (per camera)
        athr = GROUP_THR["dog"][1] if animal_thr is None else animal_thr
        weak_band = vthr > WEAK_VEHICLE_THR                                         # weak [WEAK_VEHICLE_THR, vthr) only above the weak floor
        floor = {}
        for n, (g, v) in GROUP_THR.items():
            floor[n] = (athr if g == "animal" else (vthr if n in WEAK_VEHICLE_CLASSES else v))
        dec = {n: (WEAK_VEHICLE_THR if n in WEAK_VEHICLE_CLASSES and weak_band else floor[n]) for n in floor}     # decode floor (per-class NMS on all, strong outrank weak)
        return {c: (pthr if c == 0 else dec[Y.LABELS17[c]]) for c in CLASSES}, floor, weak_band, vthr

    def recheck(self, img, crop, group, person_thr=0.2, vehicle_thr=None, animal_thr=None, scale=1.0):
        """Second look: ONE Coral pass on the full-resolution window (x0, y0, x1, y1 in pixels, may extend beyond the frame: the outside is grey 114) of
        `img`, only the classes of `group`. Returns ([(score, x0, y0, x1, y1, label)] in normalised FULL-frame coordinates, clamped to 0..1, timings)."""
        H, W = img.shape[:2]
        x0, y0, x1, y1 = crop
        thr, _, _, _ = self._floors(max(person_thr, 0.01), vehicle_thr, animal_thr)
        classes = tuple(c for c in CLASSES if _group(Y.LABELS17[c]) == group)
        thr = {c: max(0.01, v * scale) for c, v in thr.items()}               # scale < 1: decode lower so a weak positive match is seen
        t0 = time.perf_counter()
        th, tw = y1 - y0, x1 - x0
        sub = img[y0:y1, x0:x1] if (x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H) else np.full((th, tw, 3), 114, np.uint8)
        if sub.shape[:2] != (th, tw) or not (x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H):
            ox0, oy0, ox1, oy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)       # overlap with the frame, copied into the grey window
            if ox1 > ox0 and oy1 > oy0:
                sub[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = img[oy0:oy1, ox0:ox1]
        canvas, r, left, top = self._letterbox(sub)
        t1 = time.perf_counter()
        px = self.model.detect_px(canvas, classes, {c: thr[c] for c in classes}, Y.NMS_IOU)
        t2 = time.perf_counter()
        tm = self.model.timing
        out = []
        for d in px:
            s, a, b, c, e, label = Y.to_normalized(d, r, left, top, tw, th)
            fa, fc = min(1.0, max(0.0, (a * tw + x0) / W)), min(1.0, max(0.0, (c * tw + x0) / W))      # x0 / y0 may be negative
            fb, fe = min(1.0, max(0.0, (b * th + y0) / H)), min(1.0, max(0.0, (e * th + y0) / H))
            if fc - fa > 0.002 and fe - fb > 0.002:
                out.append((s, fa, fb, fc, fe, label))
        return out, {"preprocess_ms": (t1 - t0) * 1e3 + tm.get("quant_ms", 0.0), "infer_ms": tm.get("infer_ms", 0.0),
                     "post_ms": tm.get("post_ms", 0.0) + (time.perf_counter() - t2) * 1e3}

    @staticmethod
    def _dets_from_raw(raw, pthr, floor, weak_band):
        dets = []
        for s, a, b, c, e, label in raw:
            g, thr_c = ("person", pthr) if label == "person" else (GROUP_THR[label][0], floor[label])
            box = [round(a, 4), round(b, 4), round(c, 4), round(e, 4)]
            if s >= thr_c:
                dets.append({"cls": label, "group": g, "score": round(float(s), 3), "box": box})
            elif label in WEAK_VEHICLE_CLASSES and weak_band and s >= WEAK_VEHICLE_THR:
                dets.append({"cls": label, "group": g, "score": round(float(s), 3), "box": box, "weak": True})
        return dets

    def _zoom_pass(self, img, thr, zoom_out):
        """ONE extra pass: the whole frame shrunk on a grey canvas of the original size. -> (raw boxes in full-frame coordinates, pre_ms, infer_ms, post_ms)."""
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        s_ = 1.0 - zoom_out
        sw, sh = max(1, round(W * s_)), max(1, round(H * s_))
        ox, oy = (W - sw) // 2, (H - sh) // 2
        zc = np.full((H, W, 3), 114, np.uint8)
        zc[oy:oy + sh, ox:ox + sw] = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        canvas, r, left, top = self._letterbox(zc)
        t1 = time.perf_counter()
        px = self.model.detect_px(canvas, CLASSES, thr, Y.NMS_IOU)
        t2 = time.perf_counter()
        tm = self.model.timing
        out = []
        for d in px:
            s, a, b, c, e, label = Y.to_normalized(d, r, left, top, W, H)
            a, c = min(1.0, max(0.0, (a * W - ox) / sw)), min(1.0, max(0.0, (c * W - ox) / sw))      # undo scale + offset, clamp
            b, e = min(1.0, max(0.0, (b * H - oy) / sh)), min(1.0, max(0.0, (e * H - oy) / sh))
            if c - a > 0.005 and e - b > 0.005:
                out.append((s, a, b, c, e, label))
        return out, (t1 - t0) * 1e3 + tm.get("quant_ms", 0.0), tm.get("infer_ms", 0.0), tm.get("post_ms", 0.0) + (time.perf_counter() - t2) * 1e3

    def add_zoom(self, img, dets, zoom_out, person_thr=0.2, vehicle_thr=None, animal_thr=None):
        """Adds the zoom-out pass to detections already computed (a frame that was first run reduced and then promoted). -> (detections, timings)."""
        pthr = max(person_thr, 0.01)
        thr, floor, weak_band, _ = self._floors(pthr, vehicle_thr, animal_thr)
        zoom_raw, pre, infer, post = self._zoom_pass(img, thr, zoom_out)
        t3 = time.perf_counter()
        raw = [(d["score"], *d["box"], d["cls"]) for d in dets] + zoom_raw
        raw = vehicle_group_nms(merge_tiles(raw))                            # zoom-out boxes act like whole-frame boxes: same merge as in detect()
        out = self._dets_from_raw(raw, pthr, floor, weak_band)
        return out, {"preprocess_ms": pre, "infer_ms": infer, "post_ms": post + (time.perf_counter() - t3) * 1e3}

    def detect(self, img, tiles="auto", person_thr=0.2, vehicle_thr=None, animal_thr=None, zoom_out=0.0):
        """img: BGR. Returns (detections, timings). detections: [{"cls","group","score","box":[x0,y0,x1,y1]}] sorted by score; person >= person_thr,
        vehicles and animals >= their defaults.DETECT_THR (car/truck/bus/motorcycle 0.40, bicycle 0.5, animals 0.3) plus weak vehicles [0.25, 0.40) flagged weak; each carries group person|vehicle|animal. timings: preprocess_ms, infer_ms, post_ms (summed over tiles)."""
        H, W = img.shape[:2]
        tiles = resolve_tiles(W, H, tiles)
        split = tiles != "1"
        hybrid = tiles == "hybrid"                                            # ranges[0] is the whole frame, the rest are tiles
        pthr = max(person_thr, 0.01)
        thr, floor, weak_band, vthr = self._floors(pthr, vehicle_thr, animal_thr)
        pre = infer = post = 0.0
        raw = []
        ranges = tile_ranges(W, H, tiles)
        first = 1 if hybrid else 0
        tl = ranges[first:]
        whole_raw = []
        for ri, (x0, x1) in enumerate(ranges):
            ti = ri - first                                                  # tile index (-1 = the whole-frame pass)
            t0 = time.perf_counter()
            crop = img[:, x0:x1] if split else img
            th, tw = crop.shape[:2]
            canvas, r, left, top = self._letterbox(crop)
            t1 = time.perf_counter()
            px = self.model.detect_px(canvas, CLASSES, thr, Y.NMS_IOU)
            t2 = time.perf_counter()
            tm = self.model.timing
            infer += tm.get("infer_ms", 0.0)
            q = tm.get("quant_ms", 0.0)
            pre += (t1 - t0) * 1e3 + q
            for d in px:
                s, a, b, c, e, label = Y.to_normalized(d, r, left, top, tw, th)
                if hybrid and ti < 0:                                          # whole-frame detection: never subject to the tile rules
                    whole_raw.append((s, a, b, c, e, label))
                    continue
                if split:                                                      # inner-edge rule: the neighbour sees this object whole, its own box wins
                    m = EDGE_MARGIN
                    if ti > 0 and a < m and a * tw + x0 >= tl[ti - 1][0] and c * tw + x0 <= tl[ti - 1][1]:
                        continue
                    if ti < len(tl) - 1 and c > 1 - m and a * tw + x0 >= tl[ti + 1][0] and c * tw + x0 <= tl[ti + 1][1]:
                        continue
                raw.append((s, (a * tw + x0) / W, b, (c * tw + x0) / W, e, label))
            post += tm.get("post_ms", 0.0) + (time.perf_counter() - t2) * 1e3     # decode+NMS (in the model) + box normalisation
        zoom_raw = []
        if zoom_out > 0:
            zoom_raw, zp, zi, zq = self._zoom_pass(img, thr, zoom_out)
            pre += zp; infer += zi; post += zq
        t3 = time.perf_counter()
        if hybrid:
            whole_raw = whole_raw + zoom_raw                                   # zoom-out boxes act like whole-frame boxes
            raw = whole_raw + [d for d in raw if not _shadowed_by_whole(d, whole_raw)]
        else:
            raw = raw + zoom_raw                                               # exempt from the tile-edge rule (applied above, to tile boxes only)
        raw = merge_tiles(raw) if (split or zoom_raw) else sorted(raw, reverse=True)
        raw = vehicle_group_nms(raw)
        dets = self._dets_from_raw(raw, pthr, floor, weak_band)
        post += (time.perf_counter() - t3) * 1e3
        return dets, {"preprocess_ms": pre, "infer_ms": infer, "post_ms": post}
