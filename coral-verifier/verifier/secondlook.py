"""Second look: re-check uncertain detections of a frame on a full-resolution crop, inside the frame's own job (one queue slot, several invokes).
Candidates: detections in the borderline band of their group (person cont_thr <= s < start_thr; vehicle weak or strong floor <= s < 0.6; animal
animal_cont_thr <= s < animal_start_thr) plus detections that start a NEW track (no live track of the group overlaps them) while their score < 0.8.
At most SECOND_LOOK["max"] (2) Coral passes per frame, most uncertain first (closest to the confirm threshold). A re-checked object is not looked at again
for ttl_s: later frames reuse its result by box overlap (no invoke). Result per candidate: the best same-group detection of the crop whose centre lies
inside the original box replaces the score (`first_score` keeps the original, `second_look` = confirmed|lowered); none found = `rejected`."""
from defaults import SECOND_LOOK as _SL, DETECT_THR, WEAK_VEHICLE_CLASSES

MATCH_IOU = 0.3


def group_of(d):
    return d.get("group") or ("person" if d["cls"] == "person" else ("vehicle" if d["cls"] in DETECT_THR["vehicle"] else "animal"))


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def confirm_thr(g, cfg):
    return cfg.start_thr if g == "person" else (_SL["vehicle_confirm"] if g == "vehicle" else cfg.animal_start_thr)


def in_band(d, cfg):
    g, s = group_of(d), d["score"]
    if g == "person":
        return cfg.cont_thr <= s < cfg.start_thr
    if g == "vehicle":
        floor = DETECT_THR["vehicle"]["bicycle"] if d["cls"] == "bicycle" else cfg.vehicle_thr
        return bool(d.get("weak")) or floor <= s < _SL["vehicle_confirm"]
    return cfg.animal_cont_thr <= s < cfg.animal_start_thr


def has_track(d, boxes):
    b = d["box"]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return any(iou(b, t) > 0.2 or (t[0] <= cx <= t[2] and t[1] <= cy <= t[3]) for t in boxes)


def select(dets, cfg, tracks):
    """-> candidates [(priority, index)] sorted most uncertain first."""
    out = []
    for i, d in enumerate(dets):
        g = group_of(d)
        if in_band(d, cfg) or (d["score"] < _SL["new_track_max"] and not has_track(d, tracks.get(g, []))):
            out.append((abs(d["score"] - confirm_thr(g, cfg)), -d["score"], i))
    out.sort()
    return [i for _, _, i in out]


def crop_for(box, W, H, margin=None, min_px=None):
    """Pixel crop (x0, y0, x1, y1) around a normalised box: expanded by `margin` x box size on each side, square-ish (side = the larger expanded side,
    at least min_px), centred on the box, kept inside the frame."""
    margin = _SL["margin"] if margin is None else margin
    min_px = _SL["min_crop_px"] if min_px is None else min_px
    bw, bh = (box[2] - box[0]) * W, (box[3] - box[1]) * H
    side = max(bw * (1 + 2 * margin), bh * (1 + 2 * margin), float(min_px))
    cw, ch = int(round(min(side, W))), int(round(min(side, H)))
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    x0 = int(round(min(max(cx - cw / 2, 0), W - cw)))
    y0 = int(round(min(max(cy - ch / 2, 0), H - ch)))
    return x0, y0, x0 + cw, y0 + ch


def _apply(d, status, score, cfg):
    d["second_look"] = status
    if status == "rejected":
        return
    d["first_score"] = d["score"]
    d["score"] = round(float(score), 3)
    if d["cls"] in WEAK_VEHICLE_CLASSES and group_of(d) == "vehicle":
        if d["score"] < cfg.vehicle_thr:
            d["weak"] = True
        else:
            d.pop("weak", None)


def second_look(engine, img, dets, cfg, tracks, cache, now):
    """-> (detections with second-look marks, info). `cache`: entries {group, box, t, status, score} of earlier re-checks of this camera."""
    H, W = img.shape[:2]
    ttl = _SL["ttl_s"]
    cache = [e for e in cache if now - e["t"] <= ttl]
    dets = [dict(d) for d in dets]
    info = {"preprocess_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "cache": cache, "counts": {}, "n": 0}
    count = lambda k, n=1: info["counts"].__setitem__(k, info["counts"].get(k, 0) + n)
    for i in select(dets, cfg, tracks):
        d, g = dets[i], group_of(dets[i])
        hit = next((e for e in cache if e["group"] == g and iou(e["box"], d["box"]) > MATCH_IOU), None)
        if hit is not None:                                              # re-checked recently: reuse the result, follow the object
            hit["box"] = tuple(d["box"])
            _apply(d, hit["status"], hit["score"], cfg)
            count("sl_reused")
            continue
        if info["n"] >= _SL["max"]:
            continue
        crop = crop_for(d["box"], W, H)
        res, tm = engine.recheck(img, crop, g, cfg.cont_thr, cfg.vehicle_thr, cfg.animal_cont_thr)
        for k in ("preprocess_ms", "infer_ms", "post_ms"):
            info[k] += tm[k]
        info["n"] += 1
        b = d["box"]
        inside = [r for r in res if b[0] <= (r[1] + r[3]) / 2 <= b[2] and b[1] <= (r[2] + r[4]) / 2 <= b[3]]
        best = max(inside, key=lambda r: r[0]) if inside else None
        if best is None:
            status, score = "rejected", 0.0
        else:
            status, score = ("confirmed" if best[0] >= d["score"] else "lowered"), best[0]
        _apply(d, status, score, cfg)
        cache.append({"group": g, "box": tuple(b), "t": now, "status": status, "score": score})
        count("sl"); count("sl_" + status)
    return dets, info
