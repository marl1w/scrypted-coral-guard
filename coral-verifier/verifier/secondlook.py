"""Second look: re-check uncertain detections of a frame on a full-resolution window, inside the frame's own job (one queue slot, several invokes).
Candidates: detections in the borderline band of their group (person cont_thr <= s < start_thr; vehicle weak or strong floor <= s < 0.6; animal
animal_cont_thr <= s < animal_start_thr) plus detections that start a NEW track (no live track of the group overlaps them) while their score < 0.8.
At most SECOND_LOOK["max"] (2) Coral passes per frame, most uncertain first (closest to the confirm threshold). A re-checked object is not looked at again
for ttl_s: later frames reuse its result by box overlap (no invoke). Result per candidate: the best same-group detection of the crop whose centre lies
inside the original box replaces the score (`first_score` keeps the original, `second_look` = confirmed|lowered); nothing found = `not_found` (the detection is kept unchanged, e.g. half-occluded parked cars); found again below reject_factor x the keep floor
while the first score was below the confirm threshold = `rejected` (dropped from the trackers)."""
from defaults import SECOND_LOOK as _SL, DETECT_THR, WEAK_VEHICLE_CLASSES
from geom import iou, area

MATCH_IOU = 0.3


def group_of(d):
    return d.get("group") or ("person" if d["cls"] == "person" else ("vehicle" if d["cls"] in DETECT_THR["vehicle"] else "animal"))


def confirm_thr(g, cfg):
    return cfg.start_thr if g == "person" else (_SL["vehicle_confirm"] if g == "vehicle" else cfg.animal_start_thr)


def keep_thr(d, cfg):
    """The floor under which the trackers would not keep the detection: person cont_thr, vehicle strong floor, animal animal_cont_thr."""
    g = group_of(d)
    if g == "person":
        return cfg.cont_thr
    if g == "vehicle":
        return DETECT_THR["vehicle"]["bicycle"] if d["cls"] == "bicycle" else cfg.vehicle_thr
    return cfg.animal_cont_thr


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


def mandatory(d, cfg, tracks):
    """A person/animal detection at or above its start threshold that is not just continuing an already CONFIRMED track would start or confirm one."""
    g = group_of(d)
    if g not in ("person", "animal"):
        return False
    return d["score"] >= confirm_thr(g, cfg) and not has_track(d, tracks.get(g + "_conf", []))


def doubt_cap(d, cfg):
    return (cfg.cont_thr if group_of(d) == "person" else cfg.animal_cont_thr) + 0.05


def tracker_view(d, cfg):
    """The detection as the tracker sees it: a doubtful one is capped so it can continue a confirmed track but never start or confirm one."""
    if d.get("second_look_doubt"):
        return {**d, "score": min(d["score"], doubt_cap(d, cfg))}
    return d


def promote_reason(dets, cfg, promo, t_ms, tracks, release_s=2.0):
    """Load shedding: a low-priority frame on a busy service stays reduced UNLESS its camera is near a status change. Decided after the base pass:
    a person/animal session still a candidate, a base detection that would start or confirm a person/animal track, a confirmed person/animal track whose
    hold ends within release_s of this frame, or a vehicle whose moving/still classification is still pending (young track, or a strong vehicle with no track)."""
    promo = promo or {}
    if promo.get("person_state") == "candidate" or promo.get("animal_state") == "candidate":
        return "candidate"
    if any(mandatory(d, cfg, tracks) for d in dets):
        return "would_start"
    for k in ("person_hold", "animal_hold"):
        h = promo.get(k)
        if h is not None and h - t_ms <= release_s * 1000.0:
            return "release"
    if promo.get("vehicle_pending") or any(group_of(d) == "vehicle" and not d.get("weak") and d["score"] >= cfg.vehicle_thr and not has_track(d, tracks.get("vehicle", [])) for d in dets):
        return "vehicle"
    return None


def select(dets, cfg, tracks):
    """-> candidate indices: mandatory (would start/confirm a person/animal track) first, then the most uncertain (closest to the confirm threshold).
    Person boxes of at least person_skip_area of the frame are never candidates (no re-check, no doubt)."""
    out = []
    for i, d in enumerate(dets):
        g = group_of(d)
        if g == "person" and area(d["box"]) >= _SL["person_skip_area"]:
            continue                                                     # large person: the window would shrink it, never re-checked
        m = mandatory(d, cfg, tracks)
        if m or in_band(d, cfg) or (d["score"] < _SL["new_track_max"] and not has_track(d, tracks.get(g, []))):
            out.append((0 if m else 1, abs(d["score"] - confirm_thr(g, cfg)), -d["score"], i))
    out.sort()
    return [x[-1] for x in out]


def crop_for(box, W, H, margin=None, min_px=None):
    """Pixel WINDOW (x0, y0, x1, y1) around a normalised box: expanded by `margin` x box size on each side, square (side = the larger expanded side,
    at least min_px), ALWAYS centred on the box. It is NOT clamped to the frame: coordinates may be negative or exceed W/H (the part outside the
    frame is filled grey by Engine.recheck), so the margin stays symmetric and a big box gets a window larger than the frame."""
    margin = _SL["margin"] if margin is None else margin
    min_px = _SL["min_crop_px"] if min_px is None else min_px
    bw, bh = (box[2] - box[0]) * W, (box[3] - box[1]) * H
    side = max(bw * (1 + 2 * margin), bh * (1 + 2 * margin), float(min_px))
    cw = ch = int(round(side))
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
    return x0, y0, x0 + cw, y0 + ch


def _apply(d, status, score, cfg):
    d["second_look"] = status
    if status in ("not_found", "not_upgraded"):                                           # nothing found in the window: the detection stays exactly as it was
        return
    d["first_score"] = d["score"]
    d["score"] = round(float(score), 3)
    if d["cls"] in WEAK_VEHICLE_CLASSES and group_of(d) == "vehicle":
        if d["score"] < cfg.vehicle_thr:
            d["weak"] = True
        else:
            d.pop("weak", None)


def second_look(engine, img, dets, cfg, tracks, cache, now, reduced=False):
    """-> (detections with second-look marks, info). `cache`: entries {group, box, t, status, score} of earlier re-checks of this camera."""
    H, W = img.shape[:2]
    ttl = _SL["ttl_s"]
    cache = [e for e in cache if now - e["t"] <= ttl]
    dets = [dict(d) for d in dets]
    info = {"preprocess_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "cache": cache, "counts": {}, "n": 0}
    count = lambda k, n=1: info["counts"].__setitem__(k, info["counts"].get(k, 0) + n)
    for i in select(dets, cfg, tracks):
        d, g = dets[i], group_of(dets[i])
        mand = mandatory(d, cfg, tracks)                                 # evaluated on the first-pass score
        hit = next((e for e in cache if e["group"] == g and iou(e["box"], d["box"]) > MATCH_IOU), None)
        if hit is not None:                                              # re-checked recently: reuse the result, follow the object
            hit["box"] = tuple(d["box"])
            _apply(d, hit["status"], hit["score"], cfg)
            if mand and hit.get("doubt"):
                d["second_look_doubt"] = True
            count("sl_reused")
            continue
        if info["n"] >= (0 if reduced else _SL["max"]):
            if mand:                                                     # the cap (or a reduced frame) prevented the check: it cannot count for confirmation here
                d["second_look"], d["second_look_doubt"] = "skipped", True
                count("shed" if reduced else "sl_skipped")
            elif reduced and info["counts"].get("shed", 0) < _SL["max"]:      # the passes that would have run (at most the per-frame cap)
                count("shed")
            continue
        win = crop_for(d["box"], W, H)
        res, tm = engine.recheck(img, win, g, cfg.cont_thr, cfg.vehicle_thr, cfg.animal_cont_thr, _SL["reject_factor"])
        for k in ("preprocess_ms", "infer_ms", "post_ms"):
            info[k] += tm[k]
        info["n"] += 1
        b = d["box"]
        inside = [r for r in res if b[0] <= (r[1] + r[3]) / 2 <= b[2] and b[1] <= (r[2] + r[4]) / 2 <= b[3]]
        best = max(inside, key=lambda r: r[0]) if inside else None
        if best is None:
            status, score = "not_found", 0.0                             # e.g. a half-occluded parked car: never dropped for this
        elif best[0] >= d["score"]:
            status, score = "confirmed", best[0]
        elif best[0] < _SL["reject_factor"] * keep_thr(d, cfg) and d["score"] < confirm_thr(g, cfg):
            status, score = "rejected", best[0]                          # positively found again, far below the keep floor: confident downgrade
        elif best[0] < keep_thr(d, cfg):
            status, score = "not_upgraded", 0.0                          # found, but weak: keep the detection exactly as it was
        else:
            status, score = "lowered", best[0]
        doubt = g in _SL["confirm_min"] and (best is None or best[0] < _SL["confirm_min"][g])
        _apply(d, status, score, cfg)
        if mand and doubt:
            d["second_look_doubt"] = True
        cache.append({"group": g, "box": tuple(b), "t": now, "status": status, "score": score, "doubt": doubt})
        count("sl"); count("sl_" + status)
    return dets, info
