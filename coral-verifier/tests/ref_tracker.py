"""REFERENCE implementation used by the differential test (tests/test_tracker_diff.py).
Verbatim copy (numpy replaced by a plain list, trace removed) of iou / gate_pairs / greedy / Trk / run_tracker from
the author's 1 fps reference tracker simulation. Do not "fix" it: it is the specification."""



# ---------------- tracker ----------------
def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0]); iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0: return 0.0
    i = ix * iy; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

def gate_pairs(trk_boxes, det_boxes, gate):
    """candidate (cost, ti, dj) with cost ascending = better"""
    kind, g = gate; out = []
    for i, tb in enumerate(trk_boxes):
        for j, db in enumerate(det_boxes):
            if kind == "iou":
                v = iou(tb, db)
                if v >= g: out.append((-v, i, j))
            else:   # centre distance normalised by the track box's longer side
                dx = (tb[0] + tb[2] - db[0] - db[2]) / 2; dy = (tb[1] + tb[3] - db[1] - db[3]) / 2
                n = max(tb[2] - tb[0], tb[3] - tb[1], 1e-3); v = (dx * dx + dy * dy) ** 0.5 / n
                if v <= g: out.append((v, i, j))
    return out

def greedy(trk_boxes, det_boxes, gate):
    pr = gate_pairs(trk_boxes, det_boxes, gate); pr.sort(); ut, ud = set(), set(); res = []
    for c, i, j in pr:
        if i in ut or j in ud: continue
        ut.add(i); ud.add(j); res.append((i, j))
    return res

class Trk:
    __slots__ = ("id", "box", "last", "born", "hits", "conf", "hist", "still")
def run_tracker(frames, nv, T_hi, T_lo, gate, K, Kst, confirm, trace=False):
    """frames: 60 lists of (score,x0,y0,x1,y1). Returns pres (nv bool), first_conf_t (or None), trace rows."""
    tracks = []; nid = 0; pres = [False] * nv; first_conf = None; rows = []
    hold = lambda tr: Kst if (Kst and tr.still) else K
    for t in range(nv):
        # pool: tracks that were present at t-1 (confirmed) or still tentative
        pool = []
        for tr in tracks:
            if tr.conf:
                if (t - 1) - tr.last <= hold(tr): pool.append(tr)
            elif t <= tr.born + 2: pool.append(tr)
        tracks = pool
        dets = [d for d in frames[t] if d[0] >= T_lo]
        hi = [d for d in dets if d[0] >= T_hi]; lo = [d for d in dets if d[0] < T_hi]
        matched = {}   # id(tr) -> (det, 'H'/'L')
        # 1. all tracks (tentative included) vs high dets
        ms = greedy([tr.box for tr in tracks], [d[1:] for d in hi], gate)
        used_d = set(); used_t = set()
        for i, j in ms: matched[i] = (hi[j], "H"); used_t.add(i); used_d.add(j)
        # 2. unmatched CONFIRMED tracks vs low dets
        rem = [i for i, tr in enumerate(tracks) if i not in used_t and tr.conf]
        if rem and lo:
            for ii, j in greedy([tracks[i].box for i in rem], [d[1:] for d in lo], gate): matched[rem[ii]] = (lo[j], "L")
        for i, (d, kind) in matched.items():
            tr = tracks[i]; box = d[1:]; tr.box = box; tr.last = t; tr.hist.append((t, box))
            if Kst:
                tr.still = False
                for tt, bb in reversed(tr.hist[:-1]):
                    if tt <= t - 3: tr.still = iou(box, bb) > 0.7; break
            if not tr.conf:
                tr.hits += 1
                if tr.hits >= 2: tr.conf = True; first_conf = t if first_conf is None else first_conf
        # 3. new tracks from unmatched high dets
        for j, d in enumerate(hi):
            if j in used_d: continue
            tr = Trk(); tr.id = nid; nid += 1; tr.box = d[1:]; tr.last = t; tr.born = t; tr.hits = 1; tr.hist = [(t, d[1:])]; tr.still = False
            tr.conf = not confirm
            if tr.conf: first_conf = t if first_conf is None else first_conf
            tracks.append(tr)
        pres[t] = any(tr.conf and t - tr.last <= hold(tr) for tr in tracks)
        if trace:
            kinds = sorted({k for _, k in matched.values()}); live = [tr.id for tr in tracks if tr.conf and t - tr.last <= hold(tr)]
            rows.append((t, round(frames[t][0][0], 2) if frames[t] else 0.0, "".join(kinds) or "-", live))
    return pres, first_conf, rows
