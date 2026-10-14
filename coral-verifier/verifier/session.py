"""Per-camera session logic. PURE: no I/O, no clock reads (callers pass times), one lock in Registry.

Person: a streaming port of the ByteTrack-style tracker of the 1 fps reference simulation (run_tracker in tests/ref_tracker.py);
tests/ref_tracker.py is that function and tests/test_tracker_diff.py requires ZERO disagreements with it at 1 fps. The simulation counts in
integer ticks (1 tick = 1 s); here a "tick" is a whole second of FRAME time: gaps are rounded to whole seconds, and frames closer than
TICK_MERGE_MS to the first frame of the current tick belong to the SAME tick (their detections are merged, best score wins), so pre-roll bursts
and 4 fps senders cannot confirm on near-duplicate frames.
  - confirmation: >= confirm_n ticks with a box >= start_thr inside confirm_m seconds (tentative track lives while gap <= confirm_m - 1 s)
  - continuation: boxes >= cont_thr match confirmed tracks by centre distance (gate 1.5 box sides), greedy by distance
  - hold: a confirmed track is present while gap <= hold_s (still_hold_s for a stationary track), and, exactly like the reference simulation, a track can still
    be re-matched when the gap was <= hold + 1 s at the previous tick (the reference simulation's off-by-one)
  - stationary lock: IoU > 0.7 between the current box and the track's box still_s (default 3 s, as in the reference simulation) earlier, rolling
  - max_hold_s: SERVICE-SIDE safety with no counterpart in the reference simulation: a track with no detection >= start_thr for that long ends (default 600 s)
Out-of-order `t` is clamped to the last seen t. Time of the state seen by GET (`snapshot(wall_now)`) advances with the server clock.
"""
import math, re, threading

from defaults import TRACKER as _T, DETECT_THR, ANIMAL as _A, MOVE_REF_S, MOVE_DIAG_FRAC, MOVE_IOU
from dataclasses import dataclass, fields, asdict

VEHICLES = tuple(DETECT_THR["vehicle"])       # bicycle included: same tracking and parked logic as the motor vehicles
ANIMALS = tuple(DETECT_THR["animal"])
GATE = 1.5            # centre-distance gate (in units of the track's longest box side), from the reference simulation
MATCH_IOU_VEHICLE = 0.3
STILL_IOU = 0.7
PARKED_BREAK_IOU = 0.5
MAX_TRACK_DETS = 12
MAX_TRACKS = 32          # per tracker: bounds memory and the O(n*m) matching on pathological scenes (never reached by real ones)
TICK_MERGE_MS = 750
MERGE_IOU = 0.5
T_PAST_MS = 60_000    # accepted frame time window around the server clock: [now - 60 s, now + 5 s]
T_FUTURE_MS = 5_000
OVERRIDE_IDLE_S = 7 * 86400


class TooManyCameras(Exception):
    pass


@dataclass
class Config:
    start_thr: float = _T["start_thr"]
    cont_thr: float = _T["cont_thr"]
    confirm_n: int = _T["confirm_n"]
    confirm_m: int = _T["confirm_m"]
    hold_s: float = _T["hold_s"]
    still_s: float = _T["still_s"]
    still_hold_s: float = _T["still_hold_s"]
    max_hold_s: float = _T["max_hold_s"]
    parked_after_s: float = _T["parked_after_s"]
    session_ttl_s: float = _T["session_ttl_s"]
    vehicle_thr: float = _T["vehicle_thr"]
    animal_start_thr: float = _T["animal_start_thr"]
    animal_cont_thr: float = _T["animal_cont_thr"]
    zoom_out: float = _T["zoom_out"]
    second_look: bool = _T["second_look"]
    name: str = _T["name"]

    def validate(self):
        for k, v in asdict(self).items():
            if not isinstance(v, str) and not math.isfinite(v):
                raise ValueError(f"{k} must be finite")
        if len(self.name) > 64:
            raise ValueError("name must be at most 64 characters")
        for k in ("start_thr", "cont_thr"):
            if not 0 < getattr(self, k) <= 1:
                raise ValueError(f"{k} must be in (0, 1]")
        if not 0 <= self.zoom_out <= 0.6:
            raise ValueError("zoom_out must be in [0, 0.6]")
        if not 0.1 <= self.vehicle_thr <= 0.95:
            raise ValueError("vehicle_thr must be in [0.1, 0.95]")
        for k in ("animal_start_thr", "animal_cont_thr"):
            if not 0.05 <= getattr(self, k) <= 0.95:
                raise ValueError(f"{k} must be in [0.05, 0.95]")
        if self.animal_cont_thr > self.animal_start_thr:
            raise ValueError("animal_cont_thr must be <= animal_start_thr")
        if self.cont_thr > self.start_thr:
            raise ValueError("cont_thr must be <= start_thr")
        if not (1 <= self.confirm_n <= self.confirm_m <= 20):
            raise ValueError("need 1 <= confirm_n <= confirm_m <= 20")
        for k in ("hold_s", "still_s", "still_hold_s", "max_hold_s", "parked_after_s", "session_ttl_s"):
            if not 0 <= getattr(self, k) <= 86400:
                raise ValueError(f"{k} must be in [0, 86400]")
        if self.session_ttl_s <= 0:
            raise ValueError("session_ttl_s must be > 0")
        return self

    def updated(self, changes):
        """Returns a new validated Config with `changes` (dict) applied. Unknown keys / bad types raise ValueError."""
        known = {f.name: f.type for f in fields(self)}
        cur = asdict(self)
        for k, v in changes.items():
            if k not in known:
                raise ValueError(f"unknown config key {k!r}"[:80])
            if known[k] is str:
                if not isinstance(v, str):
                    raise ValueError(f"{k} must be a string")
                v = " ".join(re.sub(r"[<>\"'`&]", "", re.sub(r"[\x00-\x1f\x7f]", " ", v)).split())     # control characters -> space, markup characters removed, whitespace collapsed
                if len(v) > 64:
                    raise ValueError(f"{k} must be at most 64 characters")
                cur[k] = v
                continue
            if known[k] is bool:
                if not isinstance(v, bool):
                    raise ValueError(f"{k} must be true or false")
                cur[k] = v
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{k} must be a number")
            if not math.isfinite(v):
                raise ValueError(f"{k} must be finite")
            if known[k] is int:
                if int(v) != v:
                    raise ValueError(f"{k} must be an integer")
                v = int(v)
            else:
                v = float(v)
            cur[k] = v
        return Config(**cur).validate()

    def as_dict(self):
        return asdict(self)


def bound_t(t, now_ms):
    """Frame time sanity: -> (t_ms, clamped). Only finite t inside [now - 60 s, now + 5 s] is trusted; anything else becomes the server time."""
    try:
        ok = math.isfinite(t) and now_ms - T_PAST_MS <= t <= now_ms + T_FUTURE_MS
    except TypeError:
        ok = False
    return (int(t), False) if ok else (int(now_ms), True)


def secs(a_ms, b_ms):
    """Whole seconds between two frame times (rounded, so +-0.5 s sender jitter does not change a tick count)."""
    return math.floor((a_ms - b_ms) / 1000.0 + 0.5)


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0]); iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _cd(tb, db):
    dx = (tb[0] + tb[2] - db[0] - db[2]) / 2; dy = (tb[1] + tb[3] - db[1] - db[3]) / 2
    n = max(tb[2] - tb[0], tb[3] - tb[1], 1e-3)
    return (dx * dx + dy * dy) ** 0.5 / n


def greedy_cd(trk_boxes, det_boxes, gate=GATE):
    pr = []
    for i, tb in enumerate(trk_boxes):
        for j, db in enumerate(det_boxes):
            v = _cd(tb, db)
            if v <= gate:
                pr.append((v, i, j))
    return _greedy(pr)


def greedy_iou(trk_boxes, det_boxes, thr):
    pr = []
    for i, tb in enumerate(trk_boxes):
        for j, db in enumerate(det_boxes):
            v = iou(tb, db)
            if v >= thr:
                pr.append((-v, i, j))
    return _greedy(pr)


def _greedy(pairs):
    pairs.sort(); ut, ud, res = set(), set(), []
    for _, i, j in pairs:
        if i in ut or j in ud:
            continue
        ut.add(i); ud.add(j); res.append((i, j))
    return res


def merge_dets(a, b):
    """Detections of two frames of the same tick, [(score, box)]: a b-detection overlapping an a-detection (IoU > 0.5) is the same object, best score wins."""
    out = list(a)
    for d in b:
        best, bi = MERGE_IOU, -1
        for k in range(len(a)):
            v = iou(d[1], out[k][1])
            if v > best:
                best, bi = v, k
        if bi < 0:
            out.append(d)
        elif d[0] > out[bi][0]:
            out[bi] = d
    return out


class _PTrack:
    __slots__ = ("id", "box", "last", "last_hi", "born", "hits", "conf", "hist", "still", "score")


class PersonTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self.nid = 0
        self.last_seen = None
        self.first_conf = None

    @staticmethod
    def _clone(tr):
        c = _PTrack()
        for k in _PTrack.__slots__:
            setattr(c, k, getattr(tr, k))
        c.hist = list(tr.hist)                       # entries are immutable tuples
        return c

    def export(self):
        return [self._clone(t) for t in self.tracks], self.nid, self.last_seen, self.first_conf

    def restore(self, snap):
        tracks, self.nid, self.last_seen, self.first_conf = snap
        self.tracks = [self._clone(t) for t in tracks]

    def _hold(self, tr):
        c = self.cfg
        return c.still_hold_s if (c.still_hold_s and tr.still) else c.hold_s

    def _max_ok(self, tr, t, off):
        return secs(t, tr.last_hi) - off <= self.cfg.max_hold_s

    def step(self, t, t_latest, dets):
        """One tick. t: tick time (first frame of the tick, ms); t_latest: latest frame time of the tick; dets: [(score, (x0,y0,x1,y1))]."""
        c = self.cfg
        pool = []
        for tr in self.tracks:
            if tr.conf:
                if secs(t, tr.last) - 1 <= self._hold(tr) and self._max_ok(tr, t, 1):      # the reference simulation checks the gap at the PREVIOUS tick
                    pool.append(tr)
            elif secs(t, tr.born) <= c.confirm_m - 1:
                pool.append(tr)
        trks = self.tracks = pool
        dets = [d for d in dets if d[0] >= c.cont_thr]
        hi = [d for d in dets if d[0] >= c.start_thr]
        lo = [d for d in dets if d[0] < c.start_thr]
        matched, used_t, used_d = {}, set(), set()
        for i, j in greedy_cd([tr.box for tr in trks], [d[1] for d in hi]):
            matched[i] = (hi[j], True); used_t.add(i); used_d.add(j)
        rem = [i for i, tr in enumerate(trks) if i not in used_t and tr.conf]
        if rem and lo:
            for ii, j in greedy_cd([trks[i].box for i in rem], [d[1] for d in lo]):
                matched[rem[ii]] = (lo[j], False)
        for i, (d, is_hi) in matched.items():
            tr = trks[i]; box = d[1]
            tr.box = box; tr.last = t; tr.score = d[0]
            if is_hi:
                tr.last_hi = t
            tr.hist.append((t, box))
            tr.still = False
            for tt, bb in reversed(tr.hist[:-1]):
                if secs(t, tt) >= c.still_s:
                    tr.still = iou(box, bb) > STILL_IOU
                    break
            while len(tr.hist) >= 3 and secs(t, tr.hist[1][0]) >= c.still_s:                 # older entries can never be the reference again
                tr.hist.pop(0)
            if not tr.conf:
                tr.hits += 1
                if tr.hits >= c.confirm_n:
                    tr.conf = True
                    if self.first_conf is None:
                        self.first_conf = t
        for j, d in enumerate(hi):
            if j in used_d:
                continue
            tr = _PTrack(); tr.id = self.nid; self.nid += 1
            tr.box = d[1]; tr.last = t; tr.last_hi = t; tr.born = t; tr.hits = 1; tr.hist = [(t, d[1])]; tr.still = False; tr.score = d[0]
            tr.conf = c.confirm_n <= 1
            if tr.conf and self.first_conf is None:
                self.first_conf = t
            trks.append(tr)
        if len(trks) > MAX_TRACKS:
            trks.sort(key=lambda tr: (not tr.conf, -tr.last))
            del trks[MAX_TRACKS:]
        if any(tr.conf and tr.last == t for tr in trks):
            self.last_seen = t_latest

    def live(self, eff):
        return [tr for tr in self.tracks if tr.conf and secs(eff, tr.last) <= self._hold(tr) and self._max_ok(tr, eff, 0)]

    def state(self, eff):
        live = self.live(eff)
        if live:
            return "confirmed" if any(secs(eff, tr.last) == 0 for tr in live) else "holding"
        return "candidate" if any((not tr.conf) and secs(eff, tr.born) <= self.cfg.confirm_m - 1 for tr in self.tracks) else "idle"


class _VTrack:
    __slots__ = ("id", "cls", "box", "last", "born_t", "anchor", "stable_since", "parked", "hist", "moving")


class VehicleTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self.nid = 0

    def _alive(self, tr, t):
        hold = self.cfg.max_hold_s if tr.parked else self.cfg.hold_s
        return t - tr.last <= hold * 1000.0

    def step(self, t, dets):
        c = self.cfg
        trks = self.tracks = [tr for tr in self.tracks if self._alive(tr, t)]
        used = set()
        for i, j in greedy_iou([tr.box for tr in trks], [d[2] for d in dets], MATCH_IOU_VEHICLE):
            tr = trks[i]; d = dets[j]; used.add(j)
            tr.cls, tr.box, tr.last = d[0], d[2], t
            tr.moving = self._moved(tr, t, d[2])
            thr = PARKED_BREAK_IOU if tr.parked else STILL_IOU
            if iou(d[2], tr.anchor) > thr:
                if not tr.parked and (t - tr.stable_since) > c.parked_after_s * 1000.0:
                    tr.parked = True
            else:
                tr.anchor, tr.stable_since, tr.parked = d[2], t, False
        for j, d in enumerate(dets):
            if j in used:
                continue
            tr = _VTrack(); tr.id = self.nid; self.nid += 1
            tr.cls, tr.box, tr.anchor, tr.last, tr.born_t, tr.stable_since, tr.parked = d[0], d[2], d[2], t, t, t, False
            tr.hist, tr.moving = [(t, d[2])], False                     # a new track is "still" (unknown) until it actually moves or parks
            trks.append(tr)
        if len(trks) > MAX_TRACKS:
            trks.sort(key=lambda tr: (tr.parked, -tr.last))
            del trks[MAX_TRACKS:]

    @staticmethod
    def _moved(tr, t, box):
        """Moving = centre displacement > MOVE_DIAG_FRAC of the normalised frame diagonal, or IoU < MOVE_IOU, against the box at least MOVE_REF_S earlier."""
        h = tr.hist
        h.append((t, box))
        while len(h) > 2 and t - h[1][0] >= MOVE_REF_S * 1000.0:
            del h[0]
        t0, b0 = h[0]
        if t - t0 < MOVE_REF_S * 1000.0:
            return False
        dx = (box[0] + box[2] - b0[0] - b0[2]) / 2.0
        dy = (box[1] + box[3] - b0[1] - b0[3]) / 2.0
        return math.hypot(dx, dy) > MOVE_DIAG_FRAC * math.sqrt(2.0) or iou(box, b0) < MOVE_IOU

    def snapshot(self, eff):
        live = [tr for tr in self.tracks if self._alive(tr, eff)]
        st = lambda tr: "parked" if tr.parked else ("moving" if tr.moving else "still")
        hold = max((tr.last + (self.cfg.max_hold_s if tr.parked else self.cfg.hold_s) * 1000.0 for tr in live), default=None)
        return {"moving": any(st(tr) == "moving" for tr in live), "hold_until_ms": None if hold is None else int(hold),
                "tracks": [{"id": tr.id, "cls": tr.cls, "state": st(tr), "parked": tr.parked, "since_ms": int(tr.born_t)} for tr in live]}


class AnimalTracker:
    """Animals, fully independent of the person state: a second PersonTracker (same tick logic, animal constants from defaults.ANIMAL) decides
    `confirmed`/`present`; simple greedy IoU tracks (no parked concept) provide ids, class and since_ms and live as long as the hold."""
    def __init__(self, base):
        self.cfg = self._acfg(base)
        self.pt = PersonTracker(self.cfg)
        self.tracks = []
        self.nid = 0
        self._tick_t = self._tick_snap = self._tick_dets = None

    @staticmethod
    def _acfg(base):
        return base.updated({**_A, "start_thr": base.animal_start_thr, "cont_thr": base.animal_cont_thr})

    def set_cfg(self, base):
        self.cfg = self.pt.cfg = self._acfg(base)

    def step(self, t, dets):
        """dets: [(cls, score, box)] >= animal floor."""
        pd = [(d[1], d[2]) for d in dets]
        if self._tick_t is not None and t - self._tick_t < TICK_MERGE_MS:
            self.pt.restore(self._tick_snap)
            self._tick_dets = pd = merge_dets(self._tick_dets, pd)
        else:
            self._tick_t, self._tick_snap, self._tick_dets = t, self.pt.export(), pd
        self.pt.step(self._tick_t, t, pd)
        hold = self.cfg.hold_s * 1000.0
        trks = self.tracks = [tr for tr in self.tracks if t - tr.last <= hold]
        used = set()
        for i, j in greedy_iou([tr.box for tr in trks], [d[2] for d in dets], MATCH_IOU_VEHICLE):
            tr = trks[i]; used.add(j)
            tr.cls, tr.box, tr.last = dets[j][0], dets[j][2], t
        for j, d in enumerate(dets):
            if j not in used:
                tr = _VTrack(); tr.id = self.nid; self.nid += 1
                tr.cls, tr.box, tr.last, tr.born_t = d[0], d[2], t, t
                trks.append(tr)
        if len(trks) > MAX_TRACKS:
            trks.sort(key=lambda tr: -tr.last)
            del trks[MAX_TRACKS:]

    def snapshot(self, eff):
        """eff: tick time advanced by the server clock (same clock as the person state; both trackers see the same frames)."""
        present = self.pt.state(eff) in ("confirmed", "holding")
        live = [tr for tr in self.tracks if eff - tr.last <= self.cfg.hold_s * 1000.0] if present else []
        ptl = self.pt.live(eff) if present else []
        hold = max((tr.last + self.pt._hold(tr) * 1000.0 for tr in ptl), default=None)
        return {"present": present, "confirmed": present, "hold_until_ms": None if hold is None else int(hold),
                "tracks": [{"id": tr.id, "cls": tr.cls, "since_ms": int(tr.born_t), "last_seen_ms": int(tr.last), "box": [round(v, 4) for v in tr.box]} for tr in live]}


class Session:
    def __init__(self, cfg, wall_now=0.0):
        self.cfg = cfg
        self.person = PersonTracker(cfg)
        self.vehicle = VehicleTracker(cfg)
        self.animal = AnimalTracker(cfg)
        self.frames = 0
        self.started_ms = None
        self.last_t = None
        self.wall = wall_now          # monotonic seconds when the last frame was RECEIVED (server clock), for TTL and GET-time expiry
        self._tick_t = None
        self._tick_snap = None
        self._tick_dets = None

    def update(self, t_ms, detections, wall_now=None):
        """detections: [{"cls": str, "score": float, "box": [x0,y0,x1,y1]}]. Returns the effective (clamped) t."""
        t = int(t_ms)
        if self.last_t is not None and t < self.last_t:
            t = self.last_t
        if self.started_ms is None:
            self.started_ms = t
        self.last_t = t
        if wall_now is not None:
            self.wall = wall_now
        pdets = [(float(d["score"]), tuple(d["box"])) for d in detections if d["cls"] == "person"]
        vdets = [(d["cls"], float(d["score"]), tuple(d["box"])) for d in detections if d["cls"] in VEHICLES and not d.get("weak") and d["score"] >= (DETECT_THR["vehicle"][d["cls"]] if d["cls"] == "bicycle" else self.cfg.vehicle_thr)]
        adets = [(d["cls"], float(d["score"]), tuple(d["box"])) for d in detections if d["cls"] in ANIMALS and d["score"] >= self.cfg.animal_cont_thr]
        pdets = sorted(pdets, key=lambda d: -d[0])[:MAX_TRACK_DETS]          # bound the O(n*m) matching on pathological frames
        vdets = sorted(vdets, key=lambda d: -d[1])[:MAX_TRACK_DETS]
        adets = sorted(adets, key=lambda d: -d[1])[:MAX_TRACK_DETS]
        if self._tick_t is not None and t - self._tick_t < TICK_MERGE_MS:     # same tick: redo it with the merged detections
            self.person.restore(self._tick_snap)
            self._tick_dets = pdets = merge_dets(self._tick_dets, pdets)
        else:
            self._tick_t = t
            self._tick_snap = self.person.export()
            self._tick_dets = pdets
        self.person.step(self._tick_t, t, pdets)
        self.vehicle.step(t, vdets)
        self.animal.step(t, adets)
        self.frames += 1
        return t

    def snapshot(self, wall_now=None):
        """State as of the last frame, advanced by the server time elapsed since it was received (hold and vehicle expiry never freeze)."""
        t = self.last_t
        if t is None:
            return empty_snapshot()
        eff = self._tick_t + (max(0.0, wall_now - self.wall) * 1000.0 if wall_now is not None else 0.0)     # tick time, like the reference simulation
        p = self.person
        live = p.live(eff)
        state = p.state(eff)
        present = state in ("confirmed", "holding")
        return {
            "t": t,
            "person": {"state": state, "confirmed": present, "present": present,
                       "since_ms": int(min(tr.born for tr in live)) if live else None,
                       "last_seen_ms": p.last_seen, "tracks": len(live),
                       "tracks_detail": [{"id": tr.id, "box": [round(v, 4) for v in tr.box], "score": round(float(tr.score), 3), "last_seen_ms": int(tr.last),
                                          "since_ms": int(tr.born), "stationary": bool(tr.still)} for tr in live],
                       "hold_until_ms": int(max(tr.last + p._hold(tr) * 1000.0 for tr in live)) if live else None},
            "vehicle": self.vehicle.snapshot(eff),
            "animal": self.animal.snapshot(eff),
            "session": {"frames": self.frames, "started_ms": self.started_ms},
        }


def empty_snapshot():
    return {"t": None,
            "person": {"state": "idle", "confirmed": False, "present": False, "since_ms": None, "last_seen_ms": None, "tracks": 0, "tracks_detail": [],
                       "hold_until_ms": None},
            "vehicle": {"moving": False, "hold_until_ms": None, "tracks": []},
            "animal": {"present": False, "confirmed": False, "hold_until_ms": None, "tracks": []},
            "session": {"frames": 0, "started_ms": None}}


class Registry:
    """Sessions keyed by camera, per-camera config overrides (kept across reset/expiry), TTL sweep, bounded number of cameras. Thread-safe."""

    def __init__(self, defaults=None, max_cameras=64):
        self.defaults = defaults or Config()
        self.max_cameras = max_cameras
        self._sessions = {}
        self._sl = {}            # camera -> second-look cache entries (see secondlook.py)
        self._over = {}          # camera -> (dict of overrides, last touched wall)
        self._next_sweep = 0.0
        self.lock = threading.RLock()

    def _live(self, camera, wall_now):
        s = self._sessions.get(camera)
        if s is not None and wall_now - s.wall > s.cfg.session_ttl_s:
            del self._sessions[camera]
            return None
        return s

    def sweep(self, wall_now):
        """Amortised O(1): the full scan runs at most once a second."""
        with self.lock:
            if wall_now < self._next_sweep:
                return
            self._next_sweep = wall_now + 1.0
            for cam in [c for c, s in self._sessions.items() if wall_now - s.wall > s.cfg.session_ttl_s]:
                del self._sessions[cam]
            for cam in [c for c in self._sl if c not in self._sessions]:
                del self._sl[cam]
            for cam in [c for c, (_, w) in self._over.items() if wall_now - w > OVERRIDE_IDLE_S and c not in self._sessions]:
                del self._over[cam]

    def _admit(self, camera, wall_now):
        if camera in self._sessions or camera in self._over:
            return
        if len(set(self._sessions) | set(self._over)) >= self.max_cameras:
            raise TooManyCameras(camera)

    def count_known(self):
        with self.lock:
            return len(set(self._sessions) | set(self._over))

    def config(self, camera):
        with self.lock:
            return self.defaults.updated(self._over.get(camera, ({}, 0))[0])

    def set_config(self, camera, changes, wall_now=0.0):
        with self.lock:
            self.sweep(wall_now); self._admit(camera, wall_now)
            merged = dict(self._over.get(camera, ({}, 0))[0]); merged.update(changes)
            cfg = self.defaults.updated(merged)       # validates
            self._over[camera] = (merged, wall_now)
            s = self._sessions.get(camera)
            if s is not None:                         # live session adopts the new config (trackers share the object)
                s.cfg = s.person.cfg = s.vehicle.cfg = cfg
                s.animal.set_cfg(cfg)
            return cfg

    def update(self, camera, t_ms, detections, wall_now):
        with self.lock:
            self.sweep(wall_now)
            s = self._live(camera, wall_now)
            if s is None:
                self._admit(camera, wall_now)
                s = self._sessions[camera] = Session(self.config(camera), wall_now)
            s.update(t_ms, detections, wall_now)
            return s.snapshot(wall_now)

    def known(self, camera, wall_now):
        with self.lock:
            self.sweep(wall_now)
            return self._live(camera, wall_now) is not None or camera in self._over

    def would_admit(self, camera, wall_now):
        with self.lock:
            self.sweep(wall_now)
            try:
                self._live(camera, wall_now); self._admit(camera, wall_now)
                return True
            except TooManyCameras:
                return False

    def peek(self, camera, wall_now):
        with self.lock:
            self.sweep(wall_now)
            s = self._live(camera, wall_now)
            return s.snapshot(wall_now) if s else empty_snapshot()

    def sl_context(self, camera, wall_now):
        """For the second look (worker thread): boxes of the live tracks per group and a copy of the camera's re-check cache."""
        with self.lock:
            s = self._live(camera, wall_now)
            tracks = {} if s is None else {"person": [tuple(t.box) for t in s.person.tracks], "vehicle": [tuple(t.box) for t in s.vehicle.tracks],
                                           "animal": [tuple(t.box) for t in s.animal.tracks],
                                           "person_conf": [tuple(t.box) for t in s.person.tracks if t.conf],            # confirmed tracks only
                                           "animal_conf": [tuple(t.box) for t in s.animal.tracks] if s.animal.pt.live(s.animal._tick_t or 0) else []}
            return {"tracks": tracks, "cache": list(self._sl.get(camera, []))}

    def sl_store(self, camera, cache, wall_now):
        with self.lock:
            if cache:
                self._sl[camera] = cache[-16:]
            else:
                self._sl.pop(camera, None)

    def reset(self, camera):
        with self.lock:
            self._sl.pop(camera, None)
            return self._sessions.pop(camera, None) is not None

    def listing(self, wall_now):
        with self.lock:
            self.sweep(wall_now)
            out = []
            for cam in sorted(self._sessions):
                s = self._live(cam, wall_now)
                if s is None:
                    continue
                sn = s.snapshot(wall_now)
                out.append({"camera": cam, "t": sn["t"], "person_state": sn["person"]["state"], "present": sn["person"]["present"],
                            "vehicle_moving": sn["vehicle"]["moving"], "frames": s.frames, "age_s": round(wall_now - s.wall, 1)})
            return out

    def count(self, wall_now):
        with self.lock:
            self.sweep(wall_now)
            return sum(1 for c in list(self._sessions) if self._live(c, wall_now) is not None)
