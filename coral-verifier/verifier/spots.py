"""Known parking spots (0.3.31). PURE: no I/O, no clock reads (callers pass times); the server loads and saves the exported state.

A spot is a place where vehicles park on this camera: learnt when a vehicle track becomes parked, merged with an existing spot when the
boxes overlap (IoU >= SPOTS["iou"]), and TRUSTED once vehicles parked there on SPOTS["trusted_after"] separate occasions (an occasion is a
parked track that starts more than SPOTS["occasion_gap_s"] after the spot last saw one), so a one-off visitor never makes one. Each spot keeps
up to SPOTS["max_looks"] different looks (shape.signature of the parked vehicle: two cars that share the spot, day and IR).

A trusted spot changes two things in the vehicle tracker: a new vehicle whose look matches one of the spot's looks is parked at once
(also right after a restart), and a weak vehicle box (between WEAK_VEHICLE_THR and vehicle_thr) at the same place keeps a vehicle parked
there. Spots belong to the camera, not to the session: a reset or an expired session keeps them.
"""
import base64

import numpy as np

import shape as SH
from defaults import SPOTS


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _enc(sig):
    return base64.b64encode(sig.astype(np.float16).tobytes()).decode("ascii")


def _dec(s):
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)


class Spot:
    __slots__ = ("box", "occasions", "last_t", "last_track", "looks")

    def trusted(self):
        return self.occasions >= SPOTS["trusted_after"]

    def matches(self, sig, thr):
        return sig is not None and any(SH.similarity(sig, lk) >= thr for lk in self.looks)


class Spots:
    """The spots of one camera."""

    def __init__(self):
        self.items = []
        self.dirty = False

    def at(self, box, trusted_only=True):
        """The spot under `box` (best IoU >= SPOTS["iou"]), or None."""
        best, bi = None, SPOTS["iou"]
        for sp in self.items:
            if trusted_only and not sp.trusted():
                continue
            v = iou(box, sp.box)
            if v >= bi:
                best, bi = sp, v
        return best

    def learn(self, box, sig, t, track_key):
        """A vehicle track just became parked (or is parked and seen again): record the place and its look. `track_key` tells
        occasions apart (one parked track counts once)."""
        sp = self.at(box, trusted_only=False)
        if sp is None:
            if len(self.items) >= SPOTS["max_spots"]:
                self.items.sort(key=lambda s: (s.trusted(), s.last_t))
                self.items.pop(0)                                    # the least useful: untrusted first, then the longest unused
            sp = Spot()
            sp.box, sp.occasions, sp.last_t, sp.last_track, sp.looks = tuple(box), 1, t, track_key, []
            self.items.append(sp)
        else:
            if sp.last_track != track_key and t - sp.last_t > SPOTS["occasion_gap_s"] * 1000.0:
                sp.occasions += 1
            sp.last_track = track_key
            sp.last_t = t
            a = SPOTS["box_alpha"]
            sp.box = tuple((1 - a) * p + a * q for p, q in zip(sp.box, box))
        if sig is not None and all(SH.similarity(sig, lk) < SPOTS["new_look"] for lk in sp.looks):
            sp.looks.append(sig)
            del sp.looks[:-SPOTS["max_looks"]]
        self.dirty = True
        return sp

    def seen(self, sp, t, track_key):
        """A vehicle is still parked in `sp`: keeps it from being the one dropped when the camera has too many spots."""
        sp.last_t, sp.last_track = t, track_key

    def summary(self):
        return {"spots": len(self.items), "trusted": sum(1 for s in self.items if s.trusted())}

    def export(self):
        return [{"box": [round(v, 4) for v in sp.box], "occasions": sp.occasions, "last_t": int(sp.last_t),
                 "looks": [_enc(lk) for lk in sp.looks]} for sp in self.items]

    def describe(self):
        """For GET .../spots: the export without the looks themselves."""
        return [{"box": [round(v, 4) for v in sp.box], "occasions": sp.occasions, "trusted": sp.trusted(), "last_t": int(sp.last_t),
                 "looks": len(sp.looks)} for sp in self.items]

    @classmethod
    def restore(cls, data):
        out = cls()
        for d in data or []:
            try:
                sp = Spot()
                sp.box = tuple(float(v) for v in d["box"])
                sp.occasions, sp.last_t, sp.last_track = int(d["occasions"]), float(d["last_t"]), None
                sp.looks = [lk for lk in (_dec(x) for x in d.get("looks", [])) if lk.size == SH.SIG_W * SH.SIG_H]
                if len(sp.box) == 4:
                    out.items.append(sp)
            except (KeyError, TypeError, ValueError):
                continue                                             # one bad entry never costs the others
        del out.items[SPOTS["max_spots"]:]
        return out
