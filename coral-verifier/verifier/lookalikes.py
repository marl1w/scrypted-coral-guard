"""Known look-alikes (0.3.31). PURE: no I/O; the server learns entries from the journal and loads/saves the exported state.

When the user labels an episode `false_alarm`, the person or animal box that raised it (the best frame's strongest box of the episode's
group) is kept with its look (shape.signature): a statue, a plant, a coat on a chair. A later person/animal detection at the same place
(IoU >= LOOKALIKE["iou"], same group) whose look matches (>= LOOKALIKE["thr"], stricter than the vehicle shape hold) is that same object:
it is marked `lookalike` and never reaches the trackers. A real person in front of it changes the outline, so the match fails and the
person counts. Only the user's labels teach it (never the service's own verdicts), and relabelling the episode removes its entry.
Vehicles are left out: a vehicle false alarm is often a parked car, which must stay visible.

Known places (0.3.36): an object that moves in the wind (laundry on a rack, a plant) never looks the same twice (0.2-0.8 against its own
earlier looks), so the look match alone never catches it. Its place is kept too: a detection at a known place that does not match the look
still counts only when someone ARRIVED there, i.e. a confident detection of the same group outside every known place, next to this one,
within the last LOOKALIKE["entry_s"]. The laundry is born at its place and stays there; a person walks in from elsewhere.

0.3.44: a strong detection (LOOKALIKE["strong"] or more) is never left out for its place alone, only for
its look: one wrong Empty answer had made a doorway a known place and hid a real person there at 0.89. Look-only entries (`look_only`) are
learnt from blocked camera episodes the user answered Empty (a grill, a sofa scored as a person) and never act as places. A box covering
more than LOOKALIKE["max_area"] of the frame is never learnt.
"""
from collections import deque

import shape as SH
from defaults import LOOKALIKE
from geom import iou

GROUPS = ("person", "animal")


def capped(items):
    """At most LOOKALIKE["max"] entries learnt from false alarms / wrong classes and LOOKALIKE["max_look_only"] look-only ones, the oldest of
    each kind going first: the many Empty answers never push out what the user's own false alarms taught."""
    out, n = [], {False: 0, True: 0}
    for e in reversed(items):
        k = bool(e.get("look_only"))
        if n[k] < LOOKALIKE["max_look_only" if k else "max"]:
            n[k] += 1
            out.append(e)
    return out[::-1]


def learnable(box):
    """A box small enough to be one object: a whole-frame box (an insect on the lens, a car called a person) is never learnt."""
    return (box[2] - box[0]) * (box[3] - box[1]) <= LOOKALIKE["max_area"]


class Lookalikes:
    """The known look-alikes of one camera: [{"eid", "group", "box", "sig"}]."""

    def __init__(self):
        self.items = []
        self.dirty = False
        self.seen = deque(maxlen=64)       # (t_ms, group, box) of confident detections outside every known place (not saved)

    def saw(self, t, group, box):
        """A confident detection outside every known place of its group: someone who may be arriving at one."""
        self.seen.append((t, group, tuple(box)))

    def arrived(self, t, group, places):
        """True when a detection of `group` was seen next to one of `places` (each box grown by LOOKALIKE["near"] of its size on every side)
        within the last LOOKALIKE["entry_s"]."""
        lim = LOOKALIKE["entry_s"] * 1000.0
        while self.seen and t - self.seen[0][0] > lim:
            self.seen.popleft()
        for e in places:
            x0, y0, x1, y1 = e["box"]
            mx, my = (x1 - x0) * LOOKALIKE["near"], (y1 - y0) * LOOKALIKE["near"]
            if any(g == group and 0 <= t - ts <= lim and b[0] < x1 + mx and b[2] > x0 - mx and b[1] < y1 + my and b[3] > y0 - my
                   for ts, g, b in self.seen):
                return True
        return False

    def add(self, eid, group, box, sig, look_only=False):
        if group not in GROUPS or sig is None or not learnable(box):
            return False
        self.items = [e for e in self.items if e["eid"] != eid]
        self.items.append({"eid": eid, "group": group, "box": tuple(float(v) for v in box), "sig": sig, "look_only": bool(look_only)})
        self.items = capped(self.items)
        self.dirty = True
        return True

    def remove(self, eid):
        n = len(self.items)
        self.items = [e for e in self.items if e["eid"] != eid]
        if len(self.items) != n:
            self.dirty = True
            return True
        return False

    def known(self, eid):
        return any(e["eid"] == eid for e in self.items)

    def near(self, group, box):
        """Entries of `group` at the place of `box` (no image work: callers compute a signature only when this is not empty)."""
        return [e for e in self.items if e["group"] == group and iou(box, e["box"]) >= LOOKALIKE["iou"]]

    @staticmethod
    def places(entries, d=None):
        """The entries among `entries` that act as known places: never a look-only entry; none at all for a strong detection `d` (a second-look
        "confirmed" is no strength: it only means the zoomed pass scored at least as high)."""
        if d is not None and d["score"] >= LOOKALIKE["strong"]:
            return []
        return [e for e in entries if not e.get("look_only")]

    def match(self, group, box, gray):
        """Best look match (>= LOOKALIKE["thr"]) of a detection with the known look-alikes at its place, or None."""
        cand = self.near(group, box)
        if not cand:
            return None
        sig = SH.signature(gray, box)
        if sig is None:
            return None
        best = max(SH.similarity(sig, e["sig"]) for e in cand)
        return best if best >= LOOKALIKE["thr"] else None

    def describe(self):
        return [{"episode": e["eid"], "group": e["group"], "box": [round(v, 4) for v in e["box"]], **({"look_only": True} if e.get("look_only") else {})}
                for e in self.items]

    def export(self):
        return [{"eid": e["eid"], "group": e["group"], "box": [round(v, 4) for v in e["box"]],
                 "sig": SH.encode(e["sig"]), **({"look_only": True} if e.get("look_only") else {})} for e in self.items]

    @classmethod
    def restore(cls, data):
        out = cls()
        for d in data or []:
            try:
                sig = SH.decode(d["sig"])
                box = tuple(float(v) for v in d["box"])
                if sig.size == SH.SIG_W * SH.SIG_H and len(box) == 4 and d["group"] in GROUPS and learnable(box):
                    out.items.append({"eid": str(d["eid"]), "group": d["group"], "box": box, "sig": sig, "look_only": bool(d.get("look_only"))})
            except (KeyError, TypeError, ValueError):
                continue
        out.items = capped(out.items)
        out.dirty = len(out.items) != len(data or [])              # an entry dropped (too big, malformed, over the cap): the file is rewritten
        return out
