"""Known look-alikes (0.3.31). PURE: no I/O; the server learns entries from the journal and loads/saves the exported state.

When the user labels an episode `false_alarm`, the person or animal box that raised it (the best frame's strongest box of the episode's
group) is kept with its look (shape.signature): a statue, a plant, a coat on a chair. A later person/animal detection at the same place
(IoU >= LOOKALIKE["iou"], same group) whose look matches (>= LOOKALIKE["thr"], stricter than the vehicle shape hold) is that same object:
it is marked `lookalike` and never reaches the trackers. A real person in front of it changes the outline, so the match fails and the
person counts. Only the user's labels teach it (never the service's own verdicts), and relabelling the episode removes its entry.
Vehicles are left out: a vehicle false alarm is often a parked car, which must stay visible.
"""
import base64

import numpy as np

import shape as SH
from defaults import LOOKALIKE

GROUPS = ("person", "animal")


def _iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class Lookalikes:
    """The known look-alikes of one camera: [{"eid", "group", "box", "sig"}]."""

    def __init__(self):
        self.items = []
        self.dirty = False

    def add(self, eid, group, box, sig):
        if group not in GROUPS or sig is None:
            return False
        self.items = [e for e in self.items if e["eid"] != eid]
        self.items.append({"eid": eid, "group": group, "box": tuple(float(v) for v in box), "sig": sig})
        del self.items[:-LOOKALIKE["max"]]                         # the oldest go first
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
        return [e for e in self.items if e["group"] == group and _iou(box, e["box"]) >= LOOKALIKE["iou"]]

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
        return [{"episode": e["eid"], "group": e["group"], "box": [round(v, 4) for v in e["box"]]} for e in self.items]

    def export(self):
        return [{"eid": e["eid"], "group": e["group"], "box": [round(v, 4) for v in e["box"]],
                 "sig": base64.b64encode(e["sig"].astype(np.float16).tobytes()).decode("ascii")} for e in self.items]

    @classmethod
    def restore(cls, data):
        out = cls()
        for d in data or []:
            try:
                sig = np.frombuffer(base64.b64decode(d["sig"]), np.float16).astype(np.float32)
                box = tuple(float(v) for v in d["box"])
                if sig.size == SH.SIG_W * SH.SIG_H and len(box) == 4 and d["group"] in GROUPS:
                    out.items.append({"eid": str(d["eid"]), "group": d["group"], "box": box, "sig": sig})
            except (KeyError, TypeError, ValueError):
                continue
        del out.items[:-LOOKALIKE["max"]]
        return out
