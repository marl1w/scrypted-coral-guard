"""Auto-validation of repeat empty frames: an episode that asks "Is the frame really empty?" is labelled `correct`
without asking when every one of its frames shows the same scene as a frame the user already validated as empty.

Kept off the detection path and the journal's frame path entirely: one thread of its own at the lowest CPU priority
reads the JPEGs and sidecars the journal has already written (never the live frames), decodes them at 1/8 size, and
hands its verdict back to the journal worker as a queued call. Nothing waits for it; an episode waiting for its check
stays out of Validate only until the check is done (or is shown anyway after `auto_pending_s`).

Two guards make a match safe:
- the new episode's frames must be empty for Coral too: no person / vehicle / animal box at `auto_det_min` or more
  (boxes rejected by the second look do not count); a still person or a parked car the user may want to label is never
  auto-validated;
- references are only episodes the USER labelled `correct` (never auto-labelled ones, so a match never chains) whose
  frames pass the same emptiness test."""
import json, logging, os, queue, threading

import fingerprint as fpmod

log = logging.getLogger("autovalidate")
ROLES = ("first", "best", "last")


def eligible(m):
    """Episodes whose question is about an empty frame (vetoed camera episodes: "Is the frame really empty?" /
    "Was someone there?")."""
    return m.get("status") == "closed" and m.get("outcome") == "vetoed" and m.get("kind") != "unflagged"


def is_reference(m):
    return eligible(m) and m.get("label") == "correct" and not m.get("auto")


class AutoValidator:
    def __init__(self, journal):
        self.j = journal
        self.det_min = journal.lim["auto_det_min"]
        self.q = queue.Queue(maxsize=journal.lim["queue_max"])
        self.refs = {}                     # camera -> {episode id: [fingerprints]}
        self.cache = {}                    # episode id -> (clean, [fingerprints]) of closed episodes (frames no longer change)
        self.checked = self.matched = 0
        self.thread = threading.Thread(target=self._run, name="autovalidate", daemon=True)
        self.thread.start()

    # ---------------- called from the journal (worker or handler threads): never blocks ----------------
    def submit(self, kind, eid):
        try:
            self.q.put_nowait((kind, eid))
            return True
        except queue.Full:
            return False

    # ---------------- own thread ----------------
    def _run(self):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)       # Linux: per-thread nice
        except (AttributeError, OSError):
            pass
        try:
            self._backfill()
        except Exception:
            log.exception("auto-validation: building the references failed")
        while True:
            kind, eid = self.q.get()
            try:
                if kind == "check":
                    self._check(eid)
                elif kind == "ref":
                    self._ref(eid)
            except Exception:
                log.exception("auto-validation: %s %s failed", kind, eid)
                if kind == "check":
                    self.j._post(self.j._auto_result, eid, None)

    def _backfill(self):
        metas = self._metas()
        n = 0
        for m in metas:
            if is_reference(m):
                n += self._ref(m["id"], m)
        log.info("auto-validation: %d reference episode(s) of %d camera(s)", n, len(self.refs))
        before = self.matched
        for m in metas:                                                     # episodes already waiting in Validate (and any check cut short)
            if m.get("label") is None and eligible(m):
                self._check(m["id"], m)
        log.info("auto-validation: %d waiting episode(s) auto-validated at start", self.matched - before)

    def _metas(self):
        with self.j.lock:
            return sorted((dict(st["meta"]) for st in self.j.eps.values()), key=lambda m: m["start_ms"])

    def _meta(self, eid):
        with self.j.lock:
            st = self.j.eps.get(eid)
            return dict(st["meta"]) if st is not None else None

    def _frames(self, m):
        """-> (clean, [fingerprints]) of an episode's first / best / last frames, read from the journal's files."""
        if m["id"] in self.cache:
            return self.cache[m["id"]]
        clean, fps = True, []
        for f in m.get("frames") or []:
            if f.get("role") not in ROLES:
                continue
            with self.j.lock:
                st = self.j.eps.get(m["id"])
                d = st["dir"] if st is not None else None
            if d is None:
                return False, []
            try:
                with open(os.path.join(d, f["name"] + ".json")) as fh:
                    dets = json.load(fh).get("detections") or []
                with open(os.path.join(d, f["name"] + ".jpg"), "rb") as fh:
                    fp = fpmod.fingerprint(fh.read())
            except (OSError, ValueError):
                return False, []
            if any(x.get("score", 0) >= self.det_min and x.get("second_look") != "rejected" for x in dets):
                clean = False
            if fp is None:
                return False, []
            fps.append(fp)
        out = (clean and bool(fps), fps)
        self.cache[m["id"]] = out
        if len(self.cache) > 4000:
            self.cache.pop(next(iter(self.cache)))
        return out

    def _ref(self, eid, m=None):
        """(Re)evaluates one episode as a reference: added when the user labelled it empty and its frames are empty for
        Coral too, removed otherwise (a relabel). -> 1 when it is a reference."""
        m = m or self._meta(eid)
        if m is None:
            return 0
        cam = self.refs.setdefault(m["camera"], {})
        cam.pop(eid, None)
        if not is_reference(m):
            return 0
        clean, fps = self._frames(m)
        if not clean:
            return 0
        cam[eid] = fps
        return 1

    def _check(self, eid, m=None):
        m = m or self._meta(eid)
        if m is None or m.get("label") is not None:
            if m is not None:
                self.j._post(self.j._auto_result, eid, None)
            return
        self.checked += 1
        match = None
        clean, fps = self._frames(m)
        if clean:
            refs = self.refs.get(m["camera"], {})
            hits = []
            for fp in fps:
                hit = next((rid for rid, rfps in refs.items() if any(fpmod.same(fp, r) for r in rfps)), None)
                if hit is None:
                    break
                hits.append(hit)
            else:
                match = hits[0] if hits else None
        if match:
            self.matched += 1
        self.j._post(self.j._auto_result, eid, match)

    def stats(self):
        return {"references": sum(len(v) for v in self.refs.values()), "checked": self.checked, "matched": self.matched, "waiting": self.q.qsize()}
