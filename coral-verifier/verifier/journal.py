"""Event journal: one folder per episode (meta.json + original JPEGs with detection sidecars), retention, review queries.
Frames are linked by camera: every frame received for a camera between an episode's `start` and `end` belongs to it. Only the first frame is written at
once; best (highest score of the episode's groups) and last stay in memory until `update` / `end` / stale / a read of the episode flushes them, plus one
snapshot per snapshot_every_s of frame time (capped). JPEG bytes are stored exactly as received. Ids are validated (no path traversal)."""
import json, logging, math, os, queue, re, shutil, threading, time
from collections import deque
from defaults import JOURNAL as J

log = logging.getLogger("journal")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{5,63}$")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
MODES, KINDS = ("shadow", "gate"), ("camera", "unflagged")
OUTCOMES = ("confirmed", "vetoed", "aborted", "released", "parked")          # client-settable; the service itself adds "stale"
LABELS = ("correct", "false_alarm", "missed", "wrong_class")
ACTIONS = ("start", "update", "end")
GROUPS = ("person", "vehicle", "animal")


def _atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) > 1e15:
        raise ValueError(f"{name} must be a number")
    return int(v)


def _str(v, name, n=200):
    if not isinstance(v, str) or len(v) > n:
        raise ValueError(f"{name} must be a string of at most {n} characters")
    return v


class Journal:
    @staticmethod
    def question(m):
        """The ONE plain-language question for the annotation mode, derived from the episode's outcome, kind and groups.
        -> {key, text, group (the group to highlight, None = nothing relevant), others (the other groups of the episode), yes (label), no (label)}."""
        groups = [g for g in (m.get("groups") or []) if g in GROUPS]                  # episode order: the first is the one asked about
        g = groups[0] if groups else None
        others = groups[1:]
        if m.get("outcome") == "vetoed" and m.get("kind") != "unflagged":
            return {"key": "empty", "text": "Is the frame really empty?", "group": None, "others": [], "yes": "correct", "no": "missed"}
        if m.get("kind") == "unflagged":
            what = {"person": "a person", "vehicle": "a moving vehicle", "animal": "an animal"}.get(g, "something")
            return {"key": "missed_" + (g or "any"), "text": f"Is there {what} the camera missed?", "group": g, "others": others, "yes": "correct", "no": "false_alarm"}
        if g == "vehicle":
            return {"key": "vehicle_moving", "text": "Is this a moving vehicle?", "group": g, "others": others, "yes": "correct", "no": "false_alarm"}
        if g in ("person", "animal"):
            return {"key": g + "_present", "text": "Is there a person?" if g == "person" else "Is there an animal?", "group": g, "others": others, "yes": "correct", "no": "false_alarm"}
        return {"key": "event", "text": "Is this a real event?", "group": None, "others": [], "yes": "correct", "no": "false_alarm"}

    @staticmethod
    def valid_id(eid):
        return bool(ID_RE.match(eid or ""))

    def __init__(self, root, clock=None, **limits):
        self.root = root
        self.clock = clock or (lambda: time.time() * 1000.0)
        self.lim = {**J, **limits}
        self.lock = threading.RLock()
        self.eps = {}                  # id -> state {"meta", "dir", "bytes", ...runtime}
        self.ring = {}                 # camera -> deque of (wall_ms, t_ms, jpeg, dets)
        self.enabled = True
        self._next_maint = 0.0
        self.dropped = 0
        self._q = queue.Queue(maxsize=self.lim["queue_max"])
        self._worker = None
        try:
            os.makedirs(root, exist_ok=True)
            self._load()
        except OSError as e:
            log.error("journal disabled, cannot use %s: %s", root, e)
            self.enabled = False
        if self.enabled:
            self._purge_legacy()
            self._worker = threading.Thread(target=self._run, name="journal", daemon=True)
            self._worker.start()

    def _purge_legacy(self):
        """Unflagged episodes that never confirmed are noise (any label): removed at startup and in maintenance; the count is logged."""
        n = 0
        for st in list(self.eps.values()):
            if self._discard_if_unconfirmed_unflagged(st):
                n += 1
        if n:
            log.info("journal: removed %d unflagged episode(s) that never confirmed", n)
        return n

    # ---------------- single worker: ALL journal mutations and disk writes run here, never on the frame path ----------------
    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            fn, args, evt, box = item
            try:
                box.append(fn(*args))
            except Exception:
                log.exception("journal work failed")
                box.append((500, {"error": "journal failure"}))
            finally:
                if evt is not None:
                    evt.set()
                self._q.task_done()

    def _call(self, fn, *args):
        """Runs fn on the worker and waits for its result (the HTTP handler threads wait here, the frame path never does)."""
        if self._worker is None or threading.current_thread() is self._worker:
            return fn(*args)
        evt, box = threading.Event(), []
        try:
            self._q.put_nowait((fn, args, evt, box))
        except queue.Full:
            return 503, {"error": "journal busy"}
        if not evt.wait(15.0) or not box:
            return 503, {"error": "journal busy"}
        return box[0]

    def _post(self, fn, *args):
        """Fire and forget (frame path): non-blocking, dropped and counted when the queue is full."""
        if self._worker is None:
            return
        try:
            self._q.put_nowait((fn, args, None, []))
        except queue.Full:
            self.dropped += 1

    def drain(self):
        """Waits until the queued work is done (tests)."""
        self._q.join()

    def stop(self):
        if self._worker is not None:
            self._q.put(None)
            self._worker.join(timeout=5)
            self._worker = None

    # ---------------- persistence ----------------
    def _load(self):
        for name in os.listdir(self.root):
            d = os.path.join(self.root, name)
            try:                                                           # a damaged folder is skipped, never fatal
                with open(os.path.join(d, "meta.json")) as f:
                    meta = json.load(f)
                if not isinstance(meta, dict) or meta.get("id") != name or not ID_RE.match(name):
                    continue
                for k, t in (("frames", list), ("timeline", list), ("labels", list), ("groups", list)):
                    if not isinstance(meta.get(k), t):
                        meta[k] = []
                meta.setdefault("status", "closed"); meta.setdefault("camera", "unknown"); meta.setdefault("start_ms", 0)
                meta.setdefault("display_name", self.default_name(str(meta["camera"])))
                meta.setdefault("end_ms", None); meta.setdefault("outcome", None); meta.setdefault("label", None); meta.setdefault("last_activity_ms", 0)
                size = sum(e.stat().st_size for e in os.scandir(d) if e.is_file())
                self.eps[name] = self._state(meta, d, size)
            except Exception as e:
                log.warning("journal: skipping %s: %s", name, e)

    @staticmethod
    def _state(meta, d, size):
        return {"meta": meta, "dir": d, "bytes": size, "best": None, "last": None, "snap_t": None, "first_done": any(f["role"] == "first" for f in meta["frames"])}

    def _write_file(self, st, name, data):
        path = os.path.join(st["dir"], name)
        old = os.path.getsize(path) if os.path.exists(path) else 0
        _atomic(path, data)
        st["bytes"] += len(data) - old

    def _save_meta(self, st):
        self._write_file(st, "meta.json", json.dumps(st["meta"], separators=(",", ":")).encode())

    def _put_frame(self, st, name, role, t_ms, jpeg, dets):
        self._write_file(st, name + ".jpg", jpeg)
        self._write_file(st, name + ".json", json.dumps({"t_ms": t_ms, "detections": dets}, separators=(",", ":")).encode())
        fr = [f for f in st["meta"]["frames"] if f["name"] != name]
        fr.append({"name": name, "role": role, "t_ms": t_ms})
        st["meta"]["frames"] = fr

    def _flush(self, st):
        """Writes the pending best / last frames (when not written yet) and the meta."""
        for key in ("best", "last"):
            p = st[key]
            if p is not None and p["dirty"]:
                self._put_frame(st, key, key, p["t_ms"], p["jpeg"], p["dets"])
                p["dirty"] = False
        self._save_meta(st)

    # ---------------- episodes ----------------
    def episode(self, camera, body, display_name=None):
        return self._call(self._episode, camera, body, display_name) if self.enabled else (503, {"error": "journal unavailable"})

    @staticmethod
    def category(m):
        """One category per episode (summary buckets and the Outcome filter): unflagged by KIND, confirmed at any point, vetoed, else other."""
        if m.get("kind") == "unflagged":
            return "unflagged"
        if Journal.confirmed(m):
            return "confirmed"
        return "vetoed" if m.get("outcome") == "vetoed" else "other"

    @staticmethod
    def _outcome_ok(m, outcome):
        return Journal.category(m) == outcome if outcome in ("confirmed", "vetoed", "unflagged", "other") else m["outcome"] == outcome

    @staticmethod
    def default_name(camera):
        return re.sub(r"_\d+$", "", camera) or camera

    @staticmethod
    def confirmed(m):
        """Confirmed at any point: outcome confirmed, a timeline entry that says so, or groups present with outcome released."""
        if m.get("outcome") == "confirmed":
            return True
        if any("confirm" in str(e.get("text", "")).lower() for e in m.get("timeline") or []):
            return True
        return bool(m.get("groups")) and m.get("outcome") == "released"

    def _discard_if_unconfirmed_unflagged(self, st):
        m = st["meta"]
        if m.get("kind") == "unflagged" and m.get("status") == "closed" and not self.confirmed(m):
            self._delete(m["id"])
            return True
        return False

    def _clamp(self, v, now):
        c = self.lim["clamp_s"] * 1000.0
        return int(min(max(v, now - c), now + c))

    def _episode(self, camera, body, display_name=None):
        """POST /v1/cameras/{camera}/episodes (runs on the worker). Returns (status, json)."""
        if not self.enabled:
            return 503, {"error": "journal unavailable"}
        try:
            eid = _str(body.get("id"), "id", 64)
            if not ID_RE.match(eid):
                raise ValueError("id must be 6-64 characters of A-Z a-z 0-9 _ -")
            action = body.get("action")
            if action not in ACTIONS:
                raise ValueError("action must be start|update|end")
            t_ms = _num(body.get("t_ms"), "t_ms")
            mode = body.get("mode")
            if mode is not None and mode not in MODES:
                raise ValueError("mode must be shadow|gate")
            kind = body.get("kind")
            if kind is not None and kind not in KINDS:
                raise ValueError("kind must be camera|unflagged")
            outcome = body.get("outcome")
            if outcome is not None and outcome not in OUTCOMES:
                raise ValueError("outcome must be " + "|".join(OUTCOMES))
            groups = body.get("groups")
            if groups is not None:
                if not isinstance(groups, list) or len(groups) > 8 or any(not isinstance(g, str) or not 0 < len(g) <= 32 for g in groups):
                    raise ValueError("groups must be a list of up to 8 short strings")
            note = body.get("note")
            if note is not None:
                note = _str(note, "note", 500)
            ev = body.get("timeline_event")
            if ev is not None:
                if not isinstance(ev, dict):
                    raise ValueError("timeline_event must be an object")
                ev = {"t_ms": _num(ev.get("t_ms"), "timeline_event.t_ms"), "text": _str(ev.get("text"), "timeline_event.text", 300)}
        except ValueError as e:
            return 400, {"error": str(e)[:200]}
        with self.lock:
            st = self.eps.get(eid)
            now = self.clock()
            if action == "start":
                if st is not None:
                    if st["meta"]["camera"] != camera:
                        return 409, {"error": "id belongs to another camera"}
                    if st["meta"]["status"] == "closed":
                        return 409, {"error": "episode already ended"}
                else:
                    ongoing = [s["meta"]["camera"] for s in self.eps.values() if s["meta"]["status"] == "ongoing"]
                    if len(ongoing) >= self.lim["max_ongoing"] or ongoing.count(camera) >= self.lim["max_ongoing_per_camera"]:
                        return 429, {"error": "too many ongoing episodes"}
                    st = self._create(camera, eid, self._clamp(t_ms, now), mode or "shadow", kind or "camera", groups or [], note, now)
                    if st is not None:
                        st["meta"]["display_name"] = display_name or self.default_name(camera)
                    if st is None:
                        return 500, {"error": "journal write failed"}
            elif st is None or st["meta"]["camera"] != camera:
                return 404, {"error": "unknown episode"}
            elif st["meta"]["status"] == "closed":
                return 409, {"error": "episode already ended"}
            m = st["meta"]
            if action != "start":                       # a repeated start is idempotent: it only merges what it carries
                pass
            if mode:
                m["mode"] = mode
            if kind:
                m["kind"] = kind
            if groups is not None:
                m["groups"] = groups
            if note is not None:
                m["note"] = note
            if ev:
                ev["t_ms"] = self._clamp(ev["t_ms"], now)
                m["timeline"] = (m["timeline"] + [ev])[-200:]
            m["last_activity_ms"] = now
            try:
                if action == "end":
                    self._close(st, outcome or "released", self._clamp(t_ms, now))
                    if self._discard_if_unconfirmed_unflagged(st):                  # noise: an unflagged episode that never confirmed is not kept
                        return 200, {"id": eid, "status": "discarded", "outcome": outcome or "released"}
                else:
                    if outcome:
                        m["outcome"] = outcome
                    self._flush(st)
            except OSError as e:
                log.error("journal write failed: %s", e)
                return 500, {"error": "journal write failed"}
            self._maintain(force=action == "end")
            return 200, {"id": eid, "status": m["status"], "outcome": m["outcome"]}

    def _create(self, camera, eid, t_ms, mode, kind, groups, note, now):
        d = os.path.join(self.root, eid)
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            log.error("journal mkdir failed: %s", e)
            return None
        meta = {"id": eid, "camera": camera, "start_ms": t_ms, "end_ms": None, "mode": mode, "kind": kind, "outcome": None, "groups": groups, "note": note,
                "timeline": [], "label": None, "label_note": None, "labels": [], "frames": [], "status": "ongoing", "last_activity_ms": now}
        st = self.eps[eid] = self._state(meta, d, 0)
        for wall, ft, jpeg, dets in list(self.ring.get(camera, ())):      # adopt the frames that just arrived (the trigger frame)
            if now - wall <= self.lim["adopt_s"] * 1000.0:
                self._link(st, ft, jpeg, dets)
        return st

    def _close(self, st, outcome, end_ms):
        m = st["meta"]
        m["status"], m["outcome"], m["end_ms"] = "closed", outcome, end_ms
        self._flush(st)
        st["best"] = st["last"] = None

    # ---------------- frames ----------------
    @staticmethod
    def _relevance(dets, groups):
        sc = [d.get("score", 0) for d in dets if (not groups or d.get("group") in groups) and d.get("second_look") != "rejected"]
        return max(sc) if sc else 0.0

    def _link(self, st, t_ms, jpeg, dets):
        m = st["meta"]
        rel = self._relevance(dets, m["groups"])
        if not st["first_done"]:
            self._put_frame(st, "first", "first", t_ms, jpeg, dets)
            st["first_done"] = True
            st["snap_t"] = t_ms
            self._save_meta(st)
        if st["best"] is None or rel > st["best"]["rel"]:
            st["best"] = {"rel": rel, "t_ms": t_ms, "jpeg": jpeg, "dets": dets, "dirty": True}
        st["last"] = {"t_ms": t_ms, "jpeg": jpeg, "dets": dets, "dirty": True}
        snaps = sum(1 for f in m["frames"] if f["role"] == "snapshot")
        if t_ms - (t_ms if st["snap_t"] is None else st["snap_t"]) >= self.lim["snapshot_every_s"] * 1000.0 and snaps < self.lim["snapshot_max"]:
            self._put_frame(st, "s%02d" % (snaps + 1), "snapshot", t_ms, jpeg, dets)
            st["snap_t"] = t_ms
            self._save_meta(st)

    def record_frame_async(self, camera, t_ms, jpeg, dets):
        """Frame path entry: never blocks, never touches the disk or the lock (the work is queued for the worker; dropped and counted when full)."""
        if self.enabled:
            self._post(self.record_frame, camera, t_ms, jpeg, dets)

    def record_frame(self, camera, t_ms, jpeg, dets):
        """Links a frame (worker, or directly in tests). Cheap when no episode is ongoing for the camera."""
        if not self.enabled:
            return
        with self.lock:
            now = self.clock()
            q = self.ring.get(camera)
            if q is None:
                q = self.ring[camera] = deque(maxlen=self.lim["ring_frames"])
            q.append((now, t_ms, jpeg, dets))
            for st in self.eps.values():
                m = st["meta"]
                if m["status"] == "ongoing" and m["camera"] == camera:
                    try:
                        self._link(st, t_ms, jpeg, dets)
                        m["last_activity_ms"] = now
                    except OSError as e:
                        log.error("journal write failed: %s", e)
            self._maintain()

    # ---------------- retention ----------------
    def _delete(self, eid):
        st = self.eps.pop(eid, None)
        if st is not None:
            shutil.rmtree(st["dir"], ignore_errors=True)

    def _maintain(self, force=False):
        now = self.clock()
        if not force and now < self._next_maint:
            return
        self._next_maint = now + self.lim["maintain_every_s"] * 1000.0
        for st in list(self.eps.values()):                                   # stale ongoing episodes
            m = st["meta"]
            if m["status"] == "ongoing" and now - m["last_activity_ms"] > self.lim["stale_s"] * 1000.0:
                try:
                    self._close(st, "stale", int(m["last_activity_ms"]))
                except OSError as e:
                    log.error("journal write failed: %s", e)
        self._purge_legacy()
        cut = now - self.lim["retention_days"] * 86400_000.0
        for eid, st in list(self.eps.items()):
            m = st["meta"]
            if m["status"] == "closed" and (m["end_ms"] or m["start_ms"]) < cut:
                self._delete(eid)
        total = sum(s["bytes"] for s in self.eps.values())
        if total > self.lim["max_bytes"]:
            for eid, st in sorted(((e, s) for e, s in self.eps.items() if s["meta"]["status"] == "closed"), key=lambda kv: kv[1]["meta"]["start_ms"]):
                if total <= self.lim["max_bytes"]:
                    break
                total -= st["bytes"]
                self._delete(eid)

    def maintain(self):
        def go():
            with self.lock:
                self._maintain(force=True)
            return 200, {}
        self._call(go)

    def _kick(self):
        """Readers ask for maintenance without doing it themselves."""
        if self.clock() >= self._next_maint:
            self._post(self._maintain_locked)

    def _maintain_locked(self):
        with self.lock:
            self._maintain()

    # ---------------- review ----------------
    def _view(self, st):
        return dict(st["meta"])

    def list(self, camera=None, outcome=None, label=None, t_from=None, t_to=None, limit=50, offset=0, labelled=None, status=None, group=None):
        self._kick()
        with self.lock:
            rows = [dict(st["meta"]) for st in self.eps.values()]            # copy under the lock, filter and sort outside
        if True:
            rows = [m for m in rows if (not camera or m["camera"] == camera) and (not outcome or self._outcome_ok(m, outcome)) and (not group or group in m["groups"])
                    and (not label or (m["label"] == label if label != "none" else m["label"] is None))
                    and (labelled is None or (m["label"] is not None) == labelled) and (not status or m["status"] == status)
                    and (t_from is None or m["start_ms"] >= t_from) and (t_to is None or m["start_ms"] <= t_to)]
            rows.sort(key=lambda m: m["start_ms"], reverse=True)
            limit = max(1, min(int(limit), self.lim["max_list"]))
            offset = max(0, int(offset))
            return {"total": len(rows), "offset": offset, "items": rows[offset:offset + limit]}

    def next_unlabelled(self, after=None, camera=None, outcome=None, group=None, t_from=None, t_to=None):
        """Annotation mode: the OLDEST closed episode without a label (optionally only after episode `after` in start order, filtered by camera/outcome).
        Returns {item, remaining}: `remaining` counts all matching unlabelled closed episodes (the returned one included)."""
        with self.lock:
            rows = [dict(st["meta"]) for st in self.eps.values()
                    if st["meta"]["status"] == "closed" and st["meta"]["label"] is None and (not camera or st["meta"]["camera"] == camera)
                    and (not outcome or self._outcome_ok(st["meta"], outcome)) and (not group or group in st["meta"]["groups"])
                    and (t_from is None or st["meta"]["start_ms"] >= t_from) and (t_to is None or st["meta"]["start_ms"] <= t_to)]
            ref = self.eps.get(after) if after and ID_RE.match(after) else None
            ref_key = None if ref is None else (ref["meta"]["start_ms"], ref["meta"]["id"])
        rows.sort(key=lambda m: (m["start_ms"], m["id"]))
        pick = next((m for m in rows if ref_key is None or (m["start_ms"], m["id"]) > ref_key), None)
        if pick is not None:
            pick["question"] = self.question(pick)
        return {"item": pick, "remaining": len(rows)}

    def get(self, eid):
        with self.lock:
            st = self.eps.get(eid) if ID_RE.match(eid or "") else None
            if st is None:
                return None
            m = dict(st["meta"])
        m["question"] = self.question(m)
        return m

    def _flush_eid(self, eid):
        with self.lock:
            st = self.eps.get(eid)
            if st is not None and st["meta"]["status"] == "ongoing":
                try:
                    self._flush(st)
                except OSError as e:
                    log.error("journal write failed: %s", e)
        return 200, {}

    def file(self, eid, name, ext):
        """-> path of <name>.jpg|json of an episode, or None (ids and names are validated: no traversal)."""
        if ext not in ("jpg", "json") or not NAME_RE.match(name or ""):
            return None
        meta = self.get(eid)
        if meta is None:
            return None
        if meta["status"] == "ongoing" and name in ("best", "last"):
            self._call(self._flush_eid, eid)                                  # pending frames are written by the worker first
        with self.lock:                                                       # the episode is read under the lock (retention may delete it concurrently)
            st = self.eps.get(eid)
            if st is None or not any(f["name"] == name for f in st["meta"]["frames"]):
                return None
            p = os.path.join(st["dir"], f"{name}.{ext}")
        return p if os.path.isfile(p) else None

    def set_label(self, eid, label, note):
        return self._call(self._set_label, eid, label, note) if self.enabled else (503, {"error": "journal unavailable"})

    def _set_label(self, eid, label, note):
        if label not in LABELS:
            return 400, {"error": "label must be " + "|".join(LABELS)}
        try:
            if note is not None:
                note = _str(note, "note", 500)
        except ValueError as e:
            return 400, {"error": str(e)}
        with self.lock:
            st = self.eps.get(eid) if ID_RE.match(eid or "") else None
            if st is None:
                return 404, {"error": "unknown episode"}
            m = st["meta"]
            m["label"], m["label_note"] = label, note
            m["labels"] = (m["labels"] + [{"label": label, "note": note, "t_ms": int(self.clock())}])[-20:]
            try:
                self._save_meta(st)
            except OSError:
                return 500, {"error": "journal write failed"}
            return 200, {"id": eid, "label": label, "label_note": note}

    def summary(self, t_from=None, t_to=None, camera=None, tz_offset_min=0, group=None):
        """Dashboard numbers. Period (episode start, epoch ms) and camera filters. Returns the per camera x group `rows` (counts by outcome and label), the
        overall `totals`, one row per `cameras`, per `groups` and per local `days` (tz_offset_min east of UTC) with categories confirmed/vetoed/unflagged/other."""
        with self.lock:
            metas = [dict(st["meta"]) for st in self.eps.values()]
        metas = [m for m in metas if (t_from is None or m["start_ms"] >= t_from) and (t_to is None or m["start_ms"] <= t_to) and (not camera or m["camera"] == camera)
                 and (not group or group in m["groups"])]
        # one category per episode: unflagged by KIND (independent of the outcome); confirmed = confirmed at any point; vetoed; everything else is explicit "other"
        # (aborted, released without groups, parked, stale, ongoing)
        cat = self.category

        def acc():
            return {"events": 0, "closed": 0, "ongoing": 0, "reviewable": 0, "reviewed": 0, "correct": 0, "false_alarm": 0, "missed": 0, "wrong_class": 0, "confirmed": 0, "vetoed": 0, "unflagged": 0, "other": 0}

        def add(row, m):
            row["events"] += 1
            row[cat(m)] += 1
            # ONE definition everywhere (lists, chips, Validate, dashboard): reviewable = closed and unlabelled; ongoing episodes are counted apart
            if m["status"] == "ongoing":
                row["ongoing"] += 1
            else:
                row["closed"] += 1
                if not m["label"]:
                    row["reviewable"] += 1
            if m["label"]:
                row["reviewed"] += 1
                row[m["label"]] += 1

        def fin(row):
            row["precision"] = None if not row["reviewed"] else round(row["correct"] / row["reviewed"], 4)
            return row
        out, cams, groups, days, totals = {}, {}, {}, {}, acc()
        for m in metas:
            add(totals, m)
            crow = cams.setdefault(m["camera"], {"camera": m["camera"], "display_name": m.get("display_name") or self.default_name(m["camera"]), **acc()})
            add(crow, m)
            for g in (m["groups"] or ["none"]):
                row = out.setdefault((m["camera"], g), {"camera": m["camera"], "group": g, "total": 0, "by_outcome": {}, "by_label": {}})
                row["total"] += 1
                k = m["outcome"] or "ongoing"
                row["by_outcome"][k] = row["by_outcome"].get(k, 0) + 1
                k = m["label"] or "unlabeled"
                row["by_label"][k] = row["by_label"].get(k, 0) + 1
                add(groups.setdefault(g, {"group": g, **acc()}), m)
            day = time.strftime("%Y-%m-%d", time.gmtime(m["start_ms"] / 1000.0 + 60 * tz_offset_min))
            d = days.setdefault(day, {"day": day, "total": 0, "confirmed": 0, "vetoed": 0, "unflagged": 0, "other": 0})
            d["total"] += 1
            d[cat(m)] += 1
        return {"rows": [out[k] for k in sorted(out)], "totals": fin(totals), "cameras": [fin(cams[k]) for k in sorted(cams)],
                "groups": [fin(groups[k]) for k in sorted(groups)], "days": [days[k] for k in sorted(days)]}

    def stats(self):
        with self.lock:
            return {"enabled": self.enabled, "episodes": len(self.eps), "ongoing": sum(1 for s in self.eps.values() if s["meta"]["status"] == "ongoing"),
                    "bytes": sum(s["bytes"] for s in self.eps.values()), "queue": self._q.qsize(), "dropped": self.dropped}
