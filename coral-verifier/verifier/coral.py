"""Owns the Edge TPU THROUGH A CHILD PROCESS. The parent never loads libedgetpu: one child (multiprocessing "spawn") holds the interpreter and
runs every invoke; ONE worker thread in the parent feeds it from ONE bounded latest-wins queue.

Why a process: a hung USB invoke cannot be interrupted in a thread. Here the parent waits at most stall_s for the child's answer, then SIGKILLs
it and starts a new one (at most one attempt per recover_every_s). A hung invoke therefore never leaks a thread, never overlaps with a second
interpreter (the old child is dead before the new one starts), and stopping the service always works (the child is killed first; it also dies
with the parent through PR_SET_PDEATHSIG).

Latency policy: the queue is ordered by request ARRIVAL, bounded (max_queue_depth; when full the OLDEST arrival is shed) and a frame whose age
(since arrival, decode included) exceeds max_queue_age_ms or its own deadline_ms when it reaches the worker is dropped: 503 {"reason":"stale"}.

States: "ok", "stalled" (an invoke took > stall_s, raised, or a fresh child failed its first invoke), "missing" (the delegate could not be loaded).
Every state change is logged at ERROR with its reason. Consecutive FAILED recoveries (load error, load hang, first invoke failing, or a stall within
stable_s of the previous recovery) are counted; after fatal_after of them on_fatal() is called (the service exits non-zero so Supervisor restarts it).
"""
import bisect, collections, logging, multiprocessing, threading, time
import numpy as np
import metrics as M
import yolo_edgetpu as Y
from child import child_main
from detect import Engine
from defaults import BUSY_DEPTH

log = logging.getLogger("coral")
STAGES = ("decode_ms", "preprocess_ms", "infer_ms", "post_ms", "queue_ms", "total_ms", "second_look_ms")


class Job:
    """One frame. `finish(dets, stages)` runs in the worker (so session updates keep arrival order) and returns the reply."""

    def __init__(self, img, tiles, person_thr, finish, decode_ms=0.0, deadline_ms=None, arrival=None, low=False, thr=None, post=None):
        self.img, self.tiles, self.person_thr, self.finish = img, tiles, person_thr, finish
        self.post = post                                                    # optional (engine, img, dets) -> (dets, info): the second look, inside this job
        self.thr = thr or {}                                                # per-camera vehicle_thr / animal_cont_thr for the engine
        self.decode_ms, self.deadline_ms = decode_ms, deadline_ms
        self.low = low                                                      # monitoring frame (prio=low): shed first, served last
        self.enq = time.monotonic()
        self.arrival = self.enq if arrival is None else arrival            # when the request arrived (before decode): stale age starts here
        self.event = threading.Event()
        self.status = None
        self.body = None
        self._lock = threading.Lock()

    def complete(self, status, body):
        with self._lock:
            if self.status is not None:
                return False
            self.status, self.body = status, body
        self.event.set()
        return True


class ChildModel:
    """Parent-side proxy of the model living in the child: same detect_px()/timing/size interface as EdgeTpuYolo, so Engine uses it unchanged."""

    def __init__(self, spec, ctx, start_timeout_s, stall_s):
        self.stall_s = stall_s
        self.conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(target=child_main, args=(child_conn, spec), name="coral-child", daemon=True)
        self.proc.start()
        child_conn.close()
        self.timing = {}
        try:
            if not self.conn.poll(start_timeout_s):
                raise Y.CoralLoadError(f"Edge TPU child did not become ready within {start_timeout_s:.0f} s")
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                raise Y.CoralLoadError(f"Edge TPU child exited during start (exit code {self._exitcode()})")
            if msg[0] == "load_error":
                raise Y.CoralLoadError(msg[1])
            info = msg[1]
        except BaseException:
            self.kill()
            raise
        self.size = info["size"]
        self.cd_q, self.bd_q = info["cd_q"], info["bd_q"]
        self.n = info["scores_shape"][0]
        self.dec = Y.Decoder(self.size)

    def _exitcode(self):
        self.proc.join(1.0)
        return self.proc.exitcode

    @property
    def pid(self):
        return self.proc.pid

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.join(3.0)
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    def _timeout_kill(self):
        self.timed_out = True
        try:
            self.proc.kill()                      # unblocks any send/recv stuck on a frozen child
        except Exception:
            pass

    def detect_px(self, canvas, classes=(0,), thr=Y.THR, nms_iou=Y.NMS_IOU):
        """One exchange with the child. The WHOLE exchange (send, wait, receive) is under a hard deadline: a child frozen between requests
        (SIGSTOP, memory stall) cannot block the worker in send_bytes, the timer kills it and the call fails like a hung invoke."""
        self.timed_out = False
        timer = threading.Timer(self.stall_s + 0.5, self._timeout_kill)
        timer.daemon = True
        timer.start()
        try:
            self.conn.send_bytes(np.ascontiguousarray(canvas).reshape(-1))
            if not self.conn.poll(self.stall_s):
                raise Y.CoralStall(f"invoke took > {self.stall_s:.1f} s")
            head = self.conn.recv()
            if head[0] != "ok":
                raise Y.CoralError(head[1])
            s = np.frombuffer(self.conn.recv_bytes(), np.int8).reshape(self.n, 17)
            b = np.frombuffer(self.conn.recv_bytes(), np.int8).reshape(self.n, 64)
        except Y.CoralError:
            raise
        except (EOFError, OSError, BrokenPipeError) as e:
            if self.timed_out:
                raise Y.CoralStall(f"Edge TPU child unresponsive for > {self.stall_s + 0.5:.1f} s (frozen or blocked pipe)")
            raise Y.CoralError(f"Edge TPU child died (exit code {self._exitcode()}): {e}")
        finally:
            timer.cancel()
        t = time.perf_counter()
        r = self.dec.decode(s, b, *self.cd_q, *self.bd_q, classes=classes, thr=thr, nms_iou=nms_iou)
        self.timing = {"infer_ms": head[1], "quant_ms": head[2], "post_ms": (time.perf_counter() - t) * 1e3}
        return r


class CoralManager:
    def __init__(self, spec, stall_s=2.0, recover_every_s=10.0, attempt_timeout_s=20.0, max_queue_age_ms=400, max_queue_depth=8,
                 fatal_after=5, stable_s=30.0, on_fatal=None, clock=time.monotonic):
        """spec: picklable model description for the child ({"kind":"edgetpu","path","size"} or the test-only fake)."""
        self.spec, self.stall_s, self.recover_every_s, self.attempt_timeout_s, self.clock = spec, stall_s, recover_every_s, attempt_timeout_s, clock
        self.max_queue_age_ms, self.max_queue_depth, self.fatal_after, self.stable_s, self.on_fatal = max_queue_age_ms, max_queue_depth, fatal_after, stable_s, on_fatal
        self.ctx = multiprocessing.get_context("spawn")
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.dq = []                         # jobs ordered by arrival (oldest first)
        self.state = "missing"
        self.reason = "starting"
        self.engine = None
        self.child_pids = []                 # every child ever started (tests: at most one is alive at a time)
        self.resets = 0
        self.dropped_stale = 0
        self.failures = 0
        self.ok_since = None
        self.size = None
        self.hist = {k: collections.deque(maxlen=200) for k in STAGES}
        self.last_attempt = -1e9
        self._stop = threading.Event()
        self._fatal_called = False
        self.started = clock()
        self.roll = M.Rolling(M.ROLL_NAMES)                    # 60 s counters for /v1/metrics (wall clock seconds, independent of the test clock)

    # ---------------- lifecycle ----------------
    def start(self):
        threading.Thread(target=self._worker, name="coral-worker", daemon=True).start()
        threading.Thread(target=self._recovery, name="coral-recovery", daemon=True).start()

    def stop(self):
        self._stop.set()
        with self.cv:
            eng, self.engine = self.engine, None
            self.state, self.reason = "missing", "stopping"
            self._drain("missing")
            self.cv.notify_all()
        if eng:
            eng.model.kill()                                  # child first: docker stop / SIGTERM must always be able to exit

    # ---------------- state ----------------
    def _set_state(self, state, reason):
        changed = state != self.state
        self.state, self.reason = state, reason
        if state == "ok":
            log.info("Coral ok: %s", reason)
        else:
            log.error("Coral %s: %s%s", state, reason, "" if changed else " (still)")

    def _drain(self, state):
        for job in self.dq:
            job.complete(503, {"coral": state})
        self.dq.clear()

    # ---------------- submit ----------------
    def submit(self, job):
        with self.cv:
            if self.state != "ok":
                job.complete(503, {"coral": self.state})
                return job
            bisect.insort(self.dq, job, key=lambda j: j.arrival)           # arrival order, whatever order decode finished in
            while len(self.dq) > self.max_queue_depth:                     # latest wins: shed the OLDEST arrival
                self.dropped_stale += 1
                self.roll.add("stale")
                i = next((k for k, j in enumerate(self.dq) if j.low), 0)    # shed the oldest LOW frame first, else the oldest of all
                self.dq.pop(i).complete(503, {"reason": "stale"})
            self.cv.notify_all()
        return job

    # ---------------- worker ----------------
    def _worker(self):
        while not self._stop.is_set():
            with self.cv:
                while (not self.dq or self.engine is None) and not self._stop.is_set():
                    self.cv.wait(0.5)
                if self._stop.is_set():
                    return
                job = self.dq.pop(next((k for k, j in enumerate(self.dq) if not j.low), 0))   # normal frames first, low only when none waits
                reduced = job.low and len(self.dq) >= BUSY_DEPTH                                 # busy: a low-priority frame gets its base pass(es) only
                engine = self.engine
                started = self.clock()
                age = (started - job.arrival) * 1e3
                limit = self.max_queue_age_ms if job.deadline_ms is None else min(self.max_queue_age_ms, job.deadline_ms)
                if age > limit:
                    self.dropped_stale += 1
                    self.roll.add("stale")
                    job.complete(503, {"reason": "stale"})
                    continue
            try:
                thr = {**job.thr, "zoom_out": 0.0} if reduced else job.thr
                dets, tm = engine.detect(job.img, job.tiles, job.person_thr, **thr)
                sl_ms = 0.0
                promoted = False
                if job.post is not None:
                    t_sl = time.perf_counter()
                    try:
                        dets, ex = job.post(engine, job.img, dets, reduced)
                        for k in ("preprocess_ms", "infer_ms", "post_ms"):
                            tm[k] += ex.get(k, 0.0)
                        for name, n in ex.get("counts", {}).items():
                            self.roll.add(name, n)
                        promoted = bool(ex.get("promoted"))                                          # near a status change: the optional passes ran after all
                        sl_ms = (time.perf_counter() - t_sl) * 1e3
                    except Y.CoralError:
                        raise
                    except Exception:                  # a second-look bug costs the refinement, never the frame
                        log.exception("second look failed")
            except Y.CoralError as e:
                self._fail(engine, str(e), job)
                continue
            except Exception as e:                       # post-processing bug: this frame only
                log.exception("detect failed")
                job.complete(500, {"error": "detect failed"})
                continue
            reduced = reduced and not promoted
            if promoted:
                self.roll.add("promoted")
            elif reduced:
                self.roll.add("reduced")
                if job.thr.get("zoom_out", 0.0) > 0:
                    self.roll.add("shed")                                                             # the zoom-out pass that was not run
            done = self.clock()
            stages = {"decode_ms": job.decode_ms, "preprocess_ms": tm["preprocess_ms"], "infer_ms": tm["infer_ms"], "post_ms": tm["post_ms"],
                      "queue_ms": (started - job.enq) * 1e3, "total_ms": (done - job.arrival) * 1e3 + 0.0, "second_look_ms": sl_ms, "reduced": reduced}
            self.roll.add("busy_ms", stages["infer_ms"]); self.roll.add("inv"); self.roll.add("frames")
            with self.lock:
                for k in STAGES:
                    self.hist[k].append(stages[k])
            try:
                job.complete(200, job.finish(dets, stages))
            except Exception:
                log.exception("finish failed")
                job.complete(500, {"error": "internal error"})

    def _fail(self, engine, reason, job):
        """The child stalled or failed an invoke: kill it, fail everything queued, let the recovery thread start a new one."""
        fatal = False
        engine.model.kill()                              # SIGKILL FIRST: the old interpreter is gone before any state change lets a new one start
        with self.cv:
            if self.engine is engine:
                self.engine = None
                self.resets += 1
                if self.ok_since is not None and self.clock() - self.ok_since < self.stable_s:
                    self.failures += 1                    # flapping: a recovery that did not last is a failed recovery
                else:
                    self.failures = 0
                self.ok_since = None
                self._set_state("stalled", reason)
                fatal = bool(self.fatal_after) and self.failures >= self.fatal_after
            job.complete(503, {"coral": "stalled"})
            self._drain("stalled")
            self.cv.notify_all()
        if fatal:
            self._request_restart()

    def _request_restart(self):
        if self._fatal_called:
            return
        self._fatal_called = True
        log.critical("Coral recovery failed %d times in a row: requesting a service restart (exit code 3)", self.failures)
        self.stop()
        if self.on_fatal:
            self.on_fatal()

    # ---------------- recovery ----------------
    def _recovery(self):
        while not self._stop.is_set():
            with self.lock:
                need = self.state != "ok" and self.engine is None
            if not need or self.clock() - self.last_attempt < self.recover_every_s:
                self._stop.wait(0.1)
                continue
            self.last_attempt = self.clock()
            model = None
            try:
                model = ChildModel(self.spec, self.ctx, self.attempt_timeout_s, self.stall_s)
                self.child_pids.append(model.pid)
                eng = Engine(model)
                model.detect_px(np.full((model.size, model.size, 3), 114, np.uint8), (0,), 0.5)      # first invoke: a load that cannot invoke is a failed recovery
            except Y.CoralError as e:
                if model:
                    model.kill()
                self._attempt_failed("missing" if isinstance(e, Y.CoralLoadError) else "stalled", str(e))
                continue
            except Exception as e:
                if model:
                    model.kill()
                self._attempt_failed("stalled", f"unexpected error starting the Coral worker: {e!r}")
                continue
            with self.cv:
                if self._stop.is_set():
                    model.kill()
                    return
                self.engine = eng
                self.size = model.size
                self.ok_since = self.clock()
                self._set_state("ok", f"model input {self.size}, child pid {model.pid}")
                self.cv.notify_all()

    def _attempt_failed(self, state, reason):
        with self.cv:
            if self._stop.is_set():
                return
            self.failures += 1
            self._set_state(state, f"recovery attempt {self.failures} failed: {reason.strip()}")
            fatal = bool(self.fatal_after) and self.failures >= self.fatal_after
        if fatal:
            self._request_restart()

    # ---------------- metrics ----------------
    @staticmethod
    def _pct(xs):
        xs = sorted(xs)
        if not xs:
            return {"median": None, "p95": None}
        return {"median": round(xs[len(xs) // 2], 2), "p95": round(xs[min(len(xs) - 1, int(len(xs) * 0.95))], 2)}

    def child_pid(self):
        with self.lock:
            return self.engine.model.pid if self.engine is not None and hasattr(self.engine.model, "pid") else None

    def stats(self):
        with self.lock:
            infer = list(self.hist["infer_ms"])
            return {"coral": self.state, "reason": self.reason, "infer": self._pct(infer), "last": infer[-1] if infer else None,
                    "stages": {k: self._pct(self.hist[k]) for k in STAGES},
                    "resets": self.resets, "queue_depth": len(self.dq), "dropped_stale": self.dropped_stale}
