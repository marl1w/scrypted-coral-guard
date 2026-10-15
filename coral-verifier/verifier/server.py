"""Coral verifier HTTP service (API.md v1). stdlib ThreadingHTTPServer: one thread per connection (capped), all inference through ONE Coral worker
process (see coral.py). Hardened for a LAN service: authenticate before reading any body, JPEG-only bounded bodies, per-request deadlines against
slow clients, a connection cap, decode concurrency cap, bounded number of cameras."""
import collections, hmac, json, logging, math, os, re, signal, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

import defaults as DF
import metrics as M
import session as S
import detect as D
import secondlook as SL
from journal import Journal
from coral import CoralManager, Job

def _read_version():
    """Add-on version: VERIFIER_VERSION, else BUILD_VERSION, else the `version:` of config.yaml next to the package (or one level up), else the constant."""
    for key in ("VERIFIER_VERSION", "BUILD_VERSION"):                  # BUILD_VERSION: passed by the Supervisor as a build argument (see the Dockerfile)
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "config.yaml"), os.path.join(here, "..", "config.yaml"), "/config.yaml"):
        try:
            with open(p) as f:
                m = re.search(r'^version:\s*["\']?([0-9A-Za-z.+\-]+)', f.read(), re.M)
            if m:
                return m.group(1)
        except OSError:
            pass
    return FALLBACK_VERSION


FALLBACK_VERSION = "0.3.10"
VERSION = _read_version()
DEFAULT_PORT = DF.SERVICE["port"]
MAX_JSON_BYTES = 64 * 1024
RATE_PER_S = 4
RATE_BURST = 10
REPLY_MIN_SCORE = 0.2          # detections below this are omitted from replies unless verbose=1
JOB_WAIT_S = 10.0
IDLE_S = 15.0                  # keep-alive connection idle limit
FIRST_BYTE_S = 2.0             # a NEW connection must start its request within this (idle sockets are cut)
RESERVED_HEALTH_SLOTS = 4      # connections beyond max_connections that only GET /v1/health may use
CAM_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
MODEL_FILES = {320: "yolov9-s-relu6-best_320_int8_edgetpu.tflite", 512: "yolov9-s-relu6-tpumax_512_int8_edgetpu.tflite"}
TRACKER_KEYS = ("start_thr", "cont_thr", "confirm_n", "confirm_m", "hold_s", "still_s", "still_hold_s", "max_hold_s", "parked_after_s", "session_ttl_s", "vehicle_thr", "animal_start_thr", "animal_cont_thr", "zoom_out", "second_look", "name")
log = logging.getLogger("verifier")


class OptionsError(Exception):
    pass


def parse_options(opts):
    """Validate the settings once, at start. `opts` = the public add-on options (token, port, log_level) plus, for tests/dev only, any key of defaults.SERVICE / TRACKER.
    Precedence: defaults.py < VERIFIER_* environment < opts. Returns a settings dict; raises OptionsError with a clear message (the service then refuses to start)."""
    merged = {**DF.env_overrides(), **opts}
    D_ = DF.SERVICE

    def num(key, lo, hi, kind=float):
        v = merged.get(key, D_[key])
        try:
            v = kind(v)
        except (TypeError, ValueError):
            raise OptionsError(f"option {key}={v!r} is not a number")
        if isinstance(v, float) and not math.isfinite(v) or not lo <= v <= hi:
            raise OptionsError(f"option {key}={v} must be between {lo} and {hi}")
        return v
    token = merged.get("token") or ""
    if not isinstance(token, str):
        raise OptionsError("option token must be a string")
    allow = bool(merged.get("allow_no_token", D_["allow_no_token"]))
    if not token and not allow:
        raise OptionsError("option `token` is empty: set a token (clients send it as X-Verifier-Token)")
    st = {"token": token, "allow_no_token": allow}
    st["model"] = num("model", 320, 512, int)
    if st["model"] not in MODEL_FILES:
        raise OptionsError(f"model={st['model']} must be 320 or 512")
    st["port"] = num("port", 1, 65535, int)
    st["max_queue_age_ms"] = num("max_queue_age_ms", 50, 5000)
    st["max_queue_depth"] = num("max_queue_depth", 1, 64, int)
    st["self_restart_after"] = num("self_restart_after", 0, 100, int)
    st["max_cameras"] = num("max_cameras", 1, 1024, int)
    st["max_image_pixels"] = num("max_image_pixels", 65536, 100_000_000, int)
    st["max_body_bytes"] = num("max_body_bytes", 65536, 64 * 1024 * 1024, int)
    st["decode_concurrency"] = num("decode_concurrency", 1, 16, int)
    st["max_connections"] = num("max_connections", 2, 1024, int)
    st["request_timeout_s"] = num("request_timeout_s", 0.2, 60.0)
    jr = merged.get("jpeg_reduce", D_["jpeg_reduce"])
    st["jpeg_reduce"] = bool(jr)
    st["journal_root"] = str(merged.get("journal_root", D_["journal_root"]))
    try:
        st["defaults"] = S.Config().updated({k: merged[k] for k in TRACKER_KEYS if k in merged})
    except ValueError as e:
        raise OptionsError(f"tracker options invalid: {e}")
    return st


class RateLimiter:
    """Token bucket per camera: sustained `per_s` frames/s, bursts up to `burst` (a pre-roll burst of up to 10 frames within 1 s is accepted)."""

    def __init__(self, per_s=RATE_PER_S, burst=RATE_BURST):
        self.per_s, self.burst = per_s, burst; self.b = {}; self.lock = threading.Lock(); self.next_sweep = 0.0

    def check(self, cam, now):
        """Returns 0 if allowed, else retry_after_ms."""
        with self.lock:
            if now >= self.next_sweep:                       # amortised: drop idle cameras once a minute
                self.next_sweep = now + 60.0
                for c in [c for c, (_, l) in self.b.items() if now - l > 60]:
                    del self.b[c]
            tokens, last = self.b.get(cam, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.per_s)
            if tokens < 1.0:
                self.b[cam] = (tokens, now)
                return max(1, int((1.0 - tokens) / self.per_s * 1000) + 1)
            self.b[cam] = (tokens - 1.0, now)
            return 0


class App:
    def __init__(self, opts, spec, on_fatal=None):
        self.cfg = parse_options(opts)
        c = self.cfg
        self.token = c["token"].encode("utf-8")
        self.size = c["model"]
        self.registry = S.Registry(c["defaults"], c["max_cameras"])
        self.limiter = RateLimiter()
        self.decode_sem = threading.BoundedSemaphore(c["decode_concurrency"])
        self.coral = CoralManager(spec, max_queue_age_ms=c["max_queue_age_ms"], max_queue_depth=c["max_queue_depth"], fatal_after=c["self_restart_after"], on_fatal=on_fatal)
        self.t0 = time.monotonic()
        self.t_clamped = 0
        self.sampler = M.Sampler(self.coral.child_pid)         # started by main() (tests call sample() themselves)
        self.now_ms = lambda: time.time() * 1000.0            # test seam
        self.journal = Journal(c["journal_root"], clock=lambda: self.now_ms())
        self.ingress_ips = set(DF.JOURNAL["ingress_ips"])
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html"), "rb") as f:
            self.ui_html = f.read().replace(b"__VERSION__", VERSION.encode())       # the served page names its version (never a stale page after an update)

    def health(self):
        st = self.coral.stats()
        r1 = lambda v: None if v is None else round(v, 1)
        return {"ok": st["coral"] == "ok", "version": VERSION,
                "model": {"name": f"yolov9s_relu6_{self.size}_edgetpu", "input": self.size, "coral": st["coral"]},
                "coral_ms": {"median": r1(st["infer"]["median"]), "p95": r1(st["infer"]["p95"]), "last": r1(st["last"])},
                "stages": st["stages"], "dropped_stale": st["dropped_stale"], "t_clamped": self.t_clamped,
                "resets": st["resets"], "queue_depth": st["queue_depth"],
                "cameras_active": self.registry.count(time.monotonic()), "uptime_s": int(time.monotonic() - self.t0)}

    @staticmethod
    def _sl_metrics(roll):
        n = roll.sum("sl")
        share = lambda k: None if n <= 0 else round(roll.sum(k) / n, 3)
        return {"count_60s": int(n), "reused_60s": int(roll.sum("sl_reused")), "confirmed_share": share("sl_confirmed"),
                "lowered_share": share("sl_lowered"), "rejected_share": share("sl_rejected"), "not_found_share": share("sl_not_found"), "not_upgraded_share": share("sl_not_upgraded"), "skipped_60s": int(roll.sum("sl_skipped"))}

    def metrics(self):
        st, roll, now = self.coral.stats(), self.coral.roll, time.monotonic()
        r1 = lambda v: None if v is None else round(v, 1)
        up = now - self.t0
        win = max(1.0, min(float(M.WINDOW_S), up))
        sv = self.sampler.values()
        load1, mem = M.host_instant()
        busy = roll.sum("busy_ms")
        return {"coral": {"state": st["coral"], "invoke_ms": {"median": r1(st["infer"]["median"]), "p95": r1(st["infer"]["p95"]), "last": r1(st["last"])},
                          "utilization_pct": round(min(100.0, busy / (win * 10.0)), 1), "invocations_per_s": round(roll.sum("inv") / win, 2), "resets": st["resets"]},
                "stages": st["stages"],
                "load": {"queue_depth": st["queue_depth"], "dropped_stale_total": st["dropped_stale"], "dropped_stale_60s": int(roll.sum("stale")),
                         "rejected_429_60s": int(roll.sum("r429")), "shed_passes_60s": int(roll.sum("shed")), "reduced_frames_60s": int(roll.sum("reduced")), "promoted_frames_60s": int(roll.sum("promoted")), "cameras_active": self.registry.count(now), "frames_60s": int(roll.sum("frames"))},
                "second_look": self._sl_metrics(roll),
                "journal": self.journal.stats(),
                "host": {"cpu_pct": sv["host_cpu"], "load1": load1, "mem_used_pct": mem},
                "process": {"cpu_pct": sv["proc_cpu"], "rss_mb": M.rss_mb()},
                "gpu": {"available": sv["gpu_ok"], "busy_pct": sv["gpu_busy"]},
                "uptime_s": int(up)}


def _bad_constant(name):
    raise ValueError("non-finite number")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = None                  # deadlines are enforced by the Server's reaper (total time per phase), not per recv
    server_version = "coral-verifier"
    sys_version = ""

    # ---- connection lifecycle / deadlines ----
    def setup(self):
        super().setup()
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.deadline = time.monotonic() + FIRST_BYTE_S
        self.first = True
        self.reserved = self.server.is_reserved(self.request)
        self.server.register(self)

    def finish(self):
        self.server.unregister(self)
        try:
            super().finish()
        except OSError:
            pass

    def handle_one_request(self):
        self.deadline = time.monotonic() + (FIRST_BYTE_S if self.first else IDLE_S)
        self.first = False
        super().handle_one_request()

    def parse_request(self):
        self.deadline = time.monotonic() + self.server.app.cfg["request_timeout_s"]       # headers must arrive within this, in total
        return super().parse_request()

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    # ---- plumbing ----
    def _send(self, status, body, close=False):
        data = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if close:
            self.close_connection = True
            self.send_header("Connection", "close")
        self._headers_buffer.append(b"\r\n" + data)      # headers and body leave in ONE send
        self.flush_headers()

    def _send_raw(self, status, data, ctype, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        if cache == "no-store":
            self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self._headers_buffer.append(b"\r\n" + data)
        self.flush_headers()

    def _is_json(self):
        return (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() == "application/json"

    def _via_ingress(self):
        """Home Assistant ingress: the request comes from the Supervisor gateway and carries X-Ingress-Path. HA has authenticated the user."""
        return bool(self.headers.get("X-Ingress-Path")) and self.client_address[0] in self.server.app.ingress_ips

    def _reject(self, status, body):
        """Early rejection: the body (if any) is NOT read, the connection is closed."""
        self._send(status, body, close=True)

    def _read_body(self, limit, required):
        if self.headers.get("Transfer-Encoding") is not None:
            return self._reject(411, {"error": "Content-Length required (chunked bodies are not accepted)"})
        n = self.headers.get("Content-Length")
        if n is None:
            if required:
                return self._reject(411, {"error": "Content-Length required"})
            return b""
        try:
            n = int(n)
        except ValueError:
            n = -1
        if n < 0:
            return self._reject(400, {"error": "bad Content-Length"})
        if n > limit:
            return self._reject(413, {"error": f"body larger than {limit} bytes"})
        self.deadline = time.monotonic() + self.server.app.cfg["request_timeout_s"]       # the whole body within this, in total
        body = self.rfile.read(n) if n else b""
        if len(body) != n:
            self.close_connection = True
            return self._send(400, {"error": "incomplete body"}, close=True)
        self.deadline = time.monotonic() + JOB_WAIT_S + 20
        return body

    def _authorized(self):
        app = self.server.app
        if not app.token:
            return True
        got = (self.headers.get("X-Verifier-Token") or "").encode("latin-1", "replace")
        return hmac.compare_digest(got, app.token)

    def _dispatch(self, method):
        app = self.server.app
        try:
            u = urlsplit(self.path)
            parts = [unquote(p) for p in u.path.strip("/").split("/")]
            q = parse_qs(u.query)
            if self.reserved and parts != ["v1", "health"]:       # the reserved pool is for the watchdog only
                return self._reject(503, {"error": "too many clients"})
            if method == "GET" and parts in ([""], ["index.html"], ["ui"]):
                return self._send_raw(200, app.ui_html, "text/html; charset=utf-8")      # static page, no data in it: the API calls it makes are authenticated
            if parts[:1] != ["v1"]:
                return self._reject(404, {"error": "not found"})
            if parts == ["v1", "health"]:
                if method != "GET":
                    return self._reject(405, {"error": "method not allowed"})
                h = app.health()
                return self._send(200 if h["ok"] else 503, h)           # 503 while stalled/missing: the Supervisor watchdog and clients see an unhealthy service
            if not (self._authorized() or (parts[1:2] == ["journal"] and self._via_ingress())):      # BEFORE any body is read; ingress only for the journal routes
                return self._reject(401, {"error": "unauthorized"})
            if parts == ["v1", "metrics"]:
                if method != "GET":
                    return self._reject(405, {"error": "method not allowed"})
                return self._send(200, app.metrics())
            if parts[1:2] == ["journal"]:
                return self._journal(app, method, parts, q)
            if parts == ["v1", "cameras"]:
                if method != "GET":
                    return self._reject(405, {"error": "method not allowed"})
                cams = app.registry.listing(time.monotonic())
                return self._send(200, {"total": len(cams), "cameras": cams})
            if len(parts) in (3, 4) and parts[1] == "cameras":
                cam = parts[2]
                if not CAM_RE.match(cam):
                    return self._reject(400, {"error": "bad camera name"})
                sub = parts[3] if len(parts) == 4 else None
                if sub is None:
                    if method != "GET":
                        return self._reject(405, {"error": "method not allowed"})
                    if not app.registry.known(cam, time.monotonic()):
                        return self._send(404, {})
                    return self._send(200, {"camera": cam, **app.registry.peek(cam, time.monotonic())})
                if sub == "frame":
                    if method != "POST":
                        return self._reject(405, {"error": "method not allowed"})
                    body = self._read_body(app.cfg["max_body_bytes"], True)
                    if not isinstance(body, bytes):
                        return
                    return self._frame(app, cam, q, body)
                if sub == "config":
                    if method == "GET":
                        if not app.registry.known(cam, time.monotonic()):
                            return self._send(404, {})
                        return self._send(200, app.registry.config(cam).as_dict())
                    body = self._read_body(MAX_JSON_BYTES, False)
                    if not isinstance(body, bytes):
                        return
                    try:
                        changes = json.loads(body or b"{}", parse_constant=_bad_constant)
                        if not isinstance(changes, dict):
                            raise ValueError("body must be a JSON object")
                        return self._send(200, app.registry.set_config(cam, changes, time.monotonic()).as_dict())
                    except S.TooManyCameras:
                        return self._send(429, {"error": "too many cameras"})
                    except (json.JSONDecodeError, RecursionError):
                        return self._send(400, {"error": "invalid JSON"})
                    except ValueError as e:
                        return self._send(400, {"error": str(e)[:200]})
                if sub == "episodes":
                    if method != "POST":
                        return self._reject(405, {"error": "method not allowed"})
                    if not self._is_json():
                        return self._reject(415, {"error": "Content-Type must be application/json"})
                    body = self._read_body(MAX_JSON_BYTES, True)
                    if not isinstance(body, bytes):
                        return
                    try:
                        obj = json.loads(body, parse_constant=_bad_constant)
                        if not isinstance(obj, dict):
                            raise ValueError("body must be a JSON object")
                    except (json.JSONDecodeError, RecursionError, ValueError):
                        return self._send(400, {"error": "invalid JSON object"})
                    st, out = app.journal.episode(cam, obj, app.registry.config(cam).name)
                    return self._send(st, out)
                if sub == "reset":
                    if method != "POST":
                        return self._reject(405, {"error": "method not allowed"})
                    body = self._read_body(MAX_JSON_BYTES, False)
                    if not isinstance(body, bytes):
                        return
                    return self._send(200, {"camera": cam, "reset": True, "had_session": app.registry.reset(cam)})
            return self._reject(404, {"error": "not found"})
        except OSError:                                    # peer gone or cut by the reaper
            self.close_connection = True
        except ValueError as e:
            self.close_connection = True
            if "closed file" not in str(e):
                log.exception("request failed")
                try:
                    self._send(500, {"error": "internal error"}, close=True)
                except Exception:
                    pass
        except Exception:
            log.exception("request failed")
            try:
                self._send(500, {"error": "internal error"}, close=True)
            except Exception:
                self.close_connection = True

    def _frame(self, app, cam, q, body):
        arrival = time.monotonic()
        try:
            t_raw = float(q["t"][0]) if "t" in q else app.now_ms()
        except ValueError:
            return self._send(400, {"error": "bad t"})
        t, clamped = S.bound_t(t_raw, app.now_ms())              # only [now - 60 s, now + 5 s] is trusted; else server time
        if clamped:
            app.t_clamped += 1
        tiles = (q.get("tiles") or ["auto"])[0]
        if tiles not in ("auto", "1", "2", "3"):
            return self._send(400, {"error": "tiles must be auto|1|2|3"})
        deadline = None
        if "deadline_ms" in q:
            try:
                deadline = float(q["deadline_ms"][0])
            except ValueError:
                return self._send(400, {"error": "bad deadline_ms"})
            if not math.isfinite(deadline) or deadline <= 0:
                return self._send(400, {"error": "deadline_ms must be > 0"})
        verbose = (q.get("verbose") or ["0"])[0] == "1"
        prio = (q.get("prio") or ["normal"])[0]
        if prio not in ("normal", "low"):
            return self._send(400, {"error": "prio must be normal|low"})
        if not app.registry.would_admit(cam, time.monotonic()):
            app.coral.roll.add("r429")
            return self._send(429, {"error": "too many cameras"})
        retry = app.limiter.check(cam, time.monotonic())
        if retry:
            app.coral.roll.add("r429")
            return self._send(429, {"retry_after_ms": retry})
        state = app.coral.state
        if state != "ok":                                  # fail fast, do not even decode
            return self._send(503, {"coral": state})
        if not app.decode_sem.acquire(timeout=2.0):
            return self._send(503, {"reason": "busy"})
        t_dec = time.perf_counter()
        try:
            img, tiles = D.decode_jpeg(body, app.size, tiles, app.cfg["jpeg_reduce"], app.cfg["max_image_pixels"])
        except ValueError as e:
            return self._send(400, {"error": str(e)[:200]})
        finally:
            app.decode_sem.release()
        decode_ms = (time.perf_counter() - t_dec) * 1e3
        cfg = app.registry.config(cam)

        def sl_hook(engine, frame, dets, reduced=False):
            now = time.monotonic()
            ctx = app.registry.sl_context(cam, now)
            extra = {"preprocess_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0}
            promoted = None
            if reduced:                                                  # busy low-priority frame: promote it back to full when its camera is near a status change
                promoted = SL.promote_reason(dets, cfg, ctx["promo"], t, ctx["tracks"])
                if promoted:
                    reduced = False
                    if cfg.zoom_out > 0:
                        dets, tz = engine.add_zoom(frame, dets, cfg.zoom_out, cfg.cont_thr, cfg.vehicle_thr, cfg.animal_cont_thr)
                        for k in extra:
                            extra[k] += tz[k]
            if not cfg.second_look:
                return dets, {**extra, "counts": {}, "promoted": bool(promoted)}
            out, info = SL.second_look(engine, frame, dets, cfg, ctx["tracks"], ctx["cache"], now, reduced)
            for k in extra:
                info[k] = info.get(k, 0.0) + extra[k]
            info["promoted"] = bool(promoted)
            app.registry.sl_store(cam, info["cache"], now)
            return out, info

        def finish(dets, st):
            dets = S.person_animal_overlap(dets)                                                        # one object scored as person and animal: boost the person
            live = [SL.tracker_view(d, cfg) for d in dets if d.get("second_look") != "rejected"]      # rejected: never; doubtful person/animal: score capped
            sn = app.registry.update(cam, t, live, time.monotonic())
            try:
                app.journal.record_frame_async(cam, sn["t"], body, dets)
            except Exception:
                log.exception("journal record failed")
            shown = dets if verbose else [d for d in live if d["score"] >= REPLY_MIN_SCORE]
            return {"camera": cam, "t": sn["t"], "coral_ms": round(st["infer_ms"], 1), "queue_ms": round(st["queue_ms"], 1),
                    "decode_ms": round(st["decode_ms"], 2), "preprocess_ms": round(st["preprocess_ms"], 2), "infer_ms": round(st["infer_ms"], 2),
                    "post_ms": round(st["post_ms"], 2), "second_look_ms": round(st.get("second_look_ms", 0.0), 2), "reduced": bool(st.get("reduced")), "total_ms": round(st["total_ms"], 2), "detections": shown,
                    "person": sn["person"], "vehicle": sn["vehicle"], "animal": sn["animal"], "session": sn["session"]}

        job = app.coral.submit(Job(img, tiles, cfg.cont_thr, finish, decode_ms, deadline, arrival, low=(prio == "low"),
                      thr={"vehicle_thr": cfg.vehicle_thr, "animal_thr": cfg.animal_cont_thr, "zoom_out": cfg.zoom_out},
                      post=sl_hook if (cfg.second_look or cfg.zoom_out > 0) else None))
        if not job.event.wait(JOB_WAIT_S):
            job.complete(500, {"error": "timeout waiting for the Coral queue"})
        self._send(job.status, job.body)

    def _journal(self, app, method, parts, q):
        j = app.journal
        one = lambda k, d=None: (q.get(k) or [d])[0]

        def num(k):
            v = one(k)
            if v is None or v == "":
                return None
            try:
                f = float(v)
            except ValueError:
                raise ValueError(f"{k} must be a number")
            if not math.isfinite(f):
                raise ValueError(f"{k} must be finite")
            return f
        if not j.enabled:
            return self._send(503, {"error": "journal unavailable"})
        rest = parts[2:]
        try:
            if rest == [] and method == "GET":
                lim, off = num("limit"), num("offset")
                lab = one("labelled")
                if lab not in (None, "", "true", "false"):
                    raise ValueError("labelled must be true or false")
                grp = one("group")
                if grp not in (None, "", "person", "vehicle", "animal"):
                    raise ValueError("group must be person, vehicle or animal")
                st_ = one("status")
                if st_ not in (None, "", "ongoing", "closed"):
                    raise ValueError("status must be ongoing or closed")
                return self._send(200, j.list(one("camera"), one("outcome"), one("label"), num("from"), num("to"), 50 if lim is None else lim, 0 if off is None else off,
                                              None if not lab else lab == "true", st_ or None, grp or None))
            if rest == ["next"] and method == "GET":
                after = one("after")
                if after and not Journal.valid_id(after):
                    raise ValueError("bad after id")
                grp = one("group")
                if grp not in (None, "", "person", "vehicle", "animal"):
                    raise ValueError("group must be person, vehicle or animal")
                return self._send(200, j.next_unlabelled(after or None, one("camera") or None, one("outcome") or None, grp or None, num("from"), num("to")))
            if rest == ["summary"] and method == "GET":
                tz = num("tz")
                if tz is not None and abs(tz) > 840:
                    raise ValueError("tz must be minutes east of UTC within +-840")
                return self._send(200, j.summary(num("from"), num("to"), one("camera") or None, 0 if tz is None else int(tz),
                                                  one("group") if one("group") in ("person", "vehicle", "animal") else None))
            if len(rest) == 1 and method == "GET":
                m = j.get(rest[0])
                return self._send(200, m) if m else self._send(404, {"error": "unknown episode"})
            if len(rest) == 2 and rest[1] == "label" and method == "POST":
                if not self._is_json():
                    return self._reject(415, {"error": "Content-Type must be application/json"})
                body = self._read_body(MAX_JSON_BYTES, True)
                if not isinstance(body, bytes):
                    return
                try:
                    obj = json.loads(body, parse_constant=_bad_constant)
                    if not isinstance(obj, dict):
                        raise ValueError("object expected")
                except (json.JSONDecodeError, RecursionError, ValueError):
                    return self._send(400, {"error": "invalid JSON object"})
                st, out = j.set_label(rest[0], obj.get("label"), obj.get("note"))
                return self._send(st, out)
            if len(rest) == 2 and method == "GET" and "." in rest[1]:
                name, ext = rest[1].rsplit(".", 1)
                p = j.file(rest[0], name, ext)
                if p is None:
                    return self._send(404, {"error": "not found"})
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except OSError:                                    # deleted by the retention in between
                    return self._send(404, {"error": "not found"})
                return self._send_raw(200, data, "image/jpeg" if ext == "jpg" else "application/json", "private, max-age=300")
        except ValueError as e:
            return self._send(400, {"error": str(e)[:200]})
        return self._reject(404 if method == "GET" else 405, {"error": "not found"})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._reject(405, {"error": "method not allowed"})

    do_DELETE = do_PATCH = do_PUT


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64
    allow_reuse_address = True

    def __init__(self, addr, handler, max_connections=64, reserved=RESERVED_HEALTH_SLOTS):
        self._conn_sem = threading.BoundedSemaphore(max_connections)
        self._reserve_sem = threading.BoundedSemaphore(reserved)
        self._reserved_ids = set()
        self._handlers = set()
        self._hlock = threading.Lock()
        self._reaper_stop = threading.Event()
        super().__init__(addr, handler)
        threading.Thread(target=self._reaper, name="http-reaper", daemon=True).start()

    # connection cap: beyond max_connections only a few reserved slots remain, usable by GET /v1/health alone (the Supervisor watchdog);
    # beyond those a new connection gets an immediate 503 and is closed
    def process_request(self, request, client_address):
        if self._conn_sem.acquire(blocking=False):
            pass
        elif self._reserve_sem.acquire(blocking=False):
            with self._hlock:
                self._reserved_ids.add(id(request))
        else:
            body = b'{"error":"too many clients"}'
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode()
                                + b"\r\nConnection: close\r\n\r\n" + body)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def is_reserved(self, request):
        with self._hlock:
            return id(request) in self._reserved_ids

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._hlock:
                reserved = id(request) in self._reserved_ids
                self._reserved_ids.discard(id(request))
            (self._reserve_sem if reserved else self._conn_sem).release()

    def register(self, h):
        with self._hlock:
            self._handlers.add(h)

    def unregister(self, h):
        with self._hlock:
            self._handlers.discard(h)

    def _reaper(self):
        """Slowloris guard: a request that has not finished receiving its headers/body by its deadline (a TOTAL, not per recv) is cut."""
        while not self._reaper_stop.wait(0.2):
            now = time.monotonic()
            with self._hlock:
                late = [h for h in self._handlers if now > h.deadline]
            for h in late:
                try:
                    h.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def server_close(self):
        self._reaper_stop.set()
        super().server_close()


def make_server(opts, spec, host="0.0.0.0", port=None, on_fatal=None):
    app = App(opts, spec, on_fatal)
    httpd = Server((host, app.cfg["port"] if port is None else port), Handler, app.cfg["max_connections"])
    httpd.app = app
    return httpd, app


def default_spec(opts):
    size = int({**DF.env_overrides(), **opts}.get("model", DF.SERVICE["model"]))
    if os.environ.get("VERIFIER_FAKE_TPU") == "1":            # smoke tests only (fake_tpu.py is not in the runtime image)
        return {"kind": "fake", "size": size, "ctl": os.environ.get("VERIFIER_FAKE_CTL", "/tmp")}
    return {"kind": "edgetpu", "path": os.path.join(os.environ.get("MODEL_DIR", "/models"), MODEL_FILES[size]), "size": size}


def load_options(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main():
    opts = {k: v for k, v in load_options(os.environ.get("VERIFIER_OPTIONS", "/data/options.json")).items() if k in DF.PUBLIC_OPTIONS}   # only the public options count
    level = {"trace": "DEBUG", "debug": "DEBUG", "info": "INFO", "notice": "INFO", "warning": "WARNING", "error": "ERROR", "fatal": "CRITICAL"}
    logging.basicConfig(level=getattr(logging, level.get(str(opts.get("log_level", "info")).lower(), "INFO")),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    holder = {}

    def die():                                         # Coral recovery keeps failing: exit non-zero, Supervisor (watchdog) restarts the add-on
        logging.shutdown()
        os._exit(3)

    try:
        httpd, app = make_server(opts, default_spec(opts), on_fatal=die)
    except OptionsError as e:
        log.error("refusing to start: %s", e)
        logging.shutdown()
        sys.exit(2)
    holder["app"] = app
    app.coral.start()
    app.sampler.start()
    log.info("coral_verifier %s listening on :%s (model %s, auth %s)", VERSION, httpd.server_address[1], app.size, "token" if app.token else "NONE (allow_no_token, dev only)")
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=httpd.shutdown, daemon=True).start())
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    app.sampler.stop()
    app.coral.stop()                                    # kills the Coral child first
    app.journal.stop()
    httpd.server_close()


if __name__ == "__main__":
    main()
