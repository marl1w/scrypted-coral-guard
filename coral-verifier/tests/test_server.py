"""HTTP layer end-to-end: real server, real Coral child process, fake TPU. Endpoints, codes, token, 429, 503, stale/shed, timings, health."""
import http.client, json, os, re, threading, time
import pytest
from boxlib import Box, BASE as T0, NOW, TOKEN, jpeg, person_item, burst


@pytest.fixture
def box():
    b = Box()
    yield b
    b.close()


def test_health_shape_and_no_auth_needed(box):
    st, h = box.req("GET", "/v1/health", auth=False)
    assert st == 200 and h["ok"] is True and h["version"]
    assert h["model"] == {"name": "yolov9s_relu6_512_edgetpu", "input": 512, "coral": "ok"}
    assert set(h["coral_ms"]) == {"median", "p95", "last"} and h["resets"] == 0 and h["queue_depth"] == 0
    assert h["cameras_active"] == 0 and h["uptime_s"] >= 0 and h["t_clamped"] == 0 and h["dropped_stale"] == 0
    box.frame("c1", T0)
    h = box.req("GET", "/v1/health", auth=False)[1]
    assert h["coral_ms"]["last"] > 0 and h["cameras_active"] == 1
    assert set(h["stages"]) == {"decode_ms", "preprocess_ms", "infer_ms", "post_ms", "queue_ms", "total_ms", "second_look_ms"}
    assert all(set(v) == {"median", "p95"} and v["median"] is not None for v in h["stages"].values())


def test_frame_reply_shape_and_session(box):
    st, r = box.frame("front_door", T0)
    assert st == 200
    assert set(r) == {"camera", "t", "coral_ms", "queue_ms", "decode_ms", "preprocess_ms", "infer_ms", "post_ms", "second_look_ms", "total_ms", "detections", "person", "vehicle", "animal", "session"}
    assert r["camera"] == "front_door" and r["t"] == T0
    for k in ("decode_ms", "preprocess_ms", "infer_ms", "post_ms", "queue_ms", "total_ms", "coral_ms"):
        assert isinstance(r[k], float) and r[k] >= 0, k
    assert r["total_ms"] >= r["infer_ms"] >= 1.5 and abs(r["coral_ms"] - r["infer_ms"]) <= 0.051                  # same value, rounded to 1 and 2 decimals (no exact-equality flake on .x5)
    d = r["detections"][0]
    assert d["cls"] == "person" and 0.9 < d["score"] <= 1 and len(d["box"]) == 4 and all(0 <= v <= 1 for v in d["box"])
    assert r["person"]["state"] == "candidate" and r["person"]["present"] is False
    st, r = box.frame("front_door", T0 + 1000)
    p = r["person"]
    assert p["state"] == "confirmed" and p["confirmed"] and p["present"] and p["tracks"] == 1 and p["since_ms"] == T0 and p["last_seen_ms"] == T0 + 1000
    assert r["session"] == {"frames": 2, "started_ms": T0} and r["vehicle"] == {"moving": False, "hold_until_ms": None, "tracks": []}
    st, g = box.req("GET", "/v1/cameras/front_door")
    assert st == 200 and g["person"]["present"] and "detections" not in g and g["session"]["frames"] == 2
    st, lst = box.req("GET", "/v1/cameras")
    assert [c["camera"] for c in lst["cameras"]] == ["front_door"] and lst["cameras"][0]["present"] is True
    assert box.req("GET", "/v1/cameras/other") == (404, {})
    assert box.req("GET", "/v1/cameras/other/config") == (404, {})
    st, r = box.req("POST", "/v1/cameras/front_door/reset")
    assert st == 200 and r["reset"] is True
    assert box.req("GET", "/v1/cameras/front_door")[0] == 404
    assert box.req("GET", "/v1/cameras")[1] == {"total": 0, "cameras": []}


def test_get_state_expires_with_wall_clock_not_only_frames(box):
    box.frame("w", T0); box.frame("w", T0 + 1000)
    assert box.req("GET", "/v1/cameras/w")[1]["person"]["present"]
    box.app.registry.config("w")
    time.sleep(0.05)
    sess = box.app.registry._sessions["w"]
    sess.wall -= 20                                                    # pretend the last frame arrived 20 s ago (server clock)
    assert box.req("GET", "/v1/cameras/w")[1]["person"]["state"] == "idle"   # hold_s 8 elapsed without any new frame


def test_vehicle_detection_flows_to_session():
    b = Box(items=[["car", 3.0, 16, 8, 8, [3, 2, 3, 2]]])
    try:
        r = b.frame("g", T0)[1]
        assert r["detections"][0]["cls"] == "car" and r["vehicle"]["moving"] is False and r["vehicle"]["tracks"][0]["cls"] == "car" and r["vehicle"]["tracks"][0]["state"] == "still"
        assert r["person"]["state"] == "idle"
    finally:
        b.close()


def test_token_required_and_health_exempt():
    b = Box(opts={"token": "s3cret"})
    try:
        assert b.req("GET", "/v1/health", auth=False)[0] == 200
        assert b.frame("c", auth=False)[0] == 401
        assert b.frame("c", headers={"X-Verifier-Token": "nope"}, auth=False)[0] == 401
        assert b.frame("c", headers={"X-Verifier-Token": "s3cret"}, auth=False)[0] == 200
        assert b.req("GET", "/v1/cameras", auth=False)[0] == 401
        assert b.req("POST", "/v1/cameras/c/reset", auth=False)[0] == 401
        assert b.req("POST", "/v1/cameras/c/config", b"{}", auth=False)[0] == 401
    finally:
        b.close()


def test_non_ascii_token_header_is_401_not_500(box):
    c = http.client.HTTPConnection("127.0.0.1", box.port, timeout=5)
    c.putrequest("GET", "/v1/cameras"); c.putheader("X-Verifier-Token", "tést".encode("utf-8").decode("latin-1")); c.endheaders()
    r = c.getresponse(); r.read(); assert r.status == 401
    c.close()


def test_bad_requests_map_to_400_without_leaks(box):
    for st, resp in [box.frame("c", data=b"not a jpeg"), box.frame("c", data=b""), box.frame("c", q="&tiles=5"),
                     box.req("POST", "/v1/cameras/c/frame?t=abc", jpeg()), box.frame("bad%20name!"),
                     box.req("POST", "/v1/cameras/c/config", b"{bad"), box.req("POST", "/v1/cameras/c/config", b"[1]"),
                     box.req("POST", "/v1/cameras/c/config", b'{"hold_s": -3}'), box.req("POST", "/v1/cameras/c/config", b'{"zzz": 1}'),
                     box.req("POST", "/v1/cameras/c/config", b'{"hold_s": Infinity}'), box.req("POST", "/v1/cameras/c/config", b'{"hold_s": NaN}'),
                     box.req("POST", "/v1/cameras/c/config", b'{"cont_thr": 0.9}'), box.req("POST", "/v1/cameras/c/config", b'{"confirm_n": 5, "confirm_m": 3}'),
                     box.frame("c", q="&deadline_ms=zz"), box.frame("c", q="&deadline_ms=0"), box.frame("c", q="&deadline_ms=-5"), box.frame("c", q="&deadline_ms=nan")]:
        assert st == 400, resp
        assert "Traceback" not in json.dumps(resp) and "Error(" not in json.dumps(resp) and "object at" not in json.dumps(resp)
    assert box.req("GET", "/v1/nope")[0] == 404
    assert box.req("GET", "/other", auth=False)[0] == 404
    assert box.req("GET", "/v1/cameras/c/frame")[0] == 405
    assert box.req("DELETE", "/v1/cameras/c")[0] == 405
    assert box.req("POST", "/v1/health")[0] == 405
    assert box.frame("c")[0] == 200


def test_rate_limit_429_per_camera(box):
    codes = [box.frame("fast", T0 + i)[0] for i in range(14)]
    assert codes[:10] == [200] * 10 and 429 in codes[10:]
    st, body = box.frame("fast")
    assert st == 429 and 0 < body["retry_after_ms"] <= 1001
    assert box.frame("other")[0] == 200
    time.sleep(1.3)
    assert box.frame("fast")[0] == 200


def test_config_get_post(box):
    assert box.req("GET", "/v1/cameras/k/config") == (404, {})
    box.req("POST", "/v1/cameras/k/config", b"{}")
    st, c = box.req("GET", "/v1/cameras/k/config")
    assert st == 200 and c["start_thr"] == 0.65 and c["hold_s"] == 8 and c["confirm_n"] == 2 and c["still_s"] == 3
    st, c = box.req("POST", "/v1/cameras/k/config", json.dumps({"hold_s": 2, "confirm_n": 1}).encode())
    assert st == 200 and c["hold_s"] == 2 and c["confirm_n"] == 1
    assert box.frame("k", T0)[1]["person"]["state"] == "confirmed"


def test_defaults_from_options():
    b = Box(opts={"hold_s": 3, "start_thr": 0.6})
    try:
        b.req("POST", "/v1/cameras/z/config", b"{}")
        c = b.req("GET", "/v1/cameras/z/config")[1]
        assert c["hold_s"] == 3 and c["start_thr"] == 0.6
    finally:
        b.close()


def test_tiling_wide_frame_and_merge_through_http(box):
    wide = jpeg(1536, 576)
    n = box.calls()
    r = box.frame("duo", T0, "&tiles=1", wide)[1]
    assert box.calls() == n + 1 and len(r["detections"]) == 1
    n = box.calls()
    r = box.frame("duo", T0 + 1000, "&tiles=2", wide)[1]
    assert box.calls() == n + 2
    cx = sorted((d["box"][0] + d["box"][2]) / 2 for d in r["detections"])
    assert len(cx) == 2 and cx[0] < 0.45 and cx[1] > 0.55
    n = box.calls()
    r = box.frame("duo", T0 + 1500, "", wide)[1]               # auto, aspect 2.67 -> hybrid: whole frame + 2 tiles = 3 invokes, ONE request
    assert box.calls() == n + 3 and r["infer_ms"] > 0 and len(r["detections"]) >= 2
    n = box.calls(); box.frame("duo", T0 + 1600, "&tiles=3", jpeg(640, 480)); assert box.calls() == n + 3
    n = box.calls(); box.frame("duo", T0 + 2000, "&tiles=auto", jpeg(640, 480)); assert box.calls() == n + 1
    n = box.calls(); box.frame("duo", T0 + 3000, "&tiles=2", jpeg(640, 480)); assert box.calls() == n + 2


def test_stage_timings_and_one_invoke_at_a_time_three_cameras(box):
    results = {}
    def cam(name):
        res = []
        for i in range(12):
            res.append(box.frame(name, T0 + i * 250)[0]); time.sleep(0.26)
        results[name] = res
    ths = [threading.Thread(target=cam, args=(f"cam{i}",)) for i in range(3)]
    [t.start() for t in ths]; [t.join() for t in ths]
    assert all(c == 200 for r in results.values() for c in r), results
    tr = sorted(box.trace(), key=lambda x: x[1])
    assert len(tr) >= 36 and all(b[1] >= a[2] - 1e-3 for a, b in zip(tr, tr[1:]))     # no two invokes ever overlap
    assert box.req("GET", "/v1/health")[1]["queue_depth"] == 0


def test_stale_frames_are_dropped_not_queued():
    b = Box(opts={"max_queue_age_ms": 200}, files={"delay": "0.15"})
    try:
        out = burst(b, 6, 0.01)
        ok = [o for o in out if o[0] == 200]; stale = [o for o in out if o[0] == 503]
        assert ok and stale and len(ok) + len(stale) == 6
        assert all(o[1] == {"reason": "stale"} for o in stale)
        assert all(o[1]["queue_ms"] <= 230 for o in ok)
        h = b.req("GET", "/v1/health")[1]
        assert h["dropped_stale"] == len(stale) and h["ok"] is True
        assert b.frame("later")[0] == 200
    finally:
        b.close()


def test_client_deadline_ms():
    b = Box(opts={"max_queue_age_ms": 5000}, files={"delay": "0.2"})
    try:
        out = burst(b, 3, 0.02, q="&deadline_ms=100")
        assert out[0][0] == 200 and out[1][0] == 503 and out[1][1] == {"reason": "stale"} and out[2][0] == 503
    finally:
        b.close()


def test_bounded_queue_sheds_oldest_first():
    b = Box(opts={"max_queue_age_ms": 5000, "max_queue_depth": 2}, files={"delay": "0.3"})
    try:
        out = burst(b, 5, 0.04)
        assert [o[0] for o in out] == [200, 503, 503, 200, 200], out         # #0 in flight, #1 and #2 (oldest arrivals) shed, #3 and #4 kept
        assert out[1][2] < 0.25 and out[2][2] < 0.3
        assert b.req("GET", "/v1/health")[1]["dropped_stale"] == 2
    finally:
        b.close()


def test_queue_is_ordered_by_arrival_not_by_decode_completion():
    import coral as C
    from detect import Engine
    jobs = []
    mgr = C.CoralManager({"kind": "fake", "size": 512, "ctl": "."}, max_queue_depth=3)
    mgr.state = "ok"
    mk = lambda arr: C.Job(None, "1", 0.2, lambda d, s: {}, 0.0, None, arrival=arr)
    for arr in (5.0, 3.0, 4.0, 1.0):                                   # submitted out of arrival order (slow decode of old frames)
        jobs.append(mgr.submit(mk(arr)))
    assert [j.arrival for j in mgr.dq] == [3.0, 4.0, 5.0]               # sorted, depth 3: the OLDEST arrival (1.0) was shed even though it came last
    assert jobs[3].status == 503 and jobs[3].body == {"reason": "stale"}
    assert all(j.status is None for j in jobs[:3])


def test_low_priority_frames_are_shed_first_and_served_last():
    import coral as C
    mgr = C.CoralManager({"kind": "fake", "size": 512, "ctl": "."}, max_queue_depth=3)
    mgr.state = "ok"
    mk = lambda arr, low: C.Job(None, "1", 0.2, lambda d, s: {}, 0.0, None, arrival=arr, low=low)
    low = mgr.submit(mk(1.0, True))
    n1, n2 = mgr.submit(mk(2.0, False)), mgr.submit(mk(3.0, False))
    n3 = mgr.submit(mk(4.0, False))                                      # full: the oldest LOW (arrived first) is shed, not the oldest normal
    assert low.status == 503 and low.body == {"reason": "stale"}
    assert all(j.status is None for j in (n1, n2, n3))


def test_low_score_detections_omitted_unless_verbose():
    b = Box(items=[["person", -1.75, 16, 16, 16, [3, 3, 3, 3]]])
    try:
        b.req("POST", "/v1/cameras/v/config", b'{"cont_thr": 0.1}')
        assert b.frame("v", T0)[1]["detections"] == []
        d = b.frame("v", T0 + 1000, "&verbose=1")[1]["detections"]
        assert len(d) == 1 and d[0]["score"] < 0.2
    finally:
        b.close()


def test_response_in_one_send_and_keepalive(box):
    import socket
    s = socket.create_connection(("127.0.0.1", box.port), timeout=5)
    s.sendall(b"GET /v1/health HTTP/1.1\r\nHost: x\r\n\r\n")
    data = s.recv(65536)
    assert data.startswith(b"HTTP/1.1 200") and b'"uptime_s"' in data
    s.sendall(b"GET /v1/health HTTP/1.1\r\nHost: x\r\n\r\n")
    assert s.recv(65536).startswith(b"HTTP/1.1 200")
    s.close()


def test_reset_clears_vehicles_keeps_config():
    b = Box(items=[["car", 3.0, 16, 8, 8, [3, 2, 3, 2]]])
    try:
        b.req("POST", "/v1/cameras/g/config", b'{"hold_s": 5}')
        assert b.frame("g", T0 - 5000)[1]["vehicle"]["tracks"]
        r = b.frame("g", T0 - 6000)[1]
        assert r["t"] == T0 - 5000                                           # out of order: clamped to the last t
        assert b.req("POST", "/v1/cameras/g/reset")[0] == 200
        assert b.req("GET", "/v1/cameras/g/config")[1]["hold_s"] == 5
        assert b.req("GET", "/v1/cameras/g")[1]["vehicle"]["tracks"] == []
    finally:
        b.close()


def test_poisoned_frame_time_cannot_freeze_session():
    b = Box()
    try:
        r = b.frame("p", NOW + 3_600_000)[1]                                 # one frame stamped 1 h in the future
        assert r["t"] == NOW                                                 # replaced by server time
        for bad in ("inf", "-inf", "nan", "0", "-5", "1e400", str(NOW - 3_600_000)):
            assert b.frame("p", bad)[1]["t"] == NOW, bad
        assert b.req("GET", "/v1/health")[1]["t_clamped"] == 8
        b.now = NOW + 1000; b.frame("p", NOW + 1000); b.now = NOW + 2000
        r = b.frame("p", NOW + 2000)[1]
        assert r["person"]["present"]                                        # real frames still advance the session normally
        time.sleep(2.6)                                                      # let the rate-limit bucket refill
        b.now = NOW + 300_000
        assert b.frame("p", NOW + 300_000)[1]["person"]["state"] == "candidate"   # earlier person long gone
    finally:
        b.close()


def test_camera_cap_returns_429():
    b = Box(opts={"max_cameras": 3})
    try:
        for i in range(3):
            assert b.frame(f"c{i}", T0)[0] == 200
        st, body = b.frame("c3", T0)
        assert st == 429 and "too many" in body["error"]
        assert b.req("POST", "/v1/cameras/c4/config", b"{}")[0] == 429
        assert b.frame("c0", T0 + 1000)[0] == 200
    finally:
        b.close()


def test_jpeg_header_parse_and_reduced_decode():
    import detect as D
    for (w, h) in [(640, 360), (1920, 1080), (1536, 576)]:
        assert D.jpeg_size(jpeg(w, h)) == (w, h)
    assert D.jpeg_size(b"garbage") is None
    img, t = D.decode_jpeg(jpeg(640, 360), 512, "auto", reduce=True)
    assert img.shape[:2] == (360, 640) and t == "1"
    img, t = D.decode_jpeg(jpeg(1920, 1080), 512, "auto", reduce=True)
    assert img.shape[:2] == (540, 960) and t == "1"
    img, t = D.decode_jpeg(jpeg(1920, 1080), 512, "auto")                    # jpeg_reduce defaults to OFF
    assert img.shape[:2] == (1080, 1920)
    img, t = D.decode_jpeg(jpeg(1920, 576), 512, "2", reduce=True)
    assert t == "2" and img.shape[:2] == (288, 960)
    img, t = D.decode_jpeg(jpeg(1536, 576), 512, "auto", reduce=True)
    assert t == "hybrid" and img.shape[:2] == (576, 1536)


def test_token_is_compared_in_constant_time_on_bytes(box, monkeypatch):
    import server as SV
    seen = []
    real = SV.hmac.compare_digest
    monkeypatch.setattr(SV.hmac, "compare_digest", lambda a, b: (seen.append((type(a), type(b))), real(a, b))[1])
    assert box.frame("c", T0)[0] == 200
    assert seen and all(x == (bytes, bytes) for x in seen)


def test_stale_age_starts_at_arrival_not_after_decode(monkeypatch):
    import detect as D
    b = Box(opts={"max_queue_age_ms": 200})
    try:
        real = D.decode_jpeg
        def slow(*a, **k):
            time.sleep(0.35)
            return real(*a, **k)
        monkeypatch.setattr(D, "decode_jpeg", slow)
        st, body = b.frame("c", T0)
        assert st == 503 and body == {"reason": "stale"}                   # 350 ms of decode already used the 200 ms budget
        monkeypatch.undo()
        assert b.frame("c", T0)[0] == 200
    finally:
        b.close()


def test_version_is_read_from_build_arg_or_config_yaml(monkeypatch):
    import server as SV
    here = os.path.dirname(os.path.abspath(__file__))
    want = re.search(r'^version:\s*"?([^"\n]+)', open(os.path.join(here, "..", "config.yaml")).read(), re.M).group(1)
    monkeypatch.delenv("VERIFIER_VERSION", raising=False); monkeypatch.delenv("BUILD_VERSION", raising=False)
    assert SV._read_version() == want                                                                         # config.yaml next to the package / one level up
    monkeypatch.setenv("BUILD_VERSION", "9.8.7")
    assert SV._read_version() == "9.8.7"
    monkeypatch.setenv("VERIFIER_VERSION", "1.2.3")
    assert SV._read_version() == "1.2.3"
    monkeypatch.setenv("BUILD_VERSION", "  ")
    monkeypatch.delenv("VERIFIER_VERSION")
    assert SV._read_version() == want                                                                         # an empty build arg does not mask the file
    docker = open(os.path.join(here, "..", "Dockerfile")).read()
    assert "ARG BUILD_VERSION" in docker and docker.count("COPY config.yaml /app/config.yaml") == 2          # runtime and test stage read the real version
