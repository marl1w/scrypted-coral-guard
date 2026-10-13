"""Security / resource-exhaustion tests: decompression bombs, unauthenticated floods, slow clients, option validation, protocol edge cases."""
import json, os, resource, socket, struct, subprocess, sys, threading, time, zlib
import cv2, numpy as np
import pytest
from boxlib import Box, BASE as T0, jpeg
import detect as D
import server as SV


def png_header_bomb(w, h):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"\x00" * 64)) + chunk(b"IEND", b"")


def jpeg_with_dims(w, h):
    b = bytearray(jpeg(64, 64))
    i = b.index(b"\xff\xc0")
    b[i + 5:i + 7] = struct.pack(">H", h); b[i + 7:i + 9] = struct.pack(">H", w)
    return bytes(b)


@pytest.fixture
def box():
    b = Box()
    yield b
    b.close()


def maxrss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def test_png_and_non_jpeg_bodies_rejected_before_any_decode(box, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("cv2.imdecode must not run")
    monkeypatch.setattr(cv2, "imdecode", boom)
    before = maxrss_mb()
    for body in (png_header_bomb(20000, 20000), b"GIF89a" + b"\x00" * 100, b"\x00" * 1000):
        st, r = box.frame("c", data=body)
        assert st == 400 and r["error"] == "body must be a JPEG"            # JPEG only: FFD8 prefix required
    for body in (b"\xff\xd8", b"\xff\xd8\xff\xe0" + b"x" * 5000):
        assert box.frame("c", data=body)[0] == 400
    assert maxrss_mb() - before < 60


def test_jpeg_with_huge_declared_size_rejected_without_allocating(box, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("cv2.imdecode must not run")
    monkeypatch.setattr(cv2, "imdecode", boom)
    before = maxrss_mb()
    for dims in ((30000, 30000), (4097, 4097), (16385, 100), (5000, 5000)):
        st, r = box.frame("c", data=jpeg_with_dims(*dims))
        assert st == 400 and "unsupported image size" in r["error"], dims
    for dims in ((0, 0), (0, 480), (640, 0)):
        assert box.frame("c", data=jpeg_with_dims(*dims))[0] == 400, dims
    assert maxrss_mb() - before < 60


def test_pixel_cap_is_an_option():
    b = Box(opts={"max_image_pixels": 100_000})
    try:
        assert b.frame("c", data=jpeg(640, 480))[0] == 400
        assert b.frame("c", data=jpeg(320, 240))[0] == 200
    finally:
        b.close()


def test_opencv_env_guard_is_set():
    assert int(os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"]) <= 2 ** 24


def test_decode_concurrency_is_capped(monkeypatch):
    b = Box(opts={"decode_concurrency": 2})
    try:
        cur = {"n": 0, "peak": 0}; lock = threading.Lock(); real = D.decode_jpeg
        def slow(*a, **k):
            with lock:
                cur["n"] += 1; cur["peak"] = max(cur["peak"], cur["n"])
            time.sleep(0.2)
            try:
                return real(*a, **k)
            finally:
                with lock:
                    cur["n"] -= 1
        monkeypatch.setattr(D, "decode_jpeg", slow)
        ths = [threading.Thread(target=b.frame, args=(f"c{i}", T0)) for i in range(6)]
        [t.start() for t in ths]; [t.join(15) for t in ths]
        assert cur["peak"] == 2
    finally:
        b.close()


def test_auth_before_body_unauthenticated_client_never_gets_to_upload(box):
    s = socket.create_connection(("127.0.0.1", box.port), timeout=3)
    s.sendall(b"POST /v1/cameras/c/frame HTTP/1.1\r\nHost: x\r\nContent-Length: 1500000\r\nX-Verifier-Token: wrong\r\n\r\n")   # headers only, no body ever sent
    t = time.time()
    data = s.recv(4096)
    assert data.startswith(b"HTTP/1.1 401") and time.time() - t < 1.0 and b"Connection: close" in data
    s.close()


def test_body_size_cap_from_content_length(box):
    s = socket.create_connection(("127.0.0.1", box.port), timeout=3)
    s.sendall(f"POST /v1/cameras/c/frame HTTP/1.1\r\nHost: x\r\nContent-Length: {5 * 1024 * 1024}\r\nX-Verifier-Token: {box.token}\r\n\r\n".encode())
    assert s.recv(4096).startswith(b"HTTP/1.1 413")
    s.close()
    s = socket.create_connection(("127.0.0.1", box.port), timeout=3)
    s.sendall(f"POST /v1/cameras/c/config HTTP/1.1\r\nHost: x\r\nContent-Length: 1000000\r\nX-Verifier-Token: {box.token}\r\n\r\n".encode())
    assert s.recv(4096).startswith(b"HTTP/1.1 413")
    s.close()


def test_chunked_and_missing_length_rejected_411(box):
    for extra in (b"Transfer-Encoding: chunked\r\n", b""):
        s = socket.create_connection(("127.0.0.1", box.port), timeout=3)
        s.sendall(b"POST /v1/cameras/c/frame HTTP/1.1\r\nHost: x\r\nX-Verifier-Token: " + box.token.encode() + b"\r\n" + extra + b"\r\n")
        assert s.recv(4096).startswith(b"HTTP/1.1 411")
        s.close()


def test_slowloris_body_is_cut_by_total_deadline():
    b = Box(opts={"request_timeout_s": 0.6})
    try:
        s = socket.create_connection(("127.0.0.1", b.port), timeout=5)
        s.sendall(f"POST /v1/cameras/c/frame HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\nX-Verifier-Token: {b.token}\r\n\r\n".encode())
        t0 = time.time(); cut = False
        for _ in range(40):                                      # one byte every 0.15 s: each recv is "fast", the total is not
            try:
                s.sendall(b"x"); time.sleep(0.15)
            except OSError:
                cut = True; break
        assert cut and time.time() - t0 < 3.0
        assert b.frame("ok1")[0] == 200                          # the server is fine
    finally:
        b.close()


def test_slowloris_headers_are_cut():
    b = Box(opts={"request_timeout_s": 0.6})
    try:
        s = socket.create_connection(("127.0.0.1", b.port), timeout=5)
        s.sendall(b"GET /v1/health HTTP/1.1\r\n")
        t0 = time.time(); cut = False
        for _ in range(40):
            try:
                s.sendall(b"X-A: b\r\n"); time.sleep(0.15)
            except OSError:
                cut = True; break
        assert cut and time.time() - t0 < 3.0
    finally:
        b.close()


def test_empty_token_refuses_to_start_unless_allowed():
    with pytest.raises(SV.OptionsError) as e:
        SV.parse_options({"token": ""})
    assert "token" in str(e.value)
    with pytest.raises(SV.OptionsError):
        SV.parse_options({})
    assert SV.parse_options({"token": "", "allow_no_token": True})["token"] == ""
    assert SV.parse_options({"token": "x"})["port"] == 8750 and SV.parse_options({"token": "x"})["jpeg_reduce"] is False


def test_defaults_are_the_validated_values():
    import defaults as DF
    st = SV.parse_options({"token": "x"})
    assert st["model"] == 512 and st["port"] == 8750 and st["allow_no_token"] is False and st["jpeg_reduce"] is False
    assert (st["max_queue_age_ms"], st["max_queue_depth"], st["max_cameras"], st["max_image_pixels"]) == (400, 8, 64, 16 * 1024 * 1024)
    assert (st["decode_concurrency"], st["max_connections"], st["self_restart_after"]) == (2, 64, 5)
    d = st["defaults"]
    assert (d.start_thr, d.cont_thr, d.confirm_n, d.confirm_m, d.hold_s, d.still_s, d.still_hold_s, d.max_hold_s, d.parked_after_s, d.session_ttl_s) == \
        (0.65, 0.2, 2, 3, 8, 3, 30, 600, 120, 120)
    assert DF.PUBLIC_OPTIONS == ("token", "port", "log_level") and d.second_look is True


def test_config_yaml_exposes_only_three_options():
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "config.yaml"), "/config.yaml"):
        if os.path.exists(p):
            txt = open(p).read(); break
    else:
        pytest.skip("config.yaml not available")
    def keys(block):
        m = re.search(r"^" + block + r":\n((?:  .*\n)+)", txt, re.M)
        return [re.match(r"  (\w+):", l).group(1) for l in m.group(1).splitlines()]
    assert keys("options") == ["token", "port", "log_level"] and keys("schema") == ["token", "port", "log_level"]
    assert 'token: "password"' in txt and "port: 8750" in txt and "log_level: info" in txt


def test_env_overrides_are_dev_only_and_main_ignores_non_public_options(monkeypatch):
    monkeypatch.setenv("VERIFIER_MAX_CAMERAS", "3"); monkeypatch.setenv("VERIFIER_ALLOW_NO_TOKEN", "1"); monkeypatch.setenv("VERIFIER_HOLD_S", "2")
    st = SV.parse_options({})
    assert st["max_cameras"] == 3 and st["token"] == "" and st["defaults"].hold_s == 2
    assert SV.parse_options({"max_cameras": 5})["max_cameras"] == 5      # explicit dict beats the environment


def test_service_ignores_non_public_options_in_options_json(tmp_path):
    opts = tmp_path / "options.json"; opts.write_text(json.dumps({"token": "", "allow_no_token": True, "port": 18751}))
    r = subprocess.run([sys.executable, SV.__file__], env={**os.environ, "VERIFIER_OPTIONS": str(opts)}, capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "refusing to start" in r.stdout


def test_invalid_options_refuse_with_a_clear_message():
    for bad, frag in (({"cont_thr": 0.9, "start_thr": 0.7}, "cont_thr"), ({"confirm_n": 5, "confirm_m": 3}, "confirm_n"), ({"model": 640}, "model"),
                      ({"port": 0}, "port"), ({"hold_s": "x"}, "hold_s"), ({"max_queue_age_ms": 1}, "max_queue_age_ms")):
        with pytest.raises(SV.OptionsError) as e:
            SV.parse_options({"token": "t", **bad})
        assert frag in str(e.value), bad


def test_service_process_exits_2_with_log_line_when_token_missing(tmp_path):
    opts = tmp_path / "options.json"; opts.write_text(json.dumps({"token": "", "port": 18750}))
    r = subprocess.run([sys.executable, SV.__file__], env={**os.environ, "VERIFIER_OPTIONS": str(opts)},
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 2
    assert "refusing to start" in r.stdout and "token" in r.stdout


def test_idle_sockets_are_cut_and_health_keeps_a_reserved_slot():
    b = Box(opts={"max_connections": 3})
    try:
        held = [socket.create_connection(("127.0.0.1", b.port), timeout=5) for _ in range(3)]
        time.sleep(0.2)
        s = socket.create_connection(("127.0.0.1", b.port), timeout=5)           # beyond the cap: a RESERVED slot
        s.sendall(b"GET /v1/health HTTP/1.1\r\nHost: x\r\n\r\n")
        assert s.recv(65536).startswith(b"HTTP/1.1 200")
        s.sendall(f"GET /v1/cameras HTTP/1.1\r\nHost: x\r\nX-Verifier-Token: {b.token}\r\n\r\n".encode())
        data = s.recv(65536)
        assert data.startswith(b"HTTP/1.1 503") and b"too many clients" in data   # the reserved pool is health-only
        s.close()
        t = time.time()
        assert held[0].recv(10) == b"" and time.time() - t < 4                    # a socket that never sent a byte is cut after ~2 s
        time.sleep(0.3)
        assert b.req("GET", "/v1/cameras")[0] == 200                              # normal slots free again
        [h.close() for h in held]
    finally:
        b.close()


def test_too_many_clients_reply_has_exact_content_length():
    b = Box(opts={"max_connections": 2})
    try:
        held = [socket.create_connection(("127.0.0.1", b.port), timeout=5) for _ in range(2 + SV.RESERVED_HEALTH_SLOTS)]
        time.sleep(0.2)
        s = socket.create_connection(("127.0.0.1", b.port), timeout=5)
        data = b""
        while True:
            c = s.recv(4096)
            if not c:
                break
            data += c
        head, body = data.split(b"\r\n\r\n", 1)
        assert data.startswith(b"HTTP/1.1 503") and int([l for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")][0].split(b":")[1]) == len(body)
        assert json.loads(body) == {"error": "too many clients"}
        [h.close() for h in held]
    finally:
        b.close()


def test_deeply_nested_config_json_is_400_not_500(box):
    for body in (b"[" * 60000, b'{"a":' * 9000 + b"1" + b"}" * 9000):
        st, r = box.req("POST", "/v1/cameras/c/config", body)
        assert st == 400 and "Traceback" not in json.dumps(r), (st, r)
    assert box.frame("c")[0] == 200
