"""model_url / model_sha256 / model_token: a model from a URL replaces the built-in one only when everything checks out; otherwise the built-in
model runs and the reason is reported. No network: the download is a test fetcher."""
import hashlib, os, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import model_source as MS

BUILTIN = {"kind": "edgetpu", "path": "/models/builtin.tflite", "size": 512}
MODEL = b"\x1c\x00\x00\x00TFL3" + b"\x00" * 100
SHA = hashlib.sha256(MODEL).hexdigest()
URL = "https://api.github.com/repos/someone/models/contents/m/model_edgetpu.tflite?ref=abc123"


def fetcher(data=MODEL, calls=None):
    def f(url, token, dst):
        if calls is not None:
            calls.append((url, token))
        with open(dst, "wb") as fh:
            fh.write(data)
    return f


def test_empty_url_is_the_builtin_model_and_downloads_nothing(tmp_path):
    calls = []
    spec, origin = MS.resolve({}, BUILTIN, store=str(tmp_path), fetcher=fetcher(calls=calls))
    assert spec == BUILTIN and origin == {"source": "built-in"} and calls == []


def test_good_model_is_stored_and_used(tmp_path):
    calls = []
    spec, origin = MS.resolve({"model_url": URL, "model_sha256": SHA.upper(), "model_token": "t0k"}, BUILTIN, str(tmp_path), fetcher(calls=calls))
    assert spec["path"] == str(tmp_path / f"{SHA}.tflite") and spec["size"] == 512 and spec["kind"] == "edgetpu"
    assert origin == {"source": "url", "file": "model_edgetpu.tflite", "sha256": SHA}
    assert calls == [(URL, "t0k")]
    assert open(spec["path"], "rb").read() == MODEL


def test_stored_model_is_reused_without_the_network(tmp_path):
    MS.resolve({"model_url": URL, "model_sha256": SHA}, BUILTIN, str(tmp_path), fetcher())
    def offline(*a):
        raise OSError("no network")
    spec, origin = MS.resolve({"model_url": URL, "model_sha256": SHA}, BUILTIN, str(tmp_path), offline)
    assert origin["source"] == "url" and spec["path"].endswith(f"{SHA}.tflite")


@pytest.mark.parametrize("opts,why", [
    ({"model_url": URL, "model_sha256": "0" * 64}, "sha256"),
    ({"model_url": URL, "model_sha256": "short"}, "64 hex"),
    ({"model_url": URL}, "64 hex"),
    ({"model_url": "http://example.com/m.tflite", "model_sha256": SHA}, "https"),
])
def test_bad_options_or_hash_fall_back_to_builtin(tmp_path, opts, why):
    spec, origin = MS.resolve(opts, BUILTIN, str(tmp_path), fetcher())
    assert spec == BUILTIN and origin["source"] == "built-in" and why in origin["fallback"]
    assert not [f for f in os.listdir(tmp_path) if f.endswith((".tflite", ".part"))]       # nothing half-written is left


def test_not_a_tflite_file_falls_back(tmp_path):
    data = b"<html>not found</html>"
    spec, origin = MS.resolve({"model_url": URL, "model_sha256": hashlib.sha256(data).hexdigest()}, BUILTIN, str(tmp_path), fetcher(data))
    assert spec == BUILTIN and "not a TFLite" in origin["fallback"]


def test_download_error_falls_back_and_never_shows_the_token(tmp_path, caplog):
    def broken(url, token, dst):
        raise OSError("HTTP Error 404: Not Found")
    spec, origin = MS.resolve({"model_url": URL, "model_sha256": SHA, "model_token": "s3cret-token"}, BUILTIN, str(tmp_path), broken)
    assert spec == BUILTIN and "404" in origin["fallback"]
    assert "s3cret-token" not in caplog.text and "s3cret-token" not in str(origin)


def test_a_model_that_failed_on_the_coral_is_not_used_again(tmp_path):
    opts = {"model_url": URL, "model_sha256": SHA}
    _, origin = MS.resolve(opts, BUILTIN, str(tmp_path), fetcher())
    MS.mark_failed(origin, store=str(tmp_path))
    spec, origin2 = MS.resolve(opts, BUILTIN, str(tmp_path), fetcher())
    assert spec == BUILTIN and "made the Coral fail" in origin2["fallback"]
    MS.mark_failed(origin2, store=str(tmp_path))                     # built-in: nothing to mark
    os.remove(MS.failed_marker(str(tmp_path), SHA))
    assert MS.resolve(opts, BUILTIN, str(tmp_path), fetcher())[1]["source"] == "url"


def test_a_new_model_replaces_the_old_stored_one(tmp_path):
    MS.resolve({"model_url": URL, "model_sha256": SHA}, BUILTIN, str(tmp_path), fetcher())
    other = MODEL + b"v2"; sha2 = hashlib.sha256(other).hexdigest()
    spec, _ = MS.resolve({"model_url": URL, "model_sha256": sha2}, BUILTIN, str(tmp_path), fetcher(other))
    assert sorted(f for f in os.listdir(tmp_path) if f.endswith(".tflite")) == [f"{sha2}.tflite"]


def test_fetch_sends_token_and_raw_accept_and_caps_size(tmp_path):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.update(auth=self.headers.get("Authorization"), accept=self.headers.get("Accept"))
            body = b"x" * 3000
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/m.tflite"
        MS.fetch(url, "tok", str(tmp_path / "a"))
        assert seen["auth"] == "Bearer tok" and (tmp_path / "a").stat().st_size == 3000
        assert seen["accept"] != "application/vnd.github.raw"          # only for api.github.com
        with pytest.raises(ValueError, match="larger"):
            MS.fetch(url, "", str(tmp_path / "b"), max_bytes=1000)
    finally:
        srv.shutdown()
