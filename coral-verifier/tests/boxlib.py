"""Shared test harness: the real HTTP server + the real Coral child-process machinery with the fake TPU (controlled through files in a ctl dir)."""
import http.client, json, os, shutil, tempfile, threading, time
import cv2, numpy as np
import server as SV

BASE = 1_791_886_000_000          # frame times used by tests; the server clock is pinned 30 s later
NOW = BASE + 30_000
TOKEN = "tk-test"


def jpeg(w=640, h=480, q=88):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (np.linspace(0, 255, w, dtype=np.uint8)[None, :, None])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


class Box:
    def __init__(self, opts=None, items=None, stall_s=2.0, recover=10.0, files=None, on_fatal=None, wait="ok", start=True, attempt_timeout=None):
        self.ctl = tempfile.mkdtemp(prefix="ctl_")
        for k, v in (files or {}).items():
            self.touch(k, v)
        if items is not None:
            self.set_items(items)
        o = {"token": TOKEN}
        o.update(opts or {})
        self.token = o["token"]
        self.spec = {"kind": "fake", "size": int(o.get("model", 512)), "ctl": self.ctl}
        self.httpd, self.app = SV.make_server(o, self.spec, host="127.0.0.1", port=0, on_fatal=on_fatal)
        c = self.app.coral
        c.stall_s, c.recover_every_s = stall_s, recover
        if attempt_timeout:
            c.attempt_timeout_s = attempt_timeout
        self.app.now_ms = lambda: self.now
        self.now = NOW
        self.port = self.httpd.server_address[1]
        if start:
            c.start()
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
            if wait:
                self.wait_state(wait)

    # ---- ctl ----
    def touch(self, name, content=""):
        with open(os.path.join(self.ctl, name), "w") as f:
            f.write(content)

    def rm(self, name):
        try:
            os.unlink(os.path.join(self.ctl, name))
        except FileNotFoundError:
            pass

    def set_items(self, items):
        self.touch("items.json", json.dumps(items))

    def calls(self):
        p = os.path.join(self.ctl, "calls")
        return os.path.getsize(p) if os.path.exists(p) else 0

    def loads(self):
        p = os.path.join(self.ctl, "loads")
        return os.path.getsize(p) if os.path.exists(p) else 0

    def trace(self):
        p = os.path.join(self.ctl, "trace")
        return [tuple(map(float, l.split())) for l in open(p)] if os.path.exists(p) else []

    # ---- http ----
    def wait_state(self, st, timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            if self.app.coral.state == st:
                return
            time.sleep(0.02)
        raise AssertionError(f"coral state {self.app.coral.state} != {st} ({self.app.coral.reason})")

    def req(self, method, path, body=None, headers=None, auth=True):
        h = dict(headers or {})
        if auth and "X-Verifier-Token" not in h:
            h["X-Verifier-Token"] = self.token
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        c.request(method, path, body=body, headers=h)
        r = c.getresponse(); data = r.read(); c.close()
        return r.status, json.loads(data)

    def frame(self, cam, t=None, q="", data=None, headers=None, auth=True):
        return self.req("POST", f"/v1/cameras/{cam}/frame?t={t if t is not None else self.now}{q}", data if data is not None else jpeg(),
                        {"Content-Type": "image/jpeg", **(headers or {})}, auth=auth)

    def close(self):
        try:
            self.app.coral.stop()
            self.httpd.shutdown()
            self.httpd.server_close()
        finally:
            shutil.rmtree(self.ctl, ignore_errors=True)


def person_item(size=512, logit=3.0, gx=None, gy=None, d=3):
    return ["person", logit, 16, size // 32 if gx is None else gx, size // 32 if gy is None else gy, [d, d, d, d]]


def burst(b, n, gap, cams=None, q=""):
    out = [None] * n
    def go(i):
        t = time.time(); r = b.frame(cams[i] if cams else f"cam{i}", BASE + i, q)
        out[i] = (r[0], r[1], time.time() - t)
    ths = []
    for i in range(n):
        th = threading.Thread(target=go, args=(i,)); th.start(); ths.append(th); time.sleep(gap)
    [t.join(15) for t in ths]
    return out
