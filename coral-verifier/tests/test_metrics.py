"""GET /v1/metrics: shape, auth, rolling counters, sampler on fake /proc and sysfs (no Linux needed), graceful degradation."""
import os
import pytest
import metrics as M
from boxlib import Box, BASE as T0


@pytest.fixture
def box():
    b = Box()
    yield b
    b.close()


def w(path, text):
    with open(path, "w") as f:
        f.write(text)
    return str(path)


def test_rolling_window_and_ring_reuse():
    now = [1000.0]
    r = M.Rolling(("a",), clock=lambda: now[0])
    r.add("a", 2); now[0] += 1; r.add("a")
    assert r.sum("a") == 3
    now[0] += 58
    assert r.sum("a") == 3                       # both inside 60 s
    now[0] += 1
    assert r.sum("a") == 1                       # the first second fell out
    now[0] += 100; r.add("a")                    # the slot is reused, old content wiped
    assert r.sum("a") == 1


def test_sampler_fake_proc_and_gpu(tmp_path, monkeypatch):
    stat = tmp_path / "stat"; rc6 = tmp_path / "gt0"; own = tmp_path / "self"
    monkeypatch.setattr(M, "PROC_STAT", str(stat)); monkeypatch.setattr(M, "DRM_RC6_GLOB", str(tmp_path / "gt*"))
    monkeypatch.setattr(M, "PROC_SELF_STAT", str(own)); monkeypatch.setattr(M, "CLK_TCK", 100)
    clock = [100.0]
    s = M.Sampler(clock=lambda: clock[0])
    w(stat, "cpu 100 0 100 800 0 0 0 0\n"); w(rc6, "1000\n"); w(own, "1 (x y) S " + " ".join(["0"] * 10) + " 100 50 " + "0 " * 10)
    s.sample()
    assert s.values()["host_cpu"] is None and s.values()["gpu_ok"] is True
    clock[0] += 5.0
    w(stat, "cpu 150 0 150 900 0 0 0 0\n")       # +100 busy of +200 total
    w(rc6, "4750\n")                              # rc6 +3750 ms of 5000 -> 75% idle -> 25% busy
    w(own, "1 (x y) S " + " ".join(["0"] * 10) + " 200 100 " + "0 " * 10)   # +150 jiffies = 1.5 s over 5 s
    s.sample()
    v = s.values()
    assert v["host_cpu"] == 50.0 and v["gpu_busy"] == 25.0 and v["gpu_ok"] is True
    assert 30.0 <= v["proc_cpu"] < 32.0          # 1.5 s of CPU over 5 s; os.times children may add a hair


def test_everything_unreadable_degrades(tmp_path, monkeypatch):
    for k in ("PROC_STAT", "PROC_LOADAVG", "PROC_MEMINFO", "PROC_SELF_STAT", "PROC_SELF_STATM"):
        monkeypatch.setattr(M, k, str(tmp_path / "nope"))
    monkeypatch.setattr(M, "DRM_RC6_GLOB", str(tmp_path / "nothing*"))
    s = M.Sampler(); s.sample(); s.sample()
    v = s.values()
    assert v["host_cpu"] is None and v["gpu_busy"] is None and v["gpu_ok"] is False
    assert M.host_instant() == (None, None) and M.rss_mb() is None


def test_host_instant_fake(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "PROC_LOADAVG", w(tmp_path / "l", "1.25 0.5 0.1 1/100 5\n"))
    monkeypatch.setattr(M, "PROC_MEMINFO", w(tmp_path / "m", "MemTotal: 1000 kB\nMemFree: 10 kB\nMemAvailable: 250 kB\n"))
    monkeypatch.setattr(M, "PROC_SELF_STATM", w(tmp_path / "sm", "100 256 0 0 0 0 0\n"))
    assert M.host_instant() == (1.25, 75.0)
    assert M.rss_mb() == round(256 * os.sysconf("SC_PAGE_SIZE") / 1048576.0, 1)


def test_metrics_requires_token_and_shape(box):
    assert box.req("GET", "/v1/metrics", auth=False)[0] == 401
    st, m = box.req("GET", "/v1/metrics")
    assert st == 200 and set(m) == {"coral", "stages", "load", "host", "process", "gpu", "uptime_s"}
    assert set(m["coral"]) == {"state", "invoke_ms", "utilization_pct", "invocations_per_s", "resets"} and m["coral"]["state"] == "ok"
    assert set(m["load"]) == {"queue_depth", "dropped_stale_total", "dropped_stale_60s", "rejected_429_60s", "cameras_active", "frames_60s"}
    assert set(m["host"]) == {"cpu_pct", "load1", "mem_used_pct"} and set(m["process"]) == {"cpu_pct", "rss_mb"} and set(m["gpu"]) == {"available", "busy_pct"}
    assert m["coral"]["utilization_pct"] == 0 and m["load"]["frames_60s"] == 0
    h = box.req("GET", "/v1/health", auth=False)[1]
    assert {"ok", "version", "model", "coral_ms", "stages", "dropped_stale", "t_clamped", "resets", "queue_depth", "cameras_active", "uptime_s"} <= set(h)


def test_metrics_counts_frames_and_429(box):
    for i in range(3):
        assert box.frame("c1", T0 + i * 1000)[0] == 200
    got429 = sum(box.frame("c2", T0 + i)[0] == 429 for i in range(30))
    m = box.req("GET", "/v1/metrics")[1]
    assert m["load"]["frames_60s"] >= 3 and m["load"]["rejected_429_60s"] == got429 > 0
    assert m["load"]["cameras_active"] == 2 and m["coral"]["invocations_per_s"] > 0 and m["coral"]["utilization_pct"] > 0
    assert m["stages"]["infer_ms"]["median"] is not None and m["coral"]["invoke_ms"]["last"] > 0


def test_metrics_post_not_allowed(box):
    assert box.req("POST", "/v1/metrics")[0] == 405
