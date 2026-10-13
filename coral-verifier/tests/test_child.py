"""The Coral worker is a CHILD PROCESS: hang -> SIGKILL -> respawn, never two interpreters, reasons logged, failures counted, health 503."""
import logging, os, signal, threading, time
import pytest
from boxlib import Box, BASE as T0, jpeg


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return os.waitpid(pid, os.WNOHANG) == (0, 0)          # zombie children of ours count as dead once reaped
    except ChildProcessError:
        return True


def dead(pid, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_hung_invoke_kills_child_respawns_and_never_overlaps(caplog):
    b = Box(stall_s=0.5, recover=0.5)
    try:
        assert b.frame("c", T0)[0] == 200
        old = b.app.coral.child_pids[-1]
        b.touch("hang")
        out = {}
        def slow(n):
            t = time.time(); out[n] = (b.frame(n, T0 + 1000)[:2], time.time() - t)
        th = [threading.Thread(target=slow, args=(n,)) for n in ("a", "b")]
        th[0].start(); time.sleep(0.15); th[1].start()
        [t.join(10) for t in th]
        assert out["a"][0][0] == 503 and out["a"][0][1] == {"coral": "stalled"} and out["a"][1] < 1.5
        assert out["b"][0][0] == 503 and out["b"][1] < 1.5            # queued behind the hung invoke: failed at once
        assert dead(old), "the hung child must be SIGKILLed"
        assert b.app.coral.state in ("stalled", "missing", "ok")
        st, h = b.req("GET", "/v1/health", auth=False)
        if b.app.coral.state != "ok":
            assert st == 503 and h["ok"] is False and h["model"]["coral"] in ("stalled", "missing")      # same JSON, HTTP 503
        b.rm("hang")
        b.wait_state("ok", 10)
        new = b.app.coral.child_pids[-1]
        assert new != old and alive(new) and not alive(old)
        assert b.frame("c", T0 + 2000)[0] == 200
        assert b.req("GET", "/v1/health", auth=False)[1]["resets"] == 1
        # peak concurrency 1 by construction: invokes of ALL children are strictly sequential
        tr = sorted(b.trace(), key=lambda x: x[1])
        assert all(y[1] >= x[2] - 1e-3 for x, y in zip(tr, tr[1:]))
        assert any("stalled" in r.message and "invoke took" in r.message for r in caplog.records if r.levelno == logging.ERROR)
    finally:
        b.close()


def test_stop_kills_the_child_first_even_while_it_is_hung():
    b = Box(stall_s=30.0)
    try:
        b.frame("c", T0)
        pid = b.app.coral.child_pids[-1]
        assert alive(pid)
        b.touch("hang")
        out = []
        th = threading.Thread(target=lambda: out.append(b.frame("c2", T0)[0])); th.start()
        time.sleep(0.6)                                                  # the child is now stuck inside a (fake) invoke, the parent waits for it
        t = time.time()
        b.app.coral.stop()                                               # docker stop / SIGTERM path: the child must be killed first
        assert dead(pid, 3) and time.time() - t < 3
        th.join(10)
        assert out == [503]
    finally:
        b.close()


def test_child_dies_with_its_parent_hung_state():
    b = Box(stall_s=0.5, recover=60.0)
    try:
        b.touch("hang"); b.frame("c", T0)
        assert dead(b.app.coral.child_pids[-1])
    finally:
        b.close()


def test_invoke_error_marks_stalled_and_recovers():
    b = Box(recover=0.4)
    try:
        b.touch("fail")
        st, body = b.frame("c", T0)
        assert st == 503 and body == {"coral": "stalled"}
        assert b.req("GET", "/v1/health", auth=False)[1]["resets"] == 1
        b.rm("fail")
        b.wait_state("ok", 10)
        assert b.frame("c", T0 + 1000)[0] == 200
    finally:
        b.close()


def test_missing_coral_503_logs_reason_at_error_and_retries(caplog):
    b = Box(files={"load_fail": "cannot load Edge TPU delegate (no Edge TPU found): Failed to load delegate"}, recover=0.3, wait="missing")
    try:
        time.sleep(0.5)
        st, body = b.frame("c", T0)
        assert st == 503 and body == {"coral": "missing"}
        st, h = b.req("GET", "/v1/health", auth=False)
        assert st == 503 and h["ok"] is False and h["model"]["coral"] == "missing"
        errs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("missing" in m and "cannot load Edge TPU delegate" in m and "no Edge TPU found" in m for m in errs), errs      # reason visible at boot
        b.rm("load_fail")
        b.wait_state("ok", 10)
        assert b.frame("c", T0)[0] == 200 and b.req("GET", "/v1/health", auth=False)[0] == 200
    finally:
        b.close()


def test_recovery_attempts_are_spaced():
    b = Box(files={"load_fail": "x"}, recover=1.5, wait="missing")
    try:
        time.sleep(3.3)
        assert 2 <= b.loads() <= 4, b.loads()                              # one attempt per recover_every_s, not back to back
    finally:
        b.close()


def test_self_restart_after_repeated_load_failures():
    fired = threading.Event()
    b = Box(opts={"self_restart_after": 3}, files={"load_fail": "gone"}, recover=0.2, on_fatal=fired.set, wait="missing")
    try:
        assert fired.wait(10)
        assert b.app.coral.state == "missing"
    finally:
        b.close()


def test_load_ok_but_first_invoke_fails_counts_as_failed_recovery():
    fired = threading.Event()
    b = Box(opts={"self_restart_after": 3}, files={"first_fail": "1"}, recover=0.2, on_fatal=fired.set, wait=None)
    try:
        assert fired.wait(15), "a load that succeeds but whose first invoke fails must still add up to a restart"
        assert b.app.coral.state != "ok"
    finally:
        b.close()


def test_flapping_stall_soon_after_recovery_counts_and_triggers_restart():
    fired = threading.Event()
    b = Box(opts={"self_restart_after": 2}, stall_s=0.4, recover=0.2, on_fatal=fired.set)
    try:
        for i in range(4):
            if fired.is_set():
                break
            b.wait_state("ok", 10)
            b.touch("fail"); b.frame("f", T0 + i); b.rm("fail")
        assert fired.wait(10)
    finally:
        b.close()


def test_load_hang_is_killed_after_attempt_timeout():
    b = Box(files={"load_hang": "1"}, recover=0.2, attempt_timeout=0.8, wait=None, opts={"self_restart_after": 0})
    try:
        time.sleep(2.0)
        pids = b.app.coral.child_pids
        assert b.app.coral.state != "ok"
        assert b.loads() >= 1
    finally:
        b.close()


def test_child_frozen_between_requests_is_detected_killed_and_respawned():
    b = Box(stall_s=0.5, recover=0.5)
    try:
        assert b.frame("c", T0)[0] == 200
        pid = b.app.coral.child_pids[-1]
        os.kill(pid, signal.SIGSTOP)                                    # frozen child: never reads the pipe, the 786 KB send blocks
        t = time.time()
        st, body = b.frame("c2", T0)
        assert st == 503 and body == {"coral": "stalled"} and time.time() - t < 4
        assert dead(pid, 5)
        b.wait_state("ok", 10)
        assert b.app.coral.child_pids[-1] != pid and b.frame("c", T0 + 1000)[0] == 200
        assert b.req("GET", "/v1/health", auth=False)[1]["resets"] == 1
    finally:
        b.close()
