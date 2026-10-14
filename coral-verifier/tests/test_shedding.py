"""Load shedding: a low-priority frame that finds other frames waiting runs its base pass only (no zoom-out, no second look)."""
import json, threading, time
import pytest
from boxlib import Box, BASE as T0, person_item

JS = {"Content-Type": "application/json"}


def burst(box, plan):
    """plan: [(delay_before_s, camera, t, query)]; sends them from threads in order, returns the replies in plan order."""
    out = [None] * len(plan)

    def go(i, cam, t, q):
        out[i] = box.frame(cam, t, q)
    ts = []
    for i, (wait, cam, t, q) in enumerate(plan):
        time.sleep(wait)
        th = threading.Thread(target=go, args=(i, cam, t, q)); th.start(); ts.append(th)
    for th in ts:
        th.join()
    return out


def make(zoom=True, items=None, opts=None):
    b = Box(opts={"second_look": True, **(opts or {})}, items=items if items is not None else [person_item(logit=1.4)])      # default: a person at p ~0.8
    b.touch("delay", "0.08")                                                      # a slow Coral, so frames queue behind each other
    if zoom:
        assert b.req("POST", "/v1/cameras/cam/config", json.dumps({"zoom_out": 0.3}).encode(), JS)[0] == 200
    return b


def load(b):
    return b.req("GET", "/v1/metrics")[1]["load"]


def test_busy_low_priority_in_an_empty_scene_runs_base_pass_only_and_the_last_one_runs_full():
    b = make(items=[])
    try:
        n0 = b.calls()
        r = burst(b, [(0, "slow", T0, ""), (0.02, "cam", T0, "&prio=low"), (0.01, "cam", T0 + 1000, "&prio=low"), (0.01, "cam", T0 + 2000, "&prio=low")])
        assert [x[0] for x in r] == [200, 200, 200, 200]
        a, l1, l2, l3 = (x[1] for x in r)
        assert a["reduced"] is False and l1["reduced"] is True and l2["reduced"] is True and l3["reduced"] is False        # the last one finds an empty queue
        assert b.calls() - n0 == 1 + 1 + 1 + 2                                        # slow 1, two reduced frames: base only, the idle one: base + zoom-out
        m = load(b)
        assert m["reduced_frames_60s"] == 2 and m["shed_passes_60s"] >= 2 and m["promoted_frames_60s"] == 0
    finally:
        b.close()


def test_busy_low_priority_frame_that_would_start_a_person_is_promoted_to_full():
    b = make()                                                                        # the scene has a person at ~0.8: it would start / confirm a track
    try:
        n0 = b.calls()
        r = burst(b, [(0, "slow", T0, ""), (0.02, "cam", T0, "&prio=low"), (0.01, "cam", T0 + 1000, "&prio=low")])
        l1 = r[1][1]
        assert r[1][0] == 200 and l1["reduced"] is False and l1["detections"][0]["second_look"] in ("confirmed", "lowered") and l1["second_look_ms"] > 0
        assert b.calls() - n0 >= 1 + 3 + 1                                            # the zoom-out and the second look ran after all
        m = load(b)
        assert m["promoted_frames_60s"] >= 1 and m["reduced_frames_60s"] == 0
    finally:
        b.close()


def test_idle_low_priority_frame_gets_the_full_treatment():
    b = make()
    try:
        n0 = b.calls()
        st, r = b.frame("cam", T0, "&prio=low")
        assert st == 200 and r["reduced"] is False and b.calls() - n0 >= 3
        assert r["detections"][0]["second_look"] in ("confirmed", "lowered")
        m = load(b)
        assert m["reduced_frames_60s"] == 0 and m["promoted_frames_60s"] == 0
    finally:
        b.close()


def test_normal_priority_frames_always_run_everything_even_when_busy():
    b = make()
    try:
        n0 = b.calls()
        r = burst(b, [(0, "slow", T0, ""), (0.02, "cam", T0, "")])                   # the second frame waits behind the first one
        assert [x[0] for x in r] == [200, 200] and all(x[1]["reduced"] is False for x in r)
        assert b.calls() - n0 >= 1 + 3
        assert load(b)["reduced_frames_60s"] == 0
    finally:
        b.close()


def test_reduced_frame_leaves_a_busy_low_person_unconfirmed_until_a_full_frame_confirms():
    import secondlook as SL, session as S, detect as D
    from test_engine import SeqModel
    import numpy as np
    cfg = S.Config()
    img = np.full((480, 640, 3), 90, np.uint8)
    d = {"cls": "person", "group": "person", "score": 0.8, "box": [0.4, 0.4, 0.6, 0.6]}
    out, info = SL.second_look(D.Engine(SeqModel(512, [[]] * 4)), img, [d], cfg, {}, [], 1.0, reduced=True)
    assert out[0]["second_look"] == "skipped" and out[0]["second_look_doubt"] is True and info["n"] == 0 and info["counts"] == {"shed": 1}
    sess = S.Session(cfg)
    for i in range(4):
        sess.update(1000 + i * 1000, [SL.tracker_view(out[0], cfg)])
    assert not sess.snapshot()["person"]["present"] and sess.snapshot()["person"]["tracks"] == 0                 # cannot start or confirm a track
    for i in range(4, 6):
        sess.update(1000 + i * 1000, [{"cls": "person", "score": 0.9, "box": [0.4, 0.4, 0.6, 0.6]}])
    assert sess.snapshot()["person"]["present"]
    for i in range(6, 9):
        sess.update(1000 + i * 1000, [SL.tracker_view(out[0], cfg)])
    assert sess.snapshot()["person"]["present"]                                                                  # but it may continue a confirmed one
    out2, info2 = SL.second_look(D.Engine(SeqModel(512, [[]] * 4)), img, [d], cfg, {"person_conf": [tuple(d["box"])]}, [], 1.0, reduced=True)
    assert "second_look" not in out2[0] and info2["n"] == 0


def test_promotion_reasons_each_case_and_the_quiet_case():
    import secondlook as SL, session as S
    cfg = S.Config()
    car = {"cls": "car", "group": "vehicle", "score": 0.7, "box": [0.1, 0.5, 0.4, 0.8]}
    dog = {"cls": "dog", "group": "animal", "score": 0.7, "box": [0.5, 0.5, 0.6, 0.8]}
    weak_person = {"cls": "person", "group": "person", "score": 0.4, "box": [0.4, 0.3, 0.5, 0.8]}
    quiet = {"person_state": "idle", "animal_state": "idle", "person_hold": None, "animal_hold": None, "vehicle_pending": False}
    t = 100_000
    assert SL.promote_reason([], cfg, quiet, t, {}) is None and SL.promote_reason([weak_person], cfg, quiet, t, {}) is None             # empty scene / a weak detection stay reduced
    assert SL.promote_reason([], cfg, {**quiet, "person_state": "candidate"}, t, {}) == "candidate"                                       # (1) a session that is still a candidate
    assert SL.promote_reason([], cfg, {**quiet, "animal_state": "candidate"}, t, {}) == "candidate"
    assert SL.promote_reason([{**weak_person, "score": 0.9}], cfg, quiet, t, {}) == "would_start"                                          # (1) base detection >= start_thr that would start
    assert SL.promote_reason([dog], cfg, quiet, t, {}) == "would_start"
    conf = {"person_conf": [(0.4, 0.3, 0.5, 0.8)]}
    assert SL.promote_reason([{**weak_person, "score": 0.9}], cfg, quiet, t, conf) is None                                                 # it only continues a confirmed track
    assert SL.promote_reason([], cfg, {**quiet, "person_hold": t + 1500}, t, {}) == "release"                                              # (2) hold ends within 2 s
    assert SL.promote_reason([], cfg, {**quiet, "animal_hold": t + 2000}, t, {}) == "release"
    assert SL.promote_reason([], cfg, {**quiet, "person_hold": t + 2001}, t, {}) is None                                                   # still far from release
    assert SL.promote_reason([], cfg, {**quiet, "vehicle_pending": True}, t, {}) == "vehicle"                                              # (3) classification pending
    assert SL.promote_reason([car], cfg, quiet, t, {}) == "vehicle"                                                                        # a strong vehicle with no track yet
    assert SL.promote_reason([car], cfg, quiet, t, {"vehicle": [(0.1, 0.5, 0.4, 0.8)]}) is None                                            # an existing, settled track


def test_session_promotion_state_reflects_candidate_hold_and_pending_vehicle():
    import session as S
    s = S.Session(S.Config())
    assert s.promotion_state(1.0) == {}
    s.update(T0, [{"cls": "person", "score": 0.9, "box": [0.4, 0.3, 0.5, 0.8]}, {"cls": "car", "score": 0.9, "box": [0.1, 0.5, 0.4, 0.8]}], 1.0)
    p = s.promotion_state(1.0)
    assert p["person_state"] == "candidate" and p["vehicle_pending"] is True and p["person_hold"] is None
    s.update(T0 + 1000, [{"cls": "person", "score": 0.9, "box": [0.4, 0.3, 0.5, 0.8]}], 2.0)
    p = s.promotion_state(2.0)
    assert p["person_state"] == "confirmed" and p["person_hold"] == T0 + 1000 + 8000 and p["animal_state"] == "idle"
    s.update(T0 + 6000, [{"cls": "person", "score": 0.9, "box": [0.4, 0.3, 0.5, 0.8]}, {"cls": "car", "score": 0.9, "box": [0.1, 0.5, 0.4, 0.8]}], 7.0)
    assert s.promotion_state(7.0)["vehicle_pending"] is False                                                   # the car track is older than 3 s: settled
