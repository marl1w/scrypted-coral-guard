"""Fast person confirmation (0.3.12, retuned 0.3.14): >= fast_n (2) frames >= start_thr of one IoU track at least 200 ms apart within fast_window_ms (1 s), plus movement (centre shift or size change >= fast_move 0.03) or claim=person."""
import json, threading, time

import session as S
from boxlib import Box, BASE as T0, person_item

JS = {"Content-Type": "application/json"}


def person(score, box):
    return {"cls": "person", "score": score, "box": list(box)}


def walking(i, step=0.03):
    """A person box (height 0.5) walking to the right: the centre moves `step` per frame."""
    x = 0.2 + i * step
    return (x, 0.3, x + 0.15, 0.8)


def new(**kw):
    return S.Session(S.Config().updated({"fast_confirm": True, **kw}))


def run(sess, frames, dt=200, claim=None):
    out = []
    for i, dets in enumerate(frames):
        sess.update(T0 + i * dt, dets, claim=claim)
        out.append(sess.snapshot())
    return out


def test_moving_person_confirms_on_the_second_frame_200ms_later():
    sn = run(new(), [[person(0.7, walking(i, 0.05))] for i in range(3)])          # 0.05 shift over a 0.5 box height = 0.1 >= 0.03
    assert [s["person"]["state"] for s in sn][:2] == ["candidate", "confirmed"]
    assert sn[1]["person"]["present"] and sn[1]["person"]["confirmed_by"] == "fast_move"
    assert sn[0]["person"]["confirmed_by"] is None


def test_two_frames_closer_than_200ms_do_not_qualify():
    sn = run(new(), [[person(0.7, walking(i, 0.05))] for i in range(2)], dt=120)
    assert not any(s["person"]["present"] for s in sn)


def test_walking_towards_the_camera_counts_as_movement():
    grow = lambda i: (0.4 - i * 0.01, 0.3 - i * 0.02, 0.55 + i * 0.01, 0.8 + i * 0.02)   # same centre, box grows
    sn = run(new(), [[person(0.7, grow(i))] for i in range(2)])
    assert sn[1]["person"]["present"] and sn[1]["person"]["confirmed_by"] == "fast_move"


def test_static_high_score_box_does_not_fast_confirm_and_falls_back_to_ticks():
    box = (0.4, 0.3, 0.55, 0.8)
    sess = new()
    sn = run(sess, [[person(0.87, box)]] * 4)                                      # 4 frames in 600 ms: one tick, no movement
    assert not any(s["person"]["present"] for s in sn)
    sess.update(T0 + 1000, [person(0.87, box)])                                    # the second tick confirms by the ordinary rule
    s = sess.snapshot()["person"]
    assert s["present"] and s["confirmed_by"] == "ticks"


def test_claim_person_confirms_a_static_box():
    box = (0.4, 0.3, 0.55, 0.8)
    sn = run(new(), [[person(0.7, box)]] * 3, claim="person")
    assert sn[2]["person"]["present"] and sn[2]["person"]["confirmed_by"] == "fast_claim"
    sn = run(new(), [[person(0.7, box)]] * 3, claim="vehicle")                     # only a person claim counts
    assert not sn[2]["person"]["present"]


def test_frames_spread_beyond_the_window_do_not_qualify():
    sn = run(new(), [[person(0.7, walking(i, 0.05))] for i in range(2)], dt=1100)  # 2 frames 1.1 s apart > 1 s window
    assert not sn[1]["person"]["present"] or sn[1]["person"]["confirmed_by"] == "ticks"


def test_scores_below_start_thr_do_not_count():
    sn = run(new(), [[person(0.6, walking(i, 0.08))] for i in range(4)])
    assert not any(s["person"]["present"] for s in sn)


def test_fast_disabled_keeps_todays_behaviour():
    frames = [[person(0.7, walking(i, 0.05))] for i in range(3)]
    off = S.Session(S.Config())                                                    # default: fast_confirm off
    sn = run(off, frames, claim="person")
    assert not any(s["person"]["present"] for s in sn)
    ref = S.Session(S.Config())
    seq = [[person(0.9, walking(0))], [], [person(0.8, walking(1))], [person(0.3, walking(1))], []]
    a = [(ref.update(T0 + i * 1000, d), ref.snapshot()["person"]["state"])[1] for i, d in enumerate(seq)]
    on = new()
    b = [(on.update(T0 + i * 1000, d), on.snapshot()["person"]["state"])[1] for i, d in enumerate(seq)]
    assert a == b == ["candidate", "candidate", "confirmed", "confirmed", "holding"]


def test_fast_confirmation_survives_a_later_frame_of_the_same_tick_and_holds():
    sess = new()
    run(sess, [[person(0.7, walking(i, 0.05))] for i in range(3)])
    sess.update(T0 + 600, [])                                                      # same tick (< 750 ms): the tick is re-stepped
    assert sess.snapshot()["person"]["present"]
    sess.update(T0 + 1600, [person(0.3, walking(4, 0.05))])                        # next tick: kept alive at cont_thr
    s = sess.snapshot()["person"]
    assert s["state"] == "confirmed" and s["confirmed_by"] == "fast_move"
    sess.update(T0 + 12_000, [])
    assert sess.snapshot()["person"]["state"] == "idle" and sess.snapshot()["person"]["confirmed_by"] is None


def test_config_validates_fast_keys():
    for bad in ({"fast_n": 1}, {"fast_window_ms": 50}, {"fast_move": -1}, {"fast_confirm": 1}):
        try:
            S.Config().updated(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    c = S.Config().updated({"fast_n": 4, "fast_window_ms": 900, "fast_move": 0.2, "fast_confirm": True})
    assert (c.fast_n, c.fast_window_ms, c.fast_move, c.fast_confirm) == (4, 900.0, 0.2, True)


def test_server_claim_param_and_config():
    b = Box(items=[person_item(logit=1.4)])
    try:
        assert b.req("POST", "/v1/cameras/cam/config", json.dumps({"fast_confirm": True}).encode(), JS)[0] == 200
        st, r = b.frame("cam", T0, "&claim=nobody")
        assert st == 400
        replies = [b.frame("cam", T0 + i * 200, "&claim=person") for i in range(3)]
        assert all(x[0] == 200 for x in replies)
        assert replies[-1][1]["person"]["present"] and replies[-1][1]["person"]["confirmed_by"] == "fast_claim"
    finally:
        b.close()


def test_many_cameras_bursting_at_once_degrade_gracefully_and_none_starves():
    """Several cameras posting back-to-back (one request in flight each, deadline 300 ms) through one slow Coral: each gets answers, nothing but
    200 / stale 503 / 429 comes back, and the share per camera is roughly even."""
    b = Box(items=[person_item(logit=1.4)])
    b.touch("delay", "0.04")
    cams = [f"c{i}" for i in range(6)]
    res = {c: [] for c in cams}
    stop = time.monotonic() + 1.5

    def loop(cam):
        k = 0
        while time.monotonic() < stop:
            st, _ = b.frame(cam, T0 + k * 150, "&deadline_ms=300")
            res[cam].append(st); k += 1
    try:
        ts = [threading.Thread(target=loop, args=(c,)) for c in cams]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        ok = {c: sum(1 for s in v if s == 200) for c, v in res.items()}
        assert all(set(v) <= {200, 503, 429} for v in res.values()), res
        assert min(ok.values()) >= 2, ok
        assert max(ok.values()) <= 3 * min(ok.values()) + 2, ok
    finally:
        b.close()


def test_camera_ai_person_plus_one_strong_frame_confirms_at_once():
    box = (0.4, 0.3, 0.55, 0.8)
    sn = run(new(), [[person(0.78, box)]], claim="person")
    assert sn[0]["person"]["present"] and sn[0]["person"]["confirmed_by"] == "fast_claim"
    sn = run(new(), [[person(0.70, box)]], claim="person")                           # below 0.75: needs a second frame
    assert not sn[0]["person"]["present"]
    sn = run(new(), [[person(0.90, box)]])                                           # no camera claim: one frame never confirms
    assert not sn[0]["person"]["present"]


def test_rule_b_is_reported_but_never_confirms():
    box = (0.4, 0.3, 0.55, 0.8)
    sess = new()
    sess.update(T0, [person(0.5, box)], claim="person")
    p = sess.snapshot()["person"]
    assert p["would_b_ms"] == T0 and not p["present"]                              # logged only
    sess2 = new()
    sess2.update(T0, [person(0.5, box)])                                            # no camera claim: no B
    assert sess2.snapshot()["person"]["would_b_ms"] is None
    sess3 = new()
    sess3.update(T0, [person(0.4, box)], claim="person")                             # below 0.45
    assert sess3.snapshot()["person"]["would_b_ms"] is None
