"""0.3.65: still objects (a bin, an appliance) present before anything moved never become a person; a weak first score is never raised."""
import numpy as np
import detect as D
import secondlook as SL
import session as S
from test_engine import SeqModel

CFG = S.Config()
T0 = 1_000_000
BIN = (0.26, 0.47, 0.34, 0.67)                      # a black bin by a door (night false alarms, 24 Nov)
IMG = np.full((480, 640, 3), 90, np.uint8)


def p(score, box):
    return {"cls": "person", "score": score, "box": list(box)}


def run(frames, cfg=CFG, step_ms=500):
    """frames: lists of detections, one list per frame (0.5 s apart). -> (session, the detections as the session left them)."""
    s = S.Session(cfg)
    seen = []
    for k, dets in enumerate(frames):
        dets = [dict(d) for d in dets]
        s.update(T0 + k * step_ms, dets)
        seen.append(dets)
    return s, seen


def shifted(box, dx):
    return (box[0] + dx, box[1], box[2] + dx, box[3])


def test_bin_there_before_anything_moved_never_confirms_even_at_high_scores():
    s, seen = run([[p(0.88, BIN)]] * 12)
    assert not s.snapshot()["person"]["present"]
    assert all(d.get("frozen") for frame in seen for d in frame) and s.frozen_hits == 12


def test_a_tiny_jitter_of_the_bin_box_still_counts_as_frozen():
    s, _ = run([[p(0.88, BIN)]] + [[p(0.85, shifted(BIN, 0.001 * (k % 2)))] for k in range(10)])
    assert not s.snapshot()["person"]["present"]


def test_someone_walking_in_to_the_same_place_confirms():
    s, seen = run([[]] + [[p(0.88, BIN)]] * 6)                           # nothing in the first frame: an arrival
    assert s.snapshot()["person"]["present"] and not any(d.get("frozen") for f in seen for d in f)


def test_someone_already_there_counts_as_soon_as_they_move():
    still = [[p(0.88, BIN)]] * 3
    moving = [[p(0.88, shifted(BIN, 0.01 * k))] for k in range(1, 6)]      # overlap with the first box drops below 0.95 at once
    s, seen = run(still + moving)
    assert s.snapshot()["person"]["present"]
    assert all(d.get("frozen") for f in seen[:3] for d in f) and not any(d.get("frozen") for f in seen[3:] for d in f)


def test_a_person_walking_in_while_the_bin_stays_frozen_confirms_and_the_bin_stays_out():
    walk = [[p(0.88, BIN), p(0.9, (0.5 + 0.02 * k, 0.3, 0.6 + 0.02 * k, 0.8))] for k in range(6)]
    s, seen = run([[p(0.88, BIN)]] + walk)
    assert s.snapshot()["person"]["present"]
    assert all(d.get("frozen") == (tuple(d["box"]) == BIN) or (d.get("frozen") is None and tuple(d["box"]) != BIN) for f in seen for d in f)


def test_the_reference_is_taken_again_after_a_reset():
    reg = S.Registry()
    reg.update("cam", T0, [p(0.88, BIN)], 1.0)
    reg.reset("cam", 2.0)
    for k in range(6):                                                      # new episode: the person is NOT in its first frame -> arrives, counts
        sn = reg.update("cam", T0 + 10_000 + k * 500, [] if k == 0 else [p(0.9, BIN)], 2.0 + k * 0.5)
    assert sn["person"]["present"]


def test_camera_ai_person_claim_switches_freezing_off_for_the_episode():
    s = S.Session(CFG)
    seen = []
    for k in range(8):                                                      # someone already standing still at the door when the episode began
        d = [p(0.88, BIN)]
        s.update(T0 + k * 500, d, claim="person" if k == 2 else None)       # ... and the camera's own AI says person once
        seen.append(d)
    assert s.snapshot()["person"]["present"]
    assert all(d.get("frozen") for f in seen[:2] for d in f) and not any(d.get("frozen") for f in seen[2:] for d in f)


def test_frozen_can_be_switched_off():
    import defaults
    old = defaults.FROZEN["enabled"]
    try:
        S._FZ["enabled"] = False
        s, _ = run([[p(0.88, BIN)]] * 6)
        assert s.snapshot()["person"]["present"]
    finally:
        S._FZ["enabled"] = old


# ---- second look: a near-nothing first score is never lifted to a person ----
def it(name, logit, gx=16, gy=16, ltrb=(2, 2, 2, 2)):
    return (name, logit, 16, gx, gy, ltrb)


def test_second_look_never_raises_a_first_score_below_raise_min():
    eng = D.Engine(SeqModel(512, [[it("person", 0.85)]] * 8))             # the re-check finds a person ~0.70
    d0 = {"cls": "person", "group": "person", "score": 0.22, "box": [0.45, 0.42, 0.55, 0.56]}   # 1.4 % of the frame: below person_skip_area
    out, info = SL.second_look(eng, IMG, [d0], CFG, {}, [], 100.0)
    assert out[0]["second_look"] == "not_raised" and out[0]["score"] == 0.22 and "first_score" not in out[0]
    assert info["counts"].get("sl_not_raised") == 1
    out2, _ = SL.second_look(eng, IMG, [d0], CFG, {}, info["cache"], 101.0)          # the reused result keeps it unchanged too
    assert out2[0]["second_look"] == "not_raised" and out2[0]["score"] == 0.22
    d1 = {**d0, "score": SL._SL["raise_min"] + 0.05}
    out3, _ = SL.second_look(eng, IMG, [d1], CFG, {}, [], 200.0)                    # just above raise_min: raised as before
    assert out3[0]["second_look"] == "confirmed" and out3[0]["score"] > 0.65
