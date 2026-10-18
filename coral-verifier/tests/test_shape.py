"""Shape hold (0.3.30): a still/parked vehicle the detector misses is kept while the outline in its box matches its last look."""
import cv2
import numpy as np
import pytest

import session as S
import shape as SH

T0 = 1_791_886_000_000
CAR = (0.1, 0.5, 0.45, 0.9)
W, H = 480, 300


def frame(car=True, cover=False, gain=1.0, offset=0.0, seed=0):
    """Gray frame: paving texture, optionally a car (body, windows, wheels) and a person-sized blob over it; fresh sensor noise every call."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 90, np.float32)
    for y in range(0, H, 12):
        cv2.line(img, (0, y), (W, y), 70, 1)
    if car:
        x0, y0, x1, y1 = int(CAR[0] * W), int(CAR[1] * H), int(CAR[2] * W), int(CAR[3] * H)
        cv2.rectangle(img, (x0 + 5, y0 + 20), (x1 - 5, y1 - 20), 200, -1)
        cv2.rectangle(img, (x0 + 30, y0 + 28), (x1 - 40, y0 + 55), 40, -1)
        cv2.circle(img, (x0 + 35, y1 - 18), 14, 20, -1)
        cv2.circle(img, (x1 - 35, y1 - 18), 14, 20, -1)
    if cover:
        cv2.ellipse(img, (int(0.27 * W), int(0.7 * H)), (30, 60), 0, 0, 360, 150, -1)
    img = img * gain + offset + rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def car(score=0.9):
    return {"cls": "car", "score": score, "box": list(CAR)}


def person():
    return {"cls": "person", "score": 0.8, "box": [0.22, 0.45, 0.32, 0.95]}


def new(**kw):
    return S.Session(S.Config().updated(kw))


def step(s, i, dets, img):
    s.update(T0 + i * 1000, dets, gray=img)
    return s.snapshot()["vehicle"]


def test_missed_car_with_the_same_look_survives_and_parks():
    s = new()
    ids = set()
    for i in range(140):
        seen = i % 10 == 0                                     # the detector finds it once every 10 s (night IR near the threshold)
        v = step(s, i, [car()] if seen else [], frame(seed=i))
        assert len(v["tracks"]) == 1                           # never dropped in between (hold_s is 8 s)
        ids.add(v["tracks"][0]["id"])
    assert ids == {0}
    assert v["tracks"][0]["parked"]                            # lived the 120 s it needs to park
    assert v["shape"]["held"] > 100 and v["shape"]["released"] == 0


def test_without_the_shape_hold_the_same_car_flickers():
    s = new(shape_hold=False)
    ids = set()
    for i in range(60):
        v = step(s, i, [car()] if i % 10 == 0 else [], frame(seed=i))
        ids.update(tr["id"] for tr in v["tracks"])
    assert len(ids) > 1                                        # a new track at every hit: the old behaviour


def test_a_car_that_left_is_dropped_at_once():
    s = new()
    for i in range(5):
        step(s, i, [car()], frame(seed=i))
    v = step(s, 5, [], frame(car=False, seed=5))               # the spot is empty and nothing is in front of it
    assert v["tracks"] == [] and v["shape"]["released"] == 1


def test_a_covered_car_is_held_up_to_the_cap_then_dropped():
    s = new(occlusion_hold_s=20)
    for i in range(5):
        step(s, i, [car()], frame(seed=i))
    alive = []
    for i in range(5, 60):
        v = step(s, i, [person()], frame(car=False, cover=True, seed=i))      # someone in front, the car not visible as before
        alive.append(bool(v["tracks"]))
    assert all(alive[:20])                                     # held while covered, beyond hold_s (8 s)
    assert not alive[-1]                                       # but not forever


def test_the_look_is_not_renewed_while_something_covers_the_car():
    s = new()
    for i in range(3):
        step(s, i, [car()], frame(seed=i))
    ref = s.vehicle.tracks[0].ref.copy()
    for i in range(3, 8):
        step(s, i, [car(), person()], frame(cover=True, seed=i))             # car still detected with a person in front of it
    assert np.array_equal(s.vehicle.tracks[0].ref, ref)
    v = step(s, 8, [], frame(seed=8))                          # person gone, car missed: the clean look still matches
    assert len(v["tracks"]) == 1 and v["shape"]["held"] == 1


def test_a_brightness_change_keeps_the_match():
    s = new()
    for i in range(3):
        step(s, i, [car()], frame(seed=i))
    v = step(s, 3, [], frame(gain=0.6, offset=40, seed=3))     # exposure / light level change, same car
    assert len(v["tracks"]) == 1 and v["shape"]["held"] == 1


def test_no_frame_means_the_old_behaviour():
    s = new()
    s.update(T0, [car()])
    for i in range(1, 12):
        s.update(T0 + i * 1000, [])
    assert s.snapshot()["vehicle"]["tracks"] == []


def test_signature_needs_a_box_with_detail():
    g = frame()
    assert SH.signature(g, (0.1, 0.1, 0.11, 0.11)) is None                  # too small
    assert SH.signature(np.full((H, W), 100, np.uint8), CAR) is None         # flat
    a, b = SH.signature(g, CAR), SH.signature(frame(car=False), CAR)
    assert SH.similarity(a, a) == pytest.approx(1.0, abs=1e-5)
    assert SH.similarity(a, b) < 0.6


def test_gray_small_reduces_wide_colour_frames():
    g = SH.gray_small(np.zeros((576, 928, 3), np.uint8))
    assert g.shape == (298, 480) and g.dtype == np.uint8


@pytest.mark.parametrize("kw", [{"shape_thr": 0.05}, {"shape_thr": 1.0}, {"shape_cover": 0.0}, {"occlusion_hold_s": 601}, {"shape_hold": 1}])
def test_shape_config_is_validated(kw):
    with pytest.raises(ValueError):
        S.Config().updated(kw)
