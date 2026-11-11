"""A parked car cut by the frame edge (0.3.59): the detector draws its box either around the part it sees between a post and the
car's far end, or out to the frame edge. The box flips between the two from frame to frame while nothing moves. The centre shift
of that flip (0.07 of the frame) is above the movement threshold and the IoU of the two (0.71) sits right at STILL_IOU, so before
this release the car was "moving" (a confirmed moving vehicle every few seconds, all night) and rarely stayed still long enough
to park. Boxes from a live camera that sees two parked cars, the lower one cut by the left edge."""
import numpy as np

import session as S
import spots as SP

T0 = 1_794_392_769_000
UP = (0.330, 0.203, 0.641, 0.592)
LOW_IN = (0.147, 0.558, 0.568, 0.985)                             # the part between the post and the car's front
LOW_EDGE = (0.0005, 0.558, 0.569, 0.994)                          # the same car, its box out to the frame edge


def det(box, cls="car", score=0.8):
    return {"cls": cls, "score": score, "box": list(box)}


def new(**kw):
    s = S.Session(S.Config().updated(kw))
    s.vehicle.spots = SP.Spots()
    return s


def jitter(box, rng, d=0.006):
    return tuple(float(np.clip(v + rng.uniform(-d, d), 0, 1)) for v in box)


def flipping(rng, k):
    """One frame of the two parked cars: the lower one's box flips edge/inside at random, at times weak or missed."""
    dets = [det(jitter(UP, rng), score=float(rng.uniform(0.65, 0.87)))]
    r = rng.random()
    if r < 0.15:
        return dets                                               # the lower car missed
    low = LOW_EDGE if rng.random() < 0.5 else LOW_IN
    return dets + [det(jitter(low, rng), score=float(rng.uniform(0.42, 0.85)))]


def test_a_box_flipping_to_the_frame_edge_never_moves():
    s = new()
    rng = np.random.default_rng(3)
    for k in range(600):                                          # 10 min at one frame a second, from a fresh session
        s.update(T0 + k * 1000, flipping(rng, k))
        assert not s.snapshot()["vehicle"]["moving"], f"moving at {k} s"


def test_a_car_cut_by_the_edge_still_parks():
    s = new()
    rng = np.random.default_rng(5)
    for k in range(200):
        s.update(T0 + k * 1000, flipping(rng, k))
    states = sorted(tr["state"] for tr in s.snapshot()["vehicle"]["tracks"])
    assert states == ["parked", "parked"], states


def test_a_car_driving_out_past_the_edge_moves():
    s = new()
    for k in range(130):                                          # parked first
        s.update(T0 + k * 1000, [det(UP), det(LOW_IN if k % 2 else LOW_EDGE)])
    moved = False
    for i in range(1, 9):                                         # then it backs out to the left: its right edge travels
        x1 = LOW_IN[2] - 0.05 * i
        s.update(T0 + (130 + i) * 1000, [det(UP), det((0.0, LOW_IN[1], max(x1, 0.05), LOW_IN[3]))])
        moved = moved or s.snapshot()["vehicle"]["moving"]
    assert moved


def test_a_car_coming_in_from_the_edge_moves():
    s = new()
    moved = False
    for i in range(1, 10):                                        # enters from the left edge: its right edge travels inwards
        s.update(T0 + i * 1000, [det(UP), det((0.0, 0.56, 0.06 * i, 0.99))])
        moved = moved or s.snapshot()["vehicle"]["moving"]
    assert moved


def test_a_car_moving_away_from_the_edge_moves():
    s = new()
    moved = False
    for i in range(8):                                            # the whole box shifts right, its left edge leaving the frame edge
        x0 = 0.02 * i
        s.update(T0 + i * 1000, [det(UP), det((x0, 0.56, 0.42 + 0.04 * i, 0.99))])
        moved = moved or s.snapshot()["vehicle"]["moving"]
    assert moved
