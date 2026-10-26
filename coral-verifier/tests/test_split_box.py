"""One box over two parked cars (0.3.42): it never takes over a parked track, so a parked car is not made "moving" by one odd frame;
a parked car that really leaves still moves, and its old parked track ends."""
import numpy as np

import session as S
import shape as SH
import spots as SP
from defaults import SPOTS
from test_shape import frame

T0 = 1_791_886_000_000
LOW = (0.16, 0.548, 0.568, 0.988)                                 # two cars parked side by side, as a camera sees them
UP = (0.327, 0.203, 0.629, 0.592)
BOTH = (0.007, 0.201, 0.629, 0.982)                               # the box the detector sometimes draws over both
BOTH_TALL = (0.057, 0.084, 0.650, 0.986)                          # another one seen over the same two cars
GAP = (SPOTS["occasion_gap_s"] + 60) * 1000


def det(box, cls="car", score=0.8):
    return {"cls": cls, "score": score, "box": list(box)}


def new(spots=None, **kw):
    s = S.Session(S.Config().updated(kw))
    s.vehicle.spots = spots if spots is not None else SP.Spots()
    return s


def parked_pair(s, secs=125):
    for i in range(secs):
        s.update(T0 + i * 1000, [det(LOW), det(UP, score=0.7)])
    assert [tr["state"] for tr in s.snapshot()["vehicle"]["tracks"]] == ["parked", "parked"]
    return secs


def jitter(box, rng, d=0.01):
    return tuple(float(np.clip(v + rng.uniform(-d, d), 0, 1)) for v in box)


def test_a_box_over_two_parked_cars_never_makes_one_move():
    s = new()
    n = parked_pair(s)
    rng = np.random.default_rng(1)
    for k in range(60):                                           # every few seconds the detector sees only one box over both, never the same
        if k % 3 == 0:
            dets = [det(jitter(BOTH if k % 2 else BOTH_TALL, rng, 0.05), ("motorcycle", "car", "truck")[k % 3 + k % 2], 0.84)]
        else:
            dets = [det(jitter(LOW, rng)), det(jitter(UP, rng), score=0.7)]
        s.update(T0 + (n + k * 2) * 1000, dets)
        v = s.snapshot()["vehicle"]
        assert not v["moving"]
        assert sum(tr["state"] == "parked" for tr in v["tracks"]) == 2


def test_half_an_hour_of_random_boxes_over_both_raises_nothing():
    s = new()
    n = parked_pair(s)
    rng = np.random.default_rng(7)
    for k in range(900):                                          # 30 min of 2 s heartbeats, one in four with a box over both
        if rng.random() < 0.25:
            dets = [det(jitter(BOTH if rng.random() < 0.5 else BOTH_TALL, rng, 0.06), "motorcycle", 0.8)]
        else:
            dets = [det(jitter(LOW, rng, 0.02)), det(jitter(UP, rng, 0.02), score=0.7)]
        s.update(T0 + (n + k * 2) * 1000, dets)
        assert not s.snapshot()["vehicle"]["moving"]
    assert [tr.id for tr in s.vehicle.tracks if tr.parked] == [0, 1]   # the same two parked tracks the whole time


def test_the_box_over_both_never_parks_or_becomes_a_spot():
    s = new()
    n = parked_pair(s)
    for k in range(200):                                          # it stays for minutes; the cars are not detected meanwhile
        s.update(T0 + (n + k) * 1000, [det(BOTH, "motorcycle", 0.84)])
    v = s.snapshot()["vehicle"]
    assert not v["moving"] and sum(tr["state"] == "parked" for tr in v["tracks"]) == 2
    assert all(SP.iou(tuple(sp.box), BOTH) < 0.5 for sp in s.vehicle.spots.items)


def test_a_spot_learnt_over_both_cars_parks_nothing_while_the_cars_are_tracked():
    sp = SP.Spots()
    look = SH.signature(frame(seed=0), BOTH)
    assert look is not None
    for box in (LOW, UP, BOTH):                                   # the spot the old tracker learnt, trusted and with its look
        sp.learn(box, SH.signature(frame(seed=0), box), T0, f"a{box}")
        sp.learn(box, SH.signature(frame(seed=1), box), T0 + GAP, f"b{box}")
    s = new(sp)
    t0 = T0 + 2 * GAP
    for i in range(3):
        s.update(t0 + i * 1000, [det(LOW), det(UP, score=0.7)], gray=frame(seed=10 + i))
    assert sum(tr["parked"] for tr in s.snapshot()["vehicle"]["tracks"]) == 2      # both back in their spots at once
    for k in range(30):
        s.update(t0 + (3 + k * 2) * 1000, [det(BOTH, "motorcycle", 0.84)], gray=frame(seed=20 + k))
        v = s.snapshot()["vehicle"]
        assert not v["moving"] and not any(tr["parked"] and tr["box"] == list(BOTH) for tr in v["tracks"])


def test_a_parked_car_that_leaves_moves_and_its_old_track_ends():
    s = new()
    n = parked_pair(s)
    moved = None
    for k in range(1, 15):                                        # the lower car backs out to the left, 0.06 of the frame a second; the other stays
        dx = 0.06 * k
        s.update(T0 + (n + k) * 1000, [det((LOW[0] - dx, LOW[1], LOW[2] - dx, LOW[3])), det(UP, score=0.7)])
        v = s.snapshot()["vehicle"]
        if v["moving"]:
            moved = k
            break
    assert moved is not None and moved <= 5
    assert sorted(tr["state"] for tr in v["tracks"]) == ["moving", "parked"]    # no parked track left behind at the empty spot


def test_a_car_arriving_next_to_a_parked_one_is_an_ordinary_track():
    s = new()
    for i in range(125):
        s.update(T0 + i * 1000, [det(LOW)])
    beside = (0.36, 0.548, 0.768, 0.988)                          # overlaps the parked car (IoU ~0.33) while it is seen itself
    for i in range(125, 260):
        s.update(T0 + i * 1000, [det(LOW), det(beside)])
    v = s.snapshot()["vehicle"]
    assert [tr["state"] for tr in v["tracks"]] == ["parked", "parked"]


def test_a_person_inside_the_split_box_is_only_delayed():
    s = new()
    n = parked_pair(s)
    s.update(T0 + n * 1000, [det(BOTH, "motorcycle", 0.84)])
    walker = {"cls": "person", "score": 0.8, "box": [0.3, 0.4, 0.4, 0.9]}
    late = T0 + (n + S.MOVE_REF_S + 1.5) * 1000                    # the split track is no longer a possible mover once it stayed still
    s.update(late - 500, [det(BOTH, "motorcycle", 0.84)])
    out = s._drop_in_vehicle(late, [dict(walker)])
    assert out and not out[0].get("in_vehicle")
