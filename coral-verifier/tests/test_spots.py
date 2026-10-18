"""Known parking spots (0.3.31): learnt from parked vehicles, trusted after two occasions, kept per camera across sessions."""
import json, os

import numpy as np

import session as S
import shape as SH
import spots as SP
from defaults import SPOTS
from test_shape import CAR, T0, frame, car
from boxlib import Box, NOW, jpeg

GAP = (SPOTS["occasion_gap_s"] + 60) * 1000


def look(**kw):
    return SH.signature(frame(**kw), CAR)


def trusted_spots():
    sp = SP.Spots()
    sp.learn(CAR, look(seed=1), T0, "a")
    sp.learn(CAR, look(seed=2), T0 + GAP, "b")
    return sp


def session(spots=None, **kw):
    s = S.Session(S.Config().updated(kw))
    s.vehicle.spots = spots if spots is not None else SP.Spots()
    return s


def test_a_spot_is_trusted_after_two_separate_occasions():
    sp = SP.Spots()
    sp.learn(CAR, look(), T0, "a")
    assert sp.at(CAR) is None and sp.at(CAR, trusted_only=False) is not None
    sp.learn(CAR, look(), T0 + 60_000, "b")                    # another track a minute later: same occasion
    assert sp.at(CAR) is None
    sp.learn(CAR, look(), T0 + 60_000 + GAP, "c")                 # the gap counts from the last sighting
    assert sp.at(CAR) is not None and sp.summary() == {"spots": 1, "trusted": 1}
    assert len(sp.items[0].looks) == 1                         # the same look is kept once


def test_parking_teaches_the_spot():
    s = session()
    for i in range(125):
        s.update(T0 + i * 1000, [car()], gray=frame(seed=i))
    v = s.snapshot()["vehicle"]
    assert v["tracks"][0]["parked"] and v["spots"] == {"spots": 1, "trusted": 0}


def test_a_known_car_back_in_its_trusted_spot_is_parked_at_once():
    s = session(trusted_spots())
    s.update(T0, [car()], gray=frame(seed=9))
    v = s.snapshot()["vehicle"]
    assert v["tracks"][0]["parked"] and v["shape"]["spot_parked"] == 1


def test_another_car_in_a_trusted_spot_parks_the_usual_way():
    s = session(trusted_spots())
    other = np.full((300, 480), 90, np.uint8)
    cv2_free = other.copy()
    cv2_free[160:260, 60:200] = 180                            # a plain box, nothing like the learnt car
    s.update(T0, [car()], gray=cv2_free)
    assert not s.snapshot()["vehicle"]["tracks"][0]["parked"]


def test_untrusted_spot_changes_nothing():
    sp = SP.Spots()
    sp.learn(CAR, look(), T0, "a")
    s = session(sp)
    s.update(T0, [car()], gray=frame(seed=3))
    assert not s.snapshot()["vehicle"]["tracks"][0]["parked"]


def test_a_weak_box_keeps_a_car_parked_in_a_trusted_spot():
    s = session(trusted_spots(), shape_hold=False)             # weak hits alone, no shape hold
    s.update(T0, [car()], gray=frame(seed=1))
    for i in range(1, 30):
        s.update(T0 + i * 1000, [car(0.3)], gray=frame(seed=i))
    v = s.snapshot()["vehicle"]
    assert len(v["tracks"]) == 1 and v["tracks"][0]["parked"] and v["shape"]["weak"] == 29


def test_a_weak_box_outside_a_trusted_spot_keeps_nothing():
    s = session(SP.Spots(), shape_hold=False)
    s.update(T0, [car()])
    for i in range(1, 12):
        s.update(T0 + i * 1000, [car(0.3)])
    assert s.snapshot()["vehicle"]["tracks"] == []


def test_parking_spots_off():
    s = session(trusted_spots(), parking_spots=False)
    s.update(T0, [car()], gray=frame(seed=9))
    assert not s.snapshot()["vehicle"]["tracks"][0]["parked"]


def test_export_restore_keeps_boxes_counts_and_looks():
    sp = trusted_spots()
    back = SP.Spots.restore(json.loads(json.dumps(sp.export())))
    assert back.summary() == sp.summary()
    assert back.items[0].matches(look(seed=7), 0.9)
    assert SP.Spots.restore([{"box": [0, 0]}, {"nope": 1}, sp.export()[0]]).summary()["spots"] == 1     # bad entries are skipped


def test_spots_survive_a_session_reset_and_are_saved_once():
    r = S.Registry()
    for i in range(125):
        r.update("c1", T0 + i * 1000, [car()], 1.0, gray=frame(seed=i))
    assert r.spots_describe("c1")[0]["occasions"] == 1
    dirty = r.spots_dirty()
    assert list(dirty) == ["c1"] and r.spots_dirty() == {}
    r.reset("c1")
    r.update("c1", T0 + 200_000, [car()], 2.0, gray=frame(seed=1))
    assert r.spots_describe("c1")[0]["occasions"] == 1          # still there after the reset
    assert r.spots_forget("c1") and r.spots_describe("c1") == [] and r.spots_dirty() == {"c1": []}


def test_a_reset_keeps_the_parked_car_and_clears_people():
    r = S.Registry()
    for i in range(125):                                         # parked after parked_after_s
        r.update("c1", T0 + i * 1000, [car(), {"cls": "person", "score": 0.9, "box": [0.7, 0.2, 0.8, 0.6]}], 1.0 + i, gray=frame(seed=i))
    sn = r.peek("c1", 125.0)
    assert sn["person"]["present"] and sn["vehicle"]["tracks"][0]["parked"]
    assert r.reset("c1", 125.5) is True
    sn = r.update("c1", T0 + 126_000, [], 126.0, gray=frame(seed=200))       # the detector misses the car on the first frame after the trigger
    assert not sn["person"]["present"] and sn["session"]["frames"] == 1
    assert [(t["id"], t["parked"]) for t in sn["vehicle"]["tracks"]] == [(0, True)]
    assert r.reset("c1", 127.0, keep_vehicles=False) is True                   # the analysed view changed: nothing lingers
    assert r.update("c1", T0 + 128_000, [], 128.0, gray=frame(car=False, seed=3))["vehicle"]["tracks"] == []


def test_a_reset_drops_moving_vehicles():
    r = S.Registry()
    for i in range(8):
        x = 0.05 * i
        r.update("c1", T0 + i * 1000, [{"cls": "car", "score": 0.9, "box": [x, 0.5, x + 0.3, 0.9]}], 1.0 + i)
    assert r.peek("c1", 8.0)["vehicle"]["moving"]
    r.reset("c1", 8.0)
    assert r.update("c1", T0 + 9000, [], 9.0)["vehicle"]["tracks"] == []


def test_server_reset_all_clears_vehicles():
    b = Box(items=[["car", 3.0, 16, 8, 8, [3, 2, 3, 2]]])
    try:
        b.frame("g", NOW - 5000)
        assert b.req("POST", "/v1/cameras/g/reset")[0] == 200                # a camera trigger keeps the still car
        assert b.app.registry._sessions["g"].vehicle.tracks
        assert b.req("POST", "/v1/cameras/g/reset", b'{"all": true}')[0] == 200
        assert "g" not in b.app.registry._sessions
    finally:
        b.close()


def test_server_saves_loads_and_forgets_spots():
    b = Box()
    try:
        b.app.registry.spots_restore("cam1", trusted_spots().export())
        b.app.registry._spots["cam1"].dirty = True
        b.app.save_spots()
        path = os.path.join(b.app.spots_dir, "cam1.json")
        assert os.path.exists(path)
        st, r = b.req("GET", "/v1/cameras/cam1/spots")
        assert st == 200 and r["spots"][0]["trusted"] and r["spots"][0]["looks"] == 1          # two sightings of the same car: one look
        b2 = Box(opts={"spots_root": b.app.spots_dir})          # a restart reads them back
        try:
            assert b2.app.registry.spots_describe("cam1")[0]["trusted"]
        finally:
            b2.close()
        st, r = b.req("POST", "/v1/cameras/cam1/spots", b"{}", {"Content-Type": "application/json"})
        assert st == 200 and r["forgotten"] and not os.path.exists(path)
        st, _ = b.req("PUT", "/v1/cameras/cam1/spots")
        assert st == 405
        st, _ = b.req("GET", "/v1/cameras/cam1/spots", auth=False)
        assert st == 401
    finally:
        b.close()
