"""Minimum size per group and camera (0.3.23): too-small boxes never reach the trackers."""
import json
import pytest
import session as S
from boxlib import Box, person_item

T0 = 1_791_886_000_000


def det(cls, box, score=0.9):
    return {"cls": cls, "score": score, "box": list(box)}


def test_too_small_per_group():
    cfg = S.Config(min_area_vehicle=0.01)
    assert S.too_small(det("car", (0.70, 0.43, 0.76, 0.51)), cfg)            # 0.06 x 0.08 = 0.0048 < 1 %
    assert not S.too_small(det("car", (0.1, 0.4, 0.4, 0.8)), cfg)            # 12 % of the frame
    assert S.too_small(det("bicycle", (0.5, 0.5, 0.55, 0.55)), cfg)          # bicycles are vehicles too
    assert not S.too_small(det("person", (0.70, 0.43, 0.71, 0.44)), cfg)     # person minimum is off
    assert not S.too_small(det("car", (0.70, 0.43, 0.76, 0.51)), S.Config())  # all off by default


@pytest.mark.parametrize("bad", [-0.01, 0.26, float("nan")])
def test_min_area_bounds(bad):
    with pytest.raises(ValueError):
        S.Config().updated({"min_area_vehicle": bad})


def test_too_small_person_is_never_tracked_over_http():
    b = Box()
    try:
        b.req("POST", "/v1/cameras/small/config", json.dumps({"confirm_n": 1, "min_area_person": 0.25}).encode())
        st, r = b.frame("small", T0, "&verbose=1")
        assert st == 200 and r["person"]["state"] == "idle"
        assert r["detections"] and all(d.get("too_small") for d in r["detections"] if d["cls"] == "person")
        assert b.frame("small", T0 + 1000)[1]["detections"] == []                # hidden from normal replies
        b.req("POST", "/v1/cameras/big/config", json.dumps({"confirm_n": 1}).encode())
        assert b.frame("big", T0)[1]["person"]["state"] == "confirmed"          # same frame, no minimum: confirmed
    finally:
        b.close()


def test_too_big_person_and_bounds():
    """0.3.35: a person box over max_area_person (default 85 %) is something on the lens: never tracked."""
    cfg = S.Config()
    assert S.too_big(det("person", (0.0, 0.0, 0.98, 0.99)), cfg)                    # 97 %
    assert not S.too_big(det("person", (0.0, 0.0, 0.7, 0.72)), cfg)                 # 50 %: the largest real person seen
    assert not S.too_big(det("car", (0.0, 0.0, 1.0, 1.0)), cfg)                     # persons only
    assert not S.too_big(det("person", (0.0, 0.0, 0.98, 0.99)), S.Config(max_area_person=1.0))
    for bad in (0.04, 1.01):
        with pytest.raises(ValueError):
            S.Config().updated({"max_area_person": bad})
    with pytest.raises(ValueError):
        S.Config().updated({"single_min_h": 1.5})


def test_too_big_person_is_never_tracked_over_http():
    b = Box(items=[person_item(d=8)])                                             # a person box of about a quarter of the frame
    try:
        b.req("POST", "/v1/cameras/lens/config", json.dumps({"confirm_n": 1, "max_area_person": 0.1}).encode())
        st, r = b.frame("lens", T0, "&verbose=1")
        assert st == 200 and r["person"]["state"] == "idle"
        assert r["detections"] and all(d.get("too_big") for d in r["detections"] if d["cls"] == "person")
    finally:
        b.close()


def test_groups_the_camera_does_not_detect():
    """0.3.44: a camera that never sees cars (detect_vehicle false) never tracks nor reports one: a grill scored as a car is gone."""
    cfg = S.Config(detect_vehicle=False)
    assert S.not_detected(det("car", (0.1, 0.4, 0.4, 0.8)), cfg) and S.not_detected(det("bicycle", (0.1, 0.4, 0.4, 0.8)), cfg)
    assert not S.not_detected(det("person", (0.1, 0.4, 0.4, 0.8)), cfg) and not S.not_detected(det("car", (0.1, 0.4, 0.4, 0.8)), S.Config())
    with pytest.raises(ValueError):
        S.Config().updated({"detect_person": False, "detect_vehicle": False, "detect_animal": False})
    b = Box()
    try:
        assert b.req("POST", "/v1/cameras/patio/config", json.dumps({"confirm_n": 1, "detect_person": False}).encode())[0] == 200
        st, r = b.frame("patio", T0, "&verbose=1")
        assert st == 200 and r["person"]["state"] == "idle"
        assert r["detections"] and all(d.get("not_detected") for d in r["detections"] if d["cls"] == "person")
        assert b.frame("patio", T0 + 1000)[1]["detections"] == []                # hidden from normal replies
    finally:
        b.close()
