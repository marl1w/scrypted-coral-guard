"""Minimum size per group and camera (0.3.23): too-small boxes never reach the trackers."""
import json
import pytest
import session as S
from boxlib import Box

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
