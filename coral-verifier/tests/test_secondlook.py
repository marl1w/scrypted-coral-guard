"""Second look: borderline / new-track candidates re-checked on a full-resolution crop inside the frame job."""
import os, re
import numpy as np
import pytest
import detect as D
import secondlook as SL
import session as S
import server as SV
from test_engine import SeqModel, engine
from test_server import Box, T0

IMG = np.full((480, 640, 3), 90, np.uint8)
CFG = S.Config()


def it(name, logit, gx=16, gy=16, ltrb=(2, 2, 2, 2)):
    return (name, logit, 16, gx, gy, ltrb)


def det(cls, score, box=(0.4, 0.4, 0.6, 0.6), **k):
    g = "person" if cls == "person" else ("vehicle" if cls in ("car", "truck", "bus", "motorcycle", "bicycle") else "animal")
    return {"cls": cls, "group": g, "score": score, "box": list(box), **k}


def test_band_selection_and_new_track_rule():
    dets = [det("person", 0.5), det("person", 0.9, (0.1, 0.1, 0.2, 0.3)),                 # band / strong + new but >= 0.8
            det("car", 0.3, (0.6, 0.1, 0.8, 0.3), weak=True), det("car", 0.5, (0.1, 0.6, 0.3, 0.9)), det("truck", 0.7, (0.5, 0.6, 0.7, 0.9)),
            det("dog", 0.4, (0.8, 0.1, 0.9, 0.3)), det("cat", 0.9, (0.8, 0.5, 0.9, 0.7))]
    got = set(SL.select(dets, CFG, {}))
    assert got == {0, 2, 3, 4, 5}                                                         # strong vehicle 0.7 is a NEW track < 0.8: candidate; 0.9 ones are not
    # an existing track under the strong 0.7 truck and the strong person removes the new-track reason; band candidates stay
    tracks = {"vehicle": [(0.5, 0.6, 0.7, 0.9)], "person": [(0.1, 0.1, 0.2, 0.3)]}
    assert set(SL.select(dets, CFG, tracks)) == {0, 2, 3, 5}
    assert SL.select([det("person", 0.7)], CFG, {"person": [(0.4, 0.4, 0.6, 0.6)]}) == []           # 0.7 >= start_thr 0.65 and tracked


def test_cap_and_priority_most_uncertain_first():
    e, m = engine(), None
    m = SeqModel(512, [[it("person", 0.0)]] * 4)
    eng = D.Engine(m)
    dets = [det("person", 0.30, (0.0, 0.0, 0.1, 0.2)), det("person", 0.60, (0.4, 0.4, 0.6, 0.6)), det("person", 0.62, (0.7, 0.4, 0.9, 0.6)),
            det("person", 0.45, (0.2, 0.7, 0.3, 0.9))]
    out, info = SL.second_look(eng, IMG, dets, CFG, {"person": [d["box"] for d in dets]}, [], 100.0)
    assert m.calls == 2 and info["n"] == 2                                               # default cap 2
    marked = [i for i, d in enumerate(out) if "second_look" in d]
    assert marked == [1, 2]                                                               # 0.62 and 0.60 are closest to start_thr 0.65


def test_crop_geometry_min_size_margin_and_inside_frame():
    x0, y0, x1, y1 = SL.crop_for((0.4, 0.4, 0.6, 0.6), 1000, 800)
    assert (x1 - x0, y1 - y0) == (440, 440) and (x0 + x1) / 2 == 500 and (y0 + y1) / 2 == 400          # box 200x160 + 0.6 on each side of the larger (200 * 2.2)
    x0, y0, x1, y1 = SL.crop_for((0.5, 0.5, 0.52, 0.52), 1000, 800)
    assert x1 - x0 == 96 and y1 - y0 == 96                                                 # minimum crop
    x0, y0, x1, y1 = SL.crop_for((0.0, 0.0, 0.2, 0.2), 1000, 800)
    assert (x0, y0) == (0, 0) and x1 - x0 == y1 - y0                                       # shifted inside the frame
    x0, y0, x1, y1 = SL.crop_for((0.0, 0.1, 1.0, 0.9), 1000, 800)
    assert (x0, y0, x1, y1) == (0, 0, 1000, 800)                                           # never larger than the frame


def test_recheck_back_mapping_to_frame_coordinates():
    m = SeqModel(512, [[it("person", 3.0, 16, 16)]])
    res, tm = D.Engine(m).recheck(IMG, (160, 120, 480, 360), "person")                      # crop 320x240 of the 640x480 frame
    assert len(res) == 1 and tm["infer_ms"] > 0
    s, a, b, c, e, label = res[0]
    assert label == "person" and abs((a + c) / 2 - 0.5) < 0.03 and abs((b + e) / 2 - 0.5) < 0.04      # crop centre is the frame centre
    assert 0.05 < c - a < 0.08                                                             # 4 cells of 16 px of 512 -> 0.125 of the crop width (0.5 of the frame) = 0.0625


def test_ttl_reuse_by_object_and_expiry():
    m = SeqModel(512, [[it("person", 2.0)]] * 10)
    eng = D.Engine(m)
    d0 = [det("person", 0.5)]
    out, info = SL.second_look(eng, IMG, d0, CFG, {}, [], 100.0)
    assert m.calls == 1 and out[0]["second_look"] == "confirmed" and out[0]["first_score"] == 0.5 and out[0]["score"] > 0.8
    moved = [det("person", 0.5, (0.41, 0.4, 0.61, 0.6))]
    out2, info2 = SL.second_look(eng, IMG, moved, CFG, {}, info["cache"], 105.0)
    assert m.calls == 1 and info2["n"] == 0 and out2[0]["second_look"] == "confirmed" and out2[0]["score"] == out[0]["score"]   # reused, no invoke
    assert info2["counts"] == {"sl_reused": 1}
    out3, info3 = SL.second_look(eng, IMG, moved, CFG, {}, info2["cache"], 111.0)           # > ttl 10 s after the real re-check
    assert m.calls == 2 and info3["n"] == 1


def test_rejected_candidate_is_marked_and_never_reaches_the_tracker():
    m = SeqModel(512, [[]] * 6)                                                              # the crop shows nothing
    eng = D.Engine(m)
    sess = S.Session(CFG)
    for i in range(3):
        out, info = SL.second_look(eng, IMG, [det("person", 0.5)], CFG, {}, [], 100.0 + i * 20)
        assert out[0]["second_look"] == "rejected" and "first_score" not in out[0] and info["counts"]["sl_rejected"] == 1
        live = [d for d in out if d.get("second_look") != "rejected"]
        sess.update(T0 + i * 1000, live)
    sn = sess.snapshot()
    assert sn["person"]["state"] == "idle" and sn["person"]["tracks"] == 0


def test_upgraded_candidate_can_confirm_and_lowered_is_marked():
    seq = [[it("person", 2.0)]] * 8
    eng = D.Engine(SeqModel(512, seq))
    sess = S.Session(CFG)
    for i in range(2):
        out, _ = SL.second_look(eng, IMG, [det("person", 0.5)], CFG, {}, [], 100.0 + i * 20)
        assert out[0]["second_look"] == "confirmed" and out[0]["score"] >= 0.65
        sess.update(T0 + i * 1000, out)
    assert sess.snapshot()["person"]["present"]                                              # 0.5 alone would never confirm (start_thr 0.65)
    eng2 = D.Engine(SeqModel(512, [[it("person", -0.6)]] * 3))
    out, _ = SL.second_look(eng2, IMG, [det("person", 0.55)], CFG, {}, [], 1.0)
    assert out[0]["second_look"] == "lowered" and out[0]["score"] < 0.55 and out[0]["first_score"] == 0.55


def test_weak_vehicle_flag_follows_the_second_look_score():
    eng = D.Engine(SeqModel(512, [[it("car", 2.0)]]))
    out, _ = SL.second_look(eng, IMG, [det("car", 0.3, weak=True)], CFG, {}, [], 1.0)
    assert out[0]["second_look"] == "confirmed" and "weak" not in out[0] and out[0]["score"] > 0.8


def test_second_look_config_key_is_a_validated_bool_default_true():
    c = S.Config()
    assert c.second_look is True and c.updated({"second_look": False}).second_look is False
    for bad in (1, 0, "yes", None, 0.5):
        with pytest.raises(ValueError):
            c.updated({"second_look": bad})
    with pytest.raises(ValueError):
        c.updated({"second_look_max": 3})                                                    # max per frame is internal
    assert SV.parse_options({"token": "x"})["defaults"].second_look is True
    assert "second_look" not in __import__("defaults").PUBLIC_OPTIONS


def test_version_comes_from_config_yaml():
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "..", "config.yaml")
    if not os.path.exists(p):
        pytest.skip("config.yaml not available")
    want = re.search(r'^version:\s*"?([^"\n]+)', open(p).read(), re.M).group(1)
    if "VERIFIER_VERSION" not in os.environ:
        assert SV.VERSION == want == "0.2.1"


def test_http_second_look_on_and_off_and_metrics():
    items = [["person", 0.0, 16, 16, 16, [2, 2, 2, 2]]]                                      # p 0.5: borderline
    off = Box(items=items)
    try:
        n = off.calls(); r = off.frame("c", T0)[1]
        assert off.calls() == n + 1 and "second_look" not in r["detections"][0] and r["second_look_ms"] == 0.0
    finally:
        off.close()
    on = Box(opts={"second_look": True}, items=items)
    try:
        st, cfg = on.req("POST", "/v1/cameras/off_cam/config", b'{"second_look": false}')
        assert st == 200 and cfg["second_look"] is False and on.req("POST", "/v1/cameras/bad/config", b'{"second_look": "no"}')[0] == 400
        n = on.calls(); on.frame("off_cam", T0)
        assert on.calls() == n + 1                                                            # per-camera override: off for this camera only
        n = on.calls(); r = on.frame("c", T0)[1]
        assert on.calls() == n + 2                                                           # normal pass + ONE second look, one request
        d = r["detections"][0]
        assert d["second_look"] == "confirmed" and d["first_score"] < 0.55 and r["second_look_ms"] > 0 and r["infer_ms"] > 0
        m = on.req("GET", "/v1/metrics")[1]["second_look"]
        assert m["count_60s"] == 1 and m["confirmed_share"] == 1.0 and m["rejected_share"] == 0.0
        n = on.calls(); on.frame("c", T0 + 1000)                                              # same object within the ttl: reused, no new invoke
        assert on.calls() == n + 1
        assert on.req("GET", "/v1/metrics")[1]["second_look"]["reused_60s"] >= 1
    finally:
        on.close()
