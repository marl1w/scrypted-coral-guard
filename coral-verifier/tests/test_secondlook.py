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
    assert got == {0, 1, 2, 3, 4, 5, 6}                                                   # 0.9 person / animal would START a track: mandatory; strong vehicle 0.7 is a NEW track < 0.8
    conf = {"person_conf": [(0.1, 0.1, 0.2, 0.3)], "animal_conf": [(0.8, 0.5, 0.9, 0.7)]}
    assert set(SL.select(dets, CFG, conf)) == {0, 2, 3, 4, 5}                             # continuing an already CONFIRMED track needs no check
    # an existing track under the strong 0.7 truck and the strong person removes the new-track reason; band candidates stay
    tracks = {"vehicle": [(0.5, 0.6, 0.7, 0.9)], "person": [(0.1, 0.1, 0.2, 0.3)]}
    assert set(SL.select(dets, CFG, tracks)) == {0, 1, 2, 3, 5, 6}                       # tentative-or-any tracks do not exempt a start-level person/animal
    assert SL.select([det("person", 0.7)], CFG, {"person": [(0.4, 0.4, 0.6, 0.6)], "person_conf": [(0.4, 0.4, 0.6, 0.6)]}) == []   # >= start_thr but continuing a confirmed track


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
    assert x1 - x0 == y1 - y0 == 440 and (x0 + x1) / 2 == 100 and (y0 + y1) / 2 == 80 and x0 < 0 and y0 < 0     # NOT shifted: symmetric margin, negative offsets
    x0, y0, x1, y1 = SL.crop_for((0.0, 0.1, 1.0, 0.9), 1000, 800)
    assert (x1 - x0, y1 - y0) == (2200, 2200) and (x0 + x1) / 2 == 500 and (y0 + y1) / 2 == 400   # larger than the frame


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


def test_not_found_keeps_the_detection_unchanged_and_feeds_the_tracker():
    m = SeqModel(512, [[]] * 6)                                                              # the window shows nothing (half-occluded parked car)
    eng = D.Engine(m)
    out, info = SL.second_look(eng, IMG, [det("car", 0.30, weak=True)], CFG, {}, [], 100.0)
    assert out[0]["second_look"] == "not_found" and out[0]["score"] == 0.30 and "first_score" not in out[0] and out[0]["weak"] is True
    assert info["counts"]["sl_not_found"] == 1
    sess = S.Session(CFG)
    for i in range(3):
        out, _ = SL.second_look(eng, IMG, [det("person", 0.9)], CFG, {}, [], 200.0 + i * 20)       # strong person: not even a candidate
        sess.update(T0 + i * 1000, [d for d in out if d.get("second_look") != "rejected"])
    assert sess.snapshot()["person"]["present"]
    kept, _ = SL.second_look(eng, IMG, [det("person", 0.5)], CFG, {}, [], 300.0)
    assert kept[0]["second_look"] == "not_found" and kept[0]["score"] == 0.5                  # never dropped: it goes to the tracker as it was
    assert [d for d in kept if d.get("second_look") != "rejected"] == kept                    # goes to the tracker


def test_rejected_only_on_a_confident_low_match():
    eng = D.Engine(SeqModel(512, [[it("car", -1.8)]] * 4))                                   # found again at p 0.14 < 0.5 x vehicle floor 0.40, first 0.5 < 0.6
    sess = S.Session(CFG)
    out, info = SL.second_look(eng, IMG, [det("car", 0.5)], CFG, {}, [], 1.0)
    assert out[0]["second_look"] == "rejected" and out[0]["first_score"] == 0.5 and out[0]["score"] < 0.2 and info["counts"]["sl_rejected"] == 1
    sess.update(T0, [d for d in out if d.get("second_look") != "rejected"])
    assert sess.snapshot()["vehicle"]["tracks"] == []                                        # never reaches the vehicle tracker
    out, _ = SL.second_look(eng, IMG, [det("car", 0.65, (0.4, 0.4, 0.6, 0.6))], CFG, {}, [], 50.0)   # first >= confirm 0.6 (new track < 0.8): never rejected
    assert out[0]["second_look"] == "not_upgraded" and out[0]["score"] == 0.65 and "first_score" not in out[0]
    mid = D.Engine(SeqModel(512, [[it("car", -0.6)]] * 4))                                   # found at p 0.35: between 0.2 and the floor 0.40
    out, info = SL.second_look(mid, IMG, [det("car", 0.5)], CFG, {}, [], 1.0)
    assert out[0]["second_look"] == "not_upgraded" and out[0]["score"] == 0.5 and info["counts"]["sl_not_upgraded"] == 1
    hi = D.Engine(SeqModel(512, [[it("car", -0.3)]] * 4))                                    # p 0.43 >= floor 0.40 but < first 0.58: lowered
    out, _ = SL.second_look(hi, IMG, [det("car", 0.58)], CFG, {}, [], 1.0)
    assert out[0]["second_look"] == "lowered" and out[0]["score"] < 0.58


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
        assert SV.VERSION == want == "0.3.14"


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


def test_window_outside_the_frame_is_grey_and_edge_box_keeps_symmetric_margin():
    img = np.full((480, 640, 3), 200, np.uint8)
    m = SeqModel(512, [[]])
    eng = D.Engine(m)
    win = SL.crop_for((0.0, 0.2, 0.2, 0.6), 640, 480)                    # box touching the left frame edge: 128 x 192 px, window 422 px square
    assert win[0] < 0 and win[2] - win[0] == 422
    eng.recheck(img, win, "person")
    cv = m.last_canvas                                                    # RGB 512 canvas of the letterboxed window
    assert tuple(cv[256, 5]) == (114, 114, 114)                           # outside side (left of the frame edge): grey
    left_edge = int(round(-win[0] * 512 / 422))                           # canvas column where the frame starts
    assert tuple(cv[256, left_edge + 20]) == (200, 200, 200) and tuple(cv[256, left_edge - 5]) == (114, 114, 114)
    big = SL.crop_for((0.0, 0.0, 1.0, 1.0), 640, 480)                     # box = whole frame: window 1408 px, frame shrunk in the middle of grey
    eng.recheck(img, big, "person")
    cv = m.last_canvas
    assert big[2] - big[0] == 1408 and tuple(cv[5, 5]) == (114, 114, 114) and tuple(cv[256, 256]) == (200, 200, 200)


def test_back_mapping_with_negative_window_offsets():
    img = np.full((480, 640, 3), 90, np.uint8)
    win = SL.crop_for((0.0, 0.0, 0.2, 0.2), 640, 480)                    # window starts at negative x and y
    assert win[0] < 0 and win[1] < 0
    m = SeqModel(512, [[it("person", 3.0, 16, 16)]])                     # object at the window centre = the box centre (0.1, 0.1 of the frame)
    res, _ = D.Engine(m).recheck(img, win, "person")
    assert len(res) == 1
    s, a, b, c, e, _l = res[0]
    assert abs((a + c) / 2 - 0.1) < 0.02 and abs((b + e) / 2 - 0.1) < 0.02 and 0 <= a < c <= 1 and 0 <= b < e <= 1
    b = (0.0, 0.0, 0.2, 0.2)
    out, _ = SL.second_look(D.Engine(SeqModel(512, [[it("person", 3.0, 16, 16)]])), img, [det("person", 0.5, b)], CFG, {}, [], 1.0)
    assert out[0]["second_look"] == "confirmed"                          # centre of the mapped detection lies inside the original edge box


# ---------------- 0.3.2: a person/animal that would start or confirm a track needs a re-check that finds it again ----------------

def feed(sess, t, out):
    sess.update(t, [SL.tracker_view(d, CFG) for d in out if d.get("second_look") != "rejected"])


def test_cushion_person_first_074_recheck_015_never_confirms():
    eng = D.Engine(SeqModel(512, [[it("person", -1.7)]] * 8))                                # the re-check finds a person at p ~0.15
    sess = S.Session(CFG)
    for i in range(4):
        out, info = SL.second_look(eng, IMG, [det("person", 0.74)], CFG, {}, [], 100.0 + i * 20)       # no cache between frames: re-checked each time
        assert out[0]["second_look_doubt"] is True and out[0]["second_look"] in ("not_upgraded", "rejected") and out[0]["score"] <= 0.74
        feed(sess, T0 + i * 1000, out)
    assert not sess.snapshot()["person"]["present"] and sess.snapshot()["person"]["state"] == "idle"
    # the 0.44 -> 0.74 story with the reuse cache: the doubt found on the 0.44 frame is reused for the 0.74 frame of the same object
    eng2 = D.Engine(SeqModel(512, [[it("person", -1.7)]] * 8))
    sess = S.Session(CFG)
    out, info = SL.second_look(eng2, IMG, [det("person", 0.44)], CFG, {}, [], 100.0)
    assert "second_look_doubt" not in out[0]                                                   # 0.44 would not start a track: not mandatory, nothing to doubt yet
    feed(sess, T0, out)
    out2, info2 = SL.second_look(eng2, IMG, [det("person", 0.74)], CFG, {}, info["cache"], 102.0)
    assert info2["n"] == 0 and out2[0]["second_look_doubt"] is True and out2[0]["score"] == 0.74            # reused, shown unchanged in the reply
    assert SL.tracker_view(out2[0], CFG)["score"] == pytest.approx(0.25)                      # cont_thr 0.2 + 0.05
    feed(sess, T0 + 1000, out2)
    assert not sess.snapshot()["person"]["present"]


def test_real_person_recheck_found_confirms():
    eng = D.Engine(SeqModel(512, [[it("person", 0.85)]] * 8))                                 # re-check p ~0.70 (>= confirm_min 0.40)
    sess = S.Session(CFG)
    cache = []
    for i in range(2):
        out, info = SL.second_look(eng, IMG, [det("person", 0.8)], CFG, {}, cache, 100.0 + i)
        cache = info["cache"]
        assert "second_look_doubt" not in out[0] and out[0]["second_look"] == "lowered" and out[0]["score"] > 0.65
        feed(sess, T0 + i * 1000, out)
    assert sess.snapshot()["person"]["present"]


def test_confirmed_track_continues_with_doubtful_frames_and_needs_no_check():
    sess = S.Session(CFG)
    for i in range(2):
        sess.update(T0 + i * 1000, [{"cls": "person", "score": 0.9, "box": [0.4, 0.3, 0.5, 0.8]}])
    assert sess.snapshot()["person"]["present"]
    box = [0.4, 0.3, 0.5, 0.8]
    d = det("person", 0.9, tuple(box))
    assert not SL.mandatory(d, CFG, {"person_conf": [tuple(box)]}) and SL.mandatory(d, CFG, {"person_conf": []})
    doubtful = {**d, "second_look_doubt": True}
    for i in range(2, 6):
        sess.update(T0 + i * 1000, [SL.tracker_view(doubtful, CFG)])
    p = sess.snapshot()["person"]
    assert p["present"] and p["tracks"] == 1                                                    # the capped score still continues the confirmed track
    eng = D.Engine(SeqModel(512, [[]] * 3))
    out, info = SL.second_look(eng, IMG, [d], CFG, {"person_conf": [tuple(box)]}, [], 1.0)
    assert info["n"] == 0 and "second_look" not in out[0]


def test_vehicle_not_found_is_unchanged_and_animal_follows_the_person_rule():
    eng = D.Engine(SeqModel(512, [[]] * 4))
    out, _ = SL.second_look(eng, IMG, [det("car", 0.7)], CFG, {}, [], 1.0)
    assert out[0]["second_look"] == "not_found" and "second_look_doubt" not in out[0] and SL.tracker_view(out[0], CFG)["score"] == 0.7
    eng = D.Engine(SeqModel(512, [[it("dog", -1.7)]] * 4))
    out, _ = SL.second_look(eng, IMG, [det("dog", 0.8)], CFG, {}, [], 2.0)
    assert out[0]["second_look_doubt"] is True and SL.tracker_view(out[0], CFG)["score"] == pytest.approx(0.35)       # animal_cont_thr 0.3 + 0.05
    eng = D.Engine(SeqModel(512, [[it("dog", 1.5)]] * 4))
    out, _ = SL.second_look(eng, IMG, [det("dog", 0.8)], CFG, {}, [], 3.0)
    assert "second_look_doubt" not in out[0]


def test_cap_prevents_check_so_it_cannot_confirm_in_this_frame():
    eng = D.Engine(SeqModel(512, [[it("person", 2.0)]] * 8))
    dets = [det("person", 0.9, (0.0, 0.2, 0.1, 0.6)), det("person", 0.8, (0.3, 0.2, 0.4, 0.6)), det("person", 0.7, (0.6, 0.2, 0.7, 0.6)), det("car", 0.5, (0.8, 0.5, 0.95, 0.9))]
    out, info = SL.second_look(eng, IMG, dets, CFG, {}, [], 1.0)
    assert info["n"] == 2 and info["counts"]["sl_skipped"] == 1                                # mandatory persons come first; the car (band) gets no check
    skipped = [d for d in out if d.get("second_look") == "skipped"]
    assert len(skipped) == 1 and skipped[0]["second_look_doubt"] is True and skipped[0]["score"] == 0.9      # the one farthest above the confirm threshold is checked last
    assert "second_look" not in out[3]
    assert SL.tracker_view(skipped[0], CFG)["score"] == pytest.approx(0.25)
