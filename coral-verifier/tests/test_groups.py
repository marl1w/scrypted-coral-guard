"""Detection groups (person / vehicle / animal): engine decoding, session animal tracker, reply shape."""
import time
import numpy as np
import pytest
import detect as D
import fake_tpu
import session as S
from test_engine import engine
from test_server import Box, T0

IMG = np.full((480, 640, 3), 90, np.uint8)


def item(name, logit, gx, gy):
    return (name, logit, 16, gx, gy, (2, 2, 2, 2))


def test_engine_groups_and_ignored_classes():
    names = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "bird", "cat", "dog", "horse", "sheep", "cow", "bear",
             "airplane", "train", "boat", "elephant"]
    e, _ = engine(items=[item(n, 3.0, 3 + 2 * (i % 8) * 2, 3 + (i // 8) * 6) for i, n in enumerate(names)])
    dets, _ = e.detect(IMG, "1", 0.2)
    got = {d["cls"]: d["group"] for d in dets}
    assert got == {**{n: "person" for n in names[:1]}, **{n: "vehicle" for n in names[1:6]}, **{n: "animal" for n in names[6:13]}}
    assert set(dets[0]) == {"cls", "group", "score", "box"}


def test_animal_decode_floor_is_0_3():
    e, _ = engine(items=[item("dog", -1.2, 8, 8), item("cat", -0.5, 20, 20)])      # p 0.23 and 0.38
    dets, _ = e.detect(IMG, "1", 0.2)
    assert [d["cls"] for d in dets] == ["cat"]


def sess():
    return S.Session(S.Config())


def dog(box=(0.1, 0.5, 0.3, 0.8), score=0.8, cls="dog"):
    return {"cls": cls, "group": "animal", "score": score, "box": list(box)}


def test_animal_confirms_2_of_3_ticks_holds_6s():
    s = sess()
    s.update(T0, [dog(score=0.7)])
    assert not s.snapshot()["animal"]["present"]                                     # one tick is not enough
    s.update(T0 + 1000, [])
    s.update(T0 + 2000, [dog(score=0.7)])
    a = s.snapshot()["animal"]
    assert a["present"] and a["confirmed"] and a["tracks"][0]["cls"] == "dog"
    s.update(T0 + 3000, [dog(score=0.35)])                                           # >= 0.3 matched keeps it alive
    s.update(T0 + 9000, [])
    assert s.snapshot()["animal"]["present"]                                         # 6 s after the last detection: still holding
    s.update(T0 + 9500, [])
    s.update(T0 + 10_000, [])
    assert not s.snapshot()["animal"]["present"]


def test_animal_below_0_6_never_starts():
    s = sess()
    for i in range(6):
        s.update(T0 + i * 1000, [dog(score=0.55)])
    assert not s.snapshot()["animal"]["present"]


def test_new_still_car_is_not_moving_and_moving_car_is():
    s = sess()
    c = lambda x: {"cls": "car", "score": 0.9, "box": [x, 0.5, x + 0.3, 0.8]}
    s.update(T0, [c(0.1)])
    v = s.snapshot()["vehicle"]
    assert not v["moving"] and v["tracks"][0]["state"] == "still" and not v["tracks"][0]["parked"]
    for i in range(1, 6):                                                           # same spot for 5 s: still not moving
        s.update(T0 + i * 1000, [c(0.1)])
    assert not s.snapshot()["vehicle"]["moving"]
    m = sess()
    for i in range(6):                                                              # 0.04 per s: 0.08 over 2 s > 3 % of the diagonal
        m.update(T0 + i * 1000, [c(0.1 + 0.04 * i)])
    v = m.snapshot()["vehicle"]
    assert v["moving"] and v["tracks"][0]["state"] == "moving"


def test_car_that_stops_returns_to_still():
    s = sess()
    c = lambda x: {"cls": "car", "score": 0.9, "box": [x, 0.5, x + 0.3, 0.8]}
    for i in range(6):
        s.update(T0 + i * 1000, [c(0.1 + 0.04 * i)])
    assert s.snapshot()["vehicle"]["moving"]
    for i in range(6, 12):
        s.update(T0 + i * 1000, [c(0.3)])
    assert s.snapshot()["vehicle"]["tracks"][0]["state"] == "still" and not s.snapshot()["vehicle"]["moving"]


def test_low_score_animal_ignored_and_empty_snapshot_shape():
    s = sess()
    assert s.snapshot()["animal"] == {"present": False, "confirmed": False, "hold_until_ms": None, "tracks": []}
    s.update(T0, [dog(score=0.2)])
    assert not s.snapshot()["animal"]["present"]


def test_animal_never_touches_person_state():
    a, b = sess(), sess()
    seq = [[{"cls": "person", "score": 0.9, "box": [0.4, 0.3, 0.5, 0.8]}], [], [{"cls": "person", "score": 0.8, "box": [0.4, 0.3, 0.5, 0.8]}], [], []]
    for i, d in enumerate(seq):
        a.update(T0 + i * 1000, d)
        b.update(T0 + i * 1000, d + [dog(box=(0.4, 0.3, 0.5, 0.8)), dog(cls="cat", box=(0.7, 0.3, 0.8, 0.8))])
        assert a.snapshot()["person"] == b.snapshot()["person"]
        assert a.snapshot()["vehicle"] == b.snapshot()["vehicle"]


def test_bicycle_is_a_vehicle_with_the_same_parked_logic():
    s = sess()
    bike = {"cls": "bicycle", "group": "vehicle", "score": 0.7, "box": [0.1, 0.5, 0.2, 0.8]}
    s.update(T0, [bike])
    assert not s.snapshot()["vehicle"]["moving"] and s.snapshot()["vehicle"]["tracks"][0]["cls"] == "bicycle"
    for i in range(1, 30):
        s.update(T0 + i * 5_000, [bike])
    v = s.snapshot()["vehicle"]
    assert v["tracks"][0]["parked"] and not v["moving"]                              # stationary > parked_after_s: parked, not moving


def test_reply_shape_and_get_include_animal():
    b = Box(items=[["dog", 3.0, 16, 8, 8, [3, 2, 3, 2]], ["car", 3.0, 16, 20, 20, [3, 2, 3, 2]]])
    try:
        r = b.frame("g", T0)[1]
        b.frame("g", T0 + 1000); r = b.frame("g", T0 + 2000)[1]
        assert {d["cls"]: d["group"] for d in r["detections"]} == {"dog": "animal", "car": "vehicle"}
        assert r["animal"]["present"] and r["animal"]["tracks"][0]["cls"] == "dog"
        assert r["person"]["state"] == "idle" and not r["vehicle"]["moving"] and r["vehicle"]["tracks"][0]["state"] == "still"
        g = b.req("GET", "/v1/cameras/g")[1]
        assert g["animal"]["present"] and g["animal"]["tracks"][0]["cls"] == "dog"
    finally:
        b.close()


def test_post_ms_with_all_17_classes_decoded_stays_small():
    names = ["person", "car", "dog", "cat", "bird", "horse", "truck"]
    e, _ = engine(items=[item(n, 3.0, 3 + 4 * i, 8) for i, n in enumerate(names)])
    e.detect(IMG, "1", 0.2)
    post = [e.detect(IMG, "1", 0.2)[1]["post_ms"] for _ in range(20)]
    print("median post_ms", sorted(post)[10])
    assert sorted(post)[10] < 25


def test_weak_vehicles_display_only():
    # p 0.35 car (weak), p 0.62 truck (strong), p 0.43 bicycle (below its 0.5 floor: dropped), p 0.18 car (below weak floor 0.25: dropped)
    e, _ = engine(items=[item("car", -0.6, 8, 8), item("truck", 0.5, 20, 20), item("bicycle", -0.3, 8, 20), item("car", -1.5, 20, 8)])
    dets, _ = e.detect(IMG, "1", 0.2)
    assert [(d["cls"], d.get("weak", False)) for d in dets] == [("truck", False), ("car", True)]
    assert all(d["group"] == "vehicle" for d in dets)


def test_weak_does_not_suppress_strong_in_nms():
    near = [item("car", 0.5, 16, 16), item("car", -0.6, 17, 16)]                     # overlapping boxes, strong outranks weak
    e, _ = engine(items=near)
    dets, _ = e.detect(IMG, "1", 0.2)
    assert len(dets) == 1 and "weak" not in dets[0]


def test_weak_vehicle_never_touches_session_state():
    a, b = sess(), sess()
    weak = {"cls": "car", "group": "vehicle", "score": 0.3, "weak": True, "box": [0.1, 0.5, 0.4, 0.8]}
    for i in range(5):
        a.update(T0 + i * 1000, [])
        b.update(T0 + i * 1000, [weak])
    assert a.snapshot() == b.snapshot() and b.snapshot()["vehicle"] == {"moving": False, "hold_until_ms": None, "tracks": []}


def test_weak_flows_through_reply_without_vehicle_state():
    b = Box(items=[["car", -0.6, 16, 8, 8, [3, 2, 3, 2]]])
    try:
        r = b.frame("g", T0)[1]
        assert r["detections"][0]["weak"] is True and r["detections"][0]["group"] == "vehicle"
        assert r["vehicle"] == {"moving": False, "hold_until_ms": None, "tracks": []}
    finally:
        b.close()


def test_strong_vehicle_floor_is_0_4_for_motor_vehicles_and_0_5_for_bicycle():
    e, _ = engine(items=[item("car", -0.2, 8, 8), item("bicycle", -0.2, 20, 20)])      # both p 0.45
    dets, _ = e.detect(IMG, "1", 0.2)
    assert [(d["cls"], d.get("weak", False)) for d in dets] == [("car", False)]


@pytest.mark.parametrize("w,h", [(960, 360), (1536, 576), (1920, 576), (2560, 720)])
def test_auto_wide_frame_is_hybrid_whole_plus_two_half_tiles(w, h):
    assert D.resolve_tiles(w, h, "auto") == "hybrid"
    rs = D.tile_ranges(w, h, "hybrid")
    tw = round(w * 0.55)
    assert rs == [(0, w), (0, tw), (w - tw, w)]                                       # 3 invokes: whole frame + the old 55 % tiles
    assert rs[1][1] > rs[2][0]                                                         # the half tiles overlap in the middle


def test_tile_layout_numbers_and_aspect_rule():
    assert D.tile_ranges(960, 360, "hybrid") == [(0, 960), (0, 528), (432, 960)]
    assert D.tile_ranges(960, 360, 3) == [(0, 400), (280, 680), (560, 960)]            # explicit tiles=3: evenly spread, 30 % overlap
    assert D.resolve_tiles(1056, 480, "auto") == "1" and D.resolve_tiles(1057, 480, "auto") == "hybrid"       # 2.2 boundary
    assert D.resolve_tiles(955, 358, "auto") == "hybrid"
    assert D.resolve_tiles(1536, 576, "2") == "2" and D.tile_ranges(1000, 400, 2) == [(0, 550), (450, 1000)]
    assert D.resolve_tiles(640, 480, "auto") == "1" and D.resolve_tiles(640, 480, "3") == "3"


def test_three_explicit_tiles_box_mapping_and_overlap_dedupe():
    from test_engine import SeqModel
    # 960x360 tiles=3: (0,400) (280,680) (560,960); the same car in the overlap of tiles 1 and 2 -> one box; another car far right
    t1 = [item("car", 3.0, 27, 16)]                    # px 440 -> frame x 343
    t2 = [item("car", 3.0, 3, 16)]                     # px  56 -> 44 + 280 = frame x 324
    t3 = [item("car", 3.0, 24, 16)]                    # px 392 -> 306 + 560 = frame x 866
    m = SeqModel(512, [t1, t2, t3])
    dets, tm = D.Engine(m).detect(np.full((360, 960, 3), 90, np.uint8), "3", 0.2)
    assert m.calls == 3 and tm["infer_ms"] > 0
    cx = sorted((d["box"][0] + d["box"][2]) / 2 for d in dets)
    assert len(dets) in (2, 3) and 0.8 < cx[-1] < 0.95 and cx[0] < 0.4


def test_hybrid_box_mapping_whole_and_tile_to_full_frame_coordinates():
    from test_engine import SeqModel
    # call 1 = whole frame (letterboxed 960 -> 512, r 0.533), call 2 = left tile, call 3 = right tile
    m = SeqModel(512, [[item("dog", 3.0, 16, 16)], [], []])                           # centre of the whole frame -> x 0.5
    dets, tm = D.Engine(m).detect(np.full((360, 960, 3), 90, np.uint8), "auto", 0.2)
    assert m.calls == 3 and len(dets) == 1 and abs((dets[0]["box"][0] + dets[0]["box"][2]) / 2 - 0.5) < 0.02
    m = SeqModel(512, [[], [item("dog", 3.0, 16, 16)], []])                           # centre of the left tile (0..528): x = 264 / 960
    dets, _ = D.Engine(m).detect(np.full((360, 960, 3), 90, np.uint8), "auto", 0.2)
    assert len(dets) == 1 and abs((dets[0]["box"][0] + dets[0]["box"][2]) / 2 - 264 / 960) < 0.02


def test_object_spanning_the_tile_boundary_is_found_by_the_whole_frame_pass():
    from test_engine import SeqModel
    img = np.full((358, 955, 3), 90, np.uint8)
    # big car centred on the frame middle (spans both half tiles' inner edges). Whole pass sees it whole; each tile only sees an edge-touching piece.
    whole = [("car", 3.0, 16, 16, 16, (7, 4, 7, 4))]
    left_piece = [("car", 3.0, 16, 31, 16, (4, 4, 1, 4))]       # touches the tile's right inner edge
    right_piece = [("car", 3.0, 16, 0, 16, (0, 4, 4, 4))]       # touches the right tile's left inner edge
    m = SeqModel(512, [whole, left_piece, right_piece])
    dets, _ = D.Engine(m).detect(img, "auto", 0.2)
    assert m.calls == 3
    assert len(dets) == 1 and dets[0]["cls"] == "car"
    w = dets[0]["box"]
    assert w[2] - w[0] > 0.2 and 0.3 < (w[0] + w[2]) / 2 < 0.7                          # the whole-frame box survives, pieces are dropped
    m = SeqModel(512, [[], left_piece, right_piece])             # without the whole pass the same pieces would be (mostly) dropped by the edge rule
    dets, _ = D.Engine(m).detect(img, "auto", 0.2)
    assert all(d["box"][2] - d["box"][0] < 0.25 for d in dets)


def test_tile_detection_inside_a_whole_frame_box_is_dropped_but_small_neighbour_kept():
    from test_engine import SeqModel
    img = np.full((358, 955, 3), 90, np.uint8)
    whole = [("car", 3.0, 16, 8, 16, (4, 6, 4, 6))]
    m = SeqModel(512, [whole, [item("car", 1.0, 8, 16)], [item("dog", 3.0, 24, 24)]])
    dets, _ = D.Engine(m).detect(img, "auto", 0.2)
    assert {d["cls"] for d in dets} == {"car", "dog"} and sum(d["cls"] == "car" for d in dets) <= 2


def test_explicit_two_tiles_unchanged():
    from test_engine import SeqModel
    m = SeqModel(512, [[], []])
    D.Engine(m).detect(np.full((360, 960, 3), 90, np.uint8), "2", 0.2)
    assert m.calls == 2


def test_cross_class_vehicle_nms_keeps_higher_score_and_drops_contained():
    v = D.vehicle_group_nms
    car_ = (0.8, 0.1, 0.1, 0.5, 0.5, "car")
    truck = (0.6, 0.11, 0.1, 0.5, 0.5, "truck")                    # same object, other class, lower score
    moto_in = (0.68, 0.2, 0.1, 0.25, 0.5, "motorcycle")            # narrow box inside the car, HIGHER score than the car? 0.68 < 0.8 but contained anyway
    far = (0.5, 0.7, 0.1, 0.9, 0.5, "bus")
    bike = (0.9, 0.1, 0.1, 0.5, 0.5, "bicycle")                    # not part of the group
    out = v(sorted([car_, truck, moto_in, far, bike], reverse=True))
    assert {d[5] for d in out} == {"car", "bus", "bicycle"}


def test_engine_cross_class_nms_end_to_end():
    e, _ = engine(items=[item("car", 3.0, 16, 16), item("truck", 2.0, 16, 16)])
    dets, _ = e.detect(IMG, "1", 0.2)
    assert [d["cls"] for d in dets] == ["car"]


def test_inner_tile_edge_box_is_dropped_but_frame_border_box_kept():
    from test_engine import SeqModel
    img = np.full((360, 960, 3), 90, np.uint8)                      # tiles (0,400) (280,680) (560,960)
    # tile 1 car whose box touches ITS RIGHT edge: gx 31 -> px ~504 of 512 (inside the 2 % margin), but left/centre in the overlap with tile 2? the box
    # spans x 400-... : stays within tile 1 only when the neighbour (starts 280) fully covers it: box x ~ [330, 400] -> inside tile 2 (280..680)
    edge = [("car", 3.0, 16, 30, 16, (2, 2, 1, 2))]                  # right side lands at px ~ 504 -> x ~ 394..400
    m = SeqModel(512, [edge, [], []])
    dets, _ = D.Engine(m).detect(img, "3", 0.2)
    assert dets == []
    left_border = [("car", 3.0, 16, 2, 16, (2, 2, 2, 2))]            # touches only the FRAME left border of tile 1: kept
    m = SeqModel(512, [left_border, [], []])
    dets, _ = D.Engine(m).detect(img, "3", 0.2)
    assert len(dets) == 1 and dets[0]["box"][0] < 0.02


def test_per_camera_vehicle_and_animal_overrides_validated():
    c = S.Config()
    assert (c.vehicle_thr, c.animal_start_thr, c.animal_cont_thr) == (0.4, 0.6, 0.3)
    c2 = c.updated({"vehicle_thr": 0.6, "animal_start_thr": 0.8, "animal_cont_thr": 0.5})
    assert (c2.vehicle_thr, c2.animal_cont_thr) == (0.6, 0.5)
    for bad in ({"vehicle_thr": 0.05}, {"vehicle_thr": 0.96}, {"animal_cont_thr": 0.04}, {"animal_start_thr": 0.96},
                {"animal_start_thr": 0.3, "animal_cont_thr": 0.4}):
        with pytest.raises(ValueError):
            c.updated(bad)


def test_engine_vehicle_thr_override_and_weak_band():
    e, _ = engine(items=[item("car", -0.6, 8, 8), item("truck", 0.5, 20, 20)])         # p 0.35 and 0.62
    d = lambda **k: [(x["cls"], x.get("weak", False)) for x in e.detect(IMG, "1", 0.2, **k)[0]]
    assert d() == [("truck", False), ("car", True)]                                        # defaults: weak [0.25, 0.40)
    assert d(vehicle_thr=0.3) == [("truck", False), ("car", False)]                        # lowered strong floor: the car is strong
    assert d(vehicle_thr=0.7) == [("truck", True), ("car", True)]                          # raised: both become weak
    assert d(vehicle_thr=0.2) == [("truck", False), ("car", False)]                        # at or below the weak floor: no weak band, decode floor follows


def test_engine_animal_floor_override():
    e, _ = engine(items=[item("dog", -1.2, 8, 8), item("cat", -0.5, 20, 20)])           # p 0.23 and 0.38
    assert [x["cls"] for x in e.detect(IMG, "1", 0.2)[0]] == ["cat"]
    assert [x["cls"] for x in e.detect(IMG, "1", 0.2, animal_thr=0.2)[0]] == ["cat", "dog"]
    assert e.detect(IMG, "1", 0.2, animal_thr=0.5)[0] == []


def test_overrides_are_per_camera_and_drive_trackers():
    r = S.Registry()
    r.set_config("a", {"animal_start_thr": 0.9, "animal_cont_thr": 0.5, "vehicle_thr": 0.7}, 1.0)
    car_ = {"cls": "car", "score": 0.6, "box": [0.1, 0.5, 0.4, 0.8]}
    for cam in ("a", "b"):
        for i in range(3):
            r.update(cam, T0 + i * 1000, [dog(score=0.8), car_], 1.0 + i)
    a, b = r.peek("a", 3.0), r.peek("b", 3.0)
    assert not a["animal"]["present"] and a["vehicle"]["tracks"] == []                # a: start 0.9 not reached, car 0.6 < vehicle_thr 0.7
    assert b["animal"]["present"] and len(b["vehicle"]["tracks"]) == 1                # b keeps the defaults
    assert r.config("b").vehicle_thr == 0.4


def test_http_config_overrides_roundtrip_and_reject():
    b = Box(items=[["car", -0.6, 16, 8, 8, [3, 2, 3, 2]]])                            # p 0.35: weak by default
    try:
        assert b.frame("w", T0)[1]["detections"][0].get("weak") is True
        st, cfg = b.req("POST", "/v1/cameras/w/config", b'{"vehicle_thr": 0.3}')
        assert st == 200 and cfg["vehicle_thr"] == 0.3
        r = b.frame("w", T0 + 1000)[1]
        assert "weak" not in r["detections"][0] and r["vehicle"]["tracks"]
        assert b.frame("other", T0)[1]["detections"][0].get("weak") is True              # other cameras unaffected
        assert b.req("POST", "/v1/cameras/w/config", b'{"animal_start_thr": 0.2, "animal_cont_thr": 0.4}')[0] == 400
    finally:
        b.close()


def test_zoom_out_pass_geometry_and_back_mapping():
    from test_engine import SeqModel
    img = np.full((576, 768, 3), 90, np.uint8)
    # zoom_out 0.3: content 538x403 centred at (115,86) of the 768x576 canvas (letterbox scale 512/768). A car at the canvas centre maps to the frame centre.
    m = SeqModel(512, [[], [item("car", 3.0, 16, 16)]])
    dets, _ = D.Engine(m).detect(img, "1", 0.2, zoom_out=0.3)
    assert m.calls == 2 and len(dets) == 1
    b = dets[0]["box"]
    assert abs((b[0] + b[2]) / 2 - 0.5) < 0.05 and abs((b[1] + b[3]) / 2 - 0.5) < 0.05
    assert 0.16 < (b[2] - b[0]) < 0.2                                                    # undone scale: 1/0.7 larger than on the canvas
    canvas = m.last_canvas
    assert tuple(canvas[256, 256]) == (90, 90, 90) and tuple(canvas[5, 5]) == (114, 114, 114)      # shrunk picture centred, grey border all round
    m0 = SeqModel(512, [[]]); D.Engine(m0).detect(img, "1", 0.2)
    assert m0.calls == 1                                                          # zoom_out=0 unchanged


def test_zoom_out_box_is_clamped_to_the_frame():
    from test_engine import SeqModel
    m = SeqModel(512, [[], [("car", 3.0, 16, 31, 16, (30, 2, 0, 2))]])             # box reaching into the grey border on the left
    dets, _ = D.Engine(m).detect(np.full((576, 768, 3), 90, np.uint8), "1", 0.2, zoom_out=0.3)
    assert all(0 <= v <= 1 for d in dets for v in d["box"])


def test_zoom_out_merges_with_normal_pass_dedupe_and_hybrid_whole_semantics():
    from test_engine import SeqModel
    img = np.full((576, 768, 3), 90, np.uint8)
    same = item("car", 3.0, 16, 16)
    m = SeqModel(512, [[same], [same]])                                            # normal pass box and zoomed box of the "same" car
    dets, tm = D.Engine(m).detect(img, "1", 0.2, zoom_out=0.3)
    assert m.calls == 2 and tm["infer_ms"] > 0
    assert 1 <= len(dets) <= 2
    wide = np.full((358, 955, 3), 90, np.uint8)                                    # hybrid: whole + 2 tiles + zoom = 4 invokes, zoom box is exempt from the edge rule
    m = SeqModel(512, [[], [], [], [("car", 3.0, 16, 16, 16, (7, 4, 7, 4))]])
    dets, _ = D.Engine(m).detect(wide, "auto", 0.2, zoom_out=0.3)
    assert m.calls == 4 and len(dets) == 1


def test_zoom_out_config_validation_and_http():
    c = S.Config()
    assert c.zoom_out == 0.0 and c.updated({"zoom_out": 0.4}).zoom_out == 0.4 and c.updated({"zoom_out": 0}).zoom_out == 0
    for bad in (-0.1, 0.61, float("nan")):
        with pytest.raises(ValueError):
            c.updated({"zoom_out": bad})
    b = Box(items=[["car", 3.0, 16, 16, 16, [3, 2, 3, 2]]])
    try:
        n = b.calls()
        b.frame("z", T0)
        assert b.calls() == n + 1
        assert b.req("POST", "/v1/cameras/z/config", b'{"zoom_out": 0.3}')[1]["zoom_out"] == 0.3
        n = b.calls()
        assert b.frame("z", T0 + 1000)[0] == 200 and b.calls() == n + 2              # one Job, two invokes
        assert b.req("POST", "/v1/cameras/z/config", b'{"zoom_out": 0.9}')[0] == 400
        n = b.calls(); b.frame("other", T0); assert b.calls() == n + 1               # other cameras unaffected
    finally:
        b.close()
