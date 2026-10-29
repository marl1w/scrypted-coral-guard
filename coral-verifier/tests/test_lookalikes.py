"""Known look-alikes (0.3.31): a person/animal box the user labelled false_alarm is ignored where it stands, as long as it looks the same."""
import json

import cv2
import numpy as np

import lookalikes as LA
import session as S
import shape as SH
from boxlib import Box, BASE as T0
from test_shape import W, H

BOX = (0.4, 0.3, 0.55, 0.8)


def scene(person_in_front=False, seed=0):
    """Gray frame with a statue-like shape at BOX; optionally a person-sized blob in front of it; fresh noise every call."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 100, np.float32)
    x0, y0, x1, y1 = int(BOX[0] * W), int(BOX[1] * H), int(BOX[2] * W), int(BOX[3] * H)
    cv2.rectangle(img, (x0 + 20, y0 + 60), (x1 - 20, y1 - 5), 170, -1)          # pedestal
    cv2.circle(img, ((x0 + x1) // 2, y0 + 35), 22, 210, -1)                      # head
    cv2.line(img, (x0 + 5, y0 + 70), (x1 - 5, y0 + 70), 60, 6)                   # arms
    if person_in_front:
        cv2.ellipse(img, ((x0 + x1) // 2 + 10, (y0 + y1) // 2 + 20), (28, 70), 0, 0, 360, 40, -1)
    img = img + rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def person(box=BOX, score=0.9):
    return {"cls": "person", "score": score, "box": list(box)}


def known():
    la = LA.Lookalikes()
    la.add("ep1", "person", BOX, SH.signature(scene(seed=99), BOX))
    return la


def session(la=None, **kw):
    s = S.Session(S.Config().updated(kw))
    s.lookalikes = la if la is not None else known()
    return s


def run(s, n, **kw):
    for i in range(n):
        s.update(T0 + i * 1000, [person()], gray=scene(seed=i, **kw))
    return s.snapshot()


def test_the_known_look_alike_never_confirms():
    sn = run(session(), 6)
    assert sn["person"]["state"] == "idle" and sn["session"]["lookalikes"] == 6


def test_the_detection_is_marked_for_the_journal():
    d = person()
    session().update(T0, [d], gray=scene(seed=3))
    assert d["lookalike"] >= 0.85


def test_a_person_who_walks_in_front_of_it_counts():
    s = session()
    s.update(T0, [person((0.3, 0.3, 0.45, 0.8))], gray=scene(seed=1))                # arriving from the left
    for i in range(1, 5):
        s.update(T0 + i * 1000, [person()], gray=scene(seed=i, person_in_front=True))
    sn = s.snapshot()
    assert sn["person"]["present"] and sn["session"]["lookalikes"] == 0


def test_a_different_look_at_the_place_without_an_arrival_never_confirms():
    """Laundry in the wind: never the same look twice, never seen arriving (scored like laundry: below LOOKALIKE["strong"])."""
    s = session()
    for i in range(6):
        s.update(T0 + i * 1000, [person(score=0.7)], gray=scene(seed=i, person_in_front=True))
    sn = s.snapshot()
    assert sn["person"]["state"] == "idle" and sn["session"]["lookalikes"] == 6
    d = person(score=0.7)
    session().update(T0, [d], gray=scene(seed=3, person_in_front=True))
    assert d["lookalike_place"] is True and "lookalike" not in d


def test_an_arrival_is_too_old_after_entry_s_and_must_be_the_same_group_and_confident():
    s = session()
    s.update(T0, [person((0.3, 0.3, 0.45, 0.8))], gray=scene(seed=1))
    d = person(score=0.7)
    s.update(T0 + 16000, [d], gray=scene(seed=2, person_in_front=True))
    assert d.get("lookalike_place") is True                                          # 16 s later: no longer an arrival
    for arrival in ({"cls": "dog", "score": 0.9, "box": [0.3, 0.3, 0.45, 0.8]}, person((0.3, 0.3, 0.45, 0.8), score=0.5)):
        s = session()
        s.update(T0, [arrival], gray=scene(seed=1))
        d = person(score=0.7)
        s.update(T0 + 1000, [d], gray=scene(seed=2, person_in_front=True))
        assert d.get("lookalike_place") is True                                      # a dog or a weak box is not a person arriving
    s = session()
    s.update(T0, [person((0.85, 0.3, 0.99, 0.8))], gray=scene(seed=1))               # confident, but far from the place
    d = person(score=0.7)
    s.update(T0 + 1000, [d], gray=scene(seed=2, person_in_front=True))
    assert d.get("lookalike_place") is True


def test_the_place_rule_needs_no_frame():
    s = session()
    for i in range(4):
        s.update(T0 + i * 1000, [person(score=0.7)])
    assert s.snapshot()["person"]["state"] == "idle"


def test_a_strong_detection_is_never_left_out_for_its_place_alone():
    """0.3.44: a wrong Empty answer made a doorway a known place and hid a real person there at 0.89."""
    d = person(score=0.89)
    session().update(T0, [d], gray=scene(seed=2, person_in_front=True))
    assert "lookalike_place" not in d and "lookalike" not in d
    d = person(score=0.6) | {"second_look": "confirmed"}                             # "confirmed" only means the zoom scored as high: no strength
    session().update(T0, [d], gray=scene(seed=2, person_in_front=True))
    assert d["lookalike_place"] is True
    d = person(score=0.95)
    session().update(T0, [d], gray=scene(seed=3))                                    # the look still matches: left out
    assert d["lookalike"] >= 0.85


def test_a_look_only_entry_is_never_a_place():
    la = LA.Lookalikes()
    la.add("ep1", "person", BOX, SH.signature(scene(seed=99), BOX), look_only=True)
    d = person(score=0.5)
    session(la).update(T0, [d], gray=scene(seed=2, person_in_front=True))
    assert "lookalike_place" not in d
    d = person(score=0.5)
    session(la).update(T0, [d], gray=scene(seed=3))                                  # same look: the grill
    assert d["lookalike"] >= 0.85
    back = LA.Lookalikes.restore(json.loads(json.dumps(la.export())))
    assert back.items[0]["look_only"] and back.describe()[0]["look_only"] is True


def test_a_box_covering_most_of_the_frame_is_never_learnt():
    la = LA.Lookalikes()
    big = (0.0, 0.0, 1.0, 0.98)
    assert not la.add("ep1", "person", big, SH.signature(scene(), big))
    sig = SH.encode(SH.signature(scene(), BOX))
    back = LA.Lookalikes.restore([{"eid": "e1", "group": "person", "box": list(big), "sig": sig}, {"eid": "e2", "group": "person", "box": list(BOX), "sig": sig}])
    assert [e["eid"] for e in back.items] == ["e2"]                                  # already saved ones are dropped at start


def test_elsewhere_or_other_group_counts():
    s = session()
    other = (0.05, 0.3, 0.2, 0.8)
    for i in range(4):
        s.update(T0 + i * 1000, [person(other)], gray=scene(seed=i))
    assert s.snapshot()["person"]["present"]
    assert known().match("animal", BOX, scene()) is None


def test_switched_off_changes_nothing():
    assert run(session(lookalikes=False), 4)["person"]["present"]


def test_relabel_removes_and_export_restores():
    la = known()
    back = LA.Lookalikes.restore(json.loads(json.dumps(la.export())))
    assert back.match("person", BOX, scene(seed=5)) is not None
    assert la.remove("ep1") and not la.items and not la.remove("ep1")
    assert not la.add("ep2", "vehicle", BOX, SH.signature(scene(), BOX))        # vehicles are never learnt


def textured_jpeg():
    rng = np.random.default_rng(7)
    img = (rng.random((480, 640)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def test_a_false_alarm_label_teaches_it_and_relabelling_withdraws_it():
    b = Box(items=[["person", 3.0, 16, 16, 16, [2, 2, 2, 2]]])
    try:
        JS = {"Content-Type": "application/json"}
        eid = "look-ep-0001"
        data = textured_jpeg()
        b.req("POST", "/v1/cameras/door/episodes", json.dumps({"id": eid, "action": "start", "t_ms": T0, "mode": "gate", "kind": "camera", "groups": ["person"]}).encode(), JS)
        for i in range(2):
            assert b.frame("door", T0 + i * 1000, data=data)[0] == 200
        b.app.journal.drain()
        b.req("POST", "/v1/cameras/door/episodes", json.dumps({"id": eid, "action": "end", "t_ms": T0 + 3000, "outcome": "confirmed"}).encode(), JS)
        assert b.req("POST", f"/v1/journal/{eid}/label", json.dumps({"label": "false_alarm"}).encode(), JS)[0] == 200
        assert b.app.learn_lookalike(b.app._learn_q.get(timeout=5))                  # what the learner thread does
        st, r = b.req("GET", "/v1/cameras/door/lookalikes")
        assert st == 200 and r["lookalikes"][0]["episode"] == eid
        b.req("POST", "/v1/cameras/door/reset")
        st, r = b.frame("door", T0 + 10_000, data=data)
        assert st == 200 and r["session"]["lookalikes"] == 1 and r["person"]["state"] == "idle"
        b.app.save_lookalikes()
        b2 = Box(opts={"lookalikes_root": b.app.looks_dir})                          # a restart reads it back
        try:
            assert b2.app.registry.lookalikes_describe("door")[0]["episode"] == eid
        finally:
            b2.close()
        assert b.req("POST", f"/v1/journal/{eid}/label", json.dumps({"label": "correct"}).encode(), JS)[0] == 200
        b.app.learn_lookalike(b.app._learn_q.get(timeout=5))
        assert b.req("GET", "/v1/cameras/door/lookalikes")[1]["lookalikes"] == []
        assert b.app.backfill_lookalikes() == 0                                      # nothing labelled false_alarm any more
        st, r = b.req("POST", "/v1/cameras/door/lookalikes", b"{}", JS)
        assert st == 200 and r["forgotten"] is False
    finally:
        b.close()


def test_marks_on_a_doubtful_copy_reach_the_journal_detection():
    import server as SV
    doubt = {"cls": "person", "score": 0.7, "box": [0.4, 0.3, 0.55, 0.8], "second_look_doubt": True}
    plain = {"cls": "person", "score": 0.9, "box": [0.1, 0.3, 0.2, 0.8]}
    live = [{**doubt, "score": 0.25}, plain]
    s = session()
    s.update(T0, live, gray=scene(seed=4, person_in_front=True))
    SV.marks_back([doubt, plain], live)
    assert doubt["lookalike_place"] is True and doubt["score"] == 0.7 and "lookalike_place" not in plain


def test_a_wrong_class_answer_still_teaches_the_groups_that_were_not_there():
    import server as SV
    W = SV.App.wrong_groups
    assert W({"label": "false_alarm", "groups": ["person", "vehicle"]}) == ["person"]
    assert W({"label": "wrong_class", "truth": "animal", "groups": ["person"]}) == ["person"]                # a bird called a person
    assert W({"label": "wrong_class", "label_note": "is: animal - bird", "groups": ["person", "animal"]}) == ["person"]
    assert W({"label": "wrong_class", "truth": "vehicle", "groups": ["animal"]}) == ["animal"]
    assert W({"label": "correct", "groups": ["person"]}) == [] and W({"label": "missed", "groups": ["person"]}) == []
    assert W({"label": "wrong_class", "truth": "person+vehicle", "groups": ["person", "animal"]}) == ["animal"]   # a real person is never learnt
    b = Box(items=[["person", 3.0, 16, 16, 16, [2, 2, 2, 2]]])
    try:
        JS = {"Content-Type": "application/json"}
        eid = "look-ep-0002"
        b.req("POST", "/v1/cameras/door/episodes", json.dumps({"id": eid, "action": "start", "t_ms": T0, "mode": "gate", "kind": "camera", "groups": ["person"]}).encode(), JS)
        for i in range(2):
            assert b.frame("door", T0 + i * 1000, data=textured_jpeg())[0] == 200
        b.app.journal.drain()
        b.req("POST", "/v1/cameras/door/episodes", json.dumps({"id": eid, "action": "end", "t_ms": T0 + 3000, "outcome": "confirmed"}).encode(), JS)
        st, r = b.req("POST", f"/v1/journal/{eid}/label", json.dumps({"truth": "animal"}).encode(), JS)
        assert st == 200 and r["label"] == "wrong_class"
        assert b.app.learn_lookalike(b.app._learn_q.get(timeout=5))
        assert b.req("GET", "/v1/cameras/door/lookalikes")[1]["lookalikes"][0]["episode"] == eid
        b.app.registry.lookalike_remove("door", eid)
        assert b.app.backfill_lookalikes() == 1                                     # earlier wrong-class labels are learnt at start too
    finally:
        b.close()


def test_a_blocked_episode_answered_empty_teaches_its_look_only():
    import server as SV
    E = SV.App.empty_groups
    v = {"outcome": "vetoed", "kind": "camera", "label": "correct"}
    assert E(v) == ["person", "animal"] and E(v | {"truth": "empty"}) == ["person", "animal"]
    assert E(v | {"truth": "vehicle"}) == ["person", "animal"]                                 # a parked car alone: its person box is not a person
    assert E(v | {"auto": {"ref": "empty"}}) == [] and E(v | {"label": "missed"}) == []
    assert E(v | {"kind": "unflagged"}) == [] and E(v | {"outcome": "released"}) == []
    b = Box(items=[["person", 3.0, 16, 16, 16, [2, 2, 2, 2]]])
    try:
        JS = {"Content-Type": "application/json"}
        eid = "look-ep-0003"
        b.req("POST", "/v1/cameras/door/episodes", json.dumps({"id": eid, "action": "start", "t_ms": T0, "mode": "gate", "kind": "camera"}).encode(), JS)
        for i in range(2):
            assert b.frame("door", T0 + i * 1000, data=textured_jpeg())[0] == 200
        b.app.journal.drain()
        b.req("POST", "/v1/cameras/door/episodes", json.dumps({"id": eid, "action": "end", "t_ms": T0 + 3000, "outcome": "vetoed",
                                                               "timeline_event": {"t_ms": T0 + 3000, "text": "vetoed | B would confirm +900 ms"}}).encode(), JS)
        st, r = b.req("POST", f"/v1/journal/{eid}/label", json.dumps({"truth": "empty"}).encode(), JS)
        assert st == 200 and r["label"] == "correct"
        assert b.app.learn_lookalike(b.app._learn_q.get(timeout=5))
        assert b.req("GET", "/v1/cameras/door/lookalikes")[1]["lookalikes"] == [{"episode": eid, "group": "person", "box": b.req("GET", "/v1/cameras/door/lookalikes")[1]["lookalikes"][0]["box"], "look_only": True}]
        b.app.registry.lookalike_remove("door", eid)
        assert b.app.backfill_lookalikes() == 1                                      # earlier Empty answers are learnt at start too
        assert b.req("POST", f"/v1/journal/{eid}/label", json.dumps({"truth": "person"}).encode(), JS)[0] == 200
        b.app.learn_lookalike(b.app._learn_q.get(timeout=5))
        assert b.req("GET", "/v1/cameras/door/lookalikes")[1]["lookalikes"] == []    # answered Person: withdrawn
    finally:
        b.close()


def test_walking_past_a_look_only_entry_is_an_arrival():
    """The sofa (look only) next to the laundry place: someone walking in past the sofa still arrives at the laundry place."""
    la = known()
    sofa = (0.25, 0.3, 0.4, 0.8)
    la.add("sofa", "person", sofa, SH.signature(scene(seed=98), sofa), look_only=True)
    s = session(la)
    s.update(T0, [person(sofa, score=0.7)], gray=scene(seed=1, person_in_front=True))
    d = person(score=0.7)
    s.update(T0 + 1000, [d], gray=scene(seed=2, person_in_front=True))
    assert "lookalike_place" not in d


def test_look_only_entries_are_capped_apart():
    from defaults import LOOKALIKE
    la = known()
    sig = SH.signature(scene(seed=99), BOX)
    for i in range(LOOKALIKE["max_look_only"] + 5):
        la.add(f"e{i}", "person", BOX, sig, look_only=True)
    assert la.known("ep1")                                                          # the false alarm's entry survives the Empty answers
    assert sum(e["look_only"] for e in la.items) == LOOKALIKE["max_look_only"] and not la.known("e0")
    back = LA.Lookalikes.restore(json.loads(json.dumps(la.export())))
    assert not back.dirty and len(back.items) == len(la.items)
    big = (0.0, 0.0, 1.0, 0.98)
    assert LA.Lookalikes.restore(la.export() + [{"eid": "x", "group": "person", "box": list(big), "sig": SH.encode(sig)}]).dirty


def test_a_look_only_object_is_never_an_arrival():
    """A grill learnt look-only, scored above start_thr next to the laundry place: it is the grill, not someone arriving."""
    la = known()
    grill = (0.25, 0.3, 0.4, 0.8)
    la.add("grill", "person", grill, SH.signature(scene(seed=98), grill), look_only=True)
    s = session(la)
    for i in range(4):
        g, d = person(grill, score=0.7), person(score=0.7)
        s.update(T0 + i * 1000, [g, d], gray=scene(seed=98, person_in_front=True))
        assert g.get("lookalike") is not None and d.get("lookalike_place") is True
    assert s.snapshot()["person"]["state"] == "idle"
