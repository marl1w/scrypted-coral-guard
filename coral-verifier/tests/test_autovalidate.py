"""Auto-validation of repeat empty frames: same scene as a user-validated empty episode (clock ignored) -> labelled
correct without asking; a new subject, a Coral box, or no reference -> left for the user. Counted apart."""
import json
import time
import cv2
import numpy as np
from journal import Journal
from test_journal import Clock, det, ep


def scene(clock="12:00:00", person=False, night=False):
    rng = np.random.default_rng(7)
    img = np.full((576, 1024, 3), (90, 140, 70), np.uint8)
    cv2.rectangle(img, (0, 300), (1024, 576), (150, 150, 150), -1)
    cv2.rectangle(img, (600, 120), (800, 330), (40, 40, 40), -1)
    img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))
    if person:
        cv2.rectangle(img, (300, 380), (330, 470), (200, 30, 200), -1)          # a small subject, far from the camera
    if night:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.putText(img, "15/10/2026 " + clock, (300, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def episode(j, clk, eid, frames, dets=(), **k):
    clk.t += 10_000                                                          # never adopts the previous episode's frames (adopt_s)
    ep(j, "cam", eid, "start", int(clk.t), mode="gate", kind="camera", groups=["person"], **k)
    for i, f in enumerate(frames):
        clk.t += 1000
        j.record_frame("cam", int(clk.t), f, list(dets))
    clk.t += 1000
    ep(j, "cam", eid, "end", int(clk.t), outcome="vetoed")


def settled(j, eid):
    for _ in range(200):
        m = j.get(eid)
        if not m.get("auto_pending"):
            return m
        time.sleep(0.02)
    raise AssertionError("auto-validation never answered")


def test_repeat_empty_scene_is_auto_validated_and_counted_apart(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk, auto_validate=True)
    episode(j, clk, "ref-000001", [scene("12:00:00"), scene("12:00:05")])
    assert settled(j, "ref-000001")["label"] is None                      # nothing to compare with yet: the user is asked
    assert j.next_unlabelled()["item"]["id"] == "ref-000001"
    assert j.set_label("ref-000001", "correct", None)[0] == 200             # the user says: empty
    for _ in range(100):
        if j.auto.stats()["references"]:
            break
        time.sleep(0.02)

    episode(j, clk, "same-00001", [scene("18:42:17"), scene("18:42:21")])  # same scene, only the clock differs
    m = settled(j, "same-00001")
    assert m["label"] == "correct" and m["auto"]["ref"] == "ref-000001" and m["label_note"].startswith("auto:")
    assert j.next_unlabelled()["item"] is None

    episode(j, clk, "subj-00001", [scene("18:50:00"), scene("18:50:04", person=True)])
    assert settled(j, "subj-00001")["label"] is None                      # a new subject in one frame: asked

    episode(j, clk, "box-000001", [scene("19:00:00"), scene("19:00:03")], dets=[det("person", 0.3)])
    assert settled(j, "box-000001")["label"] is None                      # Coral saw something, however weak: asked

    episode(j, clk, "night-0001", [scene("23:00:00", night=True)])
    assert settled(j, "night-0001")["label"] is None                      # infrared frames never match a daylight reference

    t = j.summary()["totals"]
    assert t["auto_validated"] == 1 and t["reviewed"] == 1 and t["correct"] == 1 and t["reviewable"] == 3

    assert j.set_label("same-00001", "missed", "someone")[0] == 200        # the user's answer replaces the automatic one
    m = j.get("same-00001")
    assert m["label"] == "missed" and "auto" not in m
    assert j.summary()["totals"]["auto_validated"] == 0


def test_a_camera_without_vehicles_needs_only_empty_frames(tmp_path):
    """0.3.44: light changes never look the same twice; on a person/animal camera empty frames are enough, with no reference."""
    clk = Clock(); j = Journal(str(tmp_path), clock=clk, auto_validate=True)
    pa = {"detect": ["person", "animal"]}
    episode(j, clk, "lamp-00001", [scene("19:00:00"), scene("19:00:04", night=True)], **pa)
    m = settled(j, "lamp-00001")
    assert m["label"] == "correct" and m["auto"]["ref"] == "empty" and m["label_note"] == "auto: no person or animal in the frames"

    episode(j, clk, "grill-0001", [scene("19:10:00")], dets=[det("vehicle", 0.5, cls="car")], **pa)
    assert settled(j, "grill-0001")["label"] == "correct"                # a car box on a camera that does not detect cars: empty

    episode(j, clk, "known-0001", [scene("19:20:00")], dets=[det("person", 0.6, lookalike=0.9)], **pa)
    assert settled(j, "known-0001")["label"] == "correct"                # a known look-alike (the grill learnt by its look): empty

    episode(j, clk, "pers-00001", [scene("19:30:00")], dets=[det("person", 0.3)], **pa)
    assert settled(j, "pers-00001")["label"] is None                     # a person box, however weak: asked

    episode(j, clk, "place-0001", [scene("19:40:00")], dets=[det("person", 0.6, lookalike_place=True)], **pa)
    assert settled(j, "place-0001")["label"] is None                     # left out only for its place: maybe someone there, asked

    episode(j, clk, "rej-000001", [scene("19:50:00")], dets=[det("person", 0.6, second_look="rejected")], **pa)
    assert settled(j, "rej-000001")["label"] == "correct"                # rejected by the second look: empty

    episode(j, clk, "veh-000001", [scene("20:00:00")], detect=["person", "vehicle"])
    assert settled(j, "veh-000001")["label"] is None                     # a vehicle camera keeps the scene match (no reference yet)


def test_the_best_frame_is_never_picked_for_a_box_coral_left_out(tmp_path):
    """A grill scored as a car (not detected by this camera) at 0.95 must not hide the weaker person of another frame: the person's
    frame is stored as best, so the episode is asked, never auto-validated as empty."""
    clk = Clock(); j = Journal(str(tmp_path), clock=clk, auto_validate=True)
    pa = {"detect": ["person", "animal"]}
    grill = lambda s: det("vehicle", s, cls="car", not_detected=True)
    ep(j, "cam", "hide-00001", "start", int(clk.t), mode="gate", kind="camera", **pa)
    for i, dets in enumerate(([grill(0.9)], [grill(0.5), det("person", 0.45)], [grill(0.95)], [det("person", 0.6, lookalike=0.9)])):
        clk.t += 1000
        j.record_frame("cam", int(clk.t), scene("20:00:0%d" % i), dets)
    clk.t += 1000
    ep(j, "cam", "hide-00001", "end", int(clk.t), outcome="vetoed")
    m = settled(j, "hide-00001")
    assert m["label"] is None
    with open(j.file("hide-00001", "best", "json")) as f:
        assert any(d["cls"] == "person" and d["score"] == 0.45 for d in json.load(f)["detections"])
