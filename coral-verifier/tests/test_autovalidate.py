"""Auto-validation of repeat empty frames: same scene as a user-validated empty episode (clock ignored) -> labelled
correct without asking; a new subject, a Coral box, or no reference -> left for the user. Counted apart."""
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


def episode(j, clk, eid, frames, dets=()):
    ep(j, "cam", eid, "start", int(clk.t), mode="gate", kind="camera", groups=["person"])
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
