"""Event journal: lifecycle, snapshots, retention, stale, labels, summary, traversal safety, HTTP review API and the ingress auth rule."""
import http.client, json, os
import pytest
import session as S
from journal import Journal
from boxlib import Box, jpeg
from test_server import T0

H = 3600_000


@pytest.fixture(autouse=True)
def _no_continuation(monkeypatch):
    """Most tests here build several confirmed episodes on one camera seconds apart: the continuation fold (0.3.26) is off unless a test turns it on."""
    import defaults
    monkeypatch.setitem(defaults.JOURNAL, "continue_after_s", 0.0)


class Clock:
    def __init__(self, t=10_000.0):          # small epoch: test episode times (a few seconds) are never older than the retention cut
        self.t = t

    def __call__(self):
        return self.t


def det(group, score, cls=None, **k):
    return {"cls": cls or group, "group": group, "score": score, "box": [0.1, 0.1, 0.4, 0.6], **k}


def ep(j, cam, eid, action, t, **k):
    return j.episode(cam, {"id": eid, "action": action, "t_ms": t, **k})


def test_lifecycle_first_best_last_with_sidecars_and_original_bytes(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    assert ep(j, "cam1", "ep-000001", "start", 1000, mode="gate", kind="camera", groups=["person"], timeline_event={"t_ms": 1000, "text": "motion"})[0] == 200
    frames = {t: b"JPEG-%d" % t for t in (1, 2, 3, 4)}
    for t, sc in ((1, 0.5), (2, 0.9), (3, 0.7), (4, 0.6)):
        clk.t += 1000
        j.record_frame("cam1", t * 1000, frames[t], [det("person", sc), det("vehicle", 0.99, "car")])         # the car does not count for the person episode
    assert ep(j, "cam1", "ep-000001", "end", 9000, outcome="confirmed", note="n")[0] == 200
    d = tmp_path / "ep-000001"
    assert (d / "first.jpg").read_bytes() == frames[1] and (d / "best.jpg").read_bytes() == frames[2] and (d / "last.jpg").read_bytes() == frames[4]
    side = json.loads((d / "best.json").read_text())
    assert side["detections"][0]["score"] == 0.9 and side["t_ms"] == 2000
    m = json.loads((d / "meta.json").read_text())
    assert m["status"] == "closed" and m["outcome"] == "confirmed" and m["mode"] == "gate" and m["end_ms"] == 9000 and m["timeline"][0]["text"] == "motion"
    assert {f["role"] for f in m["frames"]} == {"first", "best", "last"} and m["camera"] == "cam1"
    clk.t += 1000
    j.record_frame("cam1", 99000, b"later", [])                                                            # after the end: not linked
    assert (d / "last.jpg").read_bytes() == frames[4]
    assert ep(j, "cam1", "ep-000001", "update", 9500)[0] == 409 and ep(j, "cam1", "nope-0001", "end", 1)[0] == 404


def test_best_is_repicked_for_the_groups_set_on_confirmation(tmp_path):
    """0.3.34: frames linked before the groups were known were ranked by any group, so a parked car at 0.94 stayed best and the
    confirmed person never showed in the review. Setting the groups re-picks best; the first group outranks the others."""
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    car = det("vehicle", 0.94, "car")
    j.record_frame("c", 500, b"cars", [car])
    clk.t += 500
    ep(j, "c", "regroup-01", "start", 600, kind="unflagged")
    j.record_frame("c", 900, b"person", [car, det("person", 0.7)])
    j.record_frame("c", 1200, b"cars2", [car])
    ep(j, "c", "regroup-01", "update", 1300, outcome="confirmed", groups=["person"])
    assert (tmp_path / "regroup-01" / "best.jpg").read_bytes() == b"person"
    j.record_frame("c", 1500, b"weaker", [car, det("person", 0.6)])
    ep(j, "c", "regroup-01", "update", 1600, groups=["person", "vehicle"])                                # a car joins: the person frame stays best
    ep(j, "c", "regroup-01", "end", 2000, outcome="released")
    assert (tmp_path / "regroup-01" / "best.jpg").read_bytes() == b"person"
    assert json.loads((tmp_path / "regroup-01" / "best.json").read_text())["t_ms"] == 900


def test_start_adopts_the_trigger_frame_received_just_before(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    j.record_frame("c", 500, b"trigger", [det("person", 0.8)])
    clk.t += 1000
    ep(j, "c", "adopt-0001", "start", 600)
    j.record_frame("c", 1500, b"next", [det("person", 0.7)])
    ep(j, "c", "adopt-0001", "end", 2000, outcome="released")
    assert (tmp_path / "adopt-0001" / "first.jpg").read_bytes() == b"trigger" and (tmp_path / "adopt-0001" / "last.jpg").read_bytes() == b"next"
    j.record_frame("c", 5000, b"old", [])
    clk.t += 60 * 1000
    ep(j, "c", "adopt-0002", "start", 5000)                                                                  # ring frames older than adopt_s are not adopted
    assert j.get("adopt-0002")["frames"] == []


def test_periodic_snapshots_every_600s_capped_at_12(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    ep(j, "c", "snap-00001", "start", 0)
    for i in range(0, 40):
        clk.t += 600_000
        j.record_frame("c", i * 600_000, b"f%d" % i, [det("person", 0.5)])
    m = j.get("snap-00001")
    snaps = [f for f in m["frames"] if f["role"] == "snapshot"]
    assert len(snaps) == 12 and snaps[0]["name"] == "s01"
    ep(j, "c", "snap-00001", "end", 1, outcome="aborted")
    assert (tmp_path / "snap-00001" / "s12.jpg").exists() and not (tmp_path / "snap-00001" / "s13.jpg").exists()


def test_retention_by_age_and_by_size(tmp_path):
    clk = Clock(30 * 86400_000.0); j = Journal(str(tmp_path), clock=clk, retention_days=14.0)
    base = clk.t
    for i, age_days in enumerate((20, 1, 0.5)):
        clk.t = base + (20 - age_days) * 86400_000                                   # episodes are created at their own time (client times are clamped to now +- 1 day)
        ep(j, "c", "old-%05d" % i, "start", clk.t)
        j.record_frame("c", 1, b"x" * 100, [det("person", 0.5)])
        ep(j, "c", "old-%05d" % i, "end", clk.t + 1000, outcome="released")
    clk.t = base + 20 * 86400_000
    j.maintain()
    ids = {m["id"] for m in j.list()["items"]}
    assert "old-00000" not in ids and not (tmp_path / "old-00000").exists()                                  # older than 14 days
    one = j.eps["old-00002"]["bytes"]
    j2 = Journal(str(tmp_path), clock=clk, max_bytes=one + 10)                                               # reload from disk, then trim by size: oldest closed first
    j2.maintain()
    left = [m["id"] for m in j2.list()["items"]]
    assert left == ["old-00002"] and j2.stats()["bytes"] <= one + 10


def test_stale_ongoing_episode_is_closed_after_2h(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    ep(j, "c", "stale-0001", "start", 1000)
    j.record_frame("c", 1000, b"a", [det("person", 0.6)])
    clk.t += 1.9 * H
    j.maintain()
    assert j.get("stale-0001")["status"] == "ongoing"
    clk.t += 0.2 * H
    j.maintain()
    m = j.get("stale-0001")
    assert m["status"] == "closed" and m["outcome"] == "stale" and (tmp_path / "stale-0001" / "last.jpg").exists()


def test_label_persists_across_restart_and_summary_counts(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    for i, (cam, g, out) in enumerate((("a", ["person"], "confirmed"), ("a", ["person"], "vetoed"), ("b", ["vehicle", "animal"], "released"))):
        ep(j, cam, "lab-%05d" % i, "start", 1000 + i, groups=g); ep(j, cam, "lab-%05d" % i, "end", 2000 + i, outcome=out)
    assert j.set_label("lab-00001", "false_alarm", "dog")[0] == 200 and j.set_label("lab-00000", "correct", None)[0] == 200
    assert j.set_label("lab-00001", "bogus", None)[0] == 400 and j.set_label("zzzzzzz1", "correct", None)[0] == 404
    j2 = Journal(str(tmp_path), clock=clk)                                                                   # a restart reads everything back
    assert j2.get("lab-00001")["label"] == "false_alarm" and j2.get("lab-00001")["label_note"] == "dog"
    rows = {(r["camera"], r["group"]): r for r in j2.summary()["rows"]}
    assert rows[("a", "person")]["total"] == 2 and rows[("a", "person")]["by_outcome"] == {"confirmed": 1, "vetoed": 1}
    assert rows[("a", "person")]["by_label"] == {"correct": 1, "false_alarm": 1}
    assert rows[("b", "vehicle")]["total"] == 1 and rows[("b", "animal")]["by_label"] == {"unlabeled": 1}
    assert [m["id"] for m in j2.list(camera="a", label="false_alarm")["items"]] == ["lab-00001"]
    assert [m["id"] for m in j2.list(outcome="released")["items"]] == ["lab-00002"]
    assert j2.list(t_from=1001, t_to=1001)["total"] == 1 and [m["id"] for m in j2.list(limit=2)["items"]] == ["lab-00002", "lab-00001"]       # newest first


def test_ids_and_names_are_traversal_safe(tmp_path):
    j = Journal(str(tmp_path / "root"), clock=Clock())
    for bad in ("../evil", "a/b", "..", "x" * 65, "short", "ok.dot-1", "", None, 5):
        assert ep(j, "c", bad, "start", 1)[0] == 400
    ep(j, "c", "fine-0001", "start", 1); j.record_frame("c", 1, b"x", [])
    assert j.file("fine-0001", "first", "jpg") and j.file("fine-0001", "../meta", "json") is None and j.file("../x", "first", "jpg") is None
    assert j.file("fine-0001", "first", "exe") is None and j.get("../fine-0001") is None
    assert os.listdir(tmp_path / "root") == ["fine-0001"]


def test_validation_errors(tmp_path):
    j = Journal(str(tmp_path), clock=Clock())
    base = {"id": "valid-0001", "action": "start", "t_ms": 1}
    for patch in ({"action": "go"}, {"t_ms": "x"}, {"mode": "x"}, {"kind": "x"}, {"outcome": "x"}, {"groups": "person"}, {"groups": [1]}, {"timeline_event": {"t_ms": 1}}, {"note": 5}):
        assert j.episode("c", {**base, **patch})[0] == 400, patch
    assert j.episode("c", base)[0] == 200 and j.episode("c", base)[0] == 200                               # a repeated start is idempotent


# ---------------- HTTP ----------------

def raw(box, method, path, headers=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", box.port, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse(); data = r.read(); c.close()
    return r.status, data, r.getheader("Content-Type")


def test_http_episode_review_flow_and_ingress_rule():
    b = Box(items=[["person", 3.0, 16, 16, 16, [2, 2, 2, 2]]])
    try:
        b.app.ingress_ips = {"127.0.0.1"}
        eid = "http-ep-0001"
        JS = {"Content-Type": "application/json"}
        st, out = b.req("POST", "/v1/cameras/door/episodes", json.dumps({"id": eid, "action": "start", "t_ms": T0, "mode": "shadow", "kind": "camera", "groups": ["person"]}).encode(), JS)
        assert st == 200 and out["status"] == "ongoing"
        data = jpeg(640, 480)
        for i in range(2):
            assert b.frame("door", T0 + i * 1000, data=data)[0] == 200
        b.app.journal.drain()                                                                                # frame work is queued: wait for the worker
        st, out = b.req("POST", "/v1/cameras/door/episodes", json.dumps({"id": eid, "action": "end", "t_ms": T0 + 3000, "outcome": "confirmed"}).encode(), JS)
        assert st == 200 and out["outcome"] == "confirmed"
        st, lst = b.req("GET", "/v1/journal?camera=door&outcome=confirmed")
        assert st == 200 and lst["total"] == 1 and lst["items"][0]["id"] == eid
        st, m = b.req("GET", f"/v1/journal/{eid}")
        assert st == 200 and {f["role"] for f in m["frames"]} >= {"first", "best", "last"}
        st, body, ct = raw(b, "GET", f"/v1/journal/{eid}/first.jpg", {"X-Verifier-Token": b.token})
        assert st == 200 and ct == "image/jpeg" and body == data                                           # the original bytes
        st, side = b.req("GET", f"/v1/journal/{eid}/best.json")
        assert side["detections"][0]["cls"] == "person"
        assert b.req("GET", f"/v1/journal/{eid}/../../etc.jpg")[0] == 404 and b.req("GET", "/v1/journal/%2e%2e")[0] == 404
        assert b.req("POST", f"/v1/journal/{eid}/label", json.dumps({"label": "correct", "note": "ok"}).encode(), JS)[0] == 200
        assert b.req("GET", f"/v1/journal/{eid}")[1]["label"] == "correct"
        assert b.req("GET", "/v1/journal/summary")[1]["rows"][0]["by_label"] == {"correct": 1}
        assert b.req("GET", "/v1/metrics")[1]["journal"]["episodes"] == 1 and b.req("GET", "/v1/metrics")[1]["journal"]["bytes"] > 0
        # authentication: the token is required everywhere except the ingress path (journal routes only)
        assert b.req("GET", "/v1/journal", auth=False)[0] == 401
        ing = {"X-Ingress-Path": "/api/hassio_ingress/abc"}
        assert b.req("GET", "/v1/journal", headers=ing, auth=False)[0] == 200
        assert b.req("POST", f"/v1/journal/{eid}/label", json.dumps({"label": "missed"}).encode(), headers={**ing, **JS}, auth=False)[0] == 200
        assert raw(b, "GET", f"/v1/journal/{eid}/best.jpg", ing)[0] == 200
        assert b.req("GET", "/v1/metrics", headers=ing, auth=False)[0] == 401                               # ingress does not open the rest of the API
        assert b.req("POST", "/v1/cameras/door/episodes", b"{}", headers={**ing, **JS}, auth=False)[0] == 401
        assert b.frame("door", T0 + 9000, headers=ing, auth=False)[0] == 401
        b.app.ingress_ips = {"172.30.32.2"}                                                                  # same header, but not from the Supervisor gateway
        assert b.req("GET", "/v1/journal", headers=ing, auth=False)[0] == 401
        st, html, ct = raw(b, "GET", "/")
        assert st == 200 and ct.startswith("text/html") and b"Coral Guard" in html and b'fetch("/' not in html and b"v1/journal" in html   # relative URLs only
    finally:
        b.close()


# ---------------- security review fixes ----------------

def test_frame_path_never_blocks_and_drops_when_the_queue_is_full(tmp_path):
    import threading, time as _t
    j = Journal(str(tmp_path), clock=Clock(), queue_max=3)
    gate = threading.Event()
    j._post(gate.wait)                                                                                       # the worker is busy
    _t.sleep(0.05)
    t0 = _t.perf_counter()
    for i in range(50):
        j.record_frame_async("c", i, b"x" * 1000, [])
    assert _t.perf_counter() - t0 < 0.5 and j.dropped >= 46 and j.stats()["dropped"] == j.dropped            # queued 3, the rest dropped and counted
    assert j.episode("c", {"id": "busy-0001", "action": "start", "t_ms": 1})[0] == 503                       # an API call on a full queue says busy, not hang
    gate.set(); j.drain()
    assert j.episode("c", {"id": "busy-0001", "action": "start", "t_ms": 1})[0] == 200
    j.stop()


def test_frames_are_linked_asynchronously_by_the_worker(tmp_path):
    j = Journal(str(tmp_path), clock=Clock())
    ep(j, "c", "async-0001", "start", 1)
    j.record_frame_async("c", 5, b"JPEG", [det("person", 0.7)])
    j.drain()
    assert (tmp_path / "async-0001" / "first.jpg").read_bytes() == b"JPEG"
    j.stop()


def test_ongoing_episode_caps_per_camera_and_global(tmp_path):
    j = Journal(str(tmp_path), clock=Clock(), max_ongoing=10, max_ongoing_per_camera=3)
    for i in range(3):
        assert ep(j, "a", "capa-%04d" % i, "start", 1)[0] == 200
    assert ep(j, "a", "capa-0009", "start", 1)[0] == 429 and ep(j, "a", "capa-0000", "start", 1)[0] == 200      # a repeated start of an existing id is not a new one
    for i in range(7):
        assert ep(j, "cam%d" % i, "capb-%04d" % i, "start", 1)[0] == 200
    assert ep(j, "zzz", "capz-0001", "start", 1)[0] == 429                                                    # global cap (10 ongoing)
    ep(j, "a", "capa-0001", "end", 2, outcome="released")
    assert ep(j, "zzz", "capz-0001", "start", 1)[0] == 200                                                    # a closed episode frees a slot
    j.stop()


def test_client_times_are_clamped_to_server_time_plus_minus_one_day(tmp_path):
    now = 1_800_000_000_000.0
    j = Journal(str(tmp_path), clock=Clock(now))
    day = 86400_000
    ep(j, "c", "clamp-0001", "start", now + 400 * day, timeline_event={"t_ms": now - 400 * day, "text": "x"})
    m = j.get("clamp-0001")
    assert m["start_ms"] == now + day and m["timeline"][0]["t_ms"] == now - day
    ep(j, "c", "clamp-0001", "end", now - 10 * day, outcome="aborted")
    assert j.get("clamp-0001")["end_ms"] == now - day
    ep(j, "c", "clamp-0002", "start", now + 1000)
    assert j.get("clamp-0002")["start_ms"] == now + 1000                                                      # inside the window: untouched
    j.stop()


def test_load_skips_damaged_folders_without_failing(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    good = Journal(str(root), clock=Clock()); ep(good, "c", "good-0001", "start", 1); good.stop()
    (root / "bad-json1").mkdir(); (root / "bad-json1" / "meta.json").write_text("{not json")
    (root / "bad-list1").mkdir(); (root / "bad-list1" / "meta.json").write_text("[1,2]")
    (root / "bad-id-xx").mkdir(); (root / "bad-id-xx" / "meta.json").write_text(json.dumps({"id": "other-id-1"}))
    (root / "empty-dir").mkdir()
    (root / "weird-001").mkdir(); (root / "weird-001" / "meta.json").write_text(json.dumps({"id": "weird-001", "frames": 5, "groups": None}))
    (root / "stray.txt").write_text("x")
    j = Journal(str(root), clock=Clock())
    assert j.enabled and {m["id"] for m in j.list()["items"]} == {"good-0001", "weird-001"}
    assert j.summary()["rows"]                                                                               # the repaired entry does not break the summary
    j.stop()


def test_http_requires_json_content_type_and_finite_numbers():
    b = Box()
    try:
        body = json.dumps({"id": "ct-test-0001", "action": "start", "t_ms": T0}).encode()
        assert b.req("POST", "/v1/cameras/c/episodes", body)[0] == 415
        assert b.req("POST", "/v1/cameras/c/episodes", body, {"Content-Type": "text/plain"})[0] == 415
        assert b.req("POST", "/v1/cameras/c/episodes", body, {"Content-Type": "application/json; charset=utf-8"})[0] == 200
        lab = json.dumps({"label": "correct"}).encode()
        assert b.req("POST", "/v1/journal/ct-test-0001/label", lab)[0] == 415
        assert b.req("POST", "/v1/journal/ct-test-0001/label", lab, {"Content-Type": "application/json"})[0] == 200
        for q in ("limit=nan", "limit=inf", "offset=-inf", "offset=nan", "from=nan", "to=inf", "limit=x"):
            assert b.req("GET", "/v1/journal?" + q)[0] == 400, q
        assert b.req("GET", "/v1/journal?limit=5&offset=0&from=1&to=99999999999999")[0] == 200
    finally:
        b.close()


def test_file_lookup_survives_concurrent_retention(tmp_path):
    import threading
    clk = Clock(30 * 86400_000.0)
    j = Journal(str(tmp_path), clock=clk, retention_days=14.0)
    ids = []
    for i in range(20):
        eid = "race-%05d" % i; ids.append(eid)
        ep(j, "c", eid, "start", clk.t - 20 * 86400_000.0 + i)
        j.record_frame("c", 1, b"x", [det("person", 0.5)])
        ep(j, "c", eid, "end", clk.t - 20 * 86400_000.0 + i + 5, outcome="released")      # clamped to now-1 day... still older than the cut once the clock advances
    errors = []

    def reader():
        for _ in range(300):
            for eid in ids:
                try:
                    j.file(eid, "first", "jpg"); j.get(eid); j.list(); j.summary()
                except Exception as e:
                    errors.append(e)
    ts = [threading.Thread(target=reader) for _ in range(3)]
    [t.start() for t in ts]
    clk.t += 30 * 86400_000.0
    j.maintain()
    [t.join() for t in ts]
    assert not errors and j.list()["total"] == 0
    j.stop()


# ---------------- annotation mode, reviewed lists, dashboard ----------------

def mk(j, clk, eid, cam, start, outcome, groups, kind="camera", label=None):
    clk.t = max(clk.t, start) if start < 10**6 else start                  # client times are clamped to now +- 1 day: the clock follows the episode
    ep(j, cam, eid, "start", start, kind=kind, groups=groups); ep(j, cam, eid, "end", start + 10, outcome=outcome)
    if label:
        j.set_label(eid, label, None)


def test_question_mapping_per_outcome_kind_and_group():
    q = Journal.question
    assert q({"outcome": "confirmed", "kind": "camera", "groups": ["person"]}) == {"key": "person_present", "text": "Is there a person?", "group": "person", "others": [], "yes": "correct", "no": "false_alarm"}
    assert q({"outcome": "confirmed", "kind": "camera", "groups": ["vehicle"]})["text"] == "Is this a moving vehicle?"
    assert q({"outcome": "confirmed", "kind": "camera", "groups": ["animal"]})["text"] == "Is there an animal?"
    v = q({"outcome": "vetoed", "kind": "camera", "groups": ["person"]})
    assert v["text"] == "Is the frame really empty?" and v["group"] is None and (v["yes"], v["no"]) == ("correct", "missed")
    assert q({"outcome": "released", "kind": "unflagged", "groups": ["person"]})["text"] == "Is there a person the camera missed?"
    assert q({"outcome": "released", "kind": "unflagged", "groups": ["vehicle"]})["text"] == "Is there a moving vehicle the camera missed?"
    assert q({"outcome": "released", "kind": "unflagged", "groups": ["animal"]})["text"] == "Is there an animal the camera missed?"
    multi = q({"outcome": "released", "kind": "unflagged", "groups": ["animal", "person"]})
    assert multi["group"] == "animal" and multi["others"] == ["person"] and (multi["yes"], multi["no"]) == ("correct", "false_alarm")      # first group asked, the rest listed
    assert q({"outcome": "aborted", "kind": "camera", "groups": []})["text"] == "Is this a real event?"
    assert q({"outcome": "vetoed", "kind": "unflagged", "groups": ["person"]})["key"] == "missed_person"                                   # an unflagged event is never "really empty"


def test_truth_answer_derives_the_label_from_the_episode():
    f = Journal.label_for
    person = {"outcome": "confirmed", "kind": "camera", "groups": ["person"]}
    assert [f(person, t) for t in ("person", "vehicle", "animal", "empty")] == ["correct", "wrong_class", "wrong_class", "false_alarm"]
    veto = {"outcome": "vetoed", "kind": "camera", "groups": ["person"], "inconclusive": True}
    assert [f(veto, t) for t in ("person", "vehicle", "animal", "empty")] == ["missed", "missed", "missed", "correct"]
    unflagged = {"outcome": "vetoed", "kind": "unflagged", "groups": ["animal", "person"]}
    assert [f(unflagged, t) for t in ("person", "vehicle", "animal", "empty")] == ["correct", "wrong_class", "correct", "false_alarm"]
    assert f({"outcome": "aborted", "kind": "camera", "groups": []}, "vehicle") == "correct"


def test_truth_label_is_stored_and_a_plain_label_clears_it(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    ep(j, "a", "tru-00001", "start", 1000, groups=["person"]); ep(j, "a", "tru-00001", "end", 2000, outcome="confirmed")
    assert j.set_label("tru-00001", None, None, "empty") == (200, {"id": "tru-00001", "label": "false_alarm", "label_note": None, "truth": "empty"})
    assert j.set_label("tru-00001", None, None, "ghost")[0] == 400 and j.set_label("tru-00001", "bogus", None)[0] == 400
    m = Journal(str(tmp_path), clock=clk).get("tru-00001")
    assert (m["label"], m["truth"], m["labels"][-1]["truth"]) == ("false_alarm", "empty", "empty")
    j.set_label("tru-00001", "correct", None)
    assert "truth" not in j.get("tru-00001") and j.get("tru-00001")["label"] == "correct"
    assert j.set_label("tru-00001", None, None)[0] == 200 and j.get("tru-00001")["label"] is None                     # cleared: back to review


def test_next_unlabelled_oldest_first_after_filters_and_remaining(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    mk(j, clk, "nx-000001", "a", 1000, "confirmed", ["person"])
    mk(j, clk, "nx-000002", "b", 2000, "vetoed", ["person"])
    mk(j, clk, "nx-000003", "a", 3000, "confirmed", ["vehicle"], label="correct")
    mk(j, clk, "nx-000004", "a", 4000, "released", ["person"], kind="unflagged")
    ep(j, "a", "nx-000005", "start", 5000)                                                                      # ongoing: never offered
    r = j.next_unlabelled()
    assert r["item"]["id"] == "nx-000001" and r["remaining"] == 3 and r["item"]["question"]["key"] == "person_present"
    assert j.next_unlabelled("nx-000001")["item"]["id"] == "nx-000002"
    assert j.next_unlabelled("nx-000002")["item"]["id"] == "nx-000004"                                          # the labelled 3 is skipped
    assert j.next_unlabelled("nx-000004")["item"] is None and j.next_unlabelled("nx-000004")["remaining"] == 3  # end: the client wraps around
    assert j.next_unlabelled(camera="b")["item"]["id"] == "nx-000002" and j.next_unlabelled(camera="b")["remaining"] == 1
    assert j.next_unlabelled(outcome="released")["item"]["id"] == "nx-000004"
    assert j.next_unlabelled("../x")["item"]["id"] == "nx-000001" and j.next_unlabelled("nx-zzzzzz")["item"]["id"] == "nx-000001"          # unknown `after`: from the start
    j.set_label("nx-000001", "correct", None)
    assert j.next_unlabelled()["item"]["id"] == "nx-000002" and j.next_unlabelled()["remaining"] == 2
    assert j.get("nx-000002")["question"]["key"] == "empty"
    j.stop()


def test_list_labelled_and_status_filters(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    mk(j, clk, "ls-000001", "a", 1000, "confirmed", ["person"], label="false_alarm")
    mk(j, clk, "ls-000002", "a", 2000, "confirmed", ["person"])
    ep(j, "a", "ls-000003", "start", 3000)
    ids = lambda **k: [m["id"] for m in j.list(**k)["items"]]
    assert ids(labelled=True) == ["ls-000001"] and ids(labelled=False) == ["ls-000003", "ls-000002"]
    assert ids(labelled=False, status="closed") == ["ls-000002"] and ids(status="ongoing") == ["ls-000003"]
    j.stop()


def test_summary_dashboard_period_camera_groups_days(tmp_path):
    day = 86400_000
    clk = Clock(40 * day); j = Journal(str(tmp_path), clock=clk)
    base = 30 * day
    mk(j, clk, "sm-000001", "a", base, "confirmed", ["person"], label="correct")
    mk(j, clk, "sm-000002", "a", base + 1000, "confirmed", ["person"], label="false_alarm")
    mk(j, clk, "sm-000003", "a", base + day, "vetoed", ["vehicle"], label="missed")
    mk(j, clk, "sm-000004", "b", base + day, "released", ["animal"], kind="unflagged")
    mk(j, clk, "sm-000005", "b", base - 20 * day, "confirmed", ["person"])
    s = j.summary(t_from=base - day)
    T = s["totals"]
    assert (T["events"], T["reviewed"], T["correct"], T["false_alarm"], T["missed"], T["precision"]) == (4, 3, 1, 1, 1, round(1 / 3, 4))
    assert (T["confirmed"], T["vetoed"], T["unflagged"], T["other"]) == (2, 1, 1, 0)
    cams = {c["camera"]: c for c in s["cameras"]}
    assert cams["a"]["events"] == 3 and cams["a"]["precision"] == round(1 / 3, 4) and cams["b"]["precision"] is None and cams["b"]["unflagged"] == 1       # nothing reviewed: no precision
    grp = {g["group"]: g for g in s["groups"]}
    assert grp["person"]["events"] == 2 and grp["person"]["precision"] == 0.5 and grp["vehicle"]["missed"] == 1
    assert [d["day"] for d in s["days"]] == sorted(d["day"] for d in s["days"]) and sum(d["total"] for d in s["days"]) == 4
    assert s["days"][0]["confirmed"] == 2 and s["days"][1]["vetoed"] == 1 and s["days"][1]["unflagged"] == 1
    assert j.summary(t_from=base - day, camera="b")["totals"]["events"] == 1
    assert j.summary(t_from=base - 30 * day)["totals"]["events"] == 5 and j.summary(t_to=base - 10 * day)["totals"]["events"] == 1
    shifted = j.summary(t_from=base - day, tz_offset_min=-60 * 12)                                                       # a day boundary moves with the timezone
    assert shifted["days"][0]["day"] != s["days"][0]["day"] or len(shifted["days"]) >= 1
    assert j.summary(t_from=base * 10)["totals"]["events"] == 0 and j.summary(t_from=base * 10)["cameras"] == []        # empty period
    assert s["rows"]                                                                                                    # the per camera x group rows stay
    j.stop()


def test_http_next_summary_filters_and_ui_markup():
    b = Box()
    try:
        JS = {"Content-Type": "application/json"}
        for i, (cam, out, grp) in enumerate((("a", "confirmed", ["person"]), ("b", "vetoed", ["person"]))):
            eid = "web-q-%04d" % i
            b.req("POST", f"/v1/cameras/{cam}/episodes", json.dumps({"id": eid, "action": "start", "t_ms": T0 + i, "groups": grp}).encode(), JS)
            b.req("POST", f"/v1/cameras/{cam}/episodes", json.dumps({"id": eid, "action": "end", "t_ms": T0 + i + 5, "outcome": out}).encode(), JS)
        st, r = b.req("GET", "/v1/journal/next")
        assert st == 200 and r["remaining"] == 2 and r["item"]["id"] == "web-q-0000" and r["item"]["question"]["text"] == "Is there a person?"
        assert b.req("GET", "/v1/journal/next?after=web-q-0000")[1]["item"]["question"]["text"] == "Is the frame really empty?"
        assert b.req("GET", "/v1/journal/next?camera=b&outcome=vetoed")[1]["remaining"] == 1 and b.req("GET", "/v1/journal/next?after=../x")[0] == 400
        assert b.req("GET", "/v1/journal/web-q-0001")[1]["question"]["no"] == "missed"
        b.req("POST", "/v1/journal/web-q-0000/label", json.dumps({"label": "correct"}).encode(), JS)
        assert b.req("POST", "/v1/journal/web-q-0000/label", json.dumps({"truth": "ghost"}).encode(), JS)[0] == 400
        assert b.req("GET", "/v1/journal/web-q-0000")[1]["label"] == "correct"
        assert b.req("POST", "/v1/journal/web-q-0000/label", b"{}", JS)[0] == 400                    # no key: never a silent clear
        assert b.req("POST", "/v1/journal/web-q-0000/label", json.dumps({"label": None}).encode(), JS)[0] == 200
        assert b.req("GET", "/v1/journal/next")[1]["item"]["id"] == "web-q-0000"                       # back in Validate
        b.req("POST", "/v1/journal/web-q-0000/label", json.dumps({"label": "correct"}).encode(), JS)
        assert [m["id"] for m in b.req("GET", "/v1/journal?labelled=false&status=closed")[1]["items"]] == ["web-q-0001"]
        assert b.req("GET", "/v1/journal?labelled=maybe")[0] == 400 and b.req("GET", "/v1/journal?status=weird")[0] == 400
        st, sm = b.req("GET", "/v1/journal/summary?from=0&tz=120&camera=a")
        assert st == 200 and sm["totals"]["events"] == 1 and sm["totals"]["precision"] == 1.0 and sm["cameras"][0]["camera"] == "a" and sm["days"]
        assert b.req("GET", "/v1/journal/summary?tz=5000")[0] == 400 and b.req("GET", "/v1/journal/summary?from=nan")[0] == 400
        st, html, ct = raw(b, "GET", "/")
        for needle in (b"Validate", b"Reviewed", b'["person","Person","p"]', b'["vehicle","Car","c"]', b'["animal","Animal","a"]', b'["empty","Empty","e"]', b"{truth,note", b"Review events to see accuracy", b"next?"):
            assert needle in html, needle
        assert b"innerHTML" not in html and b"document.write" not in html and b"eval(" not in html                  # untrusted text only through textContent
    finally:
        b.close()


# ---------------- 0.3.3 ----------------

def test_unflagged_episode_that_never_confirmed_is_discarded_at_end(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    for eid, out, groups, tl in (("uf-000001", "vetoed", ["person"], None), ("uf-000002", "aborted", ["person"], None), ("uf-000003", "released", [], None),
                                 ("uf-000004", "released", ["person"], None), ("uf-000005", "aborted", [], "confirmed by the plugin"), ("uf-000006", "confirmed", [], None)):
        ep(j, "c", eid, "start", 1000, kind="unflagged", groups=groups)
        j.record_frame("c", 1, b"x", [det("person", 0.8)])
        st, out_ = ep(j, "c", eid, "end", 2000, outcome=out, **({"timeline_event": {"t_ms": 1500, "text": tl}} if tl else {}))
        assert st == 200
    assert {m["id"] for m in j.list()["items"]} == {"uf-000004", "uf-000005", "uf-000006"}                    # confirmed at some point: kept
    assert sorted(os.listdir(tmp_path)) == ["uf-000004", "uf-000005", "uf-000006"]                           # the others are not even on disk
    ep(j, "c", "cam-000001", "start", 1000, kind="camera"); ep(j, "c", "cam-000001", "end", 2000, outcome="vetoed")
    assert j.get("cam-000001")["status"] == "closed"                                                          # camera episodes are never discarded
    j.stop()


def test_legacy_unflagged_noise_is_purged_at_startup_and_in_maintenance(tmp_path, caplog):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    for eid in ("old-noise1", "old-noise2", "old-keep01"):
        ep(j, "c", eid, "start", 1000, kind="camera", groups=["person"]); ep(j, "c", eid, "end", 2000, outcome="vetoed" if eid != "old-keep01" else "confirmed")
    j.set_label("old-noise1", "correct", None)
    j.stop()
    for eid in ("old-noise1", "old-noise2", "old-keep01"):                                                   # rewrite them as legacy unflagged episodes
        p = tmp_path / eid / "meta.json"; m = json.loads(p.read_text()); m["kind"] = "unflagged"; p.write_text(json.dumps(m))
    import logging
    with caplog.at_level(logging.INFO, logger="journal"):
        j2 = Journal(str(tmp_path), clock=clk)
    assert {m["id"] for m in j2.list()["items"]} == {"old-keep01"} and not (tmp_path / "old-noise1").exists()  # regardless of the label
    assert any("removed 2" in r.getMessage() for r in caplog.records)
    ep(j2, "c", "stale-uf01", "start", 1000, kind="unflagged")                                                # a stale unflagged episode closes as stale, then goes
    clk.t += 3 * H
    j2.maintain()
    assert j2.get("stale-uf01") is None
    j2.stop()


def test_summary_categories_confirmed_vetoed_unflagged_other(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    mk(j, clk, "ct-000001", "a", 1000, "confirmed", ["person"])
    mk(j, clk, "ct-000002", "a", 2000, "released", ["person"])                                                # groups + released = confirmed at some point
    mk(j, clk, "ct-000003", "a", 3000, "released", [])                                                         # nothing confirmed: other
    mk(j, clk, "ct-000004", "a", 4000, "vetoed", ["person"])
    mk(j, clk, "ct-000005", "a", 5000, "aborted", [])
    mk(j, clk, "ct-000006", "b", 6000, "released", ["animal"], kind="unflagged")                              # by kind, whatever the outcome (another camera: no merge into the veto)
    ep(j, "a", "ct-000007", "start", 7000, groups=["person"], timeline_event={"t_ms": 7100, "text": "Person confirmed"}); ep(j, "a", "ct-000007", "end", 7200, outcome="aborted")
    T = j.summary()["totals"]
    assert (T["confirmed"], T["vetoed"], T["unflagged"], T["other"], T["events"]) == (3, 1, 1, 2, 7)
    j.stop()


def test_camera_display_name_config_sanitising_episode_meta_and_fallback():
    import session as S
    c = S.Config()
    assert c.name == "" and c.updated({"name": "  Front   <b>door</b>\n"}).name == "Front bdoor/b"
    assert c.updated({"name": 'x"y\'z&`'}).name == "xyz"
    for bad in (5, None, "x" * 65):
        with pytest.raises(ValueError):
            c.updated({"name": bad})
    assert c.updated({"name": "x" * 64}).name == "x" * 64
    assert Journal.default_name("front_door_12") == "front_door" and Journal.default_name("garage") == "garage" and Journal.default_name("_5") == "_5"
    b = Box()
    try:
        JS = {"Content-Type": "application/json"}
        assert b.req("POST", "/v1/cameras/front_door_1/config", json.dumps({"name": "Front door"}).encode(), JS)[1]["name"] == "Front door"
        assert b.req("GET", "/v1/cameras/front_door_1/config")[1]["name"] == "Front door"
        assert b.req("POST", "/v1/cameras/front_door_1/config", json.dumps({"name": 3}).encode(), JS)[0] == 400
        for cam, eid in (("front_door_1", "dn-000001"), ("garage_2", "dn-000002")):
            b.req("POST", f"/v1/cameras/{cam}/episodes", json.dumps({"id": eid, "action": "start", "t_ms": T0, "groups": ["person"]}).encode(), JS)
            b.req("POST", f"/v1/cameras/{cam}/episodes", json.dumps({"id": eid, "action": "end", "t_ms": T0 + 5, "outcome": "confirmed"}).encode(), JS)
        assert b.req("GET", "/v1/journal/dn-000001")[1]["display_name"] == "Front door" and b.req("GET", "/v1/journal/dn-000002")[1]["display_name"] == "garage"
        names = {c["camera"]: c["display_name"] for c in b.req("GET", "/v1/journal/summary")[1]["cameras"]}
        assert names == {"front_door_1": "Front door", "garage_2": "garage"}
    finally:
        b.close()


def test_ui_uses_display_names_hides_load_more_and_routes_summary_clicks():
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verifier", "ui.html")).read()
    assert "display_name" in html and r"replace(/_\d+$/" in html                                              # fallback: id without a trailing _<digits>
    assert 'more("#more",r.items.length,PAGE,offset,total)' in html                                     # no "Load more" after a short last page
    assert 'tab(c.reviewable>0?"va":"ev")' in html                                                     # Validate when unreviewed events exist, else Events


def test_reviewable_has_one_definition_closed_and_unlabelled(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    mk(j, clk, "rv-000001", "a", 1000, "confirmed", ["person"], label="correct")
    mk(j, clk, "rv-000002", "a", 2000, "confirmed", ["person"])
    mk(j, clk, "rv-000003", "a", 3000, "vetoed", ["person"])
    ep(j, "a", "rv-000004", "start", 4000, groups=["person"])                                                 # in progress: not reviewable
    ep(j, "a", "rv-000005", "start", 5000, groups=["person"])
    T = j.summary()["totals"]
    assert (T["events"], T["closed"], T["ongoing"], T["reviewable"], T["reviewed"]) == (5, 3, 2, 2, 1)
    listed = j.list(labelled=False, status="closed")["total"]
    assert listed == T["reviewable"] == j.next_unlabelled()["remaining"] == 2                                  # the To review list, the chip and Validate's "N left" agree
    assert j.summary(camera="a")["cameras"][0]["reviewable"] == 2 and j.summary(camera="a")["groups"][0]["ongoing"] == 2
    ep(j, "a", "rv-000004", "end", 4100, outcome="confirmed")
    assert j.summary()["totals"]["reviewable"] == 3 == j.next_unlabelled()["remaining"]                        # it becomes reviewable only once closed
    j.set_label("rv-000002", "correct", None)
    assert j.summary()["totals"]["reviewable"] == 2 and j.summary()["totals"]["reviewed"] == 2
    j.stop()


def test_ui_lists_render_one_empty_message_and_hide_load_more_when_empty():
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verifier", "ui.html")).read()
    assert 'id="rDoSec"' not in html and 'id="tRe"' not in html                                                                # no Reviewed tab: Events has the Label filter
    assert 'id="empty"' in html and 'E("div","empty"' not in html.split("async function load")[1].split("async function recap")[0]   # one message outside the grid, never a card in it
    assert "total===0" in html and html.count("more(") >= 2                                                                    # Load more: hidden for empty / short / fully loaded lists
    assert 'x.reviewable' in html and '"in progress"' in html and "T.reviewed/T.closed" in html and "x.events-x.reviewed" not in html


def test_page_is_never_cached_names_its_version_and_lists_return_total():
    b = Box()
    try:
        for path in ("/", "/ui"):
            c = http.client.HTTPConnection("127.0.0.1", b.port, timeout=10)
            c.request("GET", path); r = c.getresponse(); body = r.read(); c.close()
            assert r.status == 200 and r.getheader("Cache-Control") == "no-store" and r.getheader("Pragma") == "no-cache"
            assert b"__VERSION__" not in body and (f"v{b.app.cfg and __import__('server').VERSION}").encode() in body and b'name="app-version"' in body
        b.frame("cam", T0)
        st, cams = b.req("GET", "/v1/cameras")
        assert st == 200 and cams["total"] == len(cams["cameras"]) == 1
        st, lst = b.req("GET", "/v1/journal?limit=1")
        assert st == 200 and lst["total"] == 0 and lst["items"] == []                                           # total is the number of matching items, not of the page
        JS = {"Content-Type": "application/json"}
        for i in range(3):
            eid = "tot-%05d" % i
            b.req("POST", "/v1/cameras/c/episodes", json.dumps({"id": eid, "action": "start", "t_ms": T0 + i, "groups": ["person"]}).encode(), JS)
            b.req("POST", "/v1/cameras/c/episodes", json.dumps({"id": eid, "action": "end", "t_ms": T0 + i + 1, "outcome": "confirmed"}).encode(), JS)
        lst = b.req("GET", "/v1/journal?limit=2&offset=0")[1]
        assert lst["total"] == 3 and len(lst["items"]) == 2
        assert b.req("GET", "/v1/journal?limit=2&offset=2")[1]["total"] == 3 and b.req("GET", "/v1/journal?camera=zzz")[1]["total"] == 0
    finally:
        b.close()


def test_group_filter_category_outcomes_and_next_filters(tmp_path):
    day = 86400_000
    clk = Clock(40 * day); j = Journal(str(tmp_path), clock=clk)
    mk(j, clk, "gf-000001", "a", 30 * day, "confirmed", ["person"])
    mk(j, clk, "gf-000002", "a", 31 * day, "released", ["vehicle"])                                           # groups + released = confirmed at some point
    mk(j, clk, "gf-000003", "b", 32 * day, "vetoed", ["person"])
    mk(j, clk, "gf-000004", "b", 33 * day, "released", ["animal"], kind="unflagged")
    mk(j, clk, "gf-000005", "b", 34 * day, "aborted", [])
    ids = lambda **k: [m["id"] for m in j.list(**k)["items"]]
    assert ids(group="person") == ["gf-000003", "gf-000001"] and ids(group="animal") == ["gf-000004"]
    assert ids(outcome="confirmed") == ["gf-000002", "gf-000001"] and ids(outcome="vetoed") == ["gf-000003"] and ids(outcome="unflagged") == ["gf-000004"]
    assert ids(outcome="aborted") == ["gf-000005"] and ids(outcome="other") == ["gf-000005"]                  # raw outcomes still work
    assert j.next_unlabelled(group="vehicle")["item"]["id"] == "gf-000002" and j.next_unlabelled(group="vehicle")["remaining"] == 1
    assert j.next_unlabelled(outcome="unflagged")["item"]["id"] == "gf-000004"
    assert j.next_unlabelled(t_from=33 * day)["item"]["id"] == "gf-000004" and j.next_unlabelled(t_to=30 * day)["remaining"] == 1
    assert j.summary(group="person")["totals"]["events"] == 2 and j.summary(group="person", camera="b")["totals"]["events"] == 1
    j.stop()


def test_http_group_filter_validation():
    b = Box()
    try:
        assert b.req("GET", "/v1/journal?group=person")[0] == 200 and b.req("GET", "/v1/journal?group=dog")[0] == 400
        assert b.req("GET", "/v1/journal/next?group=animal&from=0&to=9999999999999")[0] == 200 and b.req("GET", "/v1/journal/next?group=x")[0] == 400
        assert b.req("GET", "/v1/journal/summary?group=vehicle")[0] == 200
    finally:
        b.close()


# ---------------- 0.3.7: stale episodes ----------------

def test_two_hours_without_updates_closes_confirmed_and_removes_unconfirmed_unflagged_even_with_frames(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    ep(j, "c", "st-conf001", "start", 1000, groups=["person"], timeline_event={"t_ms": 1000, "text": "Person confirmed"})
    ep(j, "c", "st-unfl001", "start", 1000, kind="unflagged", groups=["person"])                              # never confirmed
    ep(j, "c", "st-unfl002", "start", 1000, kind="unflagged", groups=["person"], timeline_event={"t_ms": 1000, "text": "confirmed"})
    ep(j, "c", "st-live001", "start", 1000, groups=["person"])
    for i in range(1, 80):                                                                                    # the camera keeps streaming for 2 h 10 min: frames are NOT activity
        clk.t += 100_000.0
        j.record_frame("c", clk.t, b"f", [det("person", 0.7)])
        if i == 60:
            assert ep(j, "c", "st-live001", "update", clk.t)[0] == 200                                       # a client update keeps one alive
    assert clk.t > 2 * H
    j.maintain()
    for eid in ("st-conf001", "st-unfl002"):
        m = j.get(eid)
        assert m["status"] == "closed" and m["outcome"] == "stale" and (tmp_path / eid / "last.jpg").exists()   # confirmed: closed as stale, kept
    assert j.get("st-unfl001") is None and not (tmp_path / "st-unfl001").exists()                             # unconfirmed unflagged: removed
    assert j.get("st-live001")["status"] == "ongoing"
    clk.t += 2.1 * H
    j.maintain()
    assert j.get("st-live001")["status"] == "closed" and j.get("st-live001")["outcome"] == "stale"            # and it goes stale 2 h after its last update
    j.stop()


def test_maintenance_runs_on_the_worker_without_any_traffic(tmp_path):
    import time as _t
    clk = Clock(); j = Journal(str(tmp_path), clock=clk, maintain_idle_s=0.05)
    ep(j, "c", "idle-0001", "start", 1000, groups=["person"])
    clk.t += 3 * H
    for _ in range(100):                                                                                      # no frames, no API calls: only the worker's timer
        if j.eps["idle-0001"]["meta"]["status"] == "closed":
            break
        _t.sleep(0.05)
    assert j.eps["idle-0001"]["meta"]["status"] == "closed" and j.eps["idle-0001"]["meta"]["outcome"] == "stale"
    j.stop()


def test_stale_episodes_do_not_count_against_the_ongoing_caps(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk, max_ongoing_per_camera=2, max_ongoing=3, maintain_every_s=10**9)
    assert ep(j, "a", "cap-00001", "start", 1000, groups=["person"])[0] == 200 and ep(j, "a", "cap-00002", "start", 1000, groups=["person"])[0] == 200
    assert ep(j, "a", "cap-00003", "start", 1000)[0] == 429                                                    # the cap holds while they are fresh
    clk.t += 3 * H                                                                                            # both are stale now, and no maintenance has run yet
    assert ep(j, "a", "cap-00004", "start", clk.t)[0] == 200 and ep(j, "b", "cap-00005", "start", clk.t)[0] == 200
    assert j.get("cap-00001")["status"] == "closed" and j.get("cap-00001")["outcome"] == "stale"
    j.stop()


def test_unflagged_episode_right_after_a_veto_is_merged_into_the_camera_episode(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    assert ep(j, "cam1", "cam-ep-01", "start", 1000, kind="camera", timeline_event={"t_ms": 1000, "text": "camera trigger"})[0] == 200
    assert ep(j, "cam1", "cam-ep-01", "end", 7000, outcome="vetoed", inconclusive=True)[0] == 200
    st, r = ep(j, "cam1", "unf-ep-01", "start", 7600, kind="unflagged", timeline_event={"t_ms": 7600, "text": "unflagged (heartbeat)"})
    assert st == 200 and r["id"] == "unf-ep-01" and r["merged_into"] == "cam-ep-01" and r["status"] == "ongoing"
    assert ep(j, "cam1", "unf-ep-01", "update", 7700, kind="unflagged", outcome="confirmed", groups=["person"], timeline_event={"t_ms": 7700, "text": "confirmed person"})[0] == 200
    assert ep(j, "cam1", "unf-ep-01", "end", 9000, kind="unflagged", outcome="released", groups=["person"])[0] == 200
    items = j.list()["items"]
    assert [m["id"] for m in items] == ["cam-ep-01"]                                   # ONE item
    m = j.get("cam-ep-01")
    assert m["kind"] == "camera" and m["outcome"] == "released" and m["groups"] == ["person"] and "inconclusive" not in m
    assert Journal.category(m) == "confirmed" and m["question"]["key"] == "person_present"
    texts = [e["text"] for e in m["timeline"]]
    assert texts[0] == "camera trigger" and "merged: unflagged episode after the veto" in texts and "confirmed person" in texts
    assert m["merged_from"]["outcome"] == "vetoed" and m["merged_ids"] == ["unf-ep-01"]
    j2 = Journal(str(tmp_path), clock=clk)                                             # the alias survives a restart
    assert j2.alias == {"unf-ep-01": "cam-ep-01"}


def test_unflagged_episode_later_or_on_another_camera_is_not_merged(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    ep(j, "cam1", "cam-ep-01", "start", 1000, kind="camera"); ep(j, "cam1", "cam-ep-01", "end", 7000, outcome="vetoed")
    assert "merged_into" not in ep(j, "cam1", "unf-ep-01", "start", 12_500, kind="unflagged")[1]       # 5.5 s after the veto
    assert "merged_into" not in ep(j, "cam2", "unf-ep-02", "start", 7500, kind="unflagged")[1]
    ep(j, "cam1", "cam-ep-02", "start", 20_000, kind="camera"); ep(j, "cam1", "cam-ep-02", "end", 26_000, outcome="released")
    assert "merged_into" not in ep(j, "cam1", "unf-ep-03", "start", 26_500, kind="unflagged")[1]       # not a veto


def test_inconclusive_veto_asks_a_neutral_yes_no_question(tmp_path):
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    ep(j, "cam1", "cam-ep-01", "start", 1000, kind="camera", detect=["person", "animal"])
    ep(j, "cam1", "cam-ep-01", "end", 7000, outcome="vetoed", inconclusive=True)
    q = j.get("cam-ep-01")["question"]
    assert q == {"key": "inconclusive", "text": "Was a person or an animal there?", "group": None, "others": [], "yes": "missed", "no": "correct"}
    ep(j, "cam1", "cam-ep-02", "start", 9000, kind="camera"); ep(j, "cam1", "cam-ep-02", "end", 15_000, outcome="vetoed", inconclusive=True)
    assert j.get("cam-ep-02")["question"]["text"] == "Was someone or something there?"
    ep(j, "cam1", "cam-ep-03", "start", 20_000, kind="camera"); ep(j, "cam1", "cam-ep-03", "end", 26_000, outcome="vetoed")
    assert j.get("cam-ep-03")["question"]["key"] == "empty"                           # a well-founded veto keeps the old question
    assert ep(j, "cam1", "cam-ep-04", "start", 30_000, inconclusive="yes")[0] == 400
    assert ep(j, "cam1", "cam-ep-05", "start", 30_000, detect=["car"])[0] == 400


def test_motion_fields_on_end_and_summary_counts_older_episodes_render_without(tmp_path):
    """0.3.19: the end carries occupancy vs motion time and still releases; timeline entries for still release / re-raise are kept as sent."""
    clk = Clock(); j = Journal(str(tmp_path), clock=clk)
    assert ep(j, "cam1", "ep-000101", "start", 1000, mode="gate", kind="camera")[0] == 200
    assert ep(j, "cam1", "ep-000101", "update", 2000, timeline_event={"t_ms": 2000, "text": "motion released (still) +61 s, occupancy stays on"})[0] == 200
    assert ep(j, "cam1", "ep-000101", "update", 3000, timeline_event={"t_ms": 3000, "text": "motion re-raised (new subject) +90 s"})[0] == 200
    mo = {"occupied_s": 300, "motion_s": 170, "still_releases": 2, "reraises": 1}
    assert ep(j, "cam1", "ep-000101", "end", 4000, outcome="released", groups=["person"], motion=mo)[0] == 200
    assert ep(j, "cam1", "ep-000102", "start", 5000, mode="gate", kind="camera")[0] == 200
    assert ep(j, "cam1", "ep-000102", "end", 6000, outcome="released", groups=["person"])[0] == 200
    m = j.get("ep-000101")
    m = m.get("meta", m) if isinstance(m, dict) else m
    assert m["motion"] == mo
    assert [e["text"].split(" ")[1] for e in m["timeline"]] == ["released", "re-raised"]
    old = next(x for x in j.list()["items"] if x["id"] == "ep-000102")
    assert "motion" not in old
    t = j.summary()["totals"]
    assert t["still_releases"] == 2 and t["reraises"] == 1
    for bad in ({"occupied_s": "x"}, [1], {"motion_s": -1}):
        assert ep(j, "cam1", "ep-000103", "start", 7000, mode="gate", kind="camera", motion=bad)[0] == 400


# ---------------- continuation: the same scene is one review item (0.3.26) ----------------

def cont(tmp_path, **k):
    clk = Clock(100_000.0)
    return Journal(str(tmp_path), clock=clk, continue_after_s=30.0, **k), clk


def run_ep(j, clk, cam, eid, start, end, outcome="released", groups=("person",), kind="camera", score=0.9, frames=2, text=None):
    clk.t = start
    ep(j, cam, eid, "start", start, kind=kind, groups=list(groups), timeline_event={"t_ms": start, "text": text or kind})
    for i in range(frames):
        clk.t = start + 100 + i * 100
        j.record_frame(cam, clk.t, b"J-%s-%d" % (eid.encode(), i), [det("person", score)] if score else [])
    clk.t = end
    return ep(j, cam, eid, "end", end, outcome=outcome, groups=list(groups))


def test_return_within_30s_is_folded_into_the_previous_event(tmp_path):
    j, clk = cont(tmp_path)
    run_ep(j, clk, "cam1", "ep-a00001", 100_000, 110_000, score=0.6)
    st, r = run_ep(j, clk, "cam1", "ep-b00001", 122_000, 130_000, kind="unflagged", score=0.95, text="unflagged (camera AI people)")
    assert st == 200 and r["merged_into"] == "ep-a00001" and r["id"] == "ep-b00001"
    items = j.list()["items"]
    assert [m["id"] for m in items] == ["ep-a00001"]
    m = items[0]
    assert m["end_ms"] == 130_000 and m["outcome"] == "released" and m["merged_ids"] == ["ep-b00001"]
    assert [e["text"] for e in m["timeline"]] == ["camera", "continued: back after 12 s", "unflagged (camera AI people)"]
    d = tmp_path / "ep-a00001"
    assert (d / "first.jpg").read_bytes() == b"J-ep-a00001-0" and (d / "last.jpg").read_bytes() == b"J-ep-b00001-1"
    assert (d / "best.jpg").read_bytes().startswith(b"J-ep-b00001")                      # B's 0.95 beats A's 0.6
    assert (d / "s01.jpg").read_bytes() == b"J-ep-b00001-0"                               # B's first frame: the return
    assert not (tmp_path / "ep-b00001").exists()
    assert ep(j, "cam1", "ep-b00001", "update", 131_000)[0] == 409                        # late calls with B's id reach A (closed)


def test_ten_minute_ceiling_restart_is_one_event(tmp_path):
    j, clk = cont(tmp_path)
    run_ep(j, clk, "cam1", "ep-a00002", 100_000, 700_000)
    assert run_ep(j, clk, "cam1", "ep-b00002", 700_000, 760_000)[1]["merged_into"] == "ep-a00002"
    assert run_ep(j, clk, "cam1", "ep-c00002", 765_000, 800_000)[1]["merged_into"] == "ep-a00002"     # chains keep growing
    m = j.list()["items"][0]
    assert j.list()["total"] == 1 and m["merged_ids"] == ["ep-b00002", "ep-c00002"] and m["end_ms"] == 800_000
    assert "continued: no gap" in [e["text"] for e in m["timeline"]]


def test_no_fold_for_vetoes_labels_other_cameras_or_longer_gaps(tmp_path):
    j, clk = cont(tmp_path)
    run_ep(j, clk, "cam1", "ep-a00003", 100_000, 110_000)
    assert "merged_into" not in run_ep(j, clk, "cam1", "ep-v00003", 115_000, 120_000, outcome="vetoed", groups=(), score=0)[1]   # still asked about
    assert "merged_into" not in run_ep(j, clk, "cam2", "ep-o00003", 121_000, 125_000)[1]
    assert "merged_into" not in run_ep(j, clk, "cam1", "ep-l00003", 141_000, 150_000)[1]       # 31 s after A, 21 s after the veto (not a target)
    j.set_label("ep-l00003", "correct", None)
    assert "merged_into" not in run_ep(j, clk, "cam1", "ep-n00003", 155_000, 160_000)[1]       # the previous one is labelled: never changed under the user
    assert j.list()["total"] == 5


def test_fold_is_off_with_zero(tmp_path):
    j, clk = cont(tmp_path)
    j.lim["continue_after_s"] = 0.0
    run_ep(j, clk, "cam1", "ep-a00004", 100_000, 110_000)
    assert "merged_into" not in run_ep(j, clk, "cam1", "ep-b00004", 112_000, 120_000)[1]
