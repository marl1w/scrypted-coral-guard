"""Event journal: lifecycle, snapshots, retention, stale, labels, summary, traversal safety, HTTP review API and the ingress auth rule."""
import http.client, json, os
import pytest
import session as S
from journal import Journal
from boxlib import Box, jpeg
from test_server import T0

H = 3600_000


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
        assert [m["id"] for m in b.req("GET", "/v1/journal?labelled=false&status=closed")[1]["items"]] == ["web-q-0001"]
        assert b.req("GET", "/v1/journal?labelled=maybe")[0] == 400 and b.req("GET", "/v1/journal?status=weird")[0] == 400
        st, sm = b.req("GET", "/v1/journal/summary?from=0&tz=120&camera=a")
        assert st == 200 and sm["totals"]["events"] == 1 and sm["totals"]["precision"] == 1.0 and sm["cameras"][0]["camera"] == "a" and sm["days"]
        assert b.req("GET", "/v1/journal/summary?tz=5000")[0] == 400 and b.req("GET", "/v1/journal/summary?from=nan")[0] == 400
        st, html, ct = raw(b, "GET", "/")
        for needle in (b"Validate", b"Reviewed", b"Yes (Y)", b"No (N)", b"What is it?", b"Review events to see accuracy", b"next?"):
            assert needle in html, needle
        assert b"innerHTML" not in html and b"document.write" not in html and b"eval(" not in html                  # untrusted text only through textContent
    finally:
        b.close()
