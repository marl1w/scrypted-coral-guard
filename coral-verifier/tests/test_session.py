"""Replay of labelled scenarios against the pure session module (no I/O)."""
import pytest
import session as S

BOX = [0.4, 0.3, 0.5, 0.8]
T0 = 1_791_886_000_000


def person(score, box=BOX):
    return {"cls": "person", "score": score, "box": list(box)}


def car(score=0.9, box=(0.1, 0.5, 0.4, 0.8), cls="car"):
    return {"cls": cls, "score": score, "box": list(box)}


def run(sess, seq, dt=1000, t0=T0, start_idx=0):
    """seq: list of detection lists, one per frame at dt ms spacing. Returns list of snapshots."""
    out = []
    for i, dets in enumerate(seq):
        sess.update(t0 + (start_idx + i) * dt, dets)
        out.append(sess.snapshot())
    return out


def states(snaps):
    return [s["person"]["state"] for s in snaps]


def new(**kw):
    return S.Session(S.Config().updated(kw))


def test_single_confident_frame_never_confirms():
    sn = run(new(), [[person(0.9)], [], [], [], []])
    assert states(sn) == ["candidate", "candidate", "candidate", "idle", "idle"]
    assert not any(s["person"]["present"] for s in sn)


def test_two_of_three_confirms():
    sn = run(new(), [[person(0.9)], [], [person(0.8)]])
    assert states(sn) == ["candidate", "candidate", "confirmed"]
    assert sn[2]["person"]["present"] and sn[2]["person"]["confirmed"] and sn[2]["person"]["tracks"] == 1
    assert sn[2]["person"]["since_ms"] == T0


def test_two_of_four_does_not_confirm():
    sn = run(new(), [[person(0.9)], [], [], [person(0.9)]])
    assert not sn[-1]["person"]["present"]


def test_below_start_thr_never_starts():
    sn = run(new(), [[person(0.64)]] * 10)
    assert set(states(sn)) == {"idle"}


def test_seated_person_low_score_stays_present_49s():
    seq = [[person(0.9)]] * 3 + [[person(0.25)]] * 49
    sn = run(new(), seq)
    assert all(s["person"]["present"] for s in sn[1:])
    assert states(sn)[-1] == "confirmed"


def test_hold_releases_after_hold_s():
    seq = [[person(0.9)]] * 3 + [[]] * 12
    sn = run(new(), seq)
    st = states(sn)
    assert st[2] == "confirmed"
    assert all(s == "holding" for s in st[3:11])          # 1..8 s of gap
    assert st[11] == "idle"                                  # 9 s > hold_s
    assert sn[11]["person"]["last_seen_ms"] == T0 + 2000


def test_person_leaving_ends_after_hold_s_custom():
    seq = [[person(0.9)]] * 3 + [[]] * 6
    sn = run(new(hold_s=3), seq)
    assert states(sn)[3:] == ["holding", "holding", "holding", "idle", "idle", "idle"]


def test_stationary_lock_survives_longer_gap_but_not_beyond_still_hold():
    # 40 s of an unmoved person (still_s 30) then a gap
    seq = [[person(0.9)]] * 40 + [[]] * 40
    sn = run(new(), seq)
    st = states(sn)
    assert st[39] == "confirmed"
    assert st[40 + 29] == "holding"                          # 30 s gap still held
    assert st[40 + 30] == "idle"                             # 31 s > still_hold_s


def test_moving_person_does_not_get_stationary_lock():
    seq = [[person(0.9, [0.05 * i, 0.3, 0.05 * i + 0.1, 0.8])] for i in range(20)] + [[]] * 12
    sn = run(new(), seq)
    assert states(sn)[19] == "confirmed"
    assert states(sn)[19 + 8] == "holding" and states(sn)[19 + 9] == "idle"


def test_max_hold_without_confident_detection():
    # low-confidence-only continuation for longer than max_hold_s ends the track
    seq = [[person(0.9)]] * 3 + [[person(0.25)]] * 20
    sn = run(new(max_hold_s=10), seq)
    assert states(sn)[10] == "confirmed" and states(sn)[-1] == "idle"


def test_second_person_gets_its_own_track():
    a = [0.1, 0.3, 0.2, 0.8]; b = [0.7, 0.3, 0.8, 0.8]
    sn = run(new(), [[person(0.9, a)], [person(0.9, a), person(0.9, b)], [person(0.9, a), person(0.9, b)]])
    assert sn[1]["person"]["tracks"] == 1 and sn[2]["person"]["tracks"] == 2


def test_parked_car_after_120s():
    seq = [[car()]] * 200
    sn = run(new(), seq)
    assert sn[0]["vehicle"]["moving"] is False and sn[0]["vehicle"]["tracks"][0]["state"] == "still"     # new, never moved: unknown, not moving
    assert sn[100]["vehicle"]["moving"] is False and sn[100]["vehicle"]["tracks"][0]["state"] == "still"
    assert sn[100]["vehicle"]["tracks"][0]["parked"] is False
    assert sn[120]["vehicle"]["tracks"][0]["parked"] is False     # exactly 120 s: not yet (needs > 120)
    assert sn[121]["vehicle"]["tracks"][0]["parked"] is True
    assert sn[121]["vehicle"]["moving"] is False and sn[121]["vehicle"]["tracks"][0]["state"] == "parked"
    assert sn[121]["vehicle"]["tracks"][0]["since_ms"] == T0 and sn[121]["vehicle"]["tracks"][0]["cls"] == "car"


def test_parked_car_survives_jitter_and_gaps():
    seq = [[car()]] * 130 + [[car(box=(0.11, 0.51, 0.41, 0.81))]] + [[]] * 100 + [[car()]]
    sn = run(new(), seq)
    assert sn[130]["vehicle"]["tracks"][0]["parked"]
    assert sn[200]["vehicle"]["tracks"][0]["parked"]
    assert sn[-1]["vehicle"]["tracks"][0]["id"] == 0 and sn[-1]["vehicle"]["tracks"][0]["parked"]


def test_driving_car_is_moving_and_unparks_when_it_leaves():
    seq = [[car(box=(0.03 * i, 0.5, 0.03 * i + 0.3, 0.8))] for i in range(30)] + [[]] * 10
    sn = run(new(), seq)
    assert sn[29]["vehicle"]["moving"] and sn[29]["vehicle"]["tracks"][0]["state"] == "moving"
    assert not sn[1]["vehicle"]["moving"]                                  # needs a reference box >= 2 s old
    assert sn[-1]["vehicle"]["tracks"] == [] and sn[-1]["vehicle"]["moving"] is False


def test_low_score_vehicle_ignored_and_truck_bus_moto_count():
    sn = run(new(), [[car(0.39)]])
    assert sn[0]["vehicle"]["tracks"] == []
    sn = run(new(), [[car(0.45)]])                                                  # strong vehicle threshold is 0.40 (live parked car scored 0.49)
    assert len(sn[0]["vehicle"]["tracks"]) == 1
    sn = run(new(), [[car(0.45, cls="bicycle")]])                                   # bicycle keeps 0.5
    assert sn[0]["vehicle"]["tracks"] == []
    sn = run(new(), [[car(0.6, cls="truck"), car(0.6, box=(0.6, 0.5, 0.9, 0.8), cls="motorcycle")]])
    assert len(sn[0]["vehicle"]["tracks"]) == 2


def test_out_of_order_t_is_clamped():
    s = new()
    s.update(T0 + 5000, [person(0.9)])
    t = s.update(T0 + 1000, [person(0.9)])
    assert t == T0 + 5000 and s.snapshot()["t"] == T0 + 5000
    assert s.snapshot()["person"]["state"] == "candidate"        # clamped to the same tick: ONE hit, near-duplicates cannot confirm
    s.update(T0 + 6000, [person(0.9)])
    assert s.snapshot()["person"]["state"] == "confirmed"


def test_high_frame_rate_uses_frame_time_for_hold():
    sn = run(new(), [[person(0.9)]] * 12 + [[]] * 40, dt=250)    # 4 fps; 10 s of silence
    st = states(sn)
    assert st[12 + 31] == "holding" and st[12 + 33] == "idle"    # 8 s hold = 32 frames at 4 fps


def test_config_validation():
    c = S.Config()
    assert c.updated({"hold_s": 5, "confirm_n": 3}).confirm_n == 3
    for bad in ({"nope": 1}, {"start_thr": 0}, {"start_thr": 1.5}, {"cont_thr": 0.9}, {"confirm_n": 5, "confirm_m": 3},
                {"hold_s": -1}, {"hold_s": "x"}, {"confirm_n": 1.5}, {"session_ttl_s": 0}, {"hold_s": True}):
        with pytest.raises(ValueError):
            c.updated(bad)


def test_registry_reset_and_ttl_and_config():
    r = S.Registry()
    r.update("a", T0, [person(0.9)], wall_now=100.0)
    r.update("a", T0 + 1000, [person(0.9)], wall_now=101.0)
    assert r.peek("a", 101.0)["person"]["present"] and r.count(101.0) == 1
    assert r.reset("a") is True and r.reset("a") is False
    assert r.peek("a", 102.0)["person"]["state"] == "idle" and r.peek("a", 102.0)["session"]["frames"] == 0
    r.update("b", T0, [person(0.9)], wall_now=200.0)
    assert r.count(319.9) == 1 and r.count(320.1) == 0          # ttl 120 s without frames
    assert r.peek("b", 321.0)["session"]["frames"] == 0


def test_registry_per_camera_config_overrides_and_persist():
    r = S.Registry()
    assert r.set_config("a", {"hold_s": 2}).hold_s == 2.0
    assert r.config("b").hold_s == 8.0
    r.update("a", T0, [person(0.9)], 1.0); r.update("a", T0 + 1000, [person(0.9)], 2.0)
    r.reset("a")
    assert r.config("a").hold_s == 2.0                           # overrides survive reset
    with pytest.raises(ValueError):
        r.set_config("a", {"bogus": 1})
    assert r.config("a").hold_s == 2.0
    # live session adopts new config
    r.update("a", T0, [person(0.9)], 3.0); r.update("a", T0 + 1000, [person(0.9)], 4.0)
    r.set_config("a", {"hold_s": 1})
    r.update("a", T0 + 4000, [], 5.0)
    assert r.peek("a", 5.0)["person"]["state"] == "idle"
    assert [c["camera"] for c in r.listing(5.0)] == ["a"]


def test_bound_t_window_and_garbage():
    now = T0
    assert S.bound_t(now - 59_000, now) == (now - 59_000, False)
    assert S.bound_t(now + 4_000, now) == (now + 4_000, False)
    for bad in (now + 3_600_000, now - 3_600_000, 0, -5, float("inf"), float("-inf"), float("nan"), 1e300):
        t, clamped = S.bound_t(bad, now)
        assert clamped and t == now, bad


def test_poisoned_future_frame_cannot_freeze_session():
    r = S.Registry(); base = T0
    r.update("a", S.bound_t(base + 3_600_000, base)[0], [], 0.0)          # the reviewer's poison frame: stamped 1 h in the future
    for i in range(3):
        r.update("a", S.bound_t(base + i * 1000, base + i * 1000)[0], [person(0.9)], 1.0 + i)
    sn = None
    for i in range(300):
        sn = r.update("a", S.bound_t(base + 3000 + i * 1000, base + 3000 + i * 1000)[0], [], 4.0 + i)
    assert sn["person"]["state"] == "idle" and not sn["person"]["present"]


def test_get_state_expires_with_server_clock():
    r = S.Registry()
    for i in range(3):
        r.update("a", T0 + i * 1000, [person(0.9)], 100.0 + i)
    assert r.peek("a", 102.5)["person"]["present"]
    assert r.peek("a", 102.0 + 8)["person"]["state"] == "holding"
    assert r.peek("a", 102.0 + 10)["person"]["state"] == "idle"           # hold_s 8 elapsed on the wall clock, no new frame needed
    r.update("g", T0, [car()], 200.0)
    assert len(r.peek("g", 200.5)["vehicle"]["tracks"]) == 1 and r.peek("g", 200.5)["vehicle"]["moving"] is False
    assert r.peek("g", 215.0)["vehicle"]["tracks"] == [] and r.peek("g", 215.0)["vehicle"]["moving"] is False
    assert r.peek("a", 102.0 + 121)["session"]["frames"] == 0              # ttl


def test_camera_cap_and_sweeps():
    r = S.Registry(max_cameras=3)
    for i in range(3):
        r.update(f"c{i}", T0, [], 1.0)
    with pytest.raises(S.TooManyCameras):
        r.update("c3", T0, [], 1.0)
    with pytest.raises(S.TooManyCameras):
        r.set_config("c4", {"hold_s": 2}, 1.0)
    assert r.would_admit("c0", 1.0) and not r.would_admit("zz", 1.0)
    r.update("c0", T0, [], 500.0)                                          # others expired by ttl: room again
    r.update("c9", T0, [], 500.0)
    assert r.count_known() <= 3
    r2 = S.Registry(max_cameras=2)
    r2.set_config("x", {"hold_s": 2}, 0.0)
    assert r2.known("x", 10.0)
    assert not r2.known("x", 10.0 + S.OVERRIDE_IDLE_S + 5)                 # idle overrides are swept


def test_sweep_is_amortised():
    class Counting(dict):
        scans = 0
        def items(self):
            Counting.scans += 1
            return super().items()
    r = S.Registry()
    for i in range(50):
        r.update(f"c{i}", T0, [], 1.0)
    r._sessions = Counting(r._sessions)
    for k in range(5000):
        r.sweep(2.0 + k * 0.0001)                                          # 0.5 s of calls: at most one full scan
    assert Counting.scans <= 2
    r.sweep(3.5); r.sweep(5.0)
    assert Counting.scans <= 6


def test_detection_cap_of_12_per_class():
    dets = [person(0.9, [0.02 * i, 0.1, 0.02 * i + 0.015, 0.2]) for i in range(30)]
    s = new()
    s.update(T0, dets)
    assert len(s.person.tracks) == S.MAX_TRACK_DETS == 12


def test_config_rejects_non_finite_and_adopts_live():
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            S.Config().updated({"hold_s": bad})
    r = S.Registry()
    r.update("a", T0, [person(0.9)], 1.0); r.update("a", T0 + 1000, [person(0.9)], 2.0)
    r.set_config("a", {"hold_s": 1})
    assert r.peek("a", 2.0)["person"]["present"]
    assert r.peek("a", 2.0 + 2.5)["person"]["state"] == "idle"             # new hold_s applies to the live session at once
    r.set_config("a", {"hold_s": 30})
    assert r.peek("a", 2.0 + 2.5)["person"]["state"] == "holding"


def test_track_count_is_bounded():
    s = new()
    for i in range(200):
        s.update(T0 + i * 1000, [person(0.9, [(i * 7 % 90) / 100, (i * 13 % 70) / 100, (i * 7 % 90) / 100 + 0.04, (i * 13 % 70) / 100 + 0.1]) for _ in range(1)]
                 + [car(0.9, (0.01 * k, 0.5, 0.01 * k + 0.2, 0.9)) for k in range(12)])
    assert len(s.person.tracks) <= S.MAX_TRACKS and len(s.vehicle.tracks) <= S.MAX_TRACKS


def test_threshold_boundaries_are_inclusive():
    s = new()                                                              # start_thr 0.65: a box scoring exactly 0.65 starts a track
    s.update(T0, [person(0.65)]); s.update(T0 + 1000, [person(0.65)])
    assert s.snapshot()["person"]["present"]
    s2 = new()
    s2.update(T0, [person(0.9)]); s2.update(T0 + 1000, [person(0.9)])
    s2.update(T0 + 2000, [person(0.2)])                                    # cont_thr 0.2 inclusive: continues the track
    assert s2.snapshot()["person"]["state"] == "confirmed"
    s2.update(T0 + 3000, [person(0.19)])
    assert s2.snapshot()["person"]["state"] == "holding"


def test_person_track_count_bounded_by_pathological_tentatives():
    s = new()
    for k in range(4):
        dets = [person(0.9, [0.02 + 0.075 * i, 0.05 + 0.2 * k, 0.04 + 0.075 * i, 0.09 + 0.2 * k]) for i in range(12)]
        s.update(T0 + k * 1000, dets)
    assert len(s.person.tracks) <= S.MAX_TRACKS
