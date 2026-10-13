"""Differential test: the service Session must make the SAME person decisions as the 1 fps reference simulation (tests/ref_tracker.py)
on thousands of random scenarios, plus the targeted cases from the adversarial review."""
import random
import pytest
import session as S
import ref_tracker as R

T0 = 1_791_886_000_000
BOX0 = (0.3, 0.3, 0.4, 0.8)


def box_at(cx, w=0.1, h=0.5, cy=0.55):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def scenario(rng):
    n = rng.randint(30, 120)
    frames = [[] for _ in range(n)]
    for _ in range(rng.choice((0, 1, 1, 2, 3))):                       # persons
        cx = rng.uniform(0.15, 0.85); start = rng.randint(0, n // 2); end = rng.randint(start + 3, n)
        mode = rng.choice(("walk", "sit", "mix", "slow"))
        base = rng.uniform(0.25, 0.95)
        drop_until = -1
        v = rng.choice((-0.03, -0.015, 0.015, 0.03)) if mode in ("walk", "mix") else (rng.choice((-0.002, 0.002)) if mode == "slow" else 0.0)
        for t in range(start, end):
            if mode == "mix" and t > (start + end) // 2:
                v = 0.0
            cx += v + (rng.uniform(-0.003, 0.003) if mode != "walk" else 0)
            cx = min(max(cx, 0.05), 0.95)
            if t > drop_until and rng.random() < 0.06:
                drop_until = t + rng.choice((1, 2, 3, 5, 8, 9, 10, 15, 25, 31, 40))
            if t <= drop_until:
                continue
            sc = min(0.99, max(0.0, base + rng.uniform(-0.2, 0.2)))
            if sc >= 0.2:
                frames[t].append((sc,) + box_at(cx))
    for _ in range(rng.choice((0, 0, 1, 3))):                          # blips (false positives / lone frames)
        t = rng.randrange(n); frames[t].append((rng.uniform(0.2, 0.95),) + box_at(rng.uniform(0.1, 0.9)))
    for f in frames:
        rng.shuffle(f)
    return frames


def run_service(frames, jitter=None, dup=1, cfg=None):
    sess = S.Session(cfg or S.Config().updated({"still_s": 3, "max_hold_s": 86400, "start_thr": 0.7}))
    pres = []; first = None
    for t, f in enumerate(frames):
        dets = [{"cls": "person", "score": d[0], "box": list(d[1:])} for d in f]
        off = 0 if jitter is None else jitter.randint(-100, 100)
        for k in range(dup):                                            # dup > 1: several near-duplicate frames inside the same second (4 fps sender)
            sess.update(T0 + t * 1000 + off + k * 250, dets)
        sn = sess.snapshot()
        pres.append(sn["person"]["present"])
        if first is None and sn["person"]["present"]:
            first = t
    return pres, first


def run_ref(frames):
    pres, first, _ = R.run_tracker(frames, len(frames), 0.7, 0.2, ("cd", 1.5), 8, 30, True)
    return pres, first


def test_differential_exact_1fps_5000_scenarios():
    rng = random.Random(20261013)
    bad = []
    seen_present = 0
    for i in range(5000):
        fr = scenario(rng)
        p, f = run_service(fr); rp, rf = run_ref(fr)
        seen_present += any(rp)
        if p != rp or f != rf:
            bad.append(i)
    assert seen_present > 1500, "scenario generator too weak"
    assert bad == [], f"{len(bad)} disagreements, first scenarios {bad[:5]}"


def test_differential_with_clock_jitter_and_burst_duplicates():
    rng = random.Random(77)
    jit = random.Random(5)
    bad = 0
    for i in range(1500):
        fr = scenario(rng)
        rp, rf = run_ref(fr)
        p, f = run_service(fr, jitter=jit)                              # +-100 ms sender jitter (frames closer than 0.75 s merge by design): same decisions
        p2, f2 = run_service(fr, dup=3)                                 # 4 fps duplicates inside each second: same decisions
        if p != rp or f != rf or p2 != rp or f2 != rf:
            bad += 1
    assert bad == 0


def person(score, box=BOX0):
    return {"cls": "person", "score": score, "box": list(box)}


def feed(sess, seq, t0=T0, dt=1000):
    out = []
    for i, d in enumerate(seq):
        sess.update(t0 + i * dt, d)
        out.append(sess.snapshot()["person"]["present"])
    return out


def sim(seq_boxes):
    fr = [[(s,) + b for s, b in f] for f in seq_boxes]
    return run_ref(fr)[0]


def test_review_case_sit_10s_then_gap_25s():
    seq = [[(0.9, BOX0)]] * 10 + [[]] * 25 + [[(0.9, BOX0)]]
    s = S.Session(S.Config().updated({"still_s": 3, "max_hold_s": 86400, "start_thr": 0.7}))
    got = feed(s, [[person(sc, b) for sc, b in f] for f in seq])
    assert got == sim(seq)
    assert got[10 + 24] is True and got[10 + 25 - 1 + 1] is True     # still lock (30 s) bridges a 25 s gap; same as the reference simulation


def test_review_case_slow_drift_then_gap():
    seq = []
    for i in range(40):
        x = 0.3 + 0.004 * i
        seq.append([(0.9, (x, 0.3, x + 0.1, 0.8))])
    seq += [[]] * 12
    s = S.Session(S.Config().updated({"still_s": 3, "max_hold_s": 86400, "start_thr": 0.7}))
    assert feed(s, [[person(sc, b) for sc, b in f] for f in seq]) == sim(seq)


def test_review_case_hold3_then_8_missed_then_detection():
    cfg = {"still_s": 3, "max_hold_s": 86400, "start_thr": 0.7}
    seq = [[(0.9, BOX0)]] * 3 + [[]] * 8 + [[(0.9, BOX0)]] + [[]] * 3
    ref = sim(seq)
    s = S.Session(S.Config().updated(cfg))
    assert feed(s, [[person(sc, b) for sc, b in f] for f in seq]) == ref


def test_off_by_one_rematch_matches_sim_exactly():
    # hold 8: a track can be re-matched at gap 9 (the pool check uses t-1) although it was not present at that gap before the match
    for gap in range(5, 14):
        seq = [[(0.9, BOX0)]] * 3 + [[]] * gap + [[(0.3, BOX0)]] * 4
        s = S.Session(S.Config().updated({"still_s": 3, "max_hold_s": 86400, "start_thr": 0.7}))
        assert feed(s, [[person(sc, b) for sc, b in f] for f in seq]) == sim(seq), gap


def test_near_duplicate_frames_do_not_confirm():
    s = S.Session(S.Config())
    for k in range(6):                                                  # six frames within 0.5 s: ONE tick, one hit
        s.update(T0 + k * 100, [person(0.9)])
    assert s.snapshot()["person"]["state"] == "candidate"
    s.update(T0 + 1000, [person(0.9)])
    assert s.snapshot()["person"]["present"]


def test_merge_keeps_best_score_per_tick():
    s = S.Session(S.Config())
    s.update(T0, [person(0.5)])                                          # below start_thr
    s.update(T0 + 300, [person(0.9)])                                    # same tick: max score wins -> a tentative track starts
    assert s.snapshot()["person"]["state"] == "candidate"
    s.update(T0 + 1300, [person(0.8)])
    assert s.snapshot()["person"]["present"]
