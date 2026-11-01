"""A weak box continues a confirmed person track only when its size is consistent with the track (session.CONT_MAX_AREA_RATIO).

Field case: a person walked past a white wall that, overexposed under the outdoor lights, the detector reads as a huge "person" at
0.20-0.45. That box never starts a track, but it took over the confirmed track of the person who had just walked by and held it,
stationary, for minutes after they left."""
import session as S

T0 = 1_791_886_000_000
WALL = (0.50, 0.0, 1.0, 0.98)                 # ~49 % of the frame


def person(cx, cy=0.4, w=0.11, h=0.4):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run(frames):
    sess = S.Session(S.Config().updated({"still_s": 3}))
    out = []
    for t, f in enumerate(frames):
        sess.update(T0 + t * 1000, [{"cls": "person", "score": sc, "box": list(b)} for sc, b in f])
        out.append(sess.snapshot()["person"]["present"])
    return out


def test_large_weak_box_does_not_take_over_a_person_who_left():
    walk = [[(0.8, person(0.2 + 0.05 * t))] for t in range(6)]          # walks right, towards the wall, then leaves
    wall = [[(0.3, WALL)] for _ in range(90)]                             # only the weak wall box remains
    pres = run(walk + wall)
    assert pres[5]
    assert not any(pres[6 + int(S.Config().hold_s) + 2:])      # gone once the normal hold after the last real frame ran out


def test_weak_box_of_the_same_size_still_continues_a_standing_person():
    walk = [[(0.8, person(0.3 + 0.02 * t))] for t in range(4)]
    stand = [[(0.3, person(0.38, h=0.36))] for _ in range(60)]           # the same person, standing, scored low
    assert all(run(walk + stand)[1:])


def test_ratio_gate_is_symmetric_and_bounded():
    small, big = (0.4, 0.4, 0.5, 0.6), (0.3, 0.2, 0.7, 0.8)               # 12x the area
    assert S.greedy_cd([small], [big], gate=10) == [(0, 0)]
    assert S.greedy_cd([small], [big], gate=10, max_ratio=S.CONT_MAX_AREA_RATIO) == []
    assert S.greedy_cd([big], [small], gate=10, max_ratio=S.CONT_MAX_AREA_RATIO) == []
    assert S.greedy_cd([small], [(0.41, 0.4, 0.52, 0.62)], max_ratio=S.CONT_MAX_AREA_RATIO) == [(0, 0)]
