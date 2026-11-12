"""Box geometry shared by the trackers, the stores and the journal. Boxes are (x0, y0, x1, y1), normalised to the frame. PURE."""


def iou(a, b):
    ix, iy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def inside(a, b):
    """Share of box a lying inside box b."""
    ix, iy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    aa = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / aa if ix > 0 and iy > 0 and aa > 0 else 0.0


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


FRAME_BORDER = 0.01   # a box side this close to the frame border is where the frame cuts the object, not where the object ends


def unclipped(a, b, border=FRAME_BORDER):
    """Two boxes of one object, compared only where both are really drawn: on a side where either box touches the frame border, both
    are cut to the inner of the two. An object partly outside the frame is drawn out to the border on one frame and around what the
    detector recognises on the next; that flip is not movement. An object entering or leaving through the border still moves its
    inner side."""
    a, b = list(a), list(b)
    for k in (0, 1):                                              # left, top
        if a[k] <= border or b[k] <= border:
            a[k] = b[k] = max(a[k], b[k])
    for k in (2, 3):                                              # right, bottom
        if a[k] >= 1.0 - border or b[k] >= 1.0 - border:
            a[k] = b[k] = min(a[k], b[k])
    return a, b


def moved(a, b, limit):
    """Two boxes of one object at different places? Compared where both are really drawn (see unclipped), it moved when two of its
    sides travelled more than `limit` (a driving car shifts or grows: at least two sides go), or, cut by the frame border, one of
    its inner sides did (a car entering or leaving through the border moves only that one). A single side alone is the
    detector redrawing the part it sees: the border, or something in front of the object (a post, a fence), cutting it at
    another place; nothing moved (0.3.60)."""
    a2, b2 = unclipped(a, b)
    cut = any(min(a[k], b[k]) <= FRAME_BORDER for k in (0, 1)) or any(max(a[k], b[k]) >= 1.0 - FRAME_BORDER for k in (2, 3))
    sides = sum(abs(a2[k] - b2[k]) > limit for k in range(4))
    return sides >= (1 if cut else 2)
