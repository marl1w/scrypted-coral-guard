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
