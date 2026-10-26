"""Shape hold (0.3.30): is the vehicle the detector just missed still there? PURE: numpy/cv2 only, no I/O, no clock.

A still vehicle near the detection threshold (night IR, a car cut by the frame edge) is found in some frames and missed in others: the
tracker used to drop it after hold_s and start a new track at the next hit, so it never lived the parked_after_s it needs to park and its
presence flipped all night. The image inside its box does not change while it stands there, so a miss is checked against the vehicle's
last look: the box's edge map (blurred Sobel magnitude, resized to a fixed size, zero mean, unit norm), compared by correlation.

Measured on 2.5 h of recordings of one camera (2026-10-17/18): the same parked car scored 0.94-0.996 at night (all 160 presence flips of
those hours), ~0.87 under a passing hard shadow and 0.82 across the IR -> colour switch; an empty parking spot against the parked car <= 0.30.
Edges, not raw pixels: a uniform brightness or exposure change keeps the outline.
"""
import base64
import math

import cv2
import numpy as np

GRAY_W = 480          # the frame is reduced to this width (gray) once per frame, before the Coral job is queued
SIG_W, SIG_H = 96, 64  # every box is resized to this before its edge map is taken: one signature is 6144 floats (24 KB)
MIN_SIDE_PX = 12      # a box smaller than this on the reduced frame has too little detail to compare: never held by shape


def gray_small(img):
    """BGR uint8 (as decoded) -> gray uint8 no wider than GRAY_W."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    if w > GRAY_W:
        g = cv2.resize(g, (GRAY_W, max(1, round(h * GRAY_W / w))), interpolation=cv2.INTER_AREA)
    return g


def signature(gray, box):
    """Edge signature of a normalised box [x0, y0, x1, y1] on a gray frame, or None (box too small, outside, or flat)."""
    h, w = gray.shape
    x0, y0 = max(0, int(box[0] * w)), max(0, int(box[1] * h))
    x1, y1 = min(w, int(math.ceil(box[2] * w))), min(h, int(math.ceil(box[3] * h)))
    if x1 - x0 < MIN_SIDE_PX or y1 - y0 < MIN_SIDE_PX:
        return None
    c = cv2.resize(gray[y0:y1, x0:x1], (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    c = cv2.GaussianBlur(c, (5, 5), 1.2)
    m = cv2.magnitude(cv2.Sobel(c, cv2.CV_32F, 1, 0), cv2.Sobel(c, cv2.CV_32F, 0, 1))
    m -= m.mean()
    n = float(np.linalg.norm(m))
    if n < 1e-3:
        return None
    return (m / n).ravel()


def similarity(a, b):
    """Correlation of two signatures, -1..1 (1 = same outline)."""
    return float(np.dot(a, b))


def covered(box, others, frac):
    """True when another box covers at least `frac` of `box` (someone or something in front of it)."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 0:
        return False
    for o in others:
        ix = min(box[2], o[2]) - max(box[0], o[0])
        iy = min(box[3], o[3]) - max(box[1], o[1])
        if ix > 0 and iy > 0 and ix * iy >= frac * area:
            return True
    return False


def encode(sig):
    """A signature as stored in the per-camera JSON stores (float16, base64)."""
    return base64.b64encode(sig.astype(np.float16).tobytes()).decode("ascii")


def decode(s):
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)
