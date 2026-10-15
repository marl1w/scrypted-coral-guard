"""Frame fingerprints for the journal: is this frame the same scene as one the user already validated?

A fingerprint is a small, blurred grayscale thumbnail with the camera's clock band (the top rows, where the OSD
draws date and time) blanked, plus a night flag (an infrared frame has no colour). Two frames are the same scene
when the whole thumbnail barely differs AND no block of it differs noticeably: a small subject far away moves the
mean very little but stands out in its own block. Pure functions, no I/O; the decode is a reduced (1/8) JPEG decode,
about a millisecond per frame."""
import base64

import cv2
import numpy as np

W, H = 160, 90                  # thumbnail size (any aspect is resized to it; frames are only compared within one camera)
CLOCK_BAND = 0.09               # top share of the frame ignored (OSD date and time)
BLOCK = 10                      # block side, in thumbnail pixels
MEAN_MAX = 4.0                  # mean absolute difference (0..255) allowed over the whole thumbnail (calibrated on 183 validated episodes)
BLOCK_MAX = 12.0                # mean absolute difference allowed in any single block
NIGHT_SAT = 6.0                 # mean HSV saturation below this = an infrared (night) frame


def fingerprint(jpeg):
    """-> {"night": bool, "g": base64 of the W x H uint8 thumbnail} or None when the JPEG does not decode."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None or img.size == 0:
        return None
    small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    night = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < NIGHT_SAT
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return {"night": night, "g": base64.b64encode(g.tobytes()).decode()}


def thumb(fp):
    """The thumbnail of a fingerprint as a float array (H x W)."""
    return np.frombuffer(base64.b64decode(fp["g"]), np.uint8).reshape(H, W).astype(np.float32)


def difference(a, b):
    """-> (mean, worst block) absolute difference of two thumbnails, the clock band excluded. Brightness is
    equalised first (the camera's exposure drifts between otherwise identical frames)."""
    top = int(round(H * CLOCK_BAND))
    a, b = a[top:], b[top:]
    b = b + (a.mean() - b.mean())
    d = np.abs(a - b)
    h, w = d.shape
    hb, wb = h // BLOCK * BLOCK, w // BLOCK * BLOCK
    blocks = d[:hb, :wb].reshape(hb // BLOCK, BLOCK, wb // BLOCK, BLOCK).mean(axis=(1, 3))
    return float(d.mean()), float(blocks.max())


def same(a, b):
    """Same scene: both fingerprints from the same light (day / night) and both differences within the limits."""
    if a["night"] != b["night"]:
        return False
    mean, block = difference(thumb(a), thumb(b))
    return mean <= MEAN_MAX and block <= BLOCK_MAX
