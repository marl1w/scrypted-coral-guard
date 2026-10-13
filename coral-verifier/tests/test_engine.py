"""Pixel-level and mapping tests of the Engine (in-process fake model): channel order, padding, repaint, tile merge, EXIF, qmin boundary."""
import struct
import cv2, numpy as np
import pytest
import detect as D
import fake_tpu
import yolo_edgetpu as Y
from boxlib import jpeg

S_SC, Z_SC = fake_tpu.S_SC, fake_tpu.Z_SC


def engine(items=None, size=512):
    m = fake_tpu.FakeModel(size, items=items, delay=0)
    return D.Engine(m), m


def test_letterbox_channel_order_is_rgb_and_pad_is_114():
    e, m = engine()
    img = np.zeros((100, 400, 3), np.uint8); img[:, :] = (255, 10, 0)           # BGR: blue 255, green 10, red 0
    canvas, r, left, top = e._letterbox(img)
    assert canvas.shape == (512, 512, 3)
    assert tuple(canvas[256, 256]) == (0, 10, 255)                             # RGB on the model canvas
    assert tuple(canvas[5, 5]) == (114, 114, 114) and tuple(canvas[506, 300]) == (114, 114, 114)     # gray border
    assert top == (512 - 128) // 2 and left == 0 and abs(r - 1.28) < 1e-6


def test_canvas_is_repainted_when_geometry_changes():
    e, m = engine()
    e._letterbox(np.full((512, 128, 3), 200, np.uint8))                          # tall image: content in the middle columns, gray left/right
    canvas, *_ = e._letterbox(np.full((100, 400, 3), 50, np.uint8))             # wide image: the old content must be gone from the padding rows
    assert tuple(canvas[5, 256]) == (114, 114, 114) and tuple(canvas[506, 256]) == (114, 114, 114)
    assert tuple(canvas[256, 256]) == (50, 50, 50)
    canvas, *_ = e._letterbox(np.full((512, 128, 3), 200, np.uint8))
    assert tuple(canvas[256, 5]) == (114, 114, 114) and tuple(canvas[256, 256]) == (200, 200, 200)


def test_input_quantisation_matches_pixel_minus_128():
    e, m = engine()
    img = np.random.RandomState(3).randint(0, 256, (512, 512, 3), np.uint8)
    canvas, *_ = e._letterbox(img)
    x = Y.EdgeTpuYolo.quantize(m, canvas)
    assert np.array_equal(x[0].astype(np.int16), canvas.astype(np.int16) - 128)


class SeqModel(fake_tpu.FakeModel):
    """Fake whose detections differ per call (call 1 = left tile, call 2 = right tile)."""
    def __init__(self, size, seq):
        super().__init__(size, delay=0)
        self.seq = seq

    def invoke_raw(self, x):
        self.items = self.seq[min(self.calls, len(self.seq) - 1)]
        return super().invoke_raw(x)


def test_tile_merge_dedups_the_overlap_and_maps_to_full_frame():
    # 1536x576 -> tiles 845 px wide; the same person seen at the right edge of the left tile and the left edge of the right tile
    left = [("person", 3.0, 16, 29, 16, (3, 3, 3, 3))]
    right = [("person", 3.0, 16, 3, 16, (3, 3, 3, 3))]
    m = SeqModel(512, [left, right])
    e = D.Engine(m)
    img = np.full((576, 1536, 3), 90, np.uint8)
    dets, tm = e.detect(img, "2", 0.2)
    assert m.calls == 2 and len(dets) == 1                                      # merged by NMS 0.5 (a skipped merge would leave 2)
    cx = (dets[0]["box"][0] + dets[0]["box"][2]) / 2
    assert 0.48 < cx < 0.53                                                      # mapped through the tile x-offset to the middle of the full frame
    far = SeqModel(512, [[("person", 3.0, 16, 3, 16, (3, 3, 3, 3))], [("person", 3.0, 16, 29, 16, (3, 3, 3, 3))]])
    dets, _ = D.Engine(far).detect(img, "2", 0.2)
    assert len(dets) == 2                                                        # far apart: both survive
    assert dets[0]["box"][0] < 0.2 or dets[1]["box"][0] < 0.2


def test_exif_orientation_is_ignored():
    j = jpeg(64, 32)
    exif = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00\x00\x00\x00\x00"        # Orientation = 6 (rotate 90)
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    data = j[:2] + app1 + j[2:]
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (64, 32), "precondition: OpenCV honours EXIF by default"
    img, _ = D.decode_jpeg(data, 512, "auto")
    assert img.shape[:2] == (32, 64)                                             # pixels as submitted, boxes refer to the submitted frame


def test_decode_threshold_boundary_is_ceil_not_floor():
    dec = Y.Decoder(320)
    n = Y.n_anchors(320)
    sc = np.full((n, 17), -128, np.int8); bx = np.zeros((n, 64), np.int8)
    for side in range(4):
        bx[:, side * 16:(side + 1) * 16] = -32; bx[:, side * 16 + 2] = 32
    a_below, a_at = 100, 200
    sc[a_below, 0] = 3      # logit (3+10)*0.0625 = 0.8125 -> p 0.6926 < 0.7
    sc[a_at, 0] = 4         # logit 0.875 -> p 0.7058 >= 0.7
    r = dec.decode(sc, bx, S_SC, Z_SC, 0.25, 0, classes=(0,), thr=0.7)
    assert len(r) == 1 and abs(r[0][1] - 1 / (1 + np.exp(-0.875))) < 1e-4


def test_vehicle_threshold_is_0_5_in_engine():
    e, m = engine(items=[("car", -0.6, 16, 8, 8, (3, 3, 3, 3)), ("car", 0.5, 16, 20, 20, (3, 3, 3, 3))])    # p 0.35 (weak, 0.25-0.40) and 0.62 (strong)
    dets, _ = e.detect(np.full((480, 640, 3), 90, np.uint8), "1", 0.2)
    assert [d["cls"] for d in dets] == ["car", "car"] and dets[0]["score"] > 0.6 and "weak" not in dets[0] and dets[1]["weak"] is True
