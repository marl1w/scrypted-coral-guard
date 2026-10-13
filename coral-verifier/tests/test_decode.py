"""Synthetic-tensor tests for the Edge TPU YOLO decode. Run: python3 tests/test_decode.py  (needs numpy + opencv)."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verifier"))
import numpy as np
import yolo_edgetpu as Y

S_SC, Z_SC = 0.0625, -10      # class logits: float = (q+10)*0.0625
S_BX, Z_BX = 0.25, 0          # DFL logits


def make(size, items, nc=17):
    """items: (anchor_index, cls, logit, ltrb distances in stride units). Background logits ~ -6, DFL flat."""
    n = Y.n_anchors(size)
    sc = np.full((n, nc), int(round(-6 / S_SC)) + Z_SC, np.int8)
    bx = np.zeros((n, 64), np.int8)
    for a, c, logit, ltrb in items:
        sc[a, c] = int(round(logit / S_SC)) + Z_SC
        for side, dist in enumerate(ltrb):           # one-hot DFL peak at the integer bin
            bx[a, side * 16:(side + 1) * 16] = -8 / S_BX
            bx[a, side * 16 + int(dist)] = 8 / S_BX
    return sc, bx


def anchor_index(size, stride, gx, gy):
    off = 0
    for s in Y.STRIDES:
        if s == stride:
            return off + gy * (size // s) + gx
        off += (size // s) ** 2


class T(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(Y.n_anchors(320), 2100)
        self.assertEqual(Y.n_anchors(512), 5376)
        a, s = Y.anchors_and_strides(320)
        self.assertEqual(a.shape, (2100, 2)); self.assertEqual(s.shape, (2100, 1))
        self.assertTrue(np.allclose(a[0], [0.5, 0.5])); self.assertTrue(np.allclose(a[1], [1.5, 0.5]))   # x fastest
        self.assertTrue(np.allclose(a[40], [0.5, 1.5]))
        self.assertTrue(np.allclose(a[1600], [0.5, 0.5])); self.assertEqual(float(s[1600, 0]), 16.0)          # second stride starts at 40*40
        self.assertEqual(float(s[2099, 0]), 32.0)

    def test_pick_tensors(self):
        d = [{"shape": [1, 2100, 17], "index": 5}, {"shape": [1, 2100, 64], "index": 4}]
        self.assertEqual(Y.pick_tensors(d), (1, 0))
        d = [{"shape": [1, 2100, 64], "index": 5}, {"shape": [1, 2100, 1], "index": 6}, {"shape": [1, 2100, 17], "index": 7}]
        self.assertEqual(Y.pick_tensors(d), (0, 2))
        with self.assertRaises(ValueError): Y.pick_tensors([{"shape": [1, 84, 2100], "index": 1}])

    def test_single_box(self):
        for size in (320, 512):
            dec = Y.Decoder(size)
            a = anchor_index(size, 16, 5, 7)             # centre (5.5, 7.5)*16 = (88, 120)
            sc, bx = make(size, [(a, 0, 3.0, (2, 3, 4, 1))])
            r = dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX)
            self.assertEqual(len(r), 1)
            c, p, x0, y0, x1, y1 = r[0]
            self.assertEqual(c, 0)
            self.assertAlmostEqual(p, 1 / (1 + np.exp(-3.0)), places=2)
            # one-hot at 20 dB: expectation within ~1e-3 of the integer bin
            self.assertAlmostEqual(x0, (5.5 - 2) * 16, places=1); self.assertAlmostEqual(y0, (7.5 - 3) * 16, places=1)
            self.assertAlmostEqual(x1, (5.5 + 4) * 16, places=1); self.assertAlmostEqual(y1, (7.5 + 1) * 16, places=1)

    def test_dfl_expectation(self):
        dec = Y.Decoder(320)
        a = anchor_index(320, 8, 10, 10)
        sc, bx = make(320, [(a, 0, 2.0, (0, 0, 0, 0))])
        for side in range(4): bx[a, side * 16:(side + 1) * 16] = 0   # flat -> uniform softmax -> expectation 7.5
        c, p, x0, y0, x1, y1 = dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX)[0]
        self.assertAlmostEqual(x0, (10.5 - 7.5) * 8, places=3); self.assertAlmostEqual(x1, (10.5 + 7.5) * 8, places=3)

    def test_threshold_and_multiclass(self):
        dec = Y.Decoder(320)
        a1 = anchor_index(320, 8, 3, 3); a2 = anchor_index(320, 32, 5, 5); a3 = anchor_index(320, 16, 2, 2)
        sc, bx = make(320, [(a1, 0, 4.0, (2, 2, 2, 2)),     # person 0.98
                            (a2, 2, 1.0, (1, 1, 1, 1)),     # car 0.73
                            (a3, 0, -3.5, (1, 1, 1, 1))])   # person 0.029 < 0.05 -> dropped
        r = dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX, classes=(0,))
        self.assertEqual([x[0] for x in r], [0])
        r = dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX, classes=tuple(range(17)))
        self.assertEqual(sorted(x[0] for x in r), [0, 2])
        r = dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX, classes=(0,), thr=0.02)
        self.assertEqual(len(r), 2)

    def test_nms(self):
        dec = Y.Decoder(320)
        a1 = anchor_index(320, 8, 10, 10); a2 = anchor_index(320, 8, 11, 10); a3 = anchor_index(320, 8, 30, 30)
        sc, bx = make(320, [(a1, 0, 4.0, (4, 4, 4, 4)), (a2, 0, 3.0, (4, 4, 4, 4)), (a3, 0, 2.0, (4, 4, 4, 4))])
        r = dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX)
        self.assertEqual(len(r), 2)                    # a2 suppressed by a1 (IoU ~0.78), a3 far away
        self.assertGreater(r[0][1], r[1][1])

    def test_empty_and_shape_error(self):
        dec = Y.Decoder(320)
        sc, bx = make(320, [])
        self.assertEqual(dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX), [])
        with self.assertRaises(ValueError): dec.decode(sc[:10], bx, S_SC, Z_SC, S_BX, Z_BX)

    def test_letterbox_and_normalize(self):
        img = np.zeros((576, 1536, 3), np.uint8)
        cv, r, left, top = Y.letterbox(img, 512)
        self.assertEqual(cv.shape, (512, 512, 3)); self.assertAlmostEqual(r, 1 / 3)
        self.assertEqual(left, 0); self.assertEqual(top, (512 - 192) // 2)
        self.assertEqual(int(cv[0, 0, 0]), 114)
        n = Y.to_normalized((0, 0.9, 0, top, 512, top + 192), r, left, top, 1536, 576)
        self.assertEqual(n[0], 0.9); self.assertAlmostEqual(n[1], 0); self.assertAlmostEqual(n[3], 1.0); self.assertAlmostEqual(n[2], 0); self.assertAlmostEqual(n[4], 1.0)
        self.assertEqual(n[5], "person")

    def test_input_lut(self):
        l = Y.input_lut(np.int8, 1 / 255, -128)
        self.assertEqual(int(l[0]), -128); self.assertEqual(int(l[255]), 127); self.assertEqual(int(l[128]), 0)
        l = Y.input_lut(np.uint8, 1 / 255, 0)
        self.assertTrue((l == np.arange(256)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestQuantize(unittest.TestCase):
    def test_lut_quantize_matches_xor128_and_reuses_buffer(self):
        import fake_tpu
        m = fake_tpu.FakeModel(320)
        c = np.random.RandomState(1).randint(0, 256, (320, 320, 3), np.uint8)
        x = Y.EdgeTpuYolo.quantize(m, c)
        self.assertEqual(x.shape, (1, 320, 320, 3)); self.assertEqual(x.dtype, np.int8)
        self.assertTrue(np.array_equal(x[0].astype(np.int16), c.astype(np.int16) - 128))
        buf = m._qbuf
        Y.EdgeTpuYolo.quantize(m, c)
        self.assertIs(m._qbuf, buf)

    def test_per_class_thresholds_and_candidate_cap(self):
        size = 320
        dec = Y.Decoder(size)
        a = anchor_index(size, 16, 5, 7); b = anchor_index(size, 16, 12, 12)
        sc, bx = make(size, [(a, 0, 1.0, (2, 2, 2, 2)), (b, 2, 1.0, (2, 2, 2, 2))])      # person and car both at prob 0.73
        r = dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX, classes=(0, 2), thr={0: 0.5, 2: 0.9})
        self.assertEqual([x[0] for x in r], [0])
        sc2 = np.full((Y.n_anchors(size), 17), 100, np.int8); bx2 = np.zeros((Y.n_anchors(size), 64), np.int8)       # everything is a candidate
        r = dec.decode(sc2, bx2, S_SC, Z_SC, S_BX, Z_BX, classes=(0,), thr=0.3)
        self.assertLessEqual(len(r), Y.MAX_CAND)


class TestCandidateCap(unittest.TestCase):
    def test_top_candidates_survive_the_cap(self):
        size = 320
        dec = Y.Decoder(size)
        n = Y.n_anchors(size)
        sc = np.full((n, 17), int(round(-6 / S_SC)) + Z_SC, np.int8); bx = np.zeros((n, 64), np.int8)
        k = 0; scores = []
        for gy in range(1, 40, 3):
            for gx in range(1, 40, 3):                                   # 169 well separated stride-8 anchors, boxes 16 px wide (no NMS overlap)
                a = anchor_index(size, 8, gx, gy)
                logit = 1.0 + 0.01 * k; k += 1
                sc[a, 0] = int(round(logit / S_SC)) + Z_SC
                for side in range(4):
                    bx[a, side * 16:(side + 1) * 16] = -8 / S_BX; bx[a, side * 16 + 1] = 8 / S_BX
        r = dec.decode(sc, bx, S_SC, Z_SC, S_BX, Z_BX, classes=(0,), thr=0.5)
        self.assertEqual(len(r), Y.MAX_CAND)
        self.assertEqual(Y.MAX_CAND, 100)
        probs = sorted((x[1] for x in r), reverse=True)
        self.assertGreater(probs[-1], 0.73)                              # the 100 best of 169 were kept, not an arbitrary subset
