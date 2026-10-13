"""NON-Coral cost per frame with the fake TPU (instant invoke, ~40 candidate boxes so decode/NMS do real work). Not a pytest file.
Run: python tests/bench_overhead.py [N]   (needs numpy + opencv; PYTHONPATH picks up ../verifier automatically)
Prints median/p95 ms for: JPEG decode, preprocess (letterbox + RGB + LUT), post (decode+NMS+normalise), JSON encode, and the sum per frame."""
import json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verifier"))
import numpy as np, cv2
import detect as D, fake_tpu, yolo_edgetpu as Y
import session as S

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = random.Random(1)
size = 512


def scene(w, h):
    img = np.random.RandomState(0).randint(0, 255, (h, w, 3), np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    cv2.rectangle(img, (w // 3, h // 4), (w // 3 + 60, h - 40), (30, 90, 200), -1)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


NITEMS = int(os.environ.get("NITEMS", "40"))      # 40 = busy worst case, 5 = typical scene
items = []
for i in range(NITEMS):
    st = rng.choice((8, 16, 32)); g = size // st
    items.append((rng.choice(("person", "car", "person", "truck")), rng.uniform(0.5, 3.5), st, rng.randrange(g), rng.randrange(g), (rng.randint(1, 4),) * 4))
model = fake_tpu.FakeModel(size, items=items, delay=0)
eng = D.Engine(model)
reg = S.Registry()
pc = lambda xs: (np.median(xs), np.percentile(xs, 95))
for name, (w, h) in {"640x360": (640, 360), "896x512": (896, 512), "1536x576 duo": (1536, 576), "1920x576 duo": (1920, 576)}.items():
    jpg = scene(w, h)
    rows = []
    for i in range(N + 20):
        t0 = time.perf_counter(); img, tiles = D.decode_jpeg(jpg, size, "auto"); t1 = time.perf_counter()
        dets, tm = eng.detect(img, tiles, 0.2); t2 = time.perf_counter()
        sn = reg.update("c", i * 250, dets, 1.0 + i); body = json.dumps({"detections": dets, "person": sn["person"], "vehicle": sn["vehicle"], "session": sn["session"]}, separators=(",", ":")); t3 = time.perf_counter()
        if i >= 20:
            rows.append(((t1 - t0) * 1e3, tm["preprocess_ms"], tm["post_ms"], (t3 - t2) * 1e3, (t3 - t0) * 1e3))
    cols = list(zip(*rows)); lab = ("decode", "preproc", "post", "session+json", "TOTAL")
    print(f"{name:14s} tiles={tiles} decoded={img.shape[1]}x{img.shape[0]} dets={len(dets)} " + "  ".join(f"{l} {pc(c)[0]:.2f}/{pc(c)[1]:.2f}" for l, c in zip(lab, cols)) + "   (median/p95 ms)")
