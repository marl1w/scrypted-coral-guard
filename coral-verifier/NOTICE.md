# Attribution and licences

This add-on is the author's own code; it bundles nothing but fetches third-party components when the image is built.

The detector weights are NOT in this repository. The Dockerfile downloads them at build time from the releases of https://github.com/dbro/frigate-detector-edgetpu-yolo9 (v1.0 for 320, v1.5 `tpumax_512` for 512) and verifies pinned sha256 values. That repository is MIT licensed; the models in it are YOLOv9-s weights trained and compiled from WongKinYiu/yolov9, which is GPL-3.0, so the weights (and anything that embeds them) should be treated as GPL-3.0 derived. Consequences: build the image locally for your own use (this is what the Supervisor local add-on flow does); do not publish a prebuilt image containing the weights without carrying the GPL-3.0 notice and source offer, and do not redistribute the `.tflite` files from this repository.

The image also contains libedgetpu (feranick/libedgetpu build, Apache-2.0), tflite_runtime (frigate-nvr/TFlite-builds, Apache-2.0), OpenCV (Apache-2.0), NumPy (BSD-3-Clause) and Debian packages under their own licences. The Edge TPU decode in `verifier/yolo_edgetpu.py` is adapted from the dbro `edgetpu_tfl.py` plugin (MIT). `tests/ref_tracker.py` is the author's reference tracker simulation, kept for the differential test.

The repository LICENSE covers the code only; it does not cover the weights, which keep the licence above.
