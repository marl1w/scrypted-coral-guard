# Coral Verifier

A small Home Assistant add-on that owns the Coral USB Edge TPU and runs YOLOv9-s int8 (512 input) as a sidecar service. A client (the Scrypted mixin plugin, optionally Home Assistant) posts JPEG frames per camera; the service replies with detections and keeps a per-camera session (person confirmation, hold, stationary lock, parked/moving vehicles incl. bicycles, confirmed animals; detections carry a `group` person|vehicle|animal). It is never in the video path.

The HTTP contract is [API.md](API.md), the source of truth for endpoints, JSON shapes, status codes, contract rulings, rate and latency limits. Licences and attribution (dbro YOLOv9 weights, GPL-3.0 lineage) are in [NOTICE.md](NOTICE.md).

## Layout

`verifier/session.py` pure session logic (no I/O, unit tested, differential-tested against a reference tracker simulation), `verifier/coral.py` the Coral manager (bounded latest-wins queue, one worker thread, child-process supervision), `verifier/child.py` the child process that owns the Edge TPU interpreter, `verifier/detect.py` JPEG guard and decode, letterbox, tiling and merge, `verifier/yolo_edgetpu.py` the Edge TPU YOLOv9 decode, `verifier/server.py` the hardened stdlib HTTP server, `verifier/fake_tpu.py` test-only fake (NOT in the runtime image). Tests are in `tests/`. The `.tflite` models are not in the repo: the Dockerfile fetches them from the `dbro/frigate-detector-edgetpu-yolo9` releases and verifies pinned sha256; the base image is pinned by digest.

## Options

Three, and only three: `token` (REQUIRED, a password: with an empty one the service refuses to start, exit 2, reason in the log; clients send it as `X-Verifier-Token`), `port` (default 8750, host network; if you change it, update the `watchdog` URL in `config.yaml`) and `log_level` (default `info`). Everything else (model, thresholds, hold times, queue and safety limits) is a validated internal default in `verifier/defaults.py`; it is not user-tunable. Tracker values can still be overridden per camera at runtime with `POST /v1/cameras/{camera}/config` (see API.md).

## Health

`GET /v1/health` needs no token and is the metrics endpoint. It returns HTTP 503 (same JSON, `ok:false`) while the Coral is stalled or missing.

`GET /v1/metrics` (token required) feeds a performance page: Coral invoke times and utilization over the last 60 s, per-stage percentiles, load (queue, stale drops, 429s, frames, cameras), host and process CPU, memory, and Intel GPU busy share from `rc6_residency_ms` (`gpu.available:false` when the container cannot read sysfs). Host/process CPU and GPU come from a 5 s background sampler, so a request never waits on them; anything unreadable is `null`. Details in API.md.

## Behaviour notes

Person logic is a streaming port of a 1 fps reference tracker (`tests/ref_tracker.py`): `tests/test_tracker_diff.py` runs both on thousands of random scenarios and requires zero disagreements on `present` and on the confirmation frame, also with sender jitter and 4 fps duplicates (frames less than 0.75 s apart are one tick). Frame time `t` is trusted only inside [now - 60 s, now + 5 s]; otherwise the server time is used and `t_clamped` in health counts it. `GET` state advances with the server clock, so it never shows a stale "present".

The Coral is owned by a CHILD PROCESS (multiprocessing spawn). The parent sends canvases over a pipe and waits at most 2 s per invoke; on a stall or error it SIGKILLs the child and starts a new one (at most one attempt per 10 s, the old child is dead first, so a hung invoke never leaks a thread and never overlaps a second interpreter). Stopping the service kills the child first; the child also dies with its parent (PR_SET_PDEATHSIG). Every state change is logged at ERROR with its reason (for example `cannot load Edge TPU delegate ...`). A failed recovery is a load error, a load hang, a first invoke that fails, or a stall within 30 s of the previous recovery; after 5 of them in a row the service exits with code 3. `GET /v1/health` is the metrics endpoint and returns HTTP 503 (same JSON, `ok:false`) while the Coral is stalled or missing.

Latency policy: bounded queue ordered by request arrival, oldest shed first, frames older than 400 ms (or their `deadline_ms`) answered `503 {"reason":"stale"}` (a skip, not an outage). Send pre-roll bursts sequentially. Per-stage timings are in every reply and as percentiles in health. A camera may burst 10 frames in a second, sustained 4 fps, otherwise 429.

Exposure hardening: JPEG only (FFD8 and a parsable frame header, dimensions checked against the pixel cap before any decode), token checked before any body is read, 2 MB body cap from Content-Length, chunked refused (411), total (not per-recv) deadlines for headers and body, 64 connection cap, 2 concurrent decodes, 64 cameras, no internals in error messages.

Micro-benchmark of the non-Coral cost per frame: `tests/bench_overhead.py` (fake TPU with instant invoke).

## Install

Add this repository to the Home Assistant add-on store (Settings, Add-ons, Add-on store, Repositories), install Coral Verifier (Supervisor builds the image locally, which takes a few minutes), set the `token` option and start it. The Coral is owned exclusively by one process: make sure nothing else uses the stick. Outside Home Assistant, build the Dockerfile and run the container with the USB device passed through, host networking (or port 8750 published) and `/data/options.json` providing `token`, `port` and `log_level`.

The add-on is `boot: manual`; switch it to auto once it has proven itself. Supervisor watches `/v1/health` (`watchdog:` in `config.yaml`, fixed to port 8750) and only restarts a failed container when the add-on's own Watchdog toggle is ON (default OFF). If the stick is absent at start, the service exits with code 3 after about 50 s and, with the watchdog on, restarts until the stick is back.

## Safety

Clients must fail open: any non-200 or a request taking more than 2 s means "service unavailable" and the client behaves as if the verifier did not exist (a stale 503 is a skip). The add-on has `usb: true` and host networking only (no privileged mode, no iGPU, no extra mounts). There is no TLS: keep the port on the LAN. Sessions are in memory and expire by themselves; a restart loses them, which clients tolerate.

## Tests

`docker buildx build --platform linux/amd64 --target test --load -t coral-verifier-test .` then `docker run --rm coral-verifier-test` runs the pytest suite (session scenarios, differential tracker test, decode and engine pixel tests, HTTP end to end with the fake TPU in a real child process, hardening, child-kill and recovery).

Without Docker: `PYTHONPATH=. python -m pytest -q tests` with numpy, opencv-python-headless and pytest installed (the fake TPU replaces the Coral).
