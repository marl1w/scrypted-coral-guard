# Coral Guard development notes

Engineering internals. The user-facing description is in [README.md](../README.md).

A Scrypted mixin that sits on a camera's MotionSensor and uses the Coral verifier service (a separate service that owns the Google Coral accelerator; the HTTP contract is in [coral-verifier/API.md](../../coral-verifier/API.md)) to decide whether a raw motion trigger is a real detection. Video recorders, smart-home bridges and automations bind the camera's `motionDetected`, so whatever this mixin publishes is what they see. The service is a sidecar: it is never in the video path and the plugin fails open everywhere. There is no CPU fallback detector: without the service the plugin passes the camera's raw motion through.

## Settings and constants

- Layout: the plugin page has the groups Overview, Performance, Service, Detection; each camera's Coral Guard section has the tabs (subgroups) Overview (Status panel), Setup (Mode, Detect, Stream, Zoom-out) and Zones. Every setting carries a group or subgroup on purpose: an ungrouped one would create an implicit 'General', which the Scrypted UI opens first.
- Provider settings: Coral service URL, Token, Disable everywhere, and the Service health line (polled every 30 s, shown when the page is opened or reloaded, refreshed after a save). The plugin never pushes a Settings event on its own (timers, frames, state changes): that would re-render the page and wipe unsaved edits. Only the user's own save notifies, once.
- Per-camera settings: Mode (Off, Shadow, Gate) and Detect (Person, Vehicle, Animal; empty means all), plus the read-only Status panel.
- Everything else is a constant in `src/defaults.ts`: frame interval (500 ms, 2 fps), pre-roll (4 s), frame width and quality, frame process, hardware decode (Auto, at most `maxHwCameras` cameras at once), confirm timeout (6 s), max hold (600 s), request timeouts, the optional stream override URL, tracker overrides, and the display thresholds.
- The camera id on the service is the name slug plus the Scrypted device id.

Detection thresholds are plugin-wide provider settings (`src/thresholds.ts`, group 'Detection', with the 'Second look' switch sent as `second_look`): clamped to 0.05-0.95, an inverted confirm/keep pair falls back to the defaults with a one-time warning. Every camera posts them to `POST /v1/cameras/{id}/config` as `start_thr`, `cont_thr`, `vehicle_thr`, `animal_start_thr`, `animal_cont_thr` at start, on a change and after a service restart (health `uptime_s` went down); a 400/422 from an older service is retried with the person keys only. The plugin uses Vehicle confirm (moving) for its vehicle guard, and the panel draws persons solid from Person confirm and dashed from Person keep, vehicles from Vehicle detect (service-flagged weak ones below, dashed) and animals from Animal detect.

Sensorless cameras: `canMixin` accepts a Camera or Doorbell with VideoCamera even without MotionSensor (still one mixin per camera) and adds MotionSensor. The mixin detects this from `mixinDeviceInterfaces`, never listens for inner motion, writes `motionDetected = false` once at start (it has no prior value; normal cameras stay transparent at start) and skips the outermost check. With no raw motion, Shadow never publishes (results are logged and counted as unflagged), Gate publishes only through the verified unflagged/heartbeat path (synthetic episode, same confirm/hold/release/max-hold rules, Detect filter, thresholds), and failing open means motion stays false. The panel says 'Sensorless: motion comes only from Coral Guard' and the Overview row is marked.

Exclusion zones (per camera, plugin-side only): the Zones subgroup has an 'Exclusion zones' combobox (type a name to add a zone, remove it to delete) and, right below it in the same Zones tab, one 'Edit zone: <name>' setting of type `clippath` per zone (key `zone:<name>`) (Scrypted's polygon editor over the camera snapshot, as in @scrypted/objectdetector). Storage key `zones` holds JSON `[{"name": "Street", "points": [[x, y], ...]}]` with x, y normalised 0..1 (x right, y down; legacy 0..100 paths are divided by 100). An external script can write the whole list with `putSetting('zones', json)` through the camera's Settings interface; malformed entries are dropped with a warning, a zone with fewer than 3 points never masks. A detection is valid only if at least `zoneMinVisible` (0.6, `src/defaults.ts`) of its box lies outside the union of the zones (sampled on a 64x64 grid per box). The service never sees the zones, so its `person.confirmed` ignores them: with zones the plugin requires a valid person box at or above Person confirm in 2 of the last 3 replies plus a valid one at or above Person keep now, `present` needs a valid box at or above Person keep, animal confirmed/present need a valid animal box, and vehicle decisions see only valid detections. This applies before every decision (confirm, hold, vehicle guard, heartbeat and unflagged) and to the panel counts and Last detection. The panel draws the zones under the boxes (fill at 25 % opacity, dashed outline) and masked boxes dimmed with '(masked)'.

Episode journal (service 0.3.0+): every camera-triggered episode (confirmed or vetoed) and every unflagged episode that confirms (its start is sent at confirmation, back-dated to the scan's trigger; unflagged scans that find nothing are never journalled), in Shadow and Gate, is posted to `POST /v1/cameras/{id}/episodes` as `start` (camera trigger / unflagged), `update` (confirmed <groups>, 'max hold reached: verifying again', 're-confirmed after max hold') and `end` (outcome released, vetoed or aborted; maxhold maps to released) with one uuid per episode. Fire-and-forget, never awaited by decisions; a 404 (older service) turns it off silently until the plugin restarts.

## What it does

On a raw-motion rising edge the mixin resets the camera's session on the service, then hands over the last frames of its warm ring as pre-roll (the last 6, sent sequentially, with true capture times) and keeps posting live frames (one request in flight, latest wins). It follows the service's tracker state.

## Modes

- Off: pass-through, no service calls.
- Shadow (default): `motionDetected` equals the raw motion exactly. One Episode state machine runs with Gate semantics in both modes (confirm, tracker-driven hold and release independent of raw ending, max-hold re-pend, heartbeat re-open, unflagged episodes, Detect, zones, claims, vehicle guard, fail-open); it exposes `wouldPublish`, which Gate publishes and Shadow only reports (summaries, counters, latencies and the panel's 'Coral Guard would publish').
- Gate: `motionDetected` stays false until a confirmation (see below), is held while the service says present, and is released when it says gone (the camera's own `motionTimeout` is ignored). If no confirmation arrives within the confirm timeout the trigger is dropped. A released episode while raw motion is still on does not reopen: it waits for raw to fall and rise again.

## Verified detections (ObjectDetector, 0.3.11)

On cameras with an ObjectDetector the mixin also provides ObjectDetector and, with "Publish verified detections" (plugin-wide, default on) in Gate mode, publishes Coral-verified detections as `verified_person` / `verified_vehicle` / `verified_animal` (`PUBLISH_CLASS` in `src/mixin.ts`), once on confirm and every `detectionRepeatMs` (10 s) while held. The camera's own AI events are not hidden and not re-published: the Scrypted server does not mask property-less events behind a mixin (`notifyInterfaceEventFromMixin` in `server/src/state.ts` has the masking disabled because of ObjectDetector), so every listener receives the camera's classes and the verified ones and tells them apart by name. The mixin still listens below with `mixinId: this.id`, so the camera's raw classes keep working as triggers and claims. 0.3.10 assumed masking and re-published raw events in Off/Shadow, which consumers then received twice.

## Decisions across detection groups, claims and packages

- A camera episode (and an unflagged scan) is confirmed by `person.confirmed`, a vehicle (guard below) or `animal.confirmed`; it is held while any of person.present, a vehicle or animal.present holds, and released when none does. Parked vehicles never confirm. The summary and the log line record which group confirmed (`confirmed: person`).
- Vehicle guard: a vehicle confirms only from a car, truck, bus or motorcycle detection (never a bicycle) at Vehicle confirm (moving) or more (default 0.6) that is not parked, in 2 of the last 3 replies; it is held while such a detection at 0.3 or more keeps arriving and released after about 6 s without. The log line names the class and best score (`confirmed: vehicle (truck 0.62)`).
- Claimed class: when the camera's own detector reports a class (ObjectsDetected events of the inner device, for example people, vehicle, dog_cat), a claim of `person` must be confirmed by a person, `vehicle` by a vehicle, `animal` by an animal. If no class is reported, any group confirms.
- Packages: the model has no such class. A claim of `package` (or any class the model cannot verify) is passed through unverified in Gate mode (fail open, logged as `camera claims <class>: not verifiable, passing through`) and held by the camera's raw motion only.
- Detect: a disabled group is removed from confirmation and from hold; a claimed class that is disabled cannot confirm (veto in Gate); packages and other unverifiable claims still pass through.
- Unflagged monitoring (pixel gate): a cheap 1 fps pixel-change check on a 128x72 gray branch decides when to ask the service about something the camera did not trigger on. In Gate mode it starts a synthetic episode through the same state machine and raises motion only when the service confirms; Shadow only logs and counts per group. Gated frames are posted at low priority.

## Continuous tracking (heartbeat) and Coral load

Any camera with no scan posting frames (Shadow or Gate, service healthy; also while the camera's raw motion stays on after a release, so a closed episode can re-open) posts a fresh warm frame about every 2 s (every fourth ring frame, jittered per camera by device id, `heartbeatMs` in `src/defaults.ts`) with `prio=low` and no reset; one request in flight per camera overall (an episode frame waits for an in-flight heartbeat). The service's tracker therefore follows standing people, animals and vehicles. The replies are judged with the same rules as an episode (person confirmed or present, animal confirmed, a vehicle in 2 of 3 consecutive replies at 0.6 or more and not parked); a confirmed group starts the verified unflagged scan in Gate mode and is only logged and counted in Shadow. A camera episode skips its reset when the session already tracks a present subject (so it is not wiped), and the first frame confirms it at once. Heartbeat failures, timeouts and stale replies are silent. With 2 s heartbeats the vehicle 2-of-3 guard spans about 4 to 6 s of idle replies; the run is broken by any scan, failed or void reply.

Load: every enabled camera adds about one Coral frame every 2 s on top of episode and pixel-gate frames (about 0.5 frames per second per camera), leaving Coral capacity for more tiles per wide frame. Low-priority frames are shed first when the service queue is full and are served only when no normal frame waits, so they never delay an episode.

## Frame pipeline

The analysis stream is the camera's default stream (`getVideoStream` with no destination and no id, 8 s prebuffer), so it attaches to the Rebroadcast session Scrypted usually keeps open (prebuffered) for other plugins and recordings instead of starting a second source (the 'low-resolution' destination often timed out in the rebroadcast and ffmpeg then failed with 'Could not find codec parameters'); `analysisDestination` in `src/defaults.ts` can switch back to 'low-resolution'. The per-camera Stream setting (below Detect) can pick another stream by id (choices read from `getVideoStreamOptions` when the settings render, labelled '<name> WxH'; this opens an extra session to the camera); changing it restarts the frame source at once, the Status Video row names the stream in use, and a chosen stream that delivers no first JPEG (probe failure, exit, 25 s without a frame) or an unknown stored id falls back to the default stream for the rest of the plugin's life with a warning. The stream actually returned (name, resolution) and the posted size are logged per start; the stream options are logged per start and only decide the wide (aspect above 2) width cap. Selection by resolution is opt-in (`analysisStreamBySize` in `src/defaults.ts`, off) and only ever requests RTSP rebroadcast ids, never RTMP: for wide pictures (aspect above 2) the smallest stream at 1280 px or more, otherwise the smallest at 896 px or more, else the largest below 1920 px wide, never above about 2 Mpx; the choice and the posted size are logged per start (`frame source: analysis stream <name> WxH, posting WxH`). The posted JPEG is capped at 1280 px wide for wide cameras (the service splits them into two overlapping tiles) and 960 px otherwise, so decode and encode cost more than before. One warm ffmpeg per enabled camera decodes every frame and samples the JPEG branch at 2 fps (an episode posts every frame while verifying and about 1 fps once confirmed, `holdIntervalMs`); from the already scaled picture a 128x72 gray branch at exactly 1 fps feeds the pixel gate. Hardware decode (VAAPI) is used when available, at most `maxHwCameras` cameras at once, with automatic software fallback per camera (sticky until the plugin restarts); hardware decode uses `-hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -hwaccel_output_format nv12` (frames come back in system memory, no hwdownload step) with the default VA driver (`vaDriver` empty: LIBVA_DRIVER_NAME is not set; i965 failed in tests), on for every camera (`maxHwCameras` 64); a start or runtime failure (stderr naming vaapi, va_, hwaccel, transfer data, sync surface or AVHWFramesContext) restarts that camera in software at once, releases its slot, and is counted on the panel ('CPU (GPU failed)'). A frame-source restart does not abort an episode in progress; only about 10 s without any frame does. The warm stream is requested with an 8 s prebuffer so the Rebroadcast prebuffer can serve a sync frame. If ffmpeg exits before the first JPEG with code 183 or 'Could not find codec parameters', the camera switches for the rest of the plugin's life to the default stream with a longer probe (`-analyzeduration 1000000 -probesize 500000`, replacing any probe options of the input arguments) and logs it.

Diagnostics: once per start the plugin logs the ffmpeg version and input (scheme://host:port, query masked), the timings to first jpeg and first gray, the decode path, and on failure the masked stderr tail.

Optional stream override: set `streamUrl` in `src/defaults.ts` to an `rtsp://` URL served by a local restreamer. The plugin then skips `getVideoStream` and runs ffmpeg directly on that URL (TCP transport, protocol whitelist); pre-roll then comes from the warm ring only.

## Failure behaviour (fail open)

Any HTTP error (except a skip, see below), timeout, malformed reply, an unhealthy `/v1/health` (`ok` false), a dead or silent ffmpeg (no frame for about 10 s), the kill switch, or mode Off makes the camera pass raw motion through immediately. A veto needs at least one verdict from the service: if the the confirm timeout expires with no reply at all, the plugin fails open instead of vetoing. When degraded, frames stop and the plugin probes the service with exponential backoff (1 s doubling, capped at 60 s, plus the 30 s global poll). The probe is health plus an authenticated call (`GET /v1/cameras`), because `/v1/health` needs no token and would otherwise report a wrong token as ready; the backoff only resets once a real frame has been answered, so a service that keeps rejecting frames (for example 400) is retried slowly, not once a second. Each state change is logged once. On recovery while raw motion is on, the published state does not drop: it holds the pass-through value, resets the session and lets the first verdict decide; an unconfirmed episode then ends as a normal veto. All requests are async with AbortController timeouts, so the plugin is never blocked by the service.

## Latency and performance design

- One request in flight per camera, latest wins: a live frame that arrives while a request is running is skipped, never queued. Only the pre-roll burst is sent in order.
- `/frame` has its own short timeout (800 ms; a late verdict is useless). A 503 with `{"reason":"stale"}` and a 429 are skipped frames, not outages; every other non-200 (400, 401, 500, other 503) fails open. Every 200 reply is shape-checked; an invalid one is a failure, never read as "person gone". Response bodies are capped at 1 MB, redirects are refused (the token is never forwarded) and the burst queue holds at most 10 frames (the contract's burst limit).
- Smallest useful frame: width capped at 640 px (never upscaled), JPEG quality 85.
- Keep warm trades constant decode CPU and one open camera stream per enabled camera for a trigger-to-first-frame of about zero; measure RAM and CPU first.
- The JPEG splitter reuses one accumulator (capped at 8 MB, resynchronising on overflow); no per-frame info logging (one line per episode). Timers and episode timestamps use a monotonic clock.
- Generations: replies and the in-flight flag belong to one scan, so a request in flight can never wedge frame posting or confirm the next episode.

## Reading the shadow summary

Each episode logs one line in the camera's console (Scrypted, camera, Console, extension "Coral Guard") and the status panel shows the recent episodes:

`shadow raw 21:04:10-21:04:35 | confirmed +420 ms (frame -1.0s, pre-roll) | released 21:05:02 | held beyond raw 27s | posted +180 ms | verdict +230 ms | service queue 3/infer 31/total 44 ms`

- posted / verdict: ms from the raw trigger to the first frame posted and to the first service verdict. `service queue/infer/total ms` are the service-reported stage timings when present. `N frames skipped` counts frames not analysed (request in flight or stale).
- raw start-end: the camera's own motion window (`ongoing` if still on when the episode closed).
- confirmed +X ms: wall-clock delay from the raw trigger until the person was confirmed. "frame -1.0s" means a pre-roll frame taken before the trigger already completed the confirmation.
- never confirmed / VETOED: Gate would have dropped this trigger. Many of these on real visits mean thresholds or the stream are wrong; on rain or insects they are the point.
- released HH:MM:SS: when the service said gone (or maxhold, or aborted by a failure or the kill switch).
- held beyond raw: seconds Gate would have kept motion on after the camera's raw motion ended.

The status panel and the Mode description also give p50/p95 over the last 50 episodes for trigger to first frame posted, to first verdict and to confirmed.

Before switching a camera to Gate, check in Shadow that real visits show confirmed with latency of a few seconds at most and that vetoes are noise.

## Build, test, deploy

```
npm install
npx tsc --noEmit # strict type-check
npm test         # unit tests with a mock of the service (no Scrypted needed)
npm run build    # scrypted-webpack with the rollup path -> out/plugin.zip
```

The plugin is built with the SDK's rollup path (`"rollup": true` in the `scrypted` block of `package.json`, `module: es2022` and `moduleResolution: bundler` in `tsconfig.json`). The older webpack/CommonJS path fails to load on Scrypted 0.147.0 (`failed to load sdk module ... Cannot convert undefined or null to object`), so do not switch it back. The test suite uses its own `tsconfig.test.json` (Node16 modules).

Deploy (done by the maintainer, not by the build): authenticate once with `npx scrypted login <host>`, then `npx scrypted-deploy <host>` from this directory. The plugin id is `scrypted-coral-guard`, so redeploying keeps per-camera settings.

Point the plugin at the service: when the service runs on the same host as Scrypted, `http://127.0.0.1:8750` works; otherwise use the service host address.

## Rollback

Disable the mixin on the camera (Extensions, untick Coral Guard) or turn on the kill switch in the plugin settings. The kill switch restores pass-through of raw motion within a second. Removing the mixin is different: once removed, a write from this mixin is delivered as an event but not persisted, so the shared `motionDetected` state is only guaranteed to be right again at the camera's next motion write. `release()` makes a best-effort write of the raw value, and you can see a short blip (a held or gated state that lingers until the next camera motion event). Do not rely on it as an instant, guaranteed reset.

## Ordering: Coral Guard must be the outermost MotionSensor mixin

Scrypted publishes the motion state of the outermost mixin that implements MotionSensor. If another MotionSensor mixin (for example the recorder or the object detector) sits outside Coral Guard, its value masks ours and gating has no effect. After publishing a gated value the plugin reads the shared state back, and if it does not match it logs once "Coral Guard is NOT the outermost MotionSensor mixin: gating has no effect" and shows the same warning in the camera's Status line. Fix: open the camera, Extensions, and drag or re-add Coral Guard so it is the last (outermost) entry, then check that the warning is gone.

## Security notes

The token travels in an `X-Verifier-Token` header over plain `http://` to the service. Use it only on a trusted network or loopback; it is not protected against anyone who can sniff that network. Stream overrides accept only `rtsp://` and `rtsps://` URLs and run ffmpeg with a protocol whitelist. ffmpeg errors are logged with credentials and query strings masked. The service camera id may only contain letters, digits, `.`, `_` and `-` (never `.` or `..`); two cameras with the same id are warned about because they would share one service session.

## Not verified without a live Scrypted

Mixin ordering relative to other MotionSensor mixins (recorder, object detector; the plugin detects and reports being masked, see Ordering), how downstream consumers react to the short pass-through to Gate transitions, whether the default stream is prebuffered by the Rebroadcast plugin (pre-roll needs that), and whether `getVideoStream` returns a low-resolution stream for each camera.
