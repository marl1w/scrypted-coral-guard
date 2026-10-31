# Coral Guard

A Scrypted plugin that checks every camera motion trigger with a Google Coral and keeps only the ones where a person, a moving vehicle or an animal is really there. It talks to the Coral verifier service, a separate service that owns the Coral.

## What you get

- Fewer false alerts: rain, insects, shadows and headlights no longer raise motion.
- Recordings that keep going while something is there, even if the camera's own motion stops.
- Near-instant confirmation: a short pre-roll of the seconds before the trigger lets a detection be confirmed within a fraction of a second.
- Presence per class: one occupancy sensor per camera for people, vehicles and animals (still and parked vehicles included), named `{Camera} {Category} Presence`, published to HomeKit and to anything reading Scrypted devices, such as Home Assistant.
- A safe rollout: every camera starts in Shadow (nothing changes, decisions are only measured) and the plugin fails open, so a problem never hides real motion.

## How it works

The plugin attaches to the default (already prebuffered) stream of each enabled camera and keeps the last few seconds of frames. When the camera reports motion, it sends those frames and the live ones (2 per second) to the Coral verifier service, which tracks people, vehicles and animals across frames and says when one is confirmed, still present or gone. Parked vehicles never confirm. Packages and other classes the model cannot verify pass through unverified. A cheap low-resolution pixel check at 1 frame per second also watches for things the camera did not trigger on and asks the service about them. While a camera is idle, one fresh frame every 2 seconds keeps the service tracking people, animals and vehicles that stand still. The per-camera Stream setting picks the analysed stream: Default uses the camera's default stream, which Scrypted usually keeps open (prebuffered) for other plugins and recordings; choosing another stream makes Scrypted open an extra session to the camera. In Gate mode the camera's motion follows the service's answer instead of the raw trigger.

## Requirements

- Scrypted with a camera that has video and motion, and ffmpeg available to Scrypted.
- The Rebroadcast prebuffer on the camera, for the pre-roll.
- A Google Coral accelerator.
- The Coral verifier service reachable from Scrypted (its HTTP contract is in [coral-verifier/API.md](../coral-verifier/API.md)).

## Install and enable

1. Build and deploy the plugin: `npm install && npm run build`, then `npx scrypted-deploy <host>`.
2. Open the plugin's settings and set the Coral service URL (default `http://127.0.0.1:8750`) and the Token if the service uses one. The health line shows whether the service answers.
3. Open a camera, Extensions, and enable Coral Guard. Nothing is enabled automatically.
4. Leave the mode on Shadow for a few days, check the status panel, then switch cameras to Gate one by one.

## Modes

- Off: motion passes through untouched, the service is not called.
- Shadow (default): motion is exactly the camera's own. Coral Guard runs the very same decisions as Gate (confirm, hold and release by the tracker, unflagged detections, zones, claims) and reports what it would publish in the status panel and the camera console.
- Gate: motion stays off until a person, a moving vehicle or an animal is confirmed (choose which in Detect), stays on while it is there, and ends when it is gone. If the camera reports its own class, the same group must confirm it; a claim of a group that is switched off in Detect vetoes the trigger. A confirmed detection can also raise motion when the camera did not trigger at all.

## Settings

- Coral service URL: where the verifier service listens.
- Token: sent with every request; leave empty if the service has none.
- Disable everywhere: kill switch, every camera passes raw motion through at once.
- Triggers (plugin-wide, all by default): Camera motion (any camera motion), Camera AI (only motion the camera labels with a class, waiting up to 1.5 s for the label; cameras without AI labels fall back to Camera motion) and Paced (idle frames every ~2 s and the pixel check). Empty means all.
- Second look (plugin-wide, on by default): re-checks uncertain detections with a zoomed-in pass on the Coral; fewer false alerts, slightly more Coral load.
- Detection thresholds (plugin-wide, 0.05 to 0.95): Person confirm 0.65, Person keep tracking 0.20, Vehicle detect 0.40, Vehicle confirm (moving) 0.60, Animal detect 0.30, Animal confirm 0.60. A confirm value below its keep/detect partner makes that pair fall back to the defaults with a warning. They are sent to the service for every camera and also decide what the status panel draws.
- Service health: read-only, refreshed every 30 seconds and when you save a setting.
- Mode (per camera): Off, Shadow or Gate.
- Detect (per camera): which categories may confirm motion (Person, Vehicle, Animal). A category that is off never confirms, never holds motion and never raises it. Empty means all.
- Status (per camera): the read-only panel described below.

Everything else (frame rate, pre-roll length, timeouts, decoding) is fixed inside the plugin on purpose. The plugin uses hardware decode when available.

Zoom-out (per camera, Off by default) adds a second analysis of each frame shrunk with a border around it, which helps when subjects are large, close or cut by the frame edge; it costs one extra Coral pass per frame for that camera.

Exclusion zones (per camera): name a zone, draw it with Scrypted's zone editor, and detections that lie mostly inside it (less than 60 % of the box outside) never confirm, hold or count; the panel shows the zones and the masked detections.

### Cameras without a motion sensor

Cameras without a motion sensor of their own (for example a derived or cropped stream) can use Coral Guard too: the camera gains a motion sensor that only Coral Guard drives. Shadow never sets it; Gate sets it only when the service confirms something on its own (the same confirm, hold and release rules and Detect filter apply). When the service is unavailable it stays off, because there is no camera motion to pass through.

## Presence sensors

For every enabled camera the plugin adds child devices named `<camera> Person Presence`, `<camera> Vehicle Presence` and `<camera> Animal Presence`, one per detection group the camera has enabled. Each is a Sensor with an occupancy state that is on while the service tracks at least one subject of that group outside the exclusion zones, including still people and still or parked vehicles. This is broader than the camera's own occupancy, which stays tied to confirmed subjects in Gate. The sensors are fed by every analysed frame (camera episodes and the 2 s idle frames) and go off when no frame has been analysed for 15 s (mode Off, kill switch, service down). They are separate devices, so HomeKit publishes them as occupancy sensors and Home Assistant shows them as binary sensors.

## Tamper sensor

Each camera also gets a child device named `<camera> Tamper`: a binary sensor that turns on when the camera is covered, turned away or blurred (defocused, sprayed) for 10 s, and off after 10 s of a clean view. It reuses the small gray frames the pixel gate already decodes (one check every 2 s, no Coral, no extra ffmpeg), so it needs the warm stream. Each check is compared with a reference of the view that follows dusk and dawn slowly, plus a long-term anchor of the sharpest view so a slow spray or defocus is still caught. References are kept per analysed stream, and for day and night separately when the camera has a `<camera> Night mode` sensor (for example from the Reolink plugin); without one, a day/night (IR) switch that keeps the structure re-learns the reference. Brief changes (someone close to the lens, headlights, rain or snow) do not count, and a flat or black view is never learnt. A view that lost its structure in only part of the frame (part of the lens taped or sprayed, a large vehicle close by, IR out of range) raises nothing, but is never learnt either. A tamper that lasts an hour on a view that is not a cover (a re-aimed camera, fog) is accepted as the new view. The status panel shows the state and the reason, and each transition is logged. The per-camera setting "Tamper detection" turns it off (the sensor is removed, as it is when Coral Guard is removed from the camera).

## The status panel

At the top of each camera's Coral Guard section. It refreshes when the page is opened or reloaded (and after you save a setting), never while you edit. It shows the current state (idle, analysing, person confirmed, service unavailable, disabled), the mode, the last analysis with the last frame and boxes for people, vehicles and animals (parked vehicles dimmed), today's totals, typical and slow latencies, the video connection, the service health, the last failure and the recent episodes.

- Last detection: each camera's panel also keeps the most recent frame that contained a detection of an enabled category, with its boxes and when it was seen.
- Overview: the plugin page shows one summary row per camera.
- Performance: the plugin page also shows a snapshot of the service and Coral load (taken every 30 s, press Refresh or reload the page) with trends over the last hour.

## Safety and rollback

- Fail open: any error, timeout, unhealthy service, dead video stream, the kill switch or mode Off makes the camera pass its raw motion through immediately.
- Rollback: set the mode to Off, untick Coral Guard in the camera's Extensions, or turn on Disable everywhere.
- Coral Guard must be the last (outermost) motion mixin on the camera; otherwise another extension masks its result.

## Troubleshooting

- Token rejected: the health line says the service rejected the token; fix the Token setting.
- "Not the outermost MotionSensor mixin" warning: move Coral Guard to the end of the camera's Extensions list.
- No frames: the status panel shows the video as not connected or "no frames" as the last failure; check that the camera streams in Scrypted and look at the camera console for the `frame source` lines.
