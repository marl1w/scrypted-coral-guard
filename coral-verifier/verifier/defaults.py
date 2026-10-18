"""Internal defaults of the Coral verifier service: the ONE place where every tuning value lives.

The add-on exposes only three options (token, port, log_level, see config.yaml). The service is validated as shipped, so everything below is
deliberately NOT user-configurable. Values are the validated ones; change them here, in code, with a new validation run.

Overrides for tests and development ONLY (never documented in config.yaml or the README): set an environment variable VERIFIER_<NAME in upper case>,
for example VERIFIER_MAX_CAMERAS=3 or VERIFIER_ALLOW_NO_TOKEN=1. Tests may also pass a dict of overrides straight to server.parse_options().
The tracker values can additionally be overridden per camera through POST /v1/cameras/{id}/config (API.md); that is a runtime contract, not an option.
"""
import os

# Public add-on options: the only keys read from /data/options.json (anything else there is ignored).
PUBLIC_OPTIONS = ("token", "port", "log_level")

# Service settings (name -> default). Ranges are enforced by server.parse_options.
SERVICE = {
    "allow_no_token": False,        # refuse to start with an empty token
    "model": 512,                   # the only validated model (int8 Edge TPU, 512 input)
    "port": 8750,
    "max_queue_age_ms": 400,        # a queued frame older than this is dropped instead of answered late
    "max_queue_depth": 8,
    "journal_root": "/data/journal",
    "spots_root": "/data/spots",    # known parking spots, one JSON per camera (overridable for tests/dev only)
    "lookalikes_root": "/data/lookalikes",    # known look-alikes, one JSON per camera (overridable for tests/dev only)
    "jpeg_reduce": False,           # JPEG reduced-size decode changes pixels on duo frames: off
    "max_cameras": 64,
    "max_image_pixels": 16 * 1024 * 1024,    # 16 MP decompression-bomb guard
    "max_body_bytes": 2 * 1024 * 1024,
    "decode_concurrency": 2,
    "max_connections": 64,
    "request_timeout_s": 5.0,
    "self_restart_after": 5,        # consecutive failed Coral recoveries before exit code 3 (watchdog restart); 0 = never
}

# Event journal (episodes with their frames, reviewed through the ingress page). `journal_root` is overridable for tests/dev only (VERIFIER_JOURNAL_ROOT).
JOURNAL = {
    "ring_frames": 6,               # recent frames kept in memory per camera (to pick first/best/last without disk churn)
    "adopt_s": 5.0,                 # on `start`, ring frames received up to this long before are adopted into the episode
    "snapshot_every_s": 600.0,      # one extra snapshot per this much frame time while an episode is ongoing
    "snapshot_max": 12,             # cap of those extra snapshots per episode
    "retention_days": 14.0,         # closed episodes older than this are deleted
    "max_bytes": 2 * 1024 ** 3,     # total journal size cap: oldest closed episodes are deleted first
    "stale_s": 7200.0,              # an ongoing episode without any update or frame for this long is closed as `stale`
    "maintain_every_s": 30.0,
    "maintain_idle_s": 300.0,       # the worker runs the maintenance at least this often, with or without traffic
    "queue_max": 256,               # bounded queue of journal work (frames, episode calls); frame work is dropped (counted) when full
    "max_ongoing": 64,              # ongoing episodes in total / per camera: more starts are refused with 429
    "max_ongoing_per_camera": 8,
    "clamp_s": 86400.0,             # client times (t_ms) are clamped to server time +- this
    "max_list": 200,
    "merge_after_veto_s": 5.0,      # an unflagged episode starting this soon after a VETOED camera episode of the same camera is merged into it
    "continue_after_s": 30.0,       # a confirmed episode starting this soon after the end of a confirmed, unlabelled one of the same camera is folded into it (0 = off)
    "auto_validate": True,          # label repeat empty frames `correct` without asking (autovalidate.py)
    "auto_det_min": 0.25,           # a box of any group at this score or more makes a frame not empty (no auto-validation, not a reference)
    "auto_pending_s": 120.0,        # an episode waiting for its auto-validation check is shown in Validate anyway after this long
    "ingress_ips": ("172.30.32.2",),   # Home Assistant Supervisor ingress gateway: requests from here with X-Ingress-Path skip the token (journal routes only)
}

# Per-camera tracker settings (name -> default); see session.Config and API.md.
TRACKER = {
    "start_thr": 0.65,              # confidence that starts/confirms a person track (int8 model)
    "cont_thr": 0.2,                # confidence that keeps an existing track alive
    "confirm_n": 2,
    "confirm_m": 3,
    "hold_s": 8.0,
    "still_s": 3.0,                 # stationary-lock reference lag
    "still_hold_s": 30.0,
    "max_hold_s": 600.0,            # service-side safety: a track without a confident detection for this long ends
    "parked_after_s": 120.0,
    "session_ttl_s": 120.0,
    "vehicle_thr": 0.40,            # strong floor of car/truck/bus/motorcycle (bicycle fixed 0.5); the weak band is [WEAK_VEHICLE_THR, vehicle_thr)
    "animal_start_thr": 0.6,        # animal confirmation threshold
    "animal_cont_thr": 0.3,         # animal continuation threshold, also the animal decode floor
    "name": "",                     # display name of the camera (journal and review page); "" = derived from the id
    "second_look": True,            # re-check borderline / brand-new detections on a full-resolution crop (one extra Coral pass each, max SECOND_LOOK["max"] per frame)
    "zoom_out": 0.0,                # 0 = off; > 0 adds ONE Coral pass with the whole frame shrunk to (1 - zoom_out) on a grey canvas (large, edge-cut objects)
    # Fast person confirmation (0.3.12, off unless the client turns it on per camera): one IoU-tracked person track with >= fast_n frames >= start_thr within
    # fast_window_ms of frame time confirms at once when its box centre moved >= fast_move x the box height between its first and last qualifying frame
    # (static look-alikes never move) or the frame carries claim=person (the camera's own AI says person). Everything else keeps the tick rule.
    "fast_confirm": False,
    "fast_n": 2,
    "fast_window_ms": 1000.0,
    "fast_move": 0.03,
    # Minimum size per group (0.3.23): a detection whose box covers less than this share of the frame area never reaches the trackers
    # (it stays in the journal and in verbose replies with too_small: true). 0 = off. Per camera, set by the client (the plugin's camera settings).
    "min_area_person": 0.0,
    "min_area_vehicle": 0.0,
    "min_area_animal": 0.0,
    # Shape hold (0.3.30, see shape.py): a still or parked vehicle the detector misses is kept while the outline inside its box still matches
    # its last look (edge correlation >= shape_thr); covered by another box (>= shape_cover of its area) it is held up to occlusion_hold_s;
    # changed with nothing in front it is dropped at once. Per camera, set by the client (the plugin's camera settings).
    "shape_hold": True,
    "shape_thr": 0.6,
    "shape_cover": 0.1,
    "occlusion_hold_s": 60.0,
    # Known parking spots (0.3.31, see spots.py): learnt from parked vehicles, kept per camera across sessions and restarts.
    "parking_spots": True,
    # Known look-alikes (0.3.31, see lookalikes.py): person/animal boxes learnt from the user's false_alarm labels are ignored where they stand.
    "lookalikes": True,
}
# Known look-alikes (0.3.31): matching constants (lookalikes.py).
LOOKALIKE = {
    "iou": 0.5,               # a detection is at a look-alike's place when it overlaps its box this much
    "thr": 0.85,              # look match that makes it the same object (stricter than the vehicle shape hold)
    "max": 64,                # per camera; the oldest go first
}
# Known parking spots (0.3.31): spot matching and learning constants (spots.py).
SPOTS = {
    "iou": 0.5,               # a box is in a spot when it overlaps it this much
    "trusted_after": 2,       # separate parked occasions before a spot is trusted
    "occasion_gap_s": 600,    # a parked track starting this long after the spot's last one is a new occasion
    "max_spots": 16,          # per camera
    "max_looks": 4,           # vehicle looks kept per spot
    "new_look": 0.95,         # a look this similar to a kept one is not added
    "box_alpha": 0.2,         # the spot box follows its parked vehicles slowly
    "weak_iou": 0.7,          # a weak vehicle box keeps a vehicle parked in a trusted spot only when it overlaps the track this much
}
# Rule B (logged in 0.3.16, live since 0.3.18): claim=person + one person box at or above this, not on a still track, confirms (confirmed_by fast_b).
FAST_B_THR = 0.45
# 0.3.15: with claim=person (the camera's own AI says person) one frame at or above this confirms at once (two independent sources agree).
FAST_SINGLE_CLAIM = 0.75
# 0.3.29: with claim=animal (the camera's own AI says dog/cat) one animal frame at or above this confirms the animal at once (a doubtful one is capped below it).
FAST_SINGLE_CLAIM_ANIMAL = 0.70
# 0.3.14: the two qualifying frames must be at least this far apart (frame time), so "seen twice" is two looks, not two copies.
FAST_MIN_SPAN_MS = 200.0
FAST_IOU = 0.3                  # fast-confirm track matching between consecutive frames (IoU), frames are ~0.15-0.5 s apart

# Detection classes beyond person (COCO names of the 17-class model). group -> {class: threshold}. Person uses the per-frame person_thr (the camera's
# cont_thr); airplane, train, boat and elephant are decoded by nobody and ignored. Internal constants, not add-on options.
DETECT_THR = {
    # animals: floor = ANIMAL["cont_thr"] (0.3); confirmation needs ANIMAL["start_thr"] (0.6)
    "vehicle": {"bicycle": 0.5, "car": TRACKER["vehicle_thr"], "motorcycle": TRACKER["vehicle_thr"], "bus": TRACKER["vehicle_thr"], "truck": TRACKER["vehicle_thr"]},
    "animal": {n: TRACKER["animal_cont_thr"] for n in ("bird", "cat", "dog", "horse", "sheep", "cow", "bear")},
}
# Animal state machine: the person tick logic run independently on animal detections (never shares state with the person tracker).
# Animal decode/reply floor is the continuation threshold (DETECT_THR["animal"] = ANIMAL["cont_thr"]).
# Tiling of wide frames. tiles=auto: width/height <= `auto_aspect` is one pass; wider frames run HYBRID = the WHOLE frame (letterboxed) plus two overlapping
# `two_tile_frac` (55 %) half tiles = 3 invokes. Tile detections never beat the whole-frame pass on big objects: a tile detection is dropped when more than
# `hybrid_contain` of its area lies inside a same-group whole-frame detection, or when it is taller than `hybrid_big_frac` of the frame height and overlaps
# one (IoU > `hybrid_big_iou`). Explicit tiles=2 = the two 55 % tiles only; explicit tiles=3 = three evenly spread tiles with `overlap` (no whole pass).
TILING = {"auto_aspect": 2.2, "overlap": 0.3, "two_tile_frac": 0.55, "hybrid_contain": 0.6, "hybrid_big_frac": 0.25, "hybrid_big_iou": 0.3}

EDGE_MARGIN = 0.02              # multi-tile frames: a box within this fraction of the tile width of an INNER tile edge, fully inside the neighbour tile, is dropped
VEHICLE_NMS_IOU = 0.5           # cross-class NMS over car/truck/bus/motorcycle
VEHICLE_CONTAIN = 0.7           # a vehicle box with more than this share of its area inside a larger vehicle box is dropped
# Vehicle movement (session): a track is `moving` when, against its box at least MOVE_REF_S of frame time earlier, the centre moved more than
# MOVE_DIAG_FRAC of the (normalised) frame diagonal or the IoU fell below MOVE_IOU. New tracks are `still` until they move or become `parked`.
MOVE_REF_S = 2.0
MOVE_DIAG_FRAC = 0.03
MOVE_IOU = 0.7

# Load shedding: a prio=low frame (heartbeat / pixel gate) that reaches the worker while at least BUSY_DEPTH other frames wait runs only its BASE pass(es)
# (single, or the hybrid whole + tiles) and skips the zoom-out pass and the second look; the reply says `reduced: true`.
BUSY_DEPTH = 1

# Second look (internal constants): `confirm_min` = the score the re-check must find the SAME group at for a person/animal detection to count toward starting or
# confirming a track (otherwise it is doubtful: fed to the tracker capped at cont_thr + 0.05); at most `max` re-checks per frame, ttl of a re-check result per object, crop margin (fraction of the box size on each side), minimum crop side in
# pixels, `reject_factor` (a re-check that finds the group again below reject_factor x its keep floor, with a first score below the confirm threshold, rejects), a NEW track is re-checked when its score is below `new_track_max`, and the vehicle confirm threshold used for the borderline band and priority.
SECOND_LOOK = {"max": 2, "confirm_min": {"person": 0.40, "animal": 0.40}, "reject_factor": 0.5, "ttl_s": 10.0, "margin": 0.6, "min_crop_px": 96, "new_track_max": 0.8, "vehicle_confirm": 0.6, "person_skip_area": 0.03}
# 0.3.24 labels: a person box covering >= person_skip_area of the frame is never re-checked. Its window (box + 0.6 margin each side) is larger than
# the frame, so the re-check SHRINKS the person; it put 5 % of real people in doubt and caught 1 false box in 12.

# Display-only weak vehicles: car/truck/bus/motorcycle (not bicycle) are decoded down to WEAK_VEHICLE_THR and reported with "weak": true when the
# score is below the vehicle threshold. They never reach the vehicle tracker or any session state.
WEAK_VEHICLE_THR = 0.25
WEAK_VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")
# Person/animal overlap: an animal-group box with IoU >= this against a person box is the SAME object (a partly hidden person scored as both);
# the person detection takes max(person, animal) for the person tracker ("boosted_by" = the animal class, "first_score" = its own score) and the
# animal box is marked "same_object" and kept out of the animal tracker. A lone animal box never creates a person.
PERSON_ANIMAL_OVERLAP = 0.8
# A person box lying at least this share inside the box of a MOVING vehicle track (or one whose movement is not known yet, its first MOVE_REF_S + 1 s)
# is the driver or a passenger: it never reaches the person tracker (marked "in_vehicle" in the journal, left out of replies). Parked and still
# vehicles are not used, so someone walking past a parked car counts.
PERSON_IN_VEHICLE = 0.8
ANIMAL = {"start_thr": TRACKER["animal_start_thr"], "cont_thr": TRACKER["animal_cont_thr"], "confirm_n": 2, "confirm_m": 3, "hold_s": 6.0, "still_hold_s": 6.0}


def env_overrides():
    """Dev/test overrides from VERIFIER_<NAME> environment variables, for every known service and tracker key."""
    out = {}
    for key, default in {**SERVICE, **TRACKER}.items():
        raw = os.environ.get("VERIFIER_" + key.upper())
        if raw is None:
            continue
        if isinstance(default, bool):
            out[key] = raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            try:
                out[key] = type(default)(raw)
            except ValueError:
                out[key] = raw                  # left as is: parse_options refuses it with a clear message
    return out
