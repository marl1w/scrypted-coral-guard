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
    "queue_max": 256,               # bounded queue of journal work (frames, episode calls); frame work is dropped (counted) when full
    "max_ongoing": 64,              # ongoing episodes in total / per camera: more starts are refused with 429
    "max_ongoing_per_camera": 8,
    "clamp_s": 86400.0,             # client times (t_ms) are clamped to server time +- this
    "max_list": 200,
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
}

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

# Second look (internal constants): `confirm_min` = the score the re-check must find the SAME group at for a person/animal detection to count toward starting or
# confirming a track (otherwise it is doubtful: fed to the tracker capped at cont_thr + 0.05); at most `max` re-checks per frame, ttl of a re-check result per object, crop margin (fraction of the box size on each side), minimum crop side in
# pixels, `reject_factor` (a re-check that finds the group again below reject_factor x its keep floor, with a first score below the confirm threshold, rejects), a NEW track is re-checked when its score is below `new_track_max`, and the vehicle confirm threshold used for the borderline band and priority.
SECOND_LOOK = {"max": 2, "confirm_min": {"person": 0.40, "animal": 0.40}, "reject_factor": 0.5, "ttl_s": 10.0, "margin": 0.6, "min_crop_px": 96, "new_track_max": 0.8, "vehicle_confirm": 0.6}

# Display-only weak vehicles: car/truck/bus/motorcycle (not bicycle) are decoded down to WEAK_VEHICLE_THR and reported with "weak": true when the
# score is below the vehicle threshold. They never reach the vehicle tracker or any session state.
WEAK_VEHICLE_THR = 0.25
WEAK_VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")
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
