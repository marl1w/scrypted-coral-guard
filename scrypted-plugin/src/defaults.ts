// Internal tuning constants. None of these is a user setting: the plugin UI only offers Mode
// (per camera) and the service URL, token and kill switch (global). To change a value, edit it
// here and rebuild. Tests override single fields on this object through the harness.

export const DEFAULTS = {
    /** JPEG sampling and posting interval during an episode, ms (2 fps; the service accepts 4 fps sustained, bursts of 10). The gray gate branch is fixed at 1 fps. */
    frameIntervalMs: 500,
    /** posting interval once an episode is confirmed (holding phase), ms: 1 fps; a re-pend (max hold) goes back to frameIntervalMs. */
    holdIntervalMs: 1000,
    /**
     * 'Fast confirm' burst: while an episode is not yet confirmed, from the scan start for at most burstMaxMs, frames are posted back to
     * back (the next as soon as the previous reply arrives, one in flight, latest wins) and the warm ffmpeg samples JPEGs at burstFps
     * instead of 1000 / frameIntervalMs (runtime switch, no idle cost). Burst frames carry deadline_ms = burstDeadlineMs so a busy
     * service sheds them instead of queueing.
     */
    burstFps: 6,
    burstMaxMs: 1500,
    burstDeadlineMs: 300,
    /** seconds of recent frames handed to the service as pre-roll at a raw trigger (0 = off). */
    preRollS: 4,
    /** frames are scaled down to this width in px, never up. */
    maxWidthNormal: 960,
    /** posted JPEG width cap for wide (aspect > 2) cameras: the service splits those into two overlapping tiles, each needs real pixels. */
    maxWidthWide: 1280,
    /** JPEG quality 1..100 of the posted frames. */
    jpegQuality: 85,
    /** 'Keep warm': one ffmpeg per enabled camera always running (pre-roll from its ring); 'Per trigger': started on raw motion. */
    frameProcess: 'Keep warm' as string,
    /**
     * 'Auto': try VAAPI (every frame is decoded, at 1 fps downstream), fall back to software per camera on failure;
     * 'Off': software only.
     */
    hwDecode: 'Auto' as string,
    /** idle cameras post a fresh frame (low priority, no reset) about this often, jittered per camera: continuous tracking of standing people, animals and vehicles; about one Coral frame every 2 s per camera, shed first when the queue is full. */
    heartbeatMs: 2000,
    /** per-class presence sensors go off when no frame has been analysed for this long (about 7 missed heartbeats). */
    presenceStaleMs: 15000,
    /** after a session reset (camera trigger) a presence sensor that is on stays on this long while the fresh trackers confirm again. */
    presenceResetGraceMs: 4000,
    /** at most this many cameras use VAAPI at once (concurrent sessions are limited); the rest decode in software. */
    maxHwCameras: 64,
    /** LIBVA_DRIVER_NAME for hardware decode ('' = leave unset). */
    vaDriver: '',
    /** Gate: drop a trigger if no person is confirmed within this many seconds (veto). */
    confirmTimeoutS: 6,
    /** safety ceiling: longest time motion is held after a confirmation, seconds. */
    maxHoldS: 600,
    /** frame request timeout, ms: a later verdict is useless, so frames fail open fast. */
    frameTimeoutMs: 800,
    /** health / config / reset request timeout, ms. */
    timeoutMs: 2000,
    /** stream requested for analysis: 'default' = the camera's default stream (no destination: attaches to the rebroadcast session Scrypted usually keeps open); 'low-resolution' = that destination. */
    analysisDestination: 'default' as 'default' | 'low-resolution',
    /** opt-in: pick the analysis stream by resolution (RTSP rebroadcast ids only); false = the 'low-resolution' destination. */
    analysisStreamBySize: false,
    /** exclusion zones: a detection counts only if at least this share of its box lies outside the zones. */
    zoneMinVisible: 0.6,
    /** optional ffmpeg input URL (rtsp:// or rtsps://) used instead of Scrypted's stream; empty = Scrypted stream. */
    streamUrl: '',
    /** pixel-gate options (src/gate.ts); empty = its defaults. Monitoring is shadow-only. */
    gate: {} as Partial<import('./gate').GateOptions>,
    /** at most one gated frame posted per camera per this many ms. */
    gatePostMinMs: 1000,
    /** log/count one 'person seen but camera did not trigger' per camera per this many ms. */
    unflaggedEveryMs: 60000,
    /** service 0.3.35+: default "Maximum person size" (% of the frame area) and "One-frame person minimum height" (% of the frame height) */
    maxPersonPct: 85,
    singleMinHeightPct: 25,
    /** verified ObjectDetector event repeated this often while a confirmed group is held, ms. */
    detectionRepeatMs: 10000,
    /** 'Last detection' picture (MQTT add-on): one per confirmed episode, at most one per camera this often, ms. */
    lastDetectionMinMs: 30000,
    /** the frame drawn must have been analysed at most this long before the confirmation, ms. */
    lastDetectionFrameMs: 15000,
    /** tracker overrides posted to the service; empty object = service defaults, nothing is posted. */
    tracker: {} as { start_thr?: number; cont_thr?: number; hold_s?: number },
};

export type Defaults = typeof DEFAULTS;
