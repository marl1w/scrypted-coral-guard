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
    /**
     * Tamper detector (src/tamper.ts), per camera, on the pixel gate's gray frames (warm stream only). Each sample is the
     * per-pixel median of the last 3 samples at 64x36, so anything shorter than about one sample interval (rain, snow,
     * headlight sweeps, someone walking past) never reaches the measures. Measures, each compared with the reference:
     * mean and std (gray levels), edge energy (RMS Laplacian), sharpness (edge / std: independent of a global gain such as
     * the IR switch) and the similarity (NCC) of a 32x18 gradient map. One reference per profile: the analysed stream, and
     * day / night when the camera has a "<camera> Night mode" sensor. Each reference has an anchor (its sharpest stable view).
     * Checks, in order: covered (whole or part of the view flat), blurred, moved, lighting step (re-learn), not explained
     * (reference frozen), clean (the reference follows).
     */
    tamper: {
        /** gray frame size as delivered to the gate (even numbers). */
        width: 128, height: 72,
        /** one sample per this many ms (gray frames arrive at 1 fps); a background check, not per detection. */
        sampleMs: 2000,
        /** a condition must hold this long before the Tamper sensor turns on; a clean view this long before it turns off. A person standing close for a few seconds stays below it. */
        onMs: 10000, offMs: 10000,
        /** reference time constant (EMA over explained clean samples only, frozen while a condition is pending or the sensor is on): follows dusk and dawn, not a sudden change. */
        refTauMs: 600000,
        /** covered: std AND edge energy both below 30 % of the reference (a hand, tape, a bag, a cap over the lens). */
        coveredStdRatio: 0.3, coveredEdgeRatio: 0.3,
        /** covered: std under 4 levels is a flat picture whatever the edges, unless the reference is itself almost flat. A view with std under 8 is never learnt. */
        coveredStdAbs: 4,
        /** covered: mean under 16 or over 240 (black, or IR glare off something on the lens) with the std below half the reference. Such a view is never learnt; with no reference for the profile it only shows 'learning: view looks flat' (never an alarm). */
        darkLevel: 16, brightLevel: 240, brightStdRatio: 0.5,
        /**
         * damage safeguard, never an alarm: when at least 25 % of the anchor's textured cells (8x6 grid) fell below 15 % of their
         * gradient energy (corrected by the global gain, from the upper quartile of the cells), the sample stays a pending
         * 'changed': no re-learn, no anchor refresh, no accept. Covers IR out of range in part of the view, a smooth van close by,
         * part of the lens sprayed or taped: nothing is learnt from it, and nothing is raised in this version.
         */
        damageFraction: 0.25, damageCellRatio: 0.15,
        /** blurred: sharpness below 40 % of the reference or of the anchor, with the mean within 40 levels or the structure lost too (defocus, spray, a smeared lens). */
        blurRatio: 0.4, blurMaxLevelDelta: 40,
        /** moved: gradient-map similarity below 0.4 (same view 0.8-1, a 20 % turn about 0) while sharpness is at least 60 % of the reference. Checked before the lighting step: turned and brightened is moved. */
        movedNcc: 0.4, sharpKeepRatio: 0.6,
        /** the similarity is the best over global shifts of up to 1 gradient-map cell (about 3 % of the frame): wind shake is not 'moved'. */
        maxShift: 1,
        /**
         * lighting change: the mean jumps more than 30 levels, or the std by more than x1.5 either way, with sharpness kept AND a
         * similarity of at least 0.6 (the gradient map ignores gain, so a real IR switch or floodlight keeps it): the reference is
         * re-learnt, never tamper. Only while the sensor is off. Any other sample under 0.6 or not sharp is not explained: the
         * reference is frozen, and re-learnt only after 10 min of that (a parked vehicle, a moved garden chair).
         */
        relearnLevels: 30, relearnStdRatio: 1.5, relearnNcc: 0.6, changedRelearnMs: 600000,
        /**
         * anchor: the reference sharpness never drops below 60 % of the anchor, and 'blurred' is also judged against the anchor, so a
         * slow spray or defocus (hours) is caught. The anchor's map follows only 2 h of unbroken clean samples at 85 % of its
         * sharpness or more; its sharpness never declines.
         */
        anchorClampRatio: 0.6, anchorRefreshMs: 7200000, anchorKeepRatio: 0.85,
        /**
         * on for this long: the view becomes the reference and the sensor clears (a re-aimed camera, fog, condensation); never a flat
         * or damaged view, nor a 'covered' one that lost its structure (a real cover stays on). An accepted 'moved' also drops the
         * other profiles of the stream (the old night or day view). 0 = never.
         */
        acceptMs: 3600000,
        /** after a day/night or stream switch the samples are not judged for this long (the picture may switch a few seconds apart); a pending tamper run is kept across the switch. */
        settleMs: 10000,
        /** more than 4 profile switches in 2 min (day/night flapping as a hand covers and uncovers the IR sensor, glare) count as 'covered'. */
        flapFlips: 4, flapWindowMs: 120000,
        /** samples further apart than this (stream restart): the median history and a pending condition start again; the references are kept. */
        gapMs: 15000,
    },
};

export type Defaults = typeof DEFAULTS;
