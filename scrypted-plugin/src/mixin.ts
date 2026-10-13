// Per-camera mixin: owns the camera's MotionSensor, listens to the raw (inner) motion event and
// lets the pure Episode state machine decide what to publish. All network I/O is async and
// bounded by AbortController timeouts in client.ts; nothing here ever blocks the plugin.

import sdk, { EventListenerRegister, MotionSensor, ScryptedInterface, Setting, SettingValue, VideoCamera } from '@scrypted/sdk';
import { SettingsMixinDeviceBase, SettingsMixinDeviceOptions } from '@scrypted/sdk/settings-mixin';
import { StorageSettings } from '@scrypted/sdk/storage-settings';
import { FrameReply, groupOf, slugify, TrackerConfig } from './client';
import { DEFAULTS } from './defaults';
import { CameraStatus, StatusState } from './status';
import { backoffMs, Effects, Episode, Group, vehicleView, VEHICLE_HOLD_WINDOW_MS, EpisodeConfig, EpisodeSummary, formatSummary, Mode } from './episode';
import { maskReply, parseZones, Zone } from './zones';
import { DEFAULT_THRESHOLDS, serviceThresholds, Thresholds } from './thresholds';
import { FrameStream, HW_FAILED_PREFIX, PROBE_FAILED_PREFIX, startFrames, streamLabels, vaapiAvailable } from './frames';
import { FrameClock } from './preroll';
import { viewOfReply } from './reply-view';
import { PixelGate } from './gate';
import { maskUrls, monoNow, validCameraId, validStreamUrl } from './util';
import type { GlobalConfig, GuardProvider } from './main';

const DETECT_CHOICES = ['Person', 'Vehicle', 'Animal'];
const MODES = new Map<string, Mode>([['Off', 'off'], ['Shadow', 'shadow'], ['Gate', 'gate']]);
/** the contract accepts a burst of up to 10 frames; never queue more than that. */
const BURST_MAX = 10;
/** warm ring hand-over at a trigger: only the last few frames, sent sequentially, keep the burst short. */
const PREROLL_BURST_MAX = 6;
const VERIFY_DELAY_MS = 300;
const NO_FRAMES_MS = 10000;
/** camera class names that are never a claim about what moved. */
const NOT_A_CLAIM = ['bicycle'];
/** unflagged scan: how long after the last gate opening it keeps scanning, and its hard cap. */
const SYNTH_QUIET_MS = 4000;
const SYNTH_MAX_MS = 20000;
/** warm start: ask the rebroadcast for this much history so it contains a sync frame (it serves from the latest one), covering long GOPs. */
const KEYFRAME_PREBUFFER_MS = 8000;
const MAX_THUMB_JPEG_BYTES = 118 * 1024;
const ZOOM_LABELS: [string, number][] = [['Off', 0], ['20 %', 0.2], ['35 %', 0.35], ['50 %', 0.5]];
const STREAM_DEFAULT_LABEL = 'Default (already open stream)';
/** a chosen (non-default) stream must deliver its first JPEG within this long, else the default stream is used. */
const STREAM_FIRST_FRAME_MS = 25000;

export class GuardMixin extends SettingsMixinDeviceBase<VideoCamera & MotionSensor> implements MotionSensor {
    storageSettings = new StorageSettings(this, {
        mode: {
            title: 'Mode',
            description: 'Off: pass-through. Shadow: motion is unchanged, the Coral decision is only logged. Gate: motion needs a confirmed person.',
            choices: [...MODES.keys()],
            defaultValue: 'Shadow',
        },
        detect: {
            title: 'Detect',
            description: 'Categories that may confirm motion. A category that is off never confirms, never holds motion and never raises it. Empty means all.',
            choices: DETECT_CHOICES,
            multiple: true,
            defaultValue: DETECT_CHOICES,
        },
        stream: {
            title: 'Stream',
            description: 'Default uses the camera\'s default stream, which Scrypted usually keeps open (prebuffered) for other plugins and recordings. Choosing another stream makes Scrypted open an extra session to the camera.',
            choices: [STREAM_DEFAULT_LABEL],
            defaultValue: 'default',
        },
        zoomOut: {
            title: 'Zoom-out',
            description: 'Adds a second analysis of the frame shrunk with a border around it. Helps when subjects are large, close or cut by the frame edge. Costs one extra Coral pass per frame for this camera.',
            choices: ZOOM_LABELS.map(z => z[0]),
            defaultValue: '0',
        },
    });
    /** label -> stream id of the last rendered Stream choices */
    private streamLabels = new Map<string, string>([[STREAM_DEFAULT_LABEL, 'default']]);
    /** the chosen stream failed (or is unknown): the default stream for the rest of the plugin's life */
    private streamFallback = false;
    private streamName = '';
    private streamFirstTimer: NodeJS.Timeout | undefined;

    private episode: Episode;
    private listener: EventListenerRegister | undefined;
    private claimListener: EventListenerRegister | undefined;
    /** the camera's own detection classes (ObjectsDetected events of the inner device), newest last. */
    private claimEvents: { at: number; groups: Group[]; other: string[] }[] = [];
    private claimFrom = 0;
    private unverifiedLogged = false;
    private released = false;
    readonly status = new CameraStatus();
    private lastPresent = false;
    private streamUp = false;
    private streamGpu = false;
    private lastState: StatusState | undefined;
    private tickTimer: NodeJS.Timeout;
    private frames: FrameStream | undefined;
    /** generation of the ffmpeg process: its frame and exit callbacks check it. */
    private srcGen = 0;
    /** generation of the current scan (episode phase): replies and the in-flight flag check it. */
    private scanGen = 0;
    private probeFallback = false;
    private loopStartedAt = 0;
    private lastFrameAt = 0;
    /** when the current ffmpeg process was requested (the watchdog never counts time before it). */
    private clock: FrameClock | undefined;
    private hwDisabled = false;
    private hwFallbacks = 0;
    private gate: PixelGate | undefined;
    private gateBusy = false;
    private lastGatePost = 0;
    private lastUnflagged = 0;
    private nextHeartbeatAt = 0;
    private heartbeatBusy = false;
    private hbTicks: boolean[] = [];
    private hbVehSeenAt = 0;
    private hbPresentAt = 0;
    /** true once a genuine event arrived from the underlying device; before that the mixin never writes motionDetected. */
    private thumbWarned = false;
    private touched = false;
    private cameraRaw = false;
    /** Gate mode: a pixel-gate opening drives the episode machine as synthetic raw motion (never raises motion by itself). */
    private synthRaised = false;
    private synthStartedAt = 0;
    private synthReason = '';
    private synthCooldownUntil = 0;
    private lastGateOpenAt = 0;
    /** pre-roll burst waiting to be sent in order; live frames are never queued. */
    private queue: { jpeg: Buffer; t: number }[] = [];
    /** recent frames of the warm process (exact arrival times), oldest first. */
    private ring: { jpeg: Buffer; t: number }[] = [];
    /** scanGen that has a /frame request in flight (one per camera, latest wins). */
    private busyGen: number | undefined;
    private scanActive = false;
    private warmRetry: NodeJS.Timeout | undefined;
    private warmAttempt = 0;
    private resetPromise: Promise<void> = Promise.resolve();
    private probeTimer: NodeJS.Timeout | undefined;
    private probeAttempt = 0;
    private configPosted = false;
    private dupWarned = false;
    private lastPublished = false;
    private notOutermost = false;
    private verifyTimer: NodeJS.Timeout | undefined;
    /** the camera has no MotionSensor of its own: motion comes only from Coral Guard */
    readonly sensorless: boolean;

    constructor(options: SettingsMixinDeviceOptions<VideoCamera & MotionSensor>, private plugin: GuardProvider) {
        super(options);
        // Startup is fully transparent: nothing is read back as raw motion, replayed or published here.
        // The mixin writes motionDetected only after a genuine event from the underlying device
        // (see `touched`), so a plugin restart never emits a value and never starts a scan.
        this.episode = new Episode(this.episodeConfig(), false);
        // Sensorless: the camera has no MotionSensor of its own, so there is no raw motion; motion comes only from the
        // verified unflagged/heartbeat path (Gate), Shadow never publishes, and failing open means motion stays false.
        this.sensorless = Array.isArray((options as any).mixinDeviceInterfaces) && !(options as any).mixinDeviceInterfaces.includes(ScryptedInterface.MotionSensor);
        this.status.setSensorless(this.sensorless);

        const device = sdk.systemManager.getDeviceById<MotionSensor>(this.id);
        if (!this.sensorless)
            this.listener = device.listen({ event: ScryptedInterface.MotionSensor, mixinId: this.id }, (_s, _d, data) => {
                this.onCameraRaw(!!data);
            });
        else {
            // no prior value exists for a sensor we add: write false once (normal cameras stay fully transparent at start)
            try { this.motionDetected = false; } catch (e) { /* ignore */ }
        }
        // the camera's own AI class (e.g. people / vehicle / dog_cat / package), when the device reports one
        this.claimListener = device.listen({ event: ScryptedInterface.ObjectDetector, mixinId: this.id }, (_s, _d, data) => this.onObjectsDetected(data));
        this.tickTimer = setInterval(() => this.onTick(), 1000);
        plugin.register(this);
        this.syncSource(false);
        this.refreshStatus();
        this.postConfig();
        this.warnDuplicate();
    }

    // ---- configuration -------------------------------------------------------------------

    private get mode(): Mode {
        return MODES.get(this.storageSettings.values.mode) ?? 'shadow';
    }

    /** id of this camera on the service: name slug plus the Scrypted device id, so two cameras never collide. */
    get verifierId(): string {
        const id = String(this.id).replace(/[^A-Za-z0-9._-]+/g, '_');
        return `${slugify(this.name || 'camera').slice(0, 40)}_${id}`.slice(0, 64);
    }

    private warnDuplicate() {
        const other = this.plugin.findDuplicate(this);
        if (other && !this.dupWarned) {
            this.dupWarned = true;
            this.console.warn(`camera id "${this.verifierId}" is also used by "${other}": both would share one service session.`);
        }
    }

    /** Enabled detection groups from the 'Detect' setting; empty or unreadable means all three, never none. */
    private detectGroups(): Group[] {
        let raw: unknown = this.storageSettings.values.detect;
        if (typeof raw === 'string' && raw.trim().startsWith('[')) {
            try { raw = JSON.parse(raw); } catch (e) { /* fall through to the comma form */ }
        }
        const list = Array.isArray(raw) ? raw : typeof raw === 'string' && raw ? raw.split(',') : [];
        const g = DETECT_CHOICES.filter(c => list.map(x => String(x).trim().toLowerCase()).includes(c.toLowerCase())).map(c => c.toLowerCase() as Group);
        return g.length ? g : ['person', 'vehicle', 'animal'];
    }

    private episodeConfig(): EpisodeConfig {
        const v = DEFAULTS;
        const g = this.plugin.config;
        return {
            mode: this.mode,
            confirmTimeoutMs: Math.max(1, v.confirmTimeoutS) * 1000,
            maxHoldMs: Math.max(1, v.maxHoldS) * 1000,
            detect: this.detectGroups(),
            killSwitch: g.killSwitch,
            vehicleConfirm: this.thresholds.vehicleConfirm,
        };
    }

    private get thresholds(): Thresholds {
        return this.plugin.thresholds ?? DEFAULT_THRESHOLDS;
    }

    private viewThresholds() {
        const t = this.thresholds;
        return { person: t.personConfirm, personKeep: t.personKeep, vehicle: t.vehicleDetect, animal: t.animalDetect, enabled: this.detectGroups() as string[], zones: this.zones, minVisible: DEFAULTS.zoneMinVisible };
    }

    // ---- exclusion zones (plugin-side; storage key 'zones', see DEVELOPMENT.md) ----
    private zoneTicks: boolean[] = [];
    private zoneWarned = '';
    get zones(): Zone[] {
        const { zones, warnings } = parseZones(this.storage.getItem('zones'));
        const w = warnings.join('; ');
        if (w && w !== this.zoneWarned) {
            this.zoneWarned = w;
            this.console.warn(w);
        }
        return zones;
    }
    private saveZones(zones: Zone[]) {
        this.storage.setItem('zones', JSON.stringify(zones));
        this.status.setZones(zones);
    }
    /** the reply as decisions see it: masked detections removed, person/animal/vehicle flags need valid boxes */
    private masked(r: FrameReply): FrameReply {
        const t = this.thresholds;
        return maskReply(r, this.zones, DEFAULTS.zoneMinVisible, { personConfirm: t.personConfirm, personKeep: t.personKeep, animalDetect: t.animalDetect }, this.zoneTicks, groupOf);
    }

    /** called when any setting (global or per camera) changes. */
    applyConfig() {
        this.apply(this.episode.setConfig(this.episodeConfig(), monoNow()));
    }

    /** Push the tracker overrides and the plugin-wide thresholds (at start, on a change, after a service restart). */
    postConfig() {
        const person = serviceThresholds(this.thresholds);
        const full: TrackerConfig = { ...DEFAULTS.tracker, ...person, zoom_out: this.zoomOut, second_look: this.plugin.secondLook ?? true };
        this.configPosted = false;
        if (this.released || this.mode === 'off' || this.plugin.config.killSwitch)
            return;
        const personOnly: TrackerConfig = { ...DEFAULTS.tracker, start_thr: person.start_thr, cont_thr: person.cont_thr };
        this.plugin.client.config(this.verifierId, full).catch((e: any) => {
            // an older service may reject the vehicle/animal keys: retry with the person keys only
            if (e?.status === 400 || e?.status === 422)
                return this.plugin.client.config(this.verifierId, personOnly);
            throw e;
        }).then(() => { this.configPosted = true; }).catch(e => {
            // best effort: a failed config post is retried on recovery, it does not change motion.
            this.console.warn('tracker config not applied:', maskUrls(String(e?.message || e)));
        });
    }

    // ---- effects -------------------------------------------------------------------------

    private apply(fx: Effects) {
        if (this.released)
            return;
        // never write before a genuine camera event, except a confirmation (true) or the release of our own true
        if ((this.touched || fx.motion || this.lastPublished) && this.motionDetected !== fx.motion)
            this.publish(fx.motion);
        for (const log of fx.logs) {
            if (log.kind === 'summary') {
                const synth = log.summary.origin === 'synth';
                this.synthRaised = false;
                if (synth)
                    this.synthCooldownUntil = monoNow() + (log.summary.confirmedAfterMs !== undefined ? 5000 : 30000);
                if (!synth || log.summary.confirmedAfterMs !== undefined) {
                    this.console.log('episode:', formatSummary(log.summary));
                    this.feedEpisode(log.summary, synth);
                }
            }
            else if (log.kind === 'degraded') {
                this.console.warn(`verifier unavailable (${maskUrls(log.reason)}); passing raw motion through, retrying with backoff.`);
                this.status.onFailure(maskUrls(log.reason));
                this.scheduleProbe();
            }
            else {
                this.console.log('verifier recovered; resuming.');
                if (!this.configPosted)
                    this.postConfig();
            }
        }
        if (fx.reset && this.hbPresentAt && monoNow() - this.hbPresentAt < 3000) {
            // continuous tracking already holds a present subject on the service: a reset would wipe it, the first frame confirms at once
            this.console.log('session already tracking a present subject, no reset');
        }
        else if (fx.reset) {
            // frames wait for this (contract: the reset must complete before the episode's frames).
            this.resetPromise = this.plugin.client.reset(this.verifierId).then(() => undefined, e => {
                this.apply(this.episode.failure(`reset failed: ${maskUrls(String(e?.message || e))}`, monoNow()));
            });
        }
        this.syncSource(fx.scanning);
        this.refreshStatus();
    }

    private feedEpisode(e: EpisodeSummary, unflagged = false) {
        this.status.onEpisode({
            kind: unflagged ? 'unflagged' : 'camera-triggered',
            rawStartMs: e.startedAt - monoNow() + Date.now(),
            rawEndMs: e.rawEndedAt === undefined ? undefined : e.rawEndedAt - monoNow() + Date.now(),
            postedAfterMs: e.postedAfterMs,
            verdictAfterMs: e.verdictAfterMs,
            confirmedAfterMs: e.confirmedAfterMs,
            confirmedByPreroll: e.confirmedFrameOffsetMs !== undefined && e.confirmedFrameOffsetMs < 0,
            heldBeyondRawS: e.heldBeyondRawS,
            groups: e.confirmedBy,
            vetoed: e.outcome === 'vetoed',
            aborted: e.outcome === 'aborted',
            note: e.note,
        });
    }

    /** Recompute the status state; notify the settings page at once when it changed. */
    private refreshStatus() {
        const g = this.plugin.config;
        const deg = this.episode.degraded;
        const state: StatusState = (this.mode === 'off' || g.killSwitch) ? 'disabled'
            : deg ? 'service unavailable (passing motion through)'
                : !this.episode.scanning ? 'idle'
                    : this.lastPresent ? 'person confirmed' : 'analysing';
        this.status.setDetect(this.detectGroups());
        this.status.setZoomOut(this.zoomOut);
        this.status.setZones(this.zones);
        this.status.setMode([...MODES].find(([, m]) => m === this.mode)?.[0] ?? 'Shadow');
        this.status.setEnabled(state !== 'disabled');
        if (state !== 'disabled')
            this.status.setState(state);
        this.lastState = state;
    }

    /** Write motionDetected and later check the write really became the shared state. */
    private publish(value: boolean) {
        this.lastPublished = value;
        this.motionDetected = value;
        clearTimeout(this.verifyTimer);
        this.verifyTimer = setTimeout(() => this.verifyPublished(), VERIFY_DELAY_MS);
        this.verifyTimer.unref?.();
    }

    /**
     * If our write is not what the shared state shows while it differs from raw motion, another
     * MotionSensor mixin sits outside us and masks it: gating then has no effect.
     */
    private verifyPublished() {
        if (this.released || this.sensorless) // sensorless: the inner device has no motion state to compare with
            return;
        const shown = !!this.mixinDevice.motionDetected;
        const gating = this.lastPublished !== this.episode.isRaw;
        if (!gating)
            return;
        const was = this.notOutermost;
        this.notOutermost = shown !== this.lastPublished;
        if (this.notOutermost && !was) {
            this.console.warn('Coral Guard is NOT the outermost MotionSensor mixin: gating has no effect. Move Coral Guard to the end of this camera\'s Extensions list (last = outermost).');
        }
    }

    /**
     * Tell Scrypted the settings changed. Called ONLY right after the user's own putMixinSetting: any other
     * notification re-renders the settings page and wipes the user's unsaved edits. The Status panel is
     * computed fresh in getMixinSettings, so it refreshes when the page is opened or reloaded.
     */
    private notifySettings() {
        deviceManagerNotify(this);
    }

    private onTick() {
        if (this.released)
            return;
        const now = monoNow();
        // Per trigger: shorter than the confirm timeout, so a dead stream fails open before any veto.
        // A source restart does not abort an episode in progress: only ~10 s without any frame does.
        const watchdog = this.warm ? NO_FRAMES_MS : Math.min(NO_FRAMES_MS, this.episodeConfig().confirmTimeoutMs * 0.75);
        const since = Math.max(this.lastFrameAt, this.loopStartedAt);
        if (this.scanActive && now - since > watchdog) {
            this.console.warn(`no frames from the camera stream: last frame ${this.lastFrameAt ? now - this.lastFrameAt + ' ms ago' : 'never since process start'}, scan ${now - this.loopStartedAt} ms old, watchdog ${Math.round(watchdog)} ms`);
            this.apply(this.episode.failure('no frames from the camera stream', now));
            return;
        }
        if (this.episode.synthetic && (now - this.lastGateOpenAt >= SYNTH_QUIET_MS || now - this.synthStartedAt >= SYNTH_MAX_MS))
            this.apply(this.episode.synthEnd(now));
        this.apply(this.episode.tick(now));
    }

    // ---- frame source ----------------------------------------------------------------------
    // One ffmpeg per camera. Per-trigger mode: started when scanning begins, stopped when it ends.
    // Keep-warm mode: running whenever the verifier is in service for this camera.

    private get warm() {
        return DEFAULTS.frameProcess === 'Keep warm';
    }

    private syncSource(scanning: boolean) {
        // a degraded service does not stop the warm process: restarting it costs seconds of cold start on recovery
        const wantWarm = this.warm && this.mode !== 'off' && !this.plugin.config.killSwitch;
        if (scanning) {
            if (!this.scanActive) {
                this.scanActive = true;
                this.scanGen++; // a new scan: replies and in-flight state of any earlier one are void
                this.hbTicks = []; // a scan breaks the run of consecutive heartbeat replies
                this.zoneTicks = [];
                this.loopStartedAt = monoNow();
                this.queue = [];
                if (this.frames && this.warm) {
                    // hand over the warm ring as the pre-roll burst (exact capture times)
                    const cutoff = monoNow() - DEFAULTS.preRollS * 1000;
                    this.queue = this.ring.filter(f => f.t >= cutoff).slice(-PREROLL_BURST_MAX);
                    this.ring = [];
                    this.kick();
                }
            }
            if (!this.frames && !this.warmRetry)
                this.startSource(); // a pending restart timer (backoff) owns the next spawn
        }
        else {
            if (this.scanActive)
                this.scanGen++; // scan over (also in Keep-warm, where the process keeps running)
            this.scanActive = false;
            this.queue = [];
            if (this.frames && !wantWarm)
                this.stopSource();
            else if (!this.frames && wantWarm && !this.warmRetry)
                this.startSource();
        }
    }

    private startSource() {
        const gen = ++this.srcGen;
        const v = DEFAULTS;
        const interval = Math.max(250, v.frameIntervalMs || 500);
        const warm = this.warm;
        const streamUrl: string = v.streamUrl || '';
        if (streamUrl && !validStreamUrl(streamUrl)) {
            this.console.warn('stream override ignored: only rtsp:// and rtsps:// URLs are accepted.');
            this.apply(this.episode.failure('invalid stream override', monoNow()));
            return;
        }
        const usePreRoll = !warm && !streamUrl && v.preRollS > 0;
        this.clock = new FrameClock(interval, v.preRollS * 1000, usePreRoll);
        this.frames = { stop: () => { }, exited: Promise.resolve(undefined) };
        const ringMax = Math.min(BURST_MAX, Math.ceil((v.preRollS * 1000) / interval) + 1);
        this.streamUp = false;
        this.gate = new PixelGate(DEFAULTS.gate);
        let hwaccel = v.hwDecode !== 'Off' && !this.hwDisabled && vaapiAvailable();
        if (hwaccel) {
            if (this.plugin.acquireHw?.(this, v.maxHwCameras) ?? true) {
                this.console.log(`GPU decode slots: ${this.plugin.hwSlotsUsed ?? '?'}/${v.maxHwCameras} used`);
            }
            else {
                hwaccel = false;
                this.console.log(`GPU decode slots: ${v.maxHwCameras}/${v.maxHwCameras} used, this camera decodes in software`);
            }
        }
        else
            this.plugin.releaseHw?.(this); // a software start holds no slot
        let gotFirst = false;
        const explicitStream = this.chosenStream();
        /** a chosen stream that delivers no first JPEG: the default stream from now on; true when it restarted the source */
        const streamFailed = (): boolean => {
            if (!explicitStream || gotFirst || this.streamFallback || gen !== this.srcGen || this.released)
                return false;
            this.streamFallback = true;
            this.console.warn(`stream ${this.streamName || explicitStream} failed, using the default stream`);
            this.stopSource();
            this.startSource();
            return true;
        };
        clearTimeout(this.streamFirstTimer);
        if (explicitStream) {
            this.streamFirstTimer = setTimeout(() => { streamFailed(); }, STREAM_FIRST_FRAME_MS);
            this.streamFirstTimer.unref?.();
        }
        startFrames({
            camera: this.mixinDevice,
            inputUrl: streamUrl || undefined,
            triggerAt: this.scanActive ? this.loopStartedAt : undefined,
            hwaccel,
            hwFailedBefore: this.hwDisabled,
            vaDriver: v.vaDriver || undefined,
            bySize: DEFAULTS.analysisStreamBySize,
            streamId: this.chosenStream(),
            onStream: st => {
                if (gen !== this.srcGen) return;
                this.streamName = st.name;
                if (st.unknown && !this.streamFallback) {
                    this.streamFallback = true;
                    this.console.warn(`stream ${this.storageSettings.values.stream} is not among the camera's streams, using the default stream`);
                }
            },
            destination: DEFAULTS.analysisDestination,
            longProbe: this.probeFallback,
            minGapMs: Math.round(interval * 0.6),
            gray: warm && this.mode !== 'off' && !this.plugin.config.killSwitch ? { width: 128, height: 72, minGapMs: 600, onGray: f => this.onGray(f) } : undefined,
            intervalMs: interval,
            prebufferMs: streamUrl ? 0 : Math.max(KEYFRAME_PREBUFFER_MS, usePreRoll ? v.preRollS * 1000 : 0),
            maxWidth: v.maxWidthNormal,
            maxWidthWide: v.maxWidthWide,
            quality: v.jpegQuality || 85,
            console: this.console,
            onFrame: (jpeg, arrivedAt) => {
                if (gen !== this.srcGen)
                    return;
                clearTimeout(this.streamFirstTimer);
                gotFirst = true;
                this.lastFrameAt = arrivedAt;
                this.status.setLiveFrame(jpeg, Date.now()); // reference only: the panel shows it until a frame is analysed
                this.warmAttempt = 0;
                if (!this.streamUp) {
                    this.streamUp = true;
                    this.streamGpu = hwaccel;
                    this.status.setStream({ name: this.streamName || undefined, connected: true, decode: hwaccel ? 'gpu' : 'cpu', warm: this.warm, gpuFailed: !hwaccel && this.hwDisabled, gpuFallbacks: this.hwFallbacks });
                }
                if (!this.scanActive) {
                    // warm and idle: keep only the last few frames, no network.
                    this.ring.push({ jpeg, t: arrivedAt });
                    this.heartbeatFrame(jpeg, arrivedAt);
                    if (this.ring.length > ringMax)
                        this.ring.shift();
                    return;
                }
                const t = this.clock!.stamp(arrivedAt);
                if (this.clock!.inBurst) {
                    // pre-roll frame: sent in order, immediately, never waiting for the burst to end.
                    this.queue.push({ jpeg, t });
                    while (this.queue.length > BURST_MAX)
                        this.queue.shift();
                    this.kick();
                }
                else if (this.busyGen === this.scanGen || this.queue.length) {
                    // one request in flight per camera, latest wins: drop, never queue.
                    this.apply(this.episode.skip(arrivedAt));
                }
                else {
                    this.queue.push({ jpeg, t });
                    this.kick();
                }
            },
        }).then(fs => {
            if (gen !== this.srcGen || this.released) {
                fs.stop();
                return;
            }
            this.frames = fs;
            fs.exited.then(err => {
                if (gen !== this.srcGen || this.released)
                    return;
                clearTimeout(this.streamFirstTimer);
                if (streamFailed())
                    return;
                this.frames = undefined;
                this.streamUp = false;
                this.status.setStream({ connected: false, decode: 'unknown', warm: this.warm });
                this.plugin.releaseHw?.(this);
                if (err && err.startsWith(PROBE_FAILED_PREFIX) && !this.probeFallback) {
                    this.probeFallback = true; // sticky until the plugin restarts
                    this.console.warn(`analysis stream could not be probed, using the default stream with a longer probe until the plugin restarts (${err})`);
                    this.status.onFailure('stream probe failed, using the default stream');
                    this.startSource();
                    return;
                }
                if (hwaccel && err && err.startsWith(HW_FAILED_PREFIX)) {
                    this.hwDisabled = true; // sticky until the plugin restarts
                    this.hwFallbacks++;
                    this.console.warn(`frame source: decode GPU failed, falling back to CPU for this camera until the plugin restarts (${maskUrls(err)})`);
                    this.status.onFailure('hardware decode failed, using software');
                    this.startSource();
                    return;
                }
                // the episode (if any) stays alive: the source restarts and the watchdog degrades after ~10 s without frames
                this.scheduleSourceRestart(`frame source ended: ${err || 'eof'}`);
            });
        }).catch(e => {
            if (gen !== this.srcGen || this.released)
                return;
            this.frames = undefined;
            this.plugin.releaseHw?.(this); // a failed start holds no slot
            clearTimeout(this.streamFirstTimer);
            if (streamFailed())
                return;
            this.scheduleSourceRestart(`frame source failed: ${maskUrls(String(e?.message || e))}`);
        });
    }

    /**
     * Continuous monitoring: the pixel gate watches the warm stream. A 'blob' or 'fraction' opening
     * outside any camera episode is only a reason to ASK the service, never to raise motion.
     * Gate mode: it starts a synthetic episode (same state machine as a camera trigger: reset, pre-roll
     * from the ring, ~1 fps frames while the gate stays open and 4 s after, hold/release by the service);
     * motion is raised only when the service confirms a person. Shadow: one frame is posted, a confirmed
     * person is logged and counted, motion is never touched.
     */
    /**
     * Heartbeat (continuous tracking): an idle camera posts its fresh warm frame about every heartbeatMs (2 s, jittered
     * per camera, low priority, never a reset, one in flight) so the service's tracker follows a standing person, an animal or a
     * vehicle. The replies are judged with the SAME rules as an episode (person.confirmed/present, animal.confirmed, a vehicle
     * 2 of 3 consecutive replies at 0.6+ and not parked). A confirmed group starts the verified unflagged scan in Gate mode
     * (hold, release and max-hold re-pend are the episode's) and is only logged and counted in Shadow. A heartbeat never
     * publishes motion by itself; failures, timeouts and stale replies are silent.
     */
    private heartbeatFrame(jpeg: Buffer, t: number) {
        const D = DEFAULTS;
        const now = monoNow();
        if (!this.nextHeartbeatAt)
            this.nextHeartbeatAt = now + heartbeatJitter(this.id, D.heartbeatMs);
        if (now < this.nextHeartbeatAt)
            return;
        if (this.mode === 'off' || this.plugin.config.killSwitch || this.episode.degraded || this.episode.scanning || this.scanActive
            || this.gateBusy || this.heartbeatBusy || this.cameraRaw)
            return;
        this.nextHeartbeatAt = now + D.heartbeatMs - 100; // frames come every ~500 ms: the first one due
        this.heartbeatBusy = true;
        const gen = this.srcGen;
        const sg = this.scanGen; // a scan that started (and maybe ended) while in flight voids this reply
        this.plugin.client.frame(this.verifierId, jpeg, t, true).then(raw => {
            if (!raw || gen !== this.srcGen || sg !== this.scanGen || this.released || this.episode.scanning || this.scanActive || this.cameraRaw
                || this.mode === 'off' || this.plugin.config.killSwitch || this.episode.degraded) {
                this.hbTicks = []; // a skipped, stale or void reply breaks the consecutive run
                return;
            }
            try {
                if (jpeg.length <= MAX_THUMB_JPEG_BYTES)
                    this.status.setLastFrame({ jpeg, boxes: [] });
                else
                    this.status.setThumbTooLarge();
                const view = viewOfReply(raw, this.viewThresholds());
                this.status.onFrameAnalysed({ atMs: Date.now(), ...view.frame, boxes: view.boxes });
            }
            catch (e) { this.warnBadReply(e); return; }
            const reply = this.masked(raw); // exclusion zones: decisions see only valid detections
            const at = monoNow();
            const enabled = this.detectGroups();
            const veh = vehicleView(reply, this.thresholds.vehicleConfirm);
            this.hbTicks.push(veh.tick);
            if (this.hbTicks.length > 3) this.hbTicks.shift();
            if (veh.hold) this.hbVehSeenAt = at;
            const vehiclePresent = veh.hold || (this.hbVehSeenAt > 0 && at - this.hbVehSeenAt <= VEHICLE_HOLD_WINDOW_MS);
            const present = (enabled.includes('person') && !!reply.person?.present) || (enabled.includes('animal') && !!reply.animal?.present) || (enabled.includes('vehicle') && vehiclePresent);
            this.hbPresentAt = present ? at : 0; // the session holds a present subject: a camera episode must not reset it away
            const seen: string[] = [];
            if (enabled.includes('person') && (reply.person?.confirmed || reply.person?.present)) seen.push('person');
            if (enabled.includes('animal') && reply.animal?.confirmed) seen.push('animal');
            if (enabled.includes('vehicle') && this.hbTicks.filter(Boolean).length >= 2) seen.push('vehicle');
            if (!seen.length)
                return;
            if (this.mode === 'gate') {
                if (at >= this.synthCooldownUntil) {
                    this.synthRaised = false;
                    this.synthStartedAt = at;
                    this.lastGateOpenAt = at; // keeps the scan open for the quiet window
                    this.synthReason = 'heartbeat';
                    this.apply(this.episode.synthStart(at));
                }
                return;
            }
            if (at - this.lastUnflagged < D.unflaggedEveryMs)
                return;
            this.lastUnflagged = at;
            this.console.log(`${seen.join('+')} seen but camera did not trigger (heartbeat)`);
            this.status.onEpisode({ kind: 'unflagged', rawStartMs: Date.now(), heldBeyondRawS: 0, groups: seen, vetoed: false, aborted: false, note: 'heartbeat' });
        }, () => { this.hbTicks = []; /* silent: a heartbeat never degrades the camera */ }).finally(() => { this.heartbeatBusy = false; });
    }

    private onGray(frame: Uint8Array) {
        const gate = this.gate;
        if (!gate || this.released || this.mode === 'off' || this.plugin.config.killSwitch || this.episode.degraded)
            return;
        const r = gate.push(frame);
        const hit = r.open && (r.reason === 'blob' || r.reason === 'fraction');
        const now = monoNow();
        if (hit)
            this.lastGateOpenAt = now;
        if (this.mode === 'gate') {
            if (hit && !this.episode.scanning && now >= this.synthCooldownUntil) {
                this.synthRaised = false;
                this.synthStartedAt = now;
                this.synthReason = `gate ${r.reason}, blob ${r.blob}px, ${(r.fraction * 100).toFixed(1)}% changed`;
                this.apply(this.episode.synthStart(now));
            }
            return;
        }
        if (!hit || this.episode.scanning || this.scanActive || this.gateBusy)
            return;
        const last = this.ring[this.ring.length - 1];
        if (!last || now - last.t > 3000 || now - this.lastGatePost < DEFAULTS.gatePostMinMs)
            return;
        this.lastGatePost = now;
        this.gateBusy = true;
        const gen = this.srcGen;
        this.plugin.client.frame(this.verifierId, last.jpeg, last.t, true).then(reply => {
            if (!reply || gen !== this.srcGen || this.released || this.episode.scanning || this.scanActive)
                return;
            const enabled = this.detectGroups();
            const seen = groupsSeen(reply).filter(g => enabled.includes(g as Group));
            if (!seen.length)
                return;
            const at = monoNow();
            if (at - this.lastUnflagged < DEFAULTS.unflaggedEveryMs)
                return;
            this.lastUnflagged = at;
            this.console.log(`${seen.join('+')} seen but camera did not trigger (gate ${r.reason}, blob ${r.blob}px, ${(r.fraction * 100).toFixed(1)}% changed)`);
            this.status.onEpisode({ kind: 'unflagged', rawStartMs: Date.now(), heldBeyondRawS: 0, groups: seen, vetoed: false, aborted: false, note: `gate ${r.reason}` });
        }, () => { /* best effort: the monitoring path never affects motion */ }).finally(() => { this.gateBusy = false; });
    }

    /** The camera's own raw motion changed (the only input that marks the mixin as 'touched'). */
    private onCameraRaw(value: boolean) {
        this.touched = true;
        const rising = value && !this.cameraRaw; // judged from the camera edge alone, whatever a pixel-gate episode is doing
        this.cameraRaw = value;
        if (rising) {
            this.claimFrom = monoNow() - 3000; // class events up to 3 s before the edge belong to this trigger
            this.unverifiedLogged = false;
        }
        this.apply(this.episode.raw(value, monoNow()));
        if (value)
            this.applyClaim();
    }

    /** ObjectsDetected from the camera itself: its claim about what triggered the motion. A hint, never a filter. */
    private onObjectsDetected(data: any) {
        if (this.released)
            return;
        const other: string[] = [];
        let verifiable = false;
        for (const d of data?.detections || []) {
            const c = String(d?.className || '').toLowerCase();
            if (!c || c === 'motion' || NOT_A_CLAIM.includes(c))
                continue;
            if (claimGroup(c)) verifiable = true; else other.push(c);
        }
        if (!verifiable && !other.length)
            return;
        const now = monoNow();
        this.claimEvents = this.claimEvents.filter(e => now - e.at < 10000).concat({ at: now, groups: verifiable ? ['person'] : [], other });
        if (this.cameraRaw)
            this.applyClaim();
    }

    /**
     * Only one claim matters: a class the model cannot verify (package, unknown) passes the camera's trigger
     * through unverified (fail open) in Gate mode. A verifiable claim is just a hint: whatever the service confirms
     * (a person under a 'dog_cat' claim, a person walking past a 'car' claim) confirms. Claims older than the camera
     * trigger minus 3 s are ignored.
     */
    private applyClaim() {
        const ev = this.claimEvents.filter(e => e.at >= this.claimFrom);
        const verifiable = ev.some(e => e.groups.length);
        const other = [...new Set(ev.flatMap(e => e.other))];
        if (!verifiable && other.length && this.mode === 'gate' && !this.unverifiedLogged) {
            this.unverifiedLogged = true;
            this.console.log(`camera claims ${other.join('+')}: not verifiable, passing through`);
            this.apply(this.episode.acceptUnverified(monoNow()));
        }
    }

    private badReplyWarned = false;
    private warnBadReply(e: unknown) {
        if (this.badReplyWarned)
            return;
        this.badReplyWarned = true;
        this.console.warn(`a service reply could not be interpreted, frame skipped: ${maskUrls(String((e as any)?.message || e))}`);
    }

    private scheduleSourceRestart(why: string) {
        if (!(this.warm || this.scanActive) || this.released)
            return;
        if (this.warmRetry)
            return; // one pending restart at a time
        const delay = backoffMs(this.warmAttempt++);
        if (this.warmAttempt === 1)
            this.console.warn(`${maskUrls(why)}; restarting the frame process with backoff.`);
        this.warmRetry = setTimeout(() => {
            this.warmRetry = undefined;
            if (!this.released && !this.frames)
                this.syncSource(this.scanActive);
        }, delay);
    }

    private stopSource() {
        this.srcGen++;
        this.scanGen++; // an in-flight request of the old process no longer blocks, its reply is void
        this.frames?.stop();
        this.frames = undefined;
        this.plugin.releaseHw?.(this);
        this.queue = [];
        this.ring = [];
        this.clock = undefined;
    }

    /** Send the next queued frame if none is in flight for this scan. */
    private kick() {
        const gen = this.scanGen;
        if (this.busyGen === gen || this.released || !this.queue.length)
            return;
        this.busyGen = gen;
        void this.send(gen);
    }

    private async send(gen: number) {
        try {
            await this.resetPromise; // never post frames before the episode's reset completed
            if (gen !== this.scanGen || this.released)
                return;
            const item = this.queue.shift();
            if (!item)
                return;
            this.episode.posted(monoNow());
            if (item.jpeg.length <= MAX_THUMB_JPEG_BYTES) // base64 happens lazily when the panel renders
                this.status.setLastFrame({ jpeg: item.jpeg, boxes: [] });
            else {
                this.status.setThumbTooLarge(); // the previous thumbnail stays; the panel says so when there is none
                if (!this.thumbWarned) {
                    this.thumbWarned = true;
                    this.console.log(`snapshot ${Math.round(item.jpeg.length / 1024)} KB is too large for the panel thumbnail (limit ${Math.round(MAX_THUMB_JPEG_BYTES / 1024)} KB), keeping the previous one`);
                }
            }
            const reply = await this.plugin.client.frame(this.verifierId, item.jpeg, item.t, this.episode.synthetic);
            if (gen !== this.scanGen || this.released)
                return;
            if (reply === null) {
                this.status.onFrameSkipped();
                this.apply(this.episode.skip(monoNow())); // stale or 429: skipped, not an outage
            }
            else {
                try {
                    const view = viewOfReply(reply, this.viewThresholds());
                    this.lastPresent = view.present;
                    this.status.onFrameAnalysed({ atMs: Date.now(), ...view.frame, boxes: view.boxes });
                }
                catch (e) { this.warnBadReply(e); }
                this.probeAttempt = 0; // a real frame succeeded: only now is the service really back
                let fx: Effects;
                try { fx = this.episode.reply(this.masked(reply), monoNow(), item.t); }
                catch (e) { this.warnBadReply(e); fx = this.episode.skip(monoNow()); } // a malformed field is a skipped frame, never an outage
                this.apply(fx);
                if (this.episode.synthetic && !this.synthRaised && this.mode === 'gate' && this.episode.motion) {
                    this.synthRaised = true;
                    this.console.log(`${groupsSeen(reply).join('+') || 'person'} seen but camera did not trigger: raised motion (${this.synthReason})`);
                }
            }
        }
        catch (e: any) {
            if (gen === this.scanGen && !this.released) {
                this.queue = [];
                this.apply(this.episode.failure(maskUrls(e?.message || String(e)), monoNow()));
            }
        }
        finally {
            if (this.busyGen === gen)
                this.busyGen = undefined;
            this.kick(); // current generation, whatever the old one was
        }
    }

    // ---- health / recovery ---------------------------------------------------------------

    /** global health poll result (every 30 s and on demand). */
    onHealth(ok: boolean, reason: string) {
        if (this.released)
            return;
        this.status.onServiceHealth({ ok, text: ok ? '' : reason });
        this.apply(ok ? this.episode.healthy(monoNow()) : this.episode.failure(reason, monoNow()));
    }

    private scheduleProbe() {
        clearTimeout(this.probeTimer);
        if (this.released || !this.episode.degraded)
            return;
        const delay = backoffMs(this.probeAttempt++);
        this.probeTimer = setTimeout(async () => {
            if (this.released || !this.episode.degraded)
                return;
            try {
                const h = await this.plugin.client.check();
                if (h.ok) {
                    this.apply(this.episode.healthy(monoNow()));
                    return;
                }
            }
            catch (e) { /* still down */ }
            this.scheduleProbe();
        }, delay);
    }

    // ---- settings ------------------------------------------------------------------------

    async getMixinSettings(): Promise<Setting[]> {
        const settings = await this.storageSettings.getSettings();
        this.refreshStatus();
        {
            const zones = this.zones;
            settings.push({ key: 'zoneNames', subgroup: 'Zones', title: 'Exclusion zones', type: 'string', multiple: true, combobox: true, choices: zones.map(z => z.name), value: zones.map(z => z.name) as any,
                description: 'Type a name and press Enter to add an exclusion zone, then draw it with its zone editor. Remove a name to delete the zone. A detection counts only if most of its box (60 %) lies outside the zones.' } as Setting);
            for (const z of zones)
                settings.push({ key: `zone:${z.name}`, subgroup: 'Zones', title: `Edit zone: ${z.name}`, type: 'clippath', value: JSON.stringify(z.points) } as Setting);
        }
        const zo = settings.find(x => x.key === 'zoomOut');
        if (zo)
            zo.choices = ZOOM_LABELS.map(z => z[0]), zo.value = ZOOM_LABELS.find(z => z[1] === this.zoomOut)?.[0] ?? 'Off';
        const st = settings.find(x => x.key === 'stream');
        if (st) {
            this.streamLabels = new Map([[STREAM_DEFAULT_LABEL, 'default']]);
            try {
                for (const l of streamLabels(await this.mixinDevice.getVideoStreamOptions?.()))
                    this.streamLabels.set(l.label, l.id);
            }
            catch (e) { /* only the default choice */ }
            st.choices = [...this.streamLabels.keys()];
            const cur = String(this.storageSettings.values.stream || 'default');
            st.value = [...this.streamLabels].find(([, id]) => id === cur)?.[0] ?? STREAM_DEFAULT_LABEL;
        }
        const mode = settings.find(x => x.key === 'mode');
        if (mode)
            mode.description = `${this.notOutermost ? 'WARNING: not the outermost MotionSensor mixin, gating has no effect. ' : ''}${(() => { const d = this.detectGroups(); return d.length === 3 ? '' : `Detect: ${d.join(', ')} only. `; })()}${this.status.summaryLine(Date.now())}`;
        // one group: the Status panel on top, then Mode
        // tabs (subgroups): Overview (the Status panel), Setup (Mode, Detect, Stream, Zoom-out), Zones
        for (const x of settings)
            if (!x.subgroup) x.subgroup = 'Setup'; // never ungrouped: no implicit 'General' tab
        return [
            { key: 'status', subgroup: 'Overview', title: 'Status', type: 'html', readonly: true, value: this.status.render(Date.now()) },
            ...settings.filter(x => x.subgroup === 'Setup'),
            ...settings.filter(x => x.subgroup !== 'Setup'),
        ];
    }

    async putMixinSetting(key: string, value: SettingValue): Promise<boolean | void> {
        if (key === 'zoneNames' || key === 'zones' || key.startsWith('zone:')) {
            let zones = this.zones;
            if (key === 'zoneNames') {
                const names = (Array.isArray(value) ? value : value === undefined || value === null || value === '' ? [] : [value]).map(v => String(v).trim().slice(0, 40)).filter(Boolean);
                zones = [...new Set(names)].map(name => zones.find(z => z.name === name) ?? { name, points: [] });
            }
            else if (key === 'zones') {
                const p = parseZones(value as any);
                if (p.warnings.length) this.console.warn(p.warnings.join('; '));
                zones = p.zones;
            }
            else {
                const name = key.slice('zone:'.length);
                let pts: unknown = value;
                if (typeof value === 'string') { try { pts = JSON.parse(value); } catch { pts = undefined; } }
                const parsed = parseZones([{ name, points: pts }]);
                if (parsed.warnings.length || !zones.some(z => z.name === name)) { this.console.warn(`zone ${name}: invalid polygon, not saved`); return true; }
                zones = zones.map(z => z.name === name ? parsed.zones[0] : z);
            }
            this.saveZones(zones);
            this.console.log(`Exclusion zones: ${zones.map(z => `${z.name} (${z.points.length} points)`).join(', ') || 'none'}`);
            this.refreshStatus();
            this.notifySettings();
            return true;
        }
        if (key === 'zoomOut') {
            const z = ZOOM_LABELS.find(x => x[0] === String(value))?.[1] ?? (Number(value) || 0);
            await this.storageSettings.putSetting(key, String(z));
            this.console.log(`Zoom-out: ${value}`);
            this.postConfig(); // re-posted at once
            this.refreshStatus();
            this.notifySettings();
            return true;
        }
        if (key === 'stream') {
            const id = this.streamLabels.get(String(value)) ?? (String(value) === STREAM_DEFAULT_LABEL ? 'default' : String(value));
            await this.storageSettings.putSetting(key, id);
            this.console.log(`Stream: ${value}`);
            this.streamFallback = false; // a new choice gets a new chance
            this.streamName = '';
            if (this.frames || this.warmRetry) {
                clearTimeout(this.warmRetry);
                this.warmRetry = undefined;
                this.stopSource(); // restart at once on the new stream
                this.syncSource(this.scanActive);
            }
            this.refreshStatus();
            this.notifySettings();
            return true;
        }
        await this.storageSettings.putSetting(key, value);
        if (key === 'detect')
            this.console.log(`Detect: ${this.detectGroups().join(', ')}`);
        else if (key === 'mode')
            this.console.log(`Mode: ${value}`);
        this.postConfig();
        this.applyConfig();
        this.refreshStatus();
        this.notifySettings();
        return true; // handled: the base class must not notify a second time
    }

    /** Zoom-out share (0 = off), only the offered values. */
    private get zoomOut(): number {
        const v = Number(this.storageSettings.values.zoomOut);
        return ZOOM_LABELS.some(z => z[1] === v) ? v : 0;
    }

    /** this camera's row in the plugin Overview */
    overviewRow() {
        this.refreshStatus();
        return { name: `${String(this.name || this.id)}${this.sensorless ? ' (sensorless)' : ''}`, ...this.status.overviewCells() };
    }

    /** the stream id to request, undefined = the default stream */
    private chosenStream(): string | undefined {
        const v = String(this.storageSettings.values.stream || 'default');
        return v === 'default' || this.streamFallback ? undefined : v;
    }

    async release() {
        this.released = true;
        clearTimeout(this.streamFirstTimer);
        this.stopSource();
        clearTimeout(this.warmRetry);
        clearInterval(this.tickTimer);
        clearTimeout(this.probeTimer);
        clearTimeout(this.verifyTimer);
        this.listener?.removeListener();
        this.claimListener?.removeListener();
        this.plugin.unregister(this);
        // Best effort only: once removed, a write from this mixin is an event, not persisted state.
        // The next motion write from the camera re-establishes the real state.
        try { if ((this.touched || this.lastPublished) && this.motionDetected !== this.episode.isRaw) this.motionDetected = this.episode.isRaw; } catch (e) { /* ignore */ }
        await super.release();
    }
}

/** Per-camera offset (0..ms) derived from the device id, so heartbeats of many cameras do not burst together. */
export function heartbeatJitter(id: string, ms: number): number {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
    return h % Math.max(1, ms);
}

/** Map a camera class name to the detection group the service can verify (undefined: not verifiable). */
export function claimGroup(c: string): Group | undefined {
    if (['people', 'person', 'face'].includes(c)) return 'person';
    if (['vehicle', 'car', 'truck', 'bus', 'motorcycle'].includes(c)) return 'vehicle';
    if (['dog_cat', 'animal', 'dog', 'cat', 'pet'].includes(c)) return 'animal';
    return undefined;
}

/** Groups the service reports present (parked vehicles are not 'moving' and never count). */
function groupsSeen(r: FrameReply): string[] {
    return [r.person?.present && 'person', r.vehicle?.moving && 'vehicle', r.animal?.present && 'animal'].filter(Boolean) as string[];
}

function deviceManagerNotify(m: GuardMixin) {
    // refresh the settings UI so Status and Rolling summary update.
    sdk.deviceManager.onMixinEvent(m.id, m, ScryptedInterface.Settings, null);
}
