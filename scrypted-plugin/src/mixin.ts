// Per-camera mixin: owns the camera's MotionSensor, listens to the raw (inner) motion event and
// lets the pure Episode state machine decide what to publish. All network I/O is async and
// bounded by AbortController timeouts in client.ts; nothing here ever blocks the plugin.

import sdk, { EventListenerRegister, MediaObject, MotionSensor, ObjectDetectionTypes, OccupancySensor, ObjectDetector, ObjectsDetected, ScryptedInterface, Setting, SettingValue, VideoCamera } from '@scrypted/sdk';
import { SettingsMixinDeviceBase, SettingsMixinDeviceOptions } from '@scrypted/sdk/settings-mixin';
import { StorageSettings } from '@scrypted/sdk/storage-settings';
import { OverviewCamera, ViewInfo, viewInfoOf } from './overview';
import { randomUUID } from 'crypto';
import { EpisodeEvent, FrameReply, groupOf, slugify, TrackerConfig } from './client';
import { DEFAULTS } from './defaults';
import { CameraStatus, StatusState } from './status';
import { backoffMs, Effects, Episode, Group, vehicleView, VEHICLE_HOLD_WINDOW_MS, EpisodeConfig, EpisodeSummary, formatSummary, Mode } from './episode';
import { isMasked, maskReply, parseZones, Zone } from './zones';
import { PRESENCE_GROUPS, PresenceGroup, presenceAfterReset, presenceOf } from './presence';
import type { NowInfo, NowReason, TrackedItem } from './status';
import { applyAnimalOverride, DEFAULT_THRESHOLDS, serviceThresholds, Thresholds } from './thresholds';
import { fetchSnapshot, FastPathTimer, SNAP_MAX_FAILS } from './fastpath';
import { FrameStream, HW_FAILED_PREFIX, PROBE_FAILED_PREFIX, startFrames, streamLabels, vaapiAvailable } from './frames';
import { FrameClock } from './preroll';
import { viewOfReply } from './reply-view';
import type { DrawBox } from './annotate';
import type { Box } from './status';
import { PixelGate } from './gate';
import { maskUrls, monoNow, validCameraId, validStreamUrl } from './util';
import type { GlobalConfig, GuardProvider } from './main';

const DETECT_CHOICES = ['Person', 'Vehicle', 'Animal'];
const MODES = new Map<string, Mode>([['Off', 'off'], ['Shadow', 'shadow'], ['Gate', 'gate']]);
/** the contract accepts a burst of up to 10 frames; never queue more than that. */
const BURST_MAX = 10;
/** warm ring hand-over at a trigger: only the newest frames (about the last 1.5 s), sent sequentially, so the live frames follow quickly. */
const PREROLL_BURST_MAX = 3;
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
/** 'Camera AI' only: how long a raw edge waits for the camera's class claim. */
const AI_CLAIM_WAIT_MS = 1500;
const STREAM_DEFAULT_LABEL = 'Default (already open stream)';
/** a chosen (non-default) stream must deliver its first JPEG within this long, else the default stream is used. */
const STREAM_FIRST_FRAME_MS = 25000;
/** camera storage key of the journal episode in progress (closed at the next start if a restart cut it). */
const OPEN_EPISODE_KEY = 'journalOpenEpisode';
/** while a camera analyses the default stream because its chosen one failed, the chosen one is retried after these delays (the last repeats). */
const STREAM_RETRY_MS = [60_000, 120_000, 300_000];
/** a retry due during an episode waits this long instead (a retry may leave the camera without frames for up to STREAM_FIRST_FRAME_MS). */
const STREAM_RETRY_BUSY_MS = 30_000;
/**
 * Class names of the verified detections published on the camera's ObjectDetector. Distinct from any camera AI name on
 * purpose: Scrypted does not let a mixin mask the camera's own ObjectDetector events (the server disables masking of
 * property-less events), so consumers always receive both and tell them apart by name.
 */
export const PUBLISH_CLASS: Record<Group, string> = { person: 'verified_person', vehicle: 'verified_vehicle', animal: 'verified_animal' };

/** GPU decode is probed again this long after a failure: 5 min, doubling per consecutive failed probe, at most 1 h. */
export function hwRetryMs(failures: number): number {
    return backoffMs(failures, 5 * 60_000, 60 * 60_000);
}
/** a GPU probe must deliver its first JPEG within this long (the GPU start of the camera's own process gets HW_FIRST_FRAME_MS). */
const HW_PROBE_FIRST_FRAME_MS = 8000;
/** hard cap of a probe, stream resolution included. */
const HW_PROBE_CAP_MS = 30_000;
/** after this many consecutive GPU failures (failed probes, GPU processes failing after a switch) the camera stays in software until the plugin restarts. */
const HW_PROBE_MAX_FAILS = 6;
/** a GPU process back after a probe resets the backoff only once it has delivered frames this long without failing. */
const HW_CLEAN_RUN_MS = 30 * 60_000;
/** a failed probe counts as a GPU failure only if the camera's own process delivered a frame within this long (else the stream, not the GPU, is the suspect). */
const HW_PROBE_HEALTHY_MS = 10_000;
/** a probe due while every GPU slot is taken waits this long and looks again. */
const HW_SLOT_BUSY_MS = 5 * 60_000;
/** a GPU process replaces the software one only after this long without a scan. */
const HW_SWITCH_QUIET_MS = 30_000;
/** a probe due during a scan waits this long and looks again. */
const HW_PROBE_BUSY_MS = 60_000;

export class GuardMixin extends SettingsMixinDeviceBase<VideoCamera & MotionSensor> implements MotionSensor, OccupancySensor, ObjectDetector {
    storageSettings = new StorageSettings(this, {
        mode: {
            title: 'Mode',
            description: 'Off, Shadow (measure only) or Gate.',
            choices: [...MODES.keys()],
            defaultValue: 'Shadow',
        },
        detect: {
            title: 'Detect',
            description: 'Categories that may raise motion.',
            choices: DETECT_CHOICES,
            multiple: true,
            defaultValue: DETECT_CHOICES,
        },
        stream: {
            title: 'Stream',
            description: 'Default reuses the open stream.',
            choices: [STREAM_DEFAULT_LABEL],
            defaultValue: 'default',
        },
        zoomOut: {
            title: 'Zoom-out',
            description: 'Extra pass for large or close subjects.',
            choices: ZOOM_LABELS.map(z => z[0]),
            defaultValue: '0',
        },
        animalStart: {
            title: 'Animal start threshold',
            description: 'Score that confirms an animal on this camera (0.05-0.95). Empty = the plugin-wide Animal confirm.',
            type: 'string',
            placeholder: 'plugin default',
        },
        animalKeep: {
            title: 'Animal keep threshold',
            description: 'Score that keeps tracking an animal on this camera (0.05-0.95, at most the start threshold). Empty = the plugin-wide Animal detect.',
            type: 'string',
            placeholder: 'plugin default',
        },
        minPerson: {
            title: 'Minimum person size',
            description: 'Smaller person boxes are ignored: % of the frame area (0-25). Empty = off.',
            type: 'string',
            placeholder: 'off',
        },
        minVehicle: {
            title: 'Minimum vehicle size',
            description: 'Smaller vehicle boxes are ignored: % of the frame area (0-25). Empty = off.',
            type: 'string',
            placeholder: 'off',
        },
        minAnimal: {
            title: 'Minimum animal size',
            description: 'Smaller animal boxes are ignored: % of the frame area (0-25). Empty = off.',
            type: 'string',
            placeholder: 'off',
        },
        maxPerson: {
            title: 'Maximum person size',
            description: 'Larger person boxes are ignored (something on the lens): % of the frame area (5-100). Empty = 85.',
            type: 'string',
            placeholder: '85',
        },
        singleMinHeight: {
            title: 'One-frame person minimum height',
            description: 'One strong person frame confirmed by the second look is enough when its box is at least this tall: % of the frame height (0-100, 0 = off). Empty = 25.',
            type: 'string',
            placeholder: '25',
        },
        shapeHold: {
            title: 'Vehicle shape hold',
            description: 'Keep a still or parked vehicle the detector misses while the outline inside its box still matches its last look (stops parked vehicles flickering at night).',
            type: 'boolean',
            defaultValue: true,
        },
        shapeThreshold: {
            title: 'Vehicle shape match',
            description: 'Outline match (0.1-0.99) that keeps a missed vehicle. Same parked vehicle scores 0.87-1, an empty spot about 0.2. Empty = 0.6.',
            type: 'string',
            placeholder: '0.6',
        },
        coveredHold: {
            title: 'Covered vehicle hold',
            description: 'Seconds a vehicle is kept while someone or something stands in front of it (0-600). Empty = 60.',
            type: 'string',
            placeholder: '60',
        },
        parkingSpots: {
            title: 'Learn parking spots',
            description: 'Remember where vehicles park on this camera. In a spot used on two separate occasions a known vehicle is parked at once (also after a restart) and a weak detection keeps it.',
            type: 'boolean',
            defaultValue: true,
        },
        forgetSpots: {
            title: 'Forget parking spots',
            description: 'Clear the parking spots learnt on this camera (e.g. after moving it).',
            type: 'button',
        },
        lookalikes: {
            title: 'Ignore known look-alikes',
            description: 'A person or animal you labelled false alarm (a statue, a plant) is ignored where it stands while it looks the same; someone in front of it still counts.',
            type: 'boolean',
            defaultValue: true,
        },
        forgetLookalikes: {
            title: 'Forget look-alikes',
            description: 'Clear the look-alikes learnt from your false alarm labels on this camera.',
            type: 'button',
        },
    });
    /** label -> stream id of the last rendered Stream choices */
    private streamLabels = new Map<string, string>([[STREAM_DEFAULT_LABEL, 'default']]);
    /** the chosen stream failed (or is unknown): the default stream for the rest of the plugin's life */
    private streamFallback = false;
    /** fallback retries (0.3.27): the chosen stream is tried again while the default one is used; back on it as soon as it delivers */
    private streamRetryTimer: NodeJS.Timeout | undefined;
    private streamRetries = 0;
    private streamRetrying = false;
    private streamRetryDelays = STREAM_RETRY_MS;
    private streamName = '';
    private streamFirstTimer: NodeJS.Timeout | undefined;

    private episode: Episode;
    private listener: EventListenerRegister | undefined;
    private claimListener: EventListenerRegister | undefined;
    /** the camera's own detection classes (ObjectsDetected events of the inner device), newest last. */
    private claimEvents: { at: number; groups: Group[]; other: string[]; person?: boolean; animal?: boolean }[] = [];
    private claimFrom = 0;
    private unverifiedLogged = false;
    private released = false;
    readonly status = new CameraStatus();
    private lastPresent = false;
    private streamUp = false;
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
    /**
     * after a GPU decode failure: when the GPU is probed next (monotonic ms; Infinity = given up). Non-zero means software decode
     * until a probe succeeds (a stream that starts slowly fails the same way, so the GPU is tried again).
     */
    private hwRetryAt = 0;
    private hwRetryTimer?: NodeJS.Timeout;
    /** GPU-to-CPU fallbacks and failed probes since the plugin started (shown in the status) */
    private hwFallbacks = 0;
    /** consecutive GPU failures (the backoff): failed probes and GPU processes failing after a switch; reset by a clean GPU run */
    private hwProbeFails = 0;
    /** a probe succeeded: a GPU failure counts toward the backoff until a GPU process has run cleanly for HW_CLEAN_RUN_MS */
    private hwOnProbation = false;
    private hwCleanTimer?: NodeJS.Timeout;
    /** clean GPU run time on probation: summed over processes (per trigger, each one is short), and the start of the current run (0 = none) */
    private hwCleanMs = 0;
    private hwCleanSince = 0;
    /** the camera keeps the GPU slot of a successful probe until its pending switch starts the GPU process (or is abandoned) */
    private hwSwitchSlot = false;
    /** the running GPU probe: a separate short-lived ffmpeg, never this.frames */
    private hwProbe: { fs?: FrameStream; timer?: NodeJS.Timeout } | undefined;
    /** a probe succeeded while the camera was busy: the software process is replaced when the scan ends or after a quiet period */
    private hwSwitchPending = false;
    private hwSwitchTimer?: NodeJS.Timeout;
    /** decode mode of the current process */
    private srcHw = false;
    private scanEndedAt = 0;
    private lastEpisodePostAt = 0;
    private gate: PixelGate | undefined;
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
    /** 'Fast confirm' burst: monotonic end of the burst phase of the current scan (0 = none), the ffmpeg burst state, the newest frame waiting for the in-flight reply. */
    private burstUntil = 0;
    private burstOn = false;
    private burstNext: { jpeg: Buffer; t: number } | undefined;
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
    /** 'Paced frames: Camera snapshots': heartbeat timer, one snapshot in flight, consecutive failures, sticky fallback. */
    private snapTimer: NodeJS.Timeout | undefined;
    private snapBusy = false;
    private snapFails = 0;
    private snapDisabled = false;
    private snapSetting = false;
    /** capture time of the snapshot that opened the current scan: older pre-roll frames are dropped (t stays monotonic). */
    private bridgeT = 0;
    /** 'Camera AI fast path': the scan generation it started or promoted (normal priority, never behind a heartbeat). */
    private fastGen: number | undefined;
    private readonly fast = new FastPathTimer();
    /** the camera has no MotionSensor of its own: motion comes only from Coral Guard */
    readonly sensorless: boolean;
    private readonly ifaces: string[] | undefined;
    /** verified groups already announced in the current confirmation, and when the last verified event went out. */
    private emittedGroups: string[] = [];
    private lastDetectionAt = 0;

    constructor(options: SettingsMixinDeviceOptions<VideoCamera & MotionSensor>, private plugin: GuardProvider) {
        super(options);
        // Startup is fully transparent: nothing is read back as raw motion, replayed or published here.
        // The mixin writes motionDetected only after a genuine event from the underlying device
        // (see `touched`), so a plugin restart never emits a value and never starts a scan.
        this.episode = new Episode(this.episodeConfig(), false);
        // Sensorless: the camera has no MotionSensor of its own, so there is no raw motion; motion comes only from the
        // verified unflagged/heartbeat path (Gate), Shadow never publishes, and failing open means motion stays false.
        this.ifaces = (options as any).mixinDeviceInterfaces;
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
        this.closeLeftoverEpisode();
        // OccupancySensor is ours on every camera: no prior value, start false (a write of false is harmless for consumers)
        try { this.occupied = false; } catch (e) { /* ignore */ }
        // the camera's own AI class (e.g. people / vehicle / dog_cat / package), when the device reports one
        // A plain listener: the Scrypted server does not mask property-less events behind a mixin (masking is disabled in
        // notifyInterfaceEventFromMixin), so a `mixinId` listener never receives the camera's detections. Every listener gets
        // the camera's events and ours; our own verified ones are skipped here (sourceId / verified_* classes).
        this.claimListener = device.listen({ event: ScryptedInterface.ObjectDetector }, (_s, _d, data) => {
            if (!data || data.sourceId === (this.plugin.id ?? 'coral-guard')
                || (data.detections || []).some((d: any) => String(d?.className || '').startsWith('verified_')))
                return;
            this.onObjectsDetected(data);
        });
        this.tickTimer = setInterval(() => this.onTick(), 1000);
        plugin.register(this);
        plugin.ensurePresence(this.id, this.name || this.id, this.detectGroups());
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
            this.console.warn(`service camera id "${this.verifierId}" is also used by camera ${other}: both would share one service session.`);
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
            graceMs: this.plugin.graceMs ?? 3000,
            stillReleaseMs: (this.plugin.stillReleaseS ?? 170) * 1000,
        };
    }

    private animalWarned = '';
    /** plugin-wide thresholds with this camera's animal overrides applied (invalid overrides are ignored, warned once). */
    private get thresholds(): Thresholds {
        const v = this.storageSettings.values;
        const r = applyAnimalOverride(this.plugin.thresholds ?? DEFAULT_THRESHOLDS, v.animalStart, v.animalKeep);
        if (r.invalid && r.invalid !== this.animalWarned)
            this.console.warn(`${r.invalid}; using the plugin-wide animal thresholds`);
        this.animalWarned = r.invalid ?? '';
        return r.t;
    }

    private viewThresholds() {
        const t = this.thresholds;
        return { person: t.personConfirm, personKeep: t.personKeep, vehicle: t.vehicleDetect, animal: t.animalDetect, enabled: this.detectGroups() as string[], zones: this.zones, minVisible: DEFAULTS.zoneMinVisible };
    }

    // ---- 'Tracked' (what the service holds, from the last analysed reply) and the 'Now' card ----
    private lastPersonBox: number[] | undefined;
    private lastTrack: { groups: string[]; sinceMs?: number; lastSeenMs?: number } | undefined;
    private trackFrom(r: FrameReply) {
        const enabled = this.detectGroups() as string[];
        const t = this.thresholds;
        const zones = this.zones;
        const dets = (r.detections || []).filter(d => Array.isArray(d.box) && d.box.length === 4 && !(zones.length && isMasked(d.box, zones, DEFAULTS.zoneMinVisible)));
        const iou = (a: number[], b: number[]) => { const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]); if (w <= 0 || h <= 0) return 0; const i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i); };
        const personDet = dets.filter(d => groupOf(d) === 'person' && d.score >= t.personKeep).sort((a, b) => b.score - a.score)[0];
        if (personDet) this.lastPersonBox = personDet.box;
        const items: TrackedItem[] = [];
        const p = r.person;
        const frameT = typeof r.t === 'number' ? r.t : Date.now();
        const left = (h?: number | null) => typeof h === 'number' ? Math.max(0, h - frameT) : undefined;
        const pState = p ? (p.state === 'holding' ? 'held' : p.state === 'confirmed' ? 'confirmed' : p.state) : '';
        if (enabled.includes('person') && p?.present && Array.isArray(p.tracks_detail) && p.tracks_detail.length) {
            // service 0.3.0+: one item per live person track, with its own box and last seen
            for (const tr of p.tracks_detail)
                items.push({ group: 'person', cls: 'person', state: tr.stationary ? `${pState}, still` : pState, sinceMs: tr.since_ms, lastSeenMs: tr.last_seen_ms, box: tr.box,
                    detected: !!tr.box && dets.some(d => groupOf(d) === 'person' && d.score >= t.personKeep && iou(d.box, tr.box!) > 0.3), holdLeftMs: left(p.hold_until_ms) });
        }
        else if (enabled.includes('person') && p?.present)
            items.push({ group: 'person', cls: 'person', state: pState, sinceMs: p.since_ms ?? undefined, lastSeenMs: p.last_seen_ms ?? undefined, box: personDet?.box ?? this.lastPersonBox, detected: !!personDet, holdLeftMs: left(p.hold_until_ms) });
        if (!p?.present) this.lastPersonBox = personDet?.box;
        if (enabled.includes('vehicle'))
            for (const v of r.vehicle?.tracks || [])
                items.push({ group: 'vehicle', cls: v.cls, state: v.state, sinceMs: v.since_ms, box: v.box, detected: !!v.box && dets.some(d => groupOf(d) === 'vehicle' && iou(d.box, v.box!) > 0.3), holdLeftMs: v.state === 'moving' ? left(r.vehicle?.hold_until_ms) : undefined });
        if (enabled.includes('animal') && r.animal?.present)
            for (const a of r.animal.tracks || [])
                items.push({ group: 'animal', cls: a.cls, state: 'held', sinceMs: a.since_ms, lastSeenMs: a.last_seen_ms, box: a.box, detected: !!a.box && dets.some(d => groupOf(d) === 'animal' && iou(d.box, a.box!) > 0.3), holdLeftMs: left(r.animal.hold_until_ms) });
        this.status.setTracked(items, Date.now());
        this.publishPresence(presenceOf(items, enabled, box => zones.length > 0 && isMasked(box, zones, DEFAULTS.zoneMinVisible)));
        const active = items.filter(i => i.state !== 'parked' && i.state !== 'still');
        this.lastTrack = active.length ? {
            groups: [...new Set(active.map(i => i.group))],
            sinceMs: Math.min(...active.map(i => i.sinceMs ?? Infinity)) === Infinity ? undefined : Math.min(...active.map(i => i.sinceMs ?? Infinity)),
            lastSeenMs: p?.present ? (p.last_seen_ms ?? undefined) : Date.now(),
        } : undefined;
        this.refreshStatus();
    }

    // ---- per-class presence sensors ---------------------------------------------------------------

    private presenceAt = 0;
    /** what the presence sensors show, and when the service session was last reset for an episode (mono ms, 0 = never). */
    private presenceShown: Record<PresenceGroup, boolean> = { person: false, vehicle: false, animal: false };
    private presenceResetAt = 0;

    /** one analysed frame: every group the tracker holds, still and parked included. A reset never turns a sensor off by itself. */
    private publishPresence(present: Record<PresenceGroup, boolean>) {
        this.presenceAt = Date.now();
        this.plugin.ensurePresence(this.id, this.name || this.id, this.detectGroups()); // cheap: repeats only on a rename
        const since = this.presenceResetAt ? monoNow() - this.presenceResetAt : Infinity;
        this.presenceShown = presenceAfterReset(present, this.presenceShown, since, DEFAULTS.presenceResetGraceMs);
        for (const g of PRESENCE_GROUPS) this.plugin.setPresence(this.id, g, this.presenceShown[g]);
    }

    /** no analysed frame for a while (mode off, service down, camera gone): nothing is known to be there. */
    private expirePresence(force = false) {
        if (!this.presenceAt || (!force && Date.now() - this.presenceAt < DEFAULTS.presenceStaleMs)) return;
        this.presenceAt = 0;
        this.presenceShown = { person: false, vehicle: false, animal: false };
        for (const g of PRESENCE_GROUPS) this.plugin.setPresence(this.id, g, false);
    }

    private nowInfo(): NowInfo {
        const g = this.plugin.config;
        const modeLabel = [...MODES].find(([, m]) => m === this.mode)?.[0] ?? 'Shadow';
        const published = this.mode === 'shadow' ? this.cameraRaw : !!this.motionDetected;
        const groups = this.episode.confirmedGroups;
        const coral = this.episode.wouldPublish;
        const reason: NowReason = this.mode === 'off' || g.killSwitch ? { text: 'Off · raw passed through' }
            : this.episode.degraded ? { text: 'Service down · raw passed through' }
                : groups ? { groups: groups.length ? groups : (this.lastTrack?.groups ?? ['subject']), sinceMs: this.lastTrack?.sinceMs, lastSeenMs: this.lastTrack?.lastSeenMs }
                    : this.episode.pending ? { text: 'Verifying' }
                        : this.lastTrack ? { text: `${this.lastTrack.groups.join('+')} tracked · unconfirmed` }
                            : { text: 'Nothing tracked' };
        const shadow = this.mode === 'shadow';
        const occOn = shadow ? this.episode.wouldOccupied : this.mode === 'gate' ? this.episode.occupied : this.cameraRaw;
        const still = this.episode.stillForMs(monoNow());
        const occupancy: NowInfo['occupancy'] = {
            on: occOn,
            ...(groups?.length ? { groups } : {}),
            ...(occOn && still !== undefined ? { stillS: still / 1000 } : {}),
            ...(this.episode.stillReleased ? { releasedAfterS: (this.plugin.stillReleaseS ?? 170) } : {}),
        };
        return { published, coral, mode: modeLabel, reason, occupancy };
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
        const snap = !!this.plugin.pacedSnapshots;
        if (snap !== this.snapSetting) {
            this.snapSetting = snap;
            this.snapDisabled = false; // a changed setting retries snapshots after a fallback
            this.snapFails = 0;
        }
        this.apply(this.episode.setConfig(this.episodeConfig(), monoNow()));
    }

    private minWarned = '';
    /** Per-camera size limits as service fractions: minimum sizes (% of the frame area, empty or invalid = 0 = off), maximum person size
     *  (% of the area) and one-frame person minimum height (% of the frame height), both empty or invalid = the default; invalid warned once. */
    private minAreas() {
        const v = this.storageSettings.values, bad: string[] = [];
        const frac = (raw: unknown, title: string) => {
            const s = String(raw ?? '').trim().replace('%', '').replace(',', '.');
            if (!s)
                return 0;
            const n = Number(s);
            if (!Number.isFinite(n) || n < 0 || n > 25) { bad.push(title); return 0; }
            return n / 100;
        };
        const pct = (raw: unknown, title: string, lo: number, dflt: number) => {
            const s = String(raw ?? '').trim().replace('%', '').replace(',', '.');
            if (!s)
                return dflt / 100;
            const n = Number(s);
            if (!Number.isFinite(n) || n < lo || n > 100) { bad.push(title); return dflt / 100; }
            return n / 100;
        };
        const r = { min_area_person: frac(v.minPerson, 'Minimum person size'), min_area_vehicle: frac(v.minVehicle, 'Minimum vehicle size'), min_area_animal: frac(v.minAnimal, 'Minimum animal size'),
            max_area_person: pct(v.maxPerson, 'Maximum person size', 5, DEFAULTS.maxPersonPct), single_min_h: pct(v.singleMinHeight, 'One-frame person minimum height', 0, DEFAULTS.singleMinHeightPct) };
        const w = bad.join(', ');
        if (w && w !== this.minWarned)
            this.console.warn(`${w}: not a number in range; ignored`);
        this.minWarned = w;
        return r;
    }

    private shapeWarned = '';
    /** Per-camera shape hold (service 0.3.30+); empty or invalid values fall back to the service defaults (invalid warned once). */
    private shapeConfig(): TrackerConfig {
        const v = this.storageSettings.values, bad: string[] = [];
        const num = (raw: unknown, title: string, lo: number, hi: number) => {
            const s = String(raw ?? '').trim().replace(',', '.');
            if (!s)
                return undefined;
            const n = Number(s);
            if (!Number.isFinite(n) || n < lo || n > hi) { bad.push(title); return undefined; }
            return n;
        };
        const sh = (v as any).shapeHold;
        const r: TrackerConfig = { shape_hold: sh === undefined || sh === null || sh === '' ? true : sh === true || sh === 'true' };
        const thr = num(v.shapeThreshold, 'Vehicle shape match', 0.1, 0.99);
        const hold = num(v.coveredHold, 'Covered vehicle hold', 0, 600);
        const ps = (v as any).parkingSpots;
        r.parking_spots = ps === undefined || ps === null || ps === '' ? true : ps === true || ps === 'true';
        const la = (v as any).lookalikes;
        r.lookalikes = la === undefined || la === null || la === '' ? true : la === true || la === 'true';
        if (thr !== undefined) r.shape_thr = thr;
        if (hold !== undefined) r.occlusion_hold_s = hold;
        const w = bad.join(', ');
        if (w && w !== this.shapeWarned)
            this.console.warn(`${w}: out of range; the service default is used`);
        this.shapeWarned = w;
        return r;
    }

    /** Push the tracker overrides and the plugin-wide thresholds (at start, on a change, after a service restart). */
    postConfig() {
        const person = serviceThresholds(this.thresholds);
        const detect = this.detectGroups();
        const groups = { detect_person: detect.includes('person'), detect_vehicle: detect.includes('vehicle'), detect_animal: detect.includes('animal') };
        const full: TrackerConfig = { ...DEFAULTS.tracker, ...person, ...this.minAreas(), ...this.shapeConfig(), ...groups, zoom_out: this.zoomOut, second_look: this.plugin.secondLook ?? true, fast_confirm: this.plugin.fastConfirm ?? true, name: String(this.name || this.id) };
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

    // ---- episode journal on the service (POST .../episodes): fire-and-forget, never affects decisions ----
    private jId: string | undefined;
    private jKind: 'camera' | 'unflagged' = 'camera';
    private jConfirmed = false;
    private journal(action: 'start' | 'update' | 'end', extra: Partial<EpisodeEvent> = {}) {
        if (!this.jId || typeof (this.plugin.client as any).episodeEvent !== 'function')
            return;
        const ev: EpisodeEvent = { id: this.jId, action, t_ms: Date.now(), mode: this.mode, kind: this.jKind, ...(action === 'start' ? { detect: this.detectGroups() } : {}), ...extra };
        if (action === 'start') this.jOpen.add(ev.id);
        if (action === 'end') { if (!this.jOpen.has(ev.id)) return; this.jOpen.delete(ev.id); } // exactly one end per id
        if (action !== 'end')
            this.saveOpenEpisode({ id: ev.id, kind: ev.kind ?? 'camera', confirmed: this.jConfirmed || extra.outcome === 'confirmed', at: Date.now() });
        // strictly in order per camera: an update or end must never overtake its start (the service would drop it)
        this.jChain = this.jChain.then(() => this.plugin.client.episodeEvent(this.verifierId, ev)).then(() => {
            if (action === 'end') this.saveOpenEpisode(undefined); // only once the end has reached the service
        }, () => { /* silent: a refused start (too many ongoing, busy) makes only that episode's later calls 404; the next episode journals again */ });
    }

    /** the open journal episode, kept in the camera's storage: a restart that cuts the episode (a deploy kills the process before its end
     * is sent) closes it at the next start instead of leaving it 'in progress' until the service's 2-hour stale rule. */
    private saveOpenEpisode(v: { id: string; kind: string; confirmed: boolean; at: number } | undefined) {
        try {
            if (v) this.storage.setItem(OPEN_EPISODE_KEY, JSON.stringify(v));
            else this.storage.removeItem(OPEN_EPISODE_KEY);
        }
        catch { /* best effort */ }
    }

    /** at start: end a journal episode a previous run left open (confirmed -> released, else aborted), at its last known time. */
    private closeLeftoverEpisode() {
        let v: any;
        try { v = JSON.parse(this.storage.getItem(OPEN_EPISODE_KEY) || 'null'); } catch { v = undefined; }
        if (!v || typeof v.id !== 'string' || typeof (this.plugin.client as any).episodeEvent !== 'function')
            return;
        const at = Number.isFinite(v.at) ? v.at : Date.now();
        const outcome = v.confirmed ? 'released' : 'aborted';
        const ev: EpisodeEvent = { id: v.id, action: 'end', t_ms: at, mode: this.mode, kind: v.kind === 'unflagged' ? 'unflagged' : 'camera', outcome,
            timeline_event: { t_ms: at, text: `${outcome} (plugin restarted during the episode)` } };
        this.jChain = this.jChain.then(() => this.plugin.client.episodeEvent(this.verifierId, ev)).then(() => undefined, () => undefined)
            .then(() => { if (!this.jId) this.saveOpenEpisode(undefined); });   // 409 (already ended) or 404 (gone) clear it too
        this.console.log(`journal: closing episode ${v.id.slice(0, 8)} left open by the previous run (${outcome})`);
    }
    private jChain: Promise<void> = Promise.resolve();
    /** journal ids started and not yet ended (reconciled on every transition) */
    private jOpen = new Set<string>();
    private journalStep(fx: Effects) {
        for (const log of fx.logs)
            if (log.kind === 'still' && this.jId)
                this.journal('update', { timeline_event: { t_ms: Date.now(), text: log.event === 'release'
                    ? `motion released (still) +${log.afterS} s, occupancy stays on`
                    : `motion re-raised (${log.reason ?? 'moving'}) +${log.afterS} s` } });
            else if (log.kind === 'summary' && this.jId) {
                const s = log.summary;
                const outcome = s.outcome === 'maxhold' ? 'released' : s.outcome;
                const motion = s.occupiedS !== undefined ? { motion: { occupied_s: s.occupiedS, motion_s: s.motionS ?? s.occupiedS, still_releases: s.stillReleases ?? 0, reraises: s.reraises ?? 0 } } : {};
                this.journal('end', { outcome: outcome as EpisodeEvent['outcome'], groups: s.confirmedBy, ...motion, ...(s.inconclusive ? { inconclusive: true } : {}), timeline_event: { t_ms: Date.now(), text: `${outcome}${s.inconclusive ? ' (inconclusive: too little evidence)' : ''}${s.note ? ` (${s.note})` : ''}${s.wouldBAfter !== undefined ? ` | B would confirm +${s.wouldBAfter} ms` : ''}` } });
                this.jId = undefined;
                this.jConfirmed = false;
            }
        // reconcile: an open journal episode that is no longer running (no summary seen) is ended now
        if (this.jId && !this.episode.scanning) {
            this.journal('end', { outcome: this.jConfirmed ? 'released' : 'aborted', timeline_event: { t_ms: Date.now(), text: this.jConfirmed ? 'released' : 'aborted (closed)' } });
            this.jId = undefined;
            this.jConfirmed = false;
        }
        const cg = this.episode.confirmedGroups;
        if (!this.episode.scanning)
            this.jUnflaggedAt = undefined;
        else if (!this.jId && this.episode.synthetic) {
            // unflagged scans are journalled only once they confirm (start back-dated to the trigger)
            this.jUnflaggedAt ??= Date.now();
            if (cg) {
                this.jId = randomUUID();
                this.jKind = 'unflagged';
                this.jConfirmed = false;
                this.journal('start', { t_ms: this.jUnflaggedAt, timeline_event: { t_ms: this.jUnflaggedAt, text: `unflagged (${this.synthReason || 'monitoring'})` } });
            }
        }
        else if (!this.jId) {
            this.jId = randomUUID();
            this.jKind = 'camera';
            this.jConfirmed = false;
            this.journal('start', { timeline_event: { t_ms: Date.now(), text: 'camera trigger' } });
        }
        if (this.jId && cg && !this.jConfirmed) {
            this.jConfirmed = true;
            this.journal('update', { outcome: 'confirmed', groups: cg, timeline_event: { t_ms: Date.now(), text: `${this.jReconfirm ? 're-confirmed after max hold' : 'confirmed'} ${cg.join('+') || 'subject'}${this.episode.lateConfirm ? ' (after raw end, grace)' : ''}` } });
            this.jReconfirm = false;
        }
        else if (this.jId && !cg && this.jConfirmed && this.episode.pending) {
            this.jConfirmed = false;
            this.jReconfirm = true;
            this.journal('update', { timeline_event: { t_ms: Date.now(), text: 'max hold reached: verifying again' } });
        }
    }
    private jReconfirm = false;

    // ---- 'Last detection' picture (MQTT add-on) ----------------------------------------------------
    /**
     * per group, the newest analysed frame showing it (mono ms): a reference to the JPEG already in hand, nothing is copied.
     * Kept only while an episode still needs its picture, so no frame outlives it.
     */
    private lastHits = new Map<DrawBox['group'], { jpeg: Buffer; boxes: DrawBox[]; at: number }>();
    private shotTaken = false;
    private shotBusy = false;
    private shotAt = -Infinity;

    /** keeps the frame for each group it shows a valid detection of: not weak, masked, parked or still. */
    private noteHit(jpeg: Buffer, boxes: Box[]) {
        if (this.shotTaken)
            return;
        const draw = boxes.filter(b => !b.weak && !b.masked && !b.parked && !b.still)
            .map((b): DrawBox => ({ x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, group: b.group ?? 'person' }));
        const at = monoNow();
        for (const g of new Set(draw.map(b => b.group)))
            this.lastHits.set(g, { jpeg, boxes: draw, at });
    }

    /**
     * One picture per confirmed episode, drawn from the newest frame showing a confirmed group, at most one per camera every
     * lastDetectionMinMs and one at a time; never awaited. A confirmation without such a frame waits for one.
     */
    private lastDetectionStep() {
        if (!this.episode.scanning) {
            this.shotTaken = false;
            this.lastHits.clear();
            return;
        }
        const cg = this.episode.confirmedGroups as DrawBox['group'][] | undefined;
        const now = monoNow();
        if (!cg || this.shotTaken || this.shotBusy || now - this.shotAt < DEFAULTS.lastDetectionMinMs)
            return;
        // the newest frame of a confirmed group: a newer frame showing only another group does not replace it
        let hit: { jpeg: Buffer; boxes: DrawBox[]; at: number } | undefined;
        for (const g of cg) {
            const h = this.lastHits.get(g);
            if (h && now - h.at <= DEFAULTS.lastDetectionFrameMs && (!hit || h.at > hit.at)) hit = h;
        }
        if (!hit)
            return;
        const run = this.plugin.publishLastDetection?.(this.id, hit.jpeg, hit.boxes.filter(b => cg.includes(b.group)));
        if (!run)
            return;
        this.shotTaken = true;
        this.shotBusy = true;
        this.shotAt = now;
        this.lastHits.clear();
        run.finally(() => { this.shotBusy = false; });
    }
    private lastOccupied = false;
    private jUnflaggedAt: number | undefined;

    private apply(fx: Effects) {
        if (this.released)
            return;
        try { this.journalStep(fx); } catch { /* journal never affects decisions */ }
        try { this.lastDetectionStep(); } catch { /* the picture never affects decisions */ }
        // never write before a genuine camera event, except a confirmation (true) or the release of our own true
        if ((this.touched || fx.motion || this.lastPublished) && this.motionDetected !== fx.motion)
            this.publish(fx.motion);
        if ((this.touched || fx.occupied || this.lastOccupied) && !!this.occupied !== fx.occupied) {
            this.lastOccupied = fx.occupied;
            this.occupied = fx.occupied;
        }
        if (this.fast.armed && this.episode.confirmedGroups) {
            const line = this.fast.verdict(`confirmed ${this.episode.confirmedGroups.join('+')}`, monoNow());
            if (line) this.console.log(line);
        }
        for (const log of fx.logs) {
            if (log.kind === 'summary' && this.fast.armed) {
                const line = this.fast.verdict(log.summary.confirmedAfterMs !== undefined ? 'confirmed' : 'not confirmed', monoNow());
                if (line) this.console.log(line);
            }
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
            else if (log.kind === 'still') {
                const would = this.mode === 'gate' ? '' : ' (would)';
                this.console.log(log.event === 'release'
                    ? `motion released (still) +${log.afterS} s${would}: occupancy stays on`
                    : `motion re-raised (${log.reason ?? 'moving'}) +${log.afterS} s${would}`);
                this.status.onStill(log.event);
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
            this.presenceResetAt = monoNow();
            this.resetPromise = this.plugin.client.reset(this.verifierId).then(() => undefined, e => {
                this.apply(this.episode.failure(`reset failed: ${maskUrls(String(e?.message || e))}`, monoNow()));
            });
        }
        this.syncSource(fx.scanning);
        this.verifiedDetections(false);
        this.refreshStatus();
    }

    // ---- ObjectDetector: verified classes for external consumers ----

    /** the mixin claims ObjectDetector only when the camera has one (see canMixin). */
    private get ownsDetector(): boolean {
        return Array.isArray(this.ifaces) && this.ifaces.includes(ScryptedInterface.ObjectDetector);
    }

    /** verified detections are published next to the camera's raw ones: Gate mode, publishing on, not disabled, service up. */
    private get publishingVerified(): boolean {
        return this.ownsDetector && this.plugin.publishVerified !== false && this.mode === 'gate'
            && !this.plugin.config.killSwitch && !this.episode.degraded && !this.released;
    }

    /** class name published for a verified group. */
    publishClass(g: Group): string {
        return PUBLISH_CLASS[g];
    }

    private emitDetections(data: ObjectsDetected) {
        sdk.deviceManager.onMixinEvent(this.id, this, ScryptedInterface.ObjectDetector, data)?.catch?.(() => { /* best effort */ });
    }

    /** confirm transition (new group): one event at once; held: repeat every DEFAULTS.detectionRepeatMs (pulse-style consumers keep seeing it); nothing otherwise. */
    private verifiedDetections(tick: boolean) {
        const cg = this.publishingVerified ? (this.episode.confirmedGroups ?? []).filter((g): g is Group => g in PUBLISH_CLASS) : [];
        if (!cg.length) {
            this.emittedGroups = [];
            return;
        }
        const now = monoNow();
        const fresh = cg.some(g => !this.emittedGroups.includes(g));
        if (!fresh && !(tick && now - this.lastDetectionAt >= DEFAULTS.detectionRepeatMs))
            return;
        this.emittedGroups = cg;
        this.lastDetectionAt = now;
        this.emitDetections({
            detections: cg.map(g => ({ className: this.publishClass(g), score: 1 })),
            timestamp: Date.now(),
            sourceId: this.plugin.id ?? 'coral-guard',
        });
    }

    /** the camera's own classes (its raw events always reach consumers) plus the verified ones this mixin publishes. */
    async getObjectTypes(): Promise<ObjectDetectionTypes> {
        const cam: ObjectDetectionTypes = (await (this.mixinDevice as any).getObjectTypes?.()) ?? {};
        if (!this.publishingVerified)
            return cam;
        const verified = this.detectGroups().map(g => this.publishClass(g));
        return { ...cam, classes: [...(cam.classes ?? []).filter(c => !verified.includes(c)), ...verified] };
    }

    async getDetectionInput(detectionId: string, eventId?: any): Promise<MediaObject> {
        return (this.mixinDevice as any).getDetectionInput(detectionId, eventId);
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
            stillReleaseAfterS: e.stillReleaseAfterS,
            reraises: e.reraises,
            occupiedS: e.occupiedS,
            motionS: e.motionS,
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
        this.status.setNow(this.nowInfo());
        this.status.setZones(this.zones);
        this.status.setMode([...MODES].find(([, m]) => m === this.mode)?.[0] ?? 'Shadow');
        this.status.setEnabled(state !== 'disabled');
        if (state !== 'disabled')
            this.status.setState(state);
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
        if (this.plugin.isLive && !this.plugin.isLive(this))
            return; // a stale instance never judges the chain
        // the CURRENT mixin list of the device decides when it is readable: last = outermost
        try {
            const mixins = (sdk.systemManager.getDeviceById(this.id) as any)?.mixins;
            const own = (this as any).mixinProviderId ?? this.plugin.id;
            if (Array.isArray(mixins) && mixins.length && own) {
                const last = String(mixins[mixins.length - 1]) === String(own);
                if (!last && !this.notOutermost)
                    this.console.warn('Coral Guard is NOT the outermost MotionSensor mixin: gating has no effect. Move Coral Guard to the end of this camera\'s Extensions list (last = outermost).');
                this.notOutermost = !last;
                return;
            }
        }
        catch (e) { /* fall back to the shared-state check */ }
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
        this.expirePresence();
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
        this.verifiedDetections(true);
    }

    // ---- frame source ----------------------------------------------------------------------
    // One ffmpeg per camera. Per-trigger mode: started when scanning begins, stopped when it ends.
    // Keep-warm mode: running whenever the verifier is in service for this camera.

    private get warm() {
        return DEFAULTS.frameProcess === 'Keep warm' && !this.snapPaced;
    }

    /** paced analysis from the camera's own snapshots (setting on, the camera has takePicture, no fallback yet). */
    private get snapPaced(): boolean {
        return !!this.plugin.pacedSnapshots && !this.snapDisabled && typeof (this.mixinDevice as any)?.takePicture === 'function';
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
                this.bridgeT = 0;
                this.fast.scan(this.loopStartedAt);
                this.burstUntil = this.plugin.fastConfirm ?? true ? this.loopStartedAt + DEFAULTS.burstMaxMs : 0;
                this.burstNext = undefined;
                this.syncBurst();
                if (this.snapPaced && !this.frames)
                    this.bridgeSnapshot(); // the stream starts cold: the latest keyframe goes first
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
            const ended = this.scanActive;
            if (ended) {
                this.scanGen++; // scan over (also in Keep-warm, where the process keeps running)
                this.scanEndedAt = monoNow();
            }
            this.scanActive = false;
            this.queue = [];
            this.burstUntil = 0;
            this.burstNext = undefined;
            this.syncBurst();
            if (this.frames && !wantWarm)
                this.stopSource();
            else if (!this.frames && wantWarm && !this.warmRetry)
                this.startSource();
            if (ended && this.hwSwitchPending)
                this.switchToHw(); // waits the quiet period after the scan
        }
        this.syncSnapTimer();
    }

    /** software decode after a GPU failure, until a probe succeeds (not a clock comparison: a timer that fires early changes nothing) */
    private get hwDisabled(): boolean {
        return this.hwRetryAt > 0;
    }

    /** software until a GPU probe in waitMs (±20 % per camera, so a fleet does not probe in the same second) */
    private armHwRetry(waitMs: number): number {
        const wait = Math.round(waitMs * 0.8 + spreadJitter(String(this.id), Math.round(waitMs * 0.4)));
        this.hwRetryAt = monoNow() + wait;
        clearTimeout(this.hwRetryTimer);
        this.hwRetryTimer = setTimeout(() => this.hwRetryDue(), wait);
        this.hwRetryTimer.unref?.();
        return wait;
    }

    private hwRetryDue() {
        this.hwRetryTimer = undefined;
        if (this.released || this.hwProbe || !Number.isFinite(this.hwRetryAt) || this.hwRetryAt <= 0)
            return;
        const left = this.hwRetryAt - monoNow();
        if (left > 0) { // fired early (timer rounding): wait the rest
            this.hwRetryTimer = setTimeout(() => this.hwRetryDue(), left);
            this.hwRetryTimer.unref?.();
            return;
        }
        if (this.frames && this.srcHw) {
            this.hwRetryAt = 0; // already decoding on the GPU: nothing to probe
            return;
        }
        if (!this.frames || this.mode === 'off' || this.plugin.config.killSwitch || this.snapPaced) {
            // no running stream (between triggers, paced snapshots, out of service): no probe opens one; the next real start tries
            // the GPU itself, and its own fallback (counted toward the backoff) handles a failure
            this.hwRetryAt = 0;
            this.hwOnProbation = true;
            this.console.log('frame source: GPU decode is tried again at the next start');
            return;
        }
        if (this.scanActive) {
            this.armHwRetry(HW_PROBE_BUSY_MS); // a probe never competes with a scan
            return;
        }
        const v = DEFAULTS;
        if (v.hwDecode === 'Off' || !vaapiAvailable()) {
            this.armHwRetry(hwRetryMs(this.hwProbeFails)); // looked at again later: the setting may change
            return;
        }
        if (!(this.plugin.acquireHw?.(this, v.maxHwCameras) ?? true)) {
            this.console.log(`frame source: GPU probe postponed, all ${v.maxHwCameras} GPU decode slots are used`);
            this.armHwRetry(HW_SLOT_BUSY_MS);
            return;
        }
        this.probeHw();
    }

    /**
     * GPU probe: a second ffmpeg on the same stream with GPU decode, stopped at its first JPEG. The camera's own (software) process
     * and its warm ring are not touched; it holds the camera's GPU slot only while it runs.
     */
    private probeHw() {
        const v = DEFAULTS;
        const probe: NonNullable<GuardMixin['hwProbe']> = {};
        this.hwProbe = probe;
        const finish = (ok: boolean, why?: string) => {
            if (this.hwProbe !== probe)
                return;
            this.endHwProbe(ok && !this.released); // a success keeps the slot for the switch
            if (this.released)
                return;
            if (ok) {
                this.hwOnProbation = true; // the backoff is kept until the GPU has run cleanly (a probe is only a few seconds)
                this.hwRetryAt = 0;
                clearTimeout(this.hwRetryTimer);
                this.console.log('frame source: GPU probe delivered a frame, GPU decode works again');
                this.switchToHw();
                return;
            }
            if (!(this.streamUp && monoNow() - this.lastFrameAt < HW_PROBE_HEALTHY_MS)) {
                // the camera's own stream is not delivering either: not a GPU verdict, same wait again, not counted
                const wait = this.armHwRetry(hwRetryMs(this.hwProbeFails));
                this.console.log(`frame source: GPU probe inconclusive, the camera stream delivers no frames; GPU again in ${Math.round(wait / 60000)} min (${maskUrls(why || '')})`);
                return;
            }
            this.hwFallbacks++;
            this.hwBackoff('GPU probe failed, staying on CPU', why || '');
            if (this.streamUp && !this.srcHw)
                this.status.setStream({ name: this.streamName || undefined, connected: true, decode: 'cpu', warm: this.warm, gpuFailed: true, gpuFallbacks: this.hwFallbacks });
        };
        this.console.log('frame source: probing GPU decode (the software process keeps running)');
        probe.timer = setTimeout(() => finish(false, `no frame within ${HW_PROBE_CAP_MS} ms`), HW_PROBE_CAP_MS);
        probe.timer.unref?.();
        const streamUrl: string = v.streamUrl || '';
        startFrames({
            camera: this.mixinDevice,
            inputUrl: streamUrl || undefined,
            hwaccel: true,
            hwFirstFrameMs: HW_PROBE_FIRST_FRAME_MS,
            vaDriver: v.vaDriver || undefined,
            bySize: DEFAULTS.analysisStreamBySize,
            streamId: this.chosenStream(),
            destination: DEFAULTS.analysisDestination,
            longProbe: this.probeFallback,
            intervalMs: Math.max(250, v.frameIntervalMs || 500),
            prebufferMs: 0,
            maxWidth: v.maxWidthNormal,
            maxWidthWide: v.maxWidthWide,
            quality: v.jpegQuality || 85,
            console: this.console,
            onFrame: () => finish(true),
        }).then(fs => {
            if (this.hwProbe !== probe) {
                fs.stop(); // already over (first frame, timeout or release)
                return;
            }
            probe.fs = fs;
            fs.exited.then(err => finish(false, err || 'ended before the first frame'));
        }).catch(e => finish(false, String(e?.message || e)));
    }

    /** one more consecutive GPU failure: the next probe waits longer, after HW_PROBE_MAX_FAILS none until the plugin restarts */
    private hwBackoff(what: string, why: string) {
        if (++this.hwProbeFails >= HW_PROBE_MAX_FAILS) {
            this.hwRetryAt = Infinity;
            clearTimeout(this.hwRetryTimer);
            this.hwRetryTimer = undefined;
            this.console.warn(`frame source: GPU decode failed ${this.hwProbeFails} times in a row, this camera decodes in software until the plugin restarts (${maskUrls(why)})`);
            return;
        }
        const wait = this.armHwRetry(hwRetryMs(this.hwProbeFails));
        this.console.warn(`frame source: ${what}; GPU again in ${Math.round(wait / 60000)} min (${maskUrls(why)})`);
    }

    /** stop the probe (if any) and free the slot it holds (kept for the switch after a success); never touches this.frames or srcGen */
    private endHwProbe(keepSlot = false) {
        const probe = this.hwProbe;
        if (!probe)
            return;
        this.hwProbe = undefined;
        clearTimeout(probe.timer);
        probe.fs?.stop();
        if (keepSlot)
            this.hwSwitchSlot = true;
        else
            this.releaseSrcHw();
    }

    /** a software process is replaced by a GPU one only while idle; without a process (per trigger) the next start uses the GPU */
    private switchToHw() {
        clearTimeout(this.hwSwitchTimer);
        this.hwSwitchTimer = undefined;
        this.hwSwitchPending = false;
        if (this.released || !this.frames || this.srcHw || this.hwDisabled) {
            this.cancelHwSwitch(); // nothing to switch: the kept slot goes back
            return;
        }
        if (this.scanActive) {
            this.hwSwitchPending = true; // syncSource looks again when the scan ends
            return;
        }
        const lastBusy = Math.max(this.loopStartedAt, this.scanEndedAt);
        const quiet = lastBusy ? HW_SWITCH_QUIET_MS - (monoNow() - lastBusy) : 0;
        if (quiet > 0) {
            this.hwSwitchTimer = setTimeout(() => this.switchToHw(), quiet);
            this.hwSwitchTimer.unref?.();
            return;
        }
        this.console.log('frame source: switching to GPU decode');
        this.hwSwitchSlot = false; // the slot passes to the GPU start (stop and start are synchronous: no other camera can take it)
        this.stopSource();
        this.startSource();
    }

    /** a pending switch is dropped (process stopped, camera released): its kept slot is freed */
    private cancelHwSwitch() {
        clearTimeout(this.hwSwitchTimer);
        this.hwSwitchTimer = undefined;
        this.hwSwitchPending = false;
        if (this.hwSwitchSlot) {
            this.hwSwitchSlot = false;
            this.releaseSrcHw();
        }
    }

    /** the camera's own process frees the GPU slot, unless a probe or a pending switch holds it */
    private releaseSrcHw() {
        if (!this.hwProbe && !this.hwSwitchSlot && !(this.frames && this.srcHw))
            this.plugin.releaseHw?.(this);
    }

    /** the current GPU run on probation ends: clean time is summed, a GPU failure starts the sum over */
    private hwCleanStop(failed: boolean) {
        clearTimeout(this.hwCleanTimer);
        this.hwCleanTimer = undefined;
        if (failed)
            this.hwCleanMs = 0;
        else if (this.hwCleanSince)
            this.hwCleanMs += monoNow() - this.hwCleanSince;
        this.hwCleanSince = 0;
    }

    /** a GPU process on probation delivers frames: the backoff is reset once the summed clean run time reaches HW_CLEAN_RUN_MS */
    private hwCleanStart() {
        clearTimeout(this.hwCleanTimer);
        this.hwCleanSince = monoNow();
        this.hwCleanTimer = setTimeout(() => this.hwCleanDone(), Math.max(0, HW_CLEAN_RUN_MS - this.hwCleanMs));
        this.hwCleanTimer.unref?.();
    }

    private hwCleanDone() {
        this.hwCleanTimer = undefined;
        if (this.released || !this.hwOnProbation)
            return;
        this.hwOnProbation = false;
        this.hwProbeFails = 0;
        this.hwCleanMs = 0;
        this.hwCleanSince = 0;
        this.console.log(`frame source: GPU decode ran cleanly for ${HW_CLEAN_RUN_MS / 60000} min, GPU failure backoff reset`);
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
        if (this.hwSwitchSlot || this.hwSwitchPending || this.hwSwitchTimer) {
            // any start replaces the pending switch (it decodes on the GPU itself when it can); its kept slot is used or freed below
            clearTimeout(this.hwSwitchTimer);
            this.hwSwitchTimer = undefined;
            this.hwSwitchPending = false;
            this.hwSwitchSlot = false;
        }
        if (hwaccel) {
            if (this.plugin.acquireHw?.(this, v.maxHwCameras) ?? true) {
                this.console.log(`GPU decode slots: ${this.plugin.hwSlotsUsed ?? '?'}/${v.maxHwCameras} used`);
            }
            else {
                hwaccel = false;
                this.console.log(`GPU decode slots: ${v.maxHwCameras}/${v.maxHwCameras} used, this camera decodes in software`);
                if (this.hwOnProbation)
                    this.armHwRetry(HW_SLOT_BUSY_MS); // back on the GPU after a probe but no slot: probed again later
            }
        }
        this.srcHw = hwaccel;
        if (!hwaccel)
            this.releaseSrcHw(); // a software start holds no slot
        let gotFirst = false;
        const explicitStream = this.chosenStream();
        /** a chosen stream that delivers no first JPEG: the default stream from now on; true when it restarted the source */
        const streamFailed = (): boolean => {
            if (!explicitStream || gotFirst || this.streamFallback || gen !== this.srcGen || this.released)
                return false;
            this.enterFallback(`stream ${this.streamName || explicitStream} failed, using the default stream`);
            this.stopSource();
            this.startSource();
            return true;
        };
        clearTimeout(this.streamFirstTimer);
        clearTimeout(this.aiWaitTimer);
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
                if (st.unknown && !this.streamFallback)
                    this.enterFallback(`stream ${this.storageSettings.values.stream} is not among the camera's streams, using the default stream`);
            },
            destination: DEFAULTS.analysisDestination,
            longProbe: this.probeFallback,
            minGapMs: Math.round(Math.min(interval, 1000 / Math.max(1, DEFAULTS.burstFps)) * 0.6),
            burstFps: warm ? DEFAULTS.burstFps : undefined,
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
                if (this.streamRetrying && explicitStream) {
                    this.streamRetrying = false;
                    this.streamRetries = 0;
                    this.console.log(`stream ${this.streamName || explicitStream} is back, using it again`);
                    this.viewChanged();
                }
                this.lastFrameAt = arrivedAt;
                this.status.setLiveFrame(jpeg, Date.now()); // reference only: the panel shows it until a frame is analysed
                this.warmAttempt = 0;
                if (!this.streamUp) {
                    this.streamUp = true;
                    if (hwaccel && this.hwOnProbation)
                        this.hwCleanStart();
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
                if (this.clock!.inBurst && t <= this.bridgeT)
                    return; // pre-roll older than the opening snapshot: capture times must stay monotonic
                this.syncBurst(arrivedAt);
                if (this.clock!.inBurst) {
                    // pre-roll frame: sent in order, immediately, never waiting for the burst to end.
                    this.queue.push({ jpeg, t });
                    while (this.queue.length > BURST_MAX)
                        this.queue.shift();
                    this.kick();
                }
                else if (this.episode.confirmedGroups && arrivedAt - this.lastEpisodePostAt < DEFAULTS.holdIntervalMs - 100) {
                    // confirmed (holding): about 1 fps is enough to follow the subject; drop, never queue
                    return;
                }
                else if (this.bursting(arrivedAt) && this.busyGen === this.scanGen && !this.queue.length) {
                    // burst: the newest frame goes as soon as the in-flight reply arrives (latest wins, the one it replaces is skipped)
                    if (this.burstNext)
                        this.apply(this.episode.skip(arrivedAt));
                    this.burstNext = { jpeg, t };
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
                this.hwCleanStop(!!(hwaccel && err && err.startsWith(HW_FAILED_PREFIX)));
                this.status.setStream({ connected: false, decode: 'unknown', warm: this.warm });
                this.releaseSrcHw();
                if (err && err.startsWith(PROBE_FAILED_PREFIX) && !this.probeFallback) {
                    this.probeFallback = true; // sticky until the plugin restarts
                    this.console.warn(`analysis stream could not be probed, using the default stream with a longer probe until the plugin restarts (${err})`);
                    this.status.onFailure('stream probe failed, using the default stream');
                    this.startSource();
                    return;
                }
                if (hwaccel && err && err.startsWith(HW_FAILED_PREFIX)) {
                    this.hwFallbacks++;
                    if (this.hwOnProbation) // back on the GPU after a probe and failing in use: counts toward the backoff and the give-up
                        this.hwBackoff('decode GPU failed again after a successful probe, back to CPU', err);
                    else {
                        const wait = this.armHwRetry(hwRetryMs(this.hwProbeFails));
                        this.console.warn(`frame source: decode GPU failed, falling back to CPU for this camera; GPU again in ${Math.round(wait / 60000)} min (${maskUrls(err)})`);
                    }
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
            this.releaseSrcHw(); // a failed start holds no slot
            clearTimeout(this.streamFirstTimer);
            if (streamFailed())
                return;
            this.scheduleSourceRestart(`frame source failed: ${maskUrls(String(e?.message || e))}`);
        });
    }

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
        if (!this.paced || this.mode === 'off' || this.plugin.config.killSwitch || this.episode.degraded || this.episode.scanning || this.scanActive
            || this.heartbeatBusy || this.busyGen !== undefined)
            return; // camera raw motion does NOT stop it: a closed (released) episode keeps being analysed and can re-open
        this.nextHeartbeatAt = now + D.heartbeatMs - 100; // frames come every ~500 ms: the first one due
        this.heartbeatBusy = true;
        const gen = this.srcGen;
        const sg = this.scanGen; // a scan that started (and maybe ended) while in flight voids this reply
        this.plugin.client.frame(this.verifierId, jpeg, t, true).then(raw => {
            if (!raw || gen !== this.srcGen || sg !== this.scanGen || this.released || this.episode.scanning || this.scanActive
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
                this.noteHit(jpeg, view.boxes);
                this.trackFrom(raw);
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
            // a person only HELD by the service (gone behind a pillar or indoors, no box in this frame) never opens a new episode
            const personBox = (reply.detections || []).some(d => groupOf(d) === 'person' && d.score >= this.thresholds.personKeep && d.lookalike === undefined && !d.lookalike_place);
            if (enabled.includes('person') && (reply.person?.confirmed || reply.person?.present) && personBox) seen.push('person');
            if (enabled.includes('animal') && reply.animal?.confirmed) seen.push('animal');
            if (enabled.includes('vehicle') && this.hbTicks.filter(Boolean).length >= 2) seen.push('vehicle');
            if (!seen.length)
                return;
            { // Shadow and Gate alike: a verified unflagged episode (Shadow only reports its would-publish value)
                if (at >= this.synthCooldownUntil) {
                    this.synthRaised = false;
                    this.synthStartedAt = at;
                    this.lastGateOpenAt = at; // keeps the scan open for the quiet window
                    this.synthReason = 'heartbeat';
                    this.apply(this.episode.synthStart(at));
                }
                return;
            }
        }, () => { this.hbTicks = []; /* silent: a heartbeat never degrades the camera */ }).finally(() => { this.heartbeatBusy = false; this.kick(); /* an episode frame waiting for the heartbeat goes now */ });
    }

    // ---- 'Paced frames: Camera snapshots' ---------------------------------------------------

    private heartbeatEligible(): boolean {
        return this.paced && this.mode !== 'off' && !this.plugin.config.killSwitch && !this.episode.degraded && !this.episode.scanning
            && !this.scanActive && !this.heartbeatBusy && this.busyGen === undefined && !this.released;
    }

    private syncSnapTimer() {
        const want = this.snapPaced && this.paced && this.mode !== 'off' && !this.plugin.config.killSwitch && !this.released;
        if (want && !this.snapTimer) {
            this.snapTimer = setTimeout(() => this.snapTick(), heartbeatJitter(this.id, DEFAULTS.heartbeatMs));
            this.snapTimer.unref?.();
        }
        else if (!want && this.snapTimer) {
            clearTimeout(this.snapTimer);
            this.snapTimer = undefined;
        }
    }

    /** one paced snapshot about every heartbeatMs, only when a heartbeat would actually be posted. */
    private snapTick() {
        this.snapTimer = setTimeout(() => this.snapTick(), DEFAULTS.heartbeatMs);
        this.snapTimer.unref?.();
        if (this.snapBusy || !this.snapPaced || !this.heartbeatEligible())
            return;
        this.snapBusy = true;
        fetchSnapshot(this.mixinDevice as any, sdk.mediaManager as any, 'periodic').then(jpeg => {
            this.snapFails = 0;
            if (this.released || !this.snapPaced)
                return;
            const t = monoNow();
            this.status.setLiveFrame(jpeg, Date.now());
            if (!this.nextHeartbeatAt)
                this.nextHeartbeatAt = t; // the timer already carries the per-camera jitter
            this.heartbeatFrame(jpeg, t);
        }, e => this.snapFailed(e)).finally(() => { this.snapBusy = false; });
    }

    /** scan start without a running stream: the camera's latest keyframe is the first frame while ffmpeg starts. */
    private bridgeSnapshot() {
        const sg = this.scanGen;
        fetchSnapshot(this.mixinDevice as any, sdk.mediaManager as any, 'event').then(jpeg => {
            this.snapFails = 0;
            if (sg !== this.scanGen || !this.scanActive || this.released || this.lastEpisodePostAt >= this.loopStartedAt || this.queue.length)
                return; // the stream was faster, or the scan is over
            const t = monoNow();
            this.bridgeT = t;
            this.lastFrameAt = t;
            this.queue.push({ jpeg, t });
            this.kick();
        }, e => this.snapFailed(e));
    }

    private snapFailed(e: unknown) {
        if (this.released || this.snapDisabled)
            return;
        this.snapFails++;
        if (this.snapFails < SNAP_MAX_FAILS)
            return;
        this.snapDisabled = true; // sticky until the setting changes or the plugin restarts
        this.console.warn(`camera snapshots failed ${this.snapFails} times (${maskUrls(String((e as any)?.message || e))}), paced frames come from the stream again`);
        this.status.onFailure('snapshots failed, using the stream');
        this.syncSource(this.scanActive);
    }

    // ---- 'Camera AI fast path' --------------------------------------------------------------

    /**
     * A verifiable class pushed by the camera (person / vehicle / animal) starts verification at once: an idle camera opens a
     * verified unflagged scan (the same machine as the pixel gate and heartbeat, so the same confirmation rules; the camera's
     * motion edge merges into it), a running low-priority unflagged scan is promoted to normal priority. Never raises motion
     * by itself, never shortens any confirmation rule.
     */
    private fastPath(data: any, now: number) {
        this.fast.expire(now);
        if (!this.plugin.aiFastPath || this.mode === 'off' || this.plugin.config.killSwitch || this.episode.degraded || this.cameraTriggers() === 'none')
            return;
        const cls = (data?.detections || []).map((d: any) => String(d?.className || '').toLowerCase()).find((c: string) => !!claimGroup(c)) || 'object';
        if (this.episode.scanning && !this.episode.synthetic)
            return; // a camera episode is already verifying at normal priority
        this.fast.claim(cls, now);
        if (this.episode.scanning) {
            this.fastGen = this.scanGen; // promote the running unflagged scan
            this.fast.scan(now);
            this.kick();
            return;
        }
        this.synthRaised = false;
        this.synthStartedAt = now;
        this.lastGateOpenAt = now; // keeps the scan open for the quiet window until the camera's edge merges in
        this.synthReason = `camera AI ${cls}`;
        this.fastGen = this.scanGen + 1; // the scan synthStart opens below
        this.apply(this.episode.synthStart(now));
        if (!this.scanActive)
            this.fastGen = undefined;
    }

    /**
     * Continuous monitoring: the pixel gate watches the warm stream. A 'blob' or 'fraction' opening
     * outside any camera episode is only a reason to ASK the service, never to raise motion.
     * Gate mode: it starts a synthetic episode (same state machine as a camera trigger: reset, pre-roll
     * from the ring, ~1 fps frames while the gate stays open and 4 s after, hold/release by the service);
     * motion is raised only when the service confirms a person. Shadow: one frame is posted, a confirmed
     * person is logged and counted, motion is never touched.
     */
    private onGray(frame: Uint8Array) {
        const gate = this.gate;
        if (!gate || this.released || this.mode === 'off' || this.plugin.config.killSwitch || this.episode.degraded || !this.paced)
            return;
        const r = gate.push(frame);
        const hit = r.open && (r.reason === 'blob' || r.reason === 'fraction');
        const now = monoNow();
        if (hit)
            this.lastGateOpenAt = now;
        // Shadow and Gate alike: a pixel-gate hit opens a verified unflagged episode (Shadow only reports would-publish)
        if (hit && !this.episode.scanning && now >= this.synthCooldownUntil) {
            this.synthRaised = false;
            this.synthStartedAt = now;
            this.synthReason = `gate ${r.reason}, blob ${r.blob}px, ${(r.fraction * 100).toFixed(1)}% changed`;
            this.apply(this.episode.synthStart(now));
        }
    }

    /** The camera's own raw motion changed (the only input that marks the mixin as 'touched'). */
    private onCameraRaw(value: boolean) {
        this.touched = true;
        const rising = value && !this.cameraRaw; // judged from the camera edge alone, whatever a pixel-gate episode is doing
        this.cameraRaw = value;
        if (rising) {
            this.fast.edge(monoNow());
            this.claimFrom = monoNow() - 3000; // class events up to 3 s before the edge belong to this trigger
            this.unverifiedLogged = false;
        }
        clearTimeout(this.aiWaitTimer);
        this.aiWaitTimer = undefined;
        const cam = this.cameraTriggers();
        if (rising && (cam === 'none' || (cam === 'ai' && !this.hasClaimSince(this.claimFrom)))) {
            // not a trigger (yet): raw is known (pass-through, Shadow publishes it) but starts nothing
            this.apply(this.episode.raw(true, monoNow(), false));
            if (cam === 'ai') { // 'Camera AI': the edge counts once the camera claims a class (up to AI_CLAIM_WAIT_MS later)
                this.aiWaitTimer = setTimeout(() => { this.aiWaitTimer = undefined; }, AI_CLAIM_WAIT_MS);
                this.aiWaitTimer.unref?.();
            }
            return;
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
        let person = false;
        let animal = false;
        for (const d of data?.detections || []) {
            const c = String(d?.className || '').toLowerCase();
            if (!c || c === 'motion' || NOT_A_CLAIM.includes(c))
                continue;
            const g = claimGroup(c);
            if (g) verifiable = true; else other.push(c);
            if (g === 'person') person = true;
            if (g === 'animal') animal = true;
        }
        if (!verifiable && !other.length)
            return;
        const now = monoNow();
        this.claimEvents = this.claimEvents.filter(e => now - e.at < 10000).concat({ at: now, groups: verifiable ? ['person'] : [], other, person, animal });
        if (this.cameraRaw && this.aiWaitTimer) {
            // the claim the held-back edge was waiting for: it becomes a trigger now
            clearTimeout(this.aiWaitTimer);
            this.aiWaitTimer = undefined;
            this.apply(this.episode.trigger(monoNow()));
        }
        if (this.cameraRaw)
            this.applyClaim();
        if (verifiable)
            this.fastPath(data, monoNow()); // after a held-back edge became a trigger: a camera episode needs no fast path
    }

    // ---- Triggers (plugin-wide): Camera motion, Camera AI, Paced ----
    private aiWaitTimer: NodeJS.Timeout | undefined;
    private aiFallbackLogged = false;
    private get triggers(): string[] { return this.plugin.triggers ?? ['Camera motion', 'Camera AI', 'Paced']; }
    private get paced(): boolean { return this.triggers.includes('Paced'); }
    /** which camera raw edges start an episode: every edge, only AI-claimed ones, or none ('Paced' only) */
    private cameraTriggers(): 'motion' | 'ai' | 'none' {
        const t = this.triggers;
        if (this.sensorless || t.includes('Camera motion'))
            return 'motion';
        if (!t.includes('Camera AI'))
            return 'none';
        const ifaces = this.ifaces;
        if (Array.isArray(ifaces) && !ifaces.includes(ScryptedInterface.ObjectDetector)) {
            if (!this.aiFallbackLogged) {
                this.aiFallbackLogged = true;
                this.console.log('Triggers: this camera reports no AI classes, Camera AI acts as Camera motion');
            }
            return 'motion';
        }
        return 'ai';
    }
    /** the camera's own AI claimed a person for the current scan (from 3 s before its start). */
    private personClaimed(): boolean {
        if (!this.scanActive) return false;
        const from = this.loopStartedAt - 3000;
        return this.claimEvents.some(e => e.person && e.at >= from);
    }

    /** the camera's own AI claimed an animal for the current scan (from 3 s before its start). */
    private animalClaimed(): boolean {
        if (!this.scanActive) return false;
        const from = this.loopStartedAt - 3000;
        return this.claimEvents.some(e => e.animal && e.at >= from);
    }

    private hasClaimSince(from: number): boolean {
        if (!this.triggers.includes('Camera AI')) return false;
        return this.claimEvents.some(e => e.at >= from && (e.groups.length > 0 || e.other.length > 0));
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
        if (!verifiable && other.length && this.mode !== 'off' && !this.unverifiedLogged) {
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

    /** burst phase of the current scan: fast confirm on, not yet confirmed, within burstMaxMs of the scan start. */
    private bursting(now = monoNow()): boolean {
        return this.scanActive && now < this.burstUntil && !this.episode.confirmedGroups && !this.episode.degraded;
    }

    /** switch the warm ffmpeg's JPEG rate to the burst rate and back (best effort; a no-op for streams without the burst filter). */
    private syncBurst(now = monoNow()) {
        const want = this.bursting(now);
        if (!want)
            this.burstNext = undefined;
        if (want === this.burstOn)
            return;
        this.burstOn = want;
        this.frames?.setBurst?.(want);
    }

    private stopSource() {
        this.srcGen++;
        this.scanGen++; // an in-flight request of the old process no longer blocks, its reply is void
        this.frames?.stop();
        this.frames = undefined;
        this.srcHw = false;
        this.cancelHwSwitch();
        this.releaseSrcHw();
        this.hwCleanStop(false);
        this.queue = [];
        this.ring = [];
        this.clock = undefined;
    }

    /** Send the next queued frame if none is in flight for this scan. */
    private kick() {
        const gen = this.scanGen;
        if (this.busyGen === gen || this.released || !this.queue.length || (this.heartbeatBusy && this.fastGen !== gen))
            return; // one request in flight per camera: an episode frame waits for an in-flight heartbeat
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
            this.lastEpisodePostAt = monoNow();
            if (this.fastGen === gen) this.fast.posted(this.lastEpisodePostAt);
            if (item.jpeg.length <= MAX_THUMB_JPEG_BYTES) // base64 happens lazily when the panel renders
                this.status.setLastFrame({ jpeg: item.jpeg, boxes: [] });
            else {
                this.status.setThumbTooLarge(); // the previous thumbnail stays; the panel says so when there is none
                if (!this.thumbWarned) {
                    this.thumbWarned = true;
                    this.console.log(`snapshot ${Math.round(item.jpeg.length / 1024)} KB is too large for the panel thumbnail (limit ${Math.round(MAX_THUMB_JPEG_BYTES / 1024)} KB), keeping the previous one`);
                }
            }
            const burst = this.bursting();
            const reply = await this.plugin.client.frame(this.verifierId, item.jpeg, item.t, {
                low: this.episode.synthetic && this.fastGen !== gen,
                deadlineMs: burst ? DEFAULTS.burstDeadlineMs : undefined,
                claim: !(this.plugin.fastConfirm ?? true) ? undefined : this.personClaimed() ? 'person' : this.animalClaimed() ? 'animal' : undefined,
            });
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
                    this.noteHit(item.jpeg, view.boxes);
                    this.trackFrom(reply);
                }
                catch (e) { this.warnBadReply(e); }
                this.probeAttempt = 0; // a real frame succeeded: only now is the service really back
                let fx: Effects;
                try { fx = this.episode.reply(this.masked(reply), monoNow(), item.t); }
                catch (e) { this.warnBadReply(e); fx = this.episode.skip(monoNow()); } // a malformed field is a skipped frame, never an outage
                this.apply(fx);
                if (this.episode.synthetic && !this.synthRaised && this.mode !== 'off' && this.episode.wouldPublish) {
                    this.synthRaised = true;
                    this.console.log(`${groupsSeen(reply).join('+') || 'person'} seen but camera did not trigger: ${this.mode === 'gate' ? 'raised motion' : 'would raise motion (Shadow)'} (${this.synthReason})`);
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
            this.syncBurst();
            if (this.burstNext && gen === this.scanGen && !this.queue.length) {
                this.queue.push(this.burstNext);
                this.burstNext = undefined;
            }
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
                description: 'Type a name to add, remove to delete.' } as Setting);
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
            mode.description = this.notOutermost ? 'WARNING: not the outermost MotionSensor mixin, gating has no effect.' : 'Off, Shadow (measure only) or Gate.';
        // one group: the Status panel on top, then Mode
        // tabs (subgroups): Overview (the Status panel), Setup (Mode, Detect, Stream, Zoom-out), Zones
        for (const x of settings)
            if (!x.subgroup) x.subgroup = 'Setup'; // never ungrouped: no implicit 'General' tab
        return [
            { key: 'refresh', subgroup: 'Overview', title: 'Refresh', type: 'button' },
            { key: 'status', subgroup: 'Overview', title: 'Status', type: 'html', readonly: true, value: this.status.render(Date.now()) },
            ...settings.filter(x => x.subgroup === 'Setup'),
            ...settings.filter(x => x.subgroup !== 'Setup'),
        ];
    }

    async putMixinSetting(key: string, value: SettingValue): Promise<boolean | void> {
        if (key === 'refresh') {
            this.notifySettings(); // user-initiated: one re-render from the in-memory state, nothing stored
            return true;
        }
        if (key === 'forgetLookalikes') {
            try {
                const r = await this.plugin.client.forgetLookalikes(this.verifierId);
                this.console.log(r?.forgotten ? 'Look-alikes forgotten' : 'No look-alikes to forget');
            }
            catch (e: any) {
                this.console.warn('look-alikes not forgotten:', maskUrls(String(e?.message || e)));
            }
            return true;
        }
        if (key === 'forgetSpots') {
            try {
                const r = await this.plugin.client.forgetSpots(this.verifierId);
                this.console.log(r?.forgotten ? 'Parking spots forgotten' : 'No parking spots to forget');
            }
            catch (e: any) {
                this.console.warn('parking spots not forgotten:', maskUrls(String(e?.message || e)));
            }
            return true;
        }
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
            this.clearStreamRetry();
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
    overviewRow(): OverviewCamera {
        this.refreshStatus();
        this.refreshViewInfo();
        const base = String(this.name || this.id);
        return { name: `${base}${this.sensorless ? ' (sensorless)' : ''}`, base, ...this.status.overviewCells(), stream: this.streamUp ? this.streamName || undefined : undefined,
            fallback: this.streamFallback, view: this.viewInfo };
    }

    /** the camera plugin's crop / mirror settings for the Overview: read in the background, at most once a minute */
    private viewInfo: ViewInfo = {};
    private viewInfoAt = -Infinity;
    private refreshViewInfo() {
        const now = monoNow();
        if (now - this.viewInfoAt < 60_000)
            return;
        this.viewInfoAt = now;
        Promise.resolve((this.mixinDevice as any)?.getSettings?.()).then((s: any) => { this.viewInfo = viewInfoOf(Array.isArray(s) ? s : []); }, () => undefined);
    }

    /** the chosen stream failed: analyse the default stream for now and retry the chosen one later (never for good). */
    private enterFallback(msg: string) {
        this.streamFallback = true;
        this.streamRetrying = false;
        const d = this.streamRetryDelays[Math.min(this.streamRetries, this.streamRetryDelays.length - 1)];
        this.console.warn(`${msg}; retrying it in ${Math.round(d / 1000)} s`);
        this.status.onFailure(`stream ${this.storageSettings.values.stream} unavailable, analysing the default stream`);
        this.viewChanged();
        clearTimeout(this.streamRetryTimer);
        this.streamRetries++;
        this.streamRetryTimer = setTimeout(() => this.retryStream(), d);
        this.streamRetryTimer.unref?.();
    }

    private retryStream() {
        this.streamRetryTimer = undefined;
        if (this.released || !this.streamFallback)
            return;
        if (this.scanActive) {
            // never during an episode: a chosen stream that is still down would leave it without frames
            this.streamRetryTimer = setTimeout(() => this.retryStream(), STREAM_RETRY_BUSY_MS);
            this.streamRetryTimer.unref?.();
            return;
        }
        this.streamFallback = false;
        this.streamRetrying = true;
        this.console.log(`retrying stream ${this.storageSettings.values.stream}`);
        if (this.frames || this.warmRetry) {
            clearTimeout(this.warmRetry);
            this.warmRetry = undefined;
            this.stopSource();
            this.syncSource(this.scanActive);
        }
    }

    private clearStreamRetry() {
        clearTimeout(this.streamRetryTimer);
        this.streamRetryTimer = undefined;
        this.streamRetries = 0;
        this.streamRetrying = false;
    }

    /** the analysed view changed (to the default stream or back): what the service tracked in the other view must not linger (parked cars of a
     * neighbouring crop held for minutes). An episode in progress keeps its session. */
    private viewChanged() {
        if (this.scanActive || this.released)
            return;
        this.plugin.client.reset(this.verifierId, true).catch(() => undefined);
    }

    /** the stream id to request, undefined = the default stream */
    private chosenStream(): string | undefined {
        const v = String(this.storageSettings.values.stream || 'default');
        return v === 'default' || this.streamFallback ? undefined : v;
    }

    async release() {
        if (this.released)
            return; // idempotent: a replaced instance may also be released later by Scrypted
        this.expirePresence(true);
        clearTimeout(this.aiWaitTimer);
        clearTimeout(this.snapTimer);
        this.snapTimer = undefined;
        if (this.jId) {
            // never leave a journal episode 'in progress' across a restart or a removal
            const outcome = this.jConfirmed ? 'released' : 'aborted';
            try { this.journal('end', { outcome, note: 'plugin restart', timeline_event: { t_ms: Date.now(), text: `${outcome} (plugin restart)` } }); } catch { /* best effort */ }
            this.jId = undefined;
        }
        this.released = true;
        clearTimeout(this.streamFirstTimer);
        this.clearStreamRetry();
        this.endHwProbe();
        this.stopSource();
        clearTimeout(this.warmRetry);
        clearTimeout(this.hwRetryTimer);
        clearInterval(this.tickTimer);
        clearTimeout(this.probeTimer);
        clearTimeout(this.verifyTimer);
        this.listener?.removeListener();
        this.claimListener?.removeListener();
        this.plugin.unregister(this);
        // Best effort only: once removed, a write from this mixin is an event, not persisted state.
        // The next motion write from the camera re-establishes the real state.
        try { if ((this.touched || this.lastPublished) && this.motionDetected !== this.episode.isRaw) this.motionDetected = this.episode.isRaw; } catch (e) { /* ignore */ }
        try { if (this.lastOccupied) this.occupied = false; } catch (e) { /* ignore */ }
        await super.release();
    }
}

/** Per-camera offset (0..ms) from a well-mixed hash of the device id: short numeric ids spread over the whole range. */
export function spreadJitter(id: string, ms: number): number {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
    h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
    h = (h ^ (h >>> 16)) >>> 0;
    return h % Math.max(1, ms);
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
