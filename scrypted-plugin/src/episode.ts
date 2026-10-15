// Pure state machine deciding the camera's motion state. No I/O, no timers, no Scrypted:
// every input carries the clock, every output is returned, so it is unit-testable.
//
// Modes
//   off     pass-through, the service is never called.
//   shadow  exactly the Gate state machine (wouldPublish); the published motion mirrors raw motion instead.
//   gate    motion false until the service confirms a person, a moving (non-parked) vehicle or an animal, held while any is present.
// Fail open: kill switch, mode off, a failed request or an unhealthy service force pass-through.

import { FrameReply, groupOf } from './client';
import { isParked } from './reply-view';

export type Mode = 'off' | 'shadow' | 'gate';

export interface EpisodeConfig {
    mode: Mode;
    confirmTimeoutMs: number;
    maxHoldMs: number;
    /** enabled detection groups (per-camera 'Detect' setting); undefined = all. A disabled group never confirms and does not hold. */
    detect?: Group[];
    killSwitch: boolean;
    /** Vehicle confirm (moving) threshold; default 0.6. */
    vehicleConfirm?: number;
    /** an unconfirmed episode whose raw motion ended keeps verifying this long after the raw end before a veto (0 = off). */
    graceMs?: number;
}

/** Detection groups the service confirms; the camera may claim one (see setClaim). Parked vehicles never confirm. */
export type Group = 'person' | 'vehicle' | 'animal';
const GROUPS: Group[] = ['person', 'vehicle', 'animal'];

const VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle'];
const VEHICLE_CONFIRM_SCORE = 0.6;
const VEHICLE_HOLD_SCORE = 0.3;
const VEHICLE_HOLD_MS = 6000;

/** Vehicle evidence of one reply: non-parked car/truck/bus/motorcycle detections only (bicycles are drawn, never decide). */
export const VEHICLE_HOLD_WINDOW_MS = VEHICLE_HOLD_MS;
export function vehicleView(r: FrameReply, confirmScore = VEHICLE_CONFIRM_SCORE): { tick: boolean; hold: boolean; best?: { cls: string; score: number } } {
    let tick = false, hold = false, best: { cls: string; score: number } | undefined;
    for (const d of r.detections || []) {
        if (!VEHICLE_CLASSES.includes(d.cls) || d.weak === true || !Array.isArray(d.box) || d.box.length !== 4 || isParked(d, r.vehicle?.tracks))
            continue;
        if (d.score >= VEHICLE_HOLD_SCORE)
            hold = true;
        if (d.score >= confirmScore) {
            tick = true;
            if (!best || d.score > best.score) best = { cls: d.cls, score: d.score };
        }
    }
    return { tick, hold, best };
}

function animalDetail(r: FrameReply): string {
    let best: { cls: string; score: number } | undefined;
    for (const d of r.detections || [])
        if (groupOf(d) === 'animal' && (!best || d.score > best.score)) best = { cls: d.cls, score: d.score };
    return best ? `animal (${best.cls} ${best.score.toFixed(2)})` : 'animal';
}

export type Outcome = 'released' | 'vetoed' | 'maxhold' | 'aborted';

export interface EpisodeSummary {
    mode: Mode;
    startedAt: number;
    /** undefined while raw motion is still on at the time the episode closed. */
    rawEndedAt?: number;
    outcome: Outcome;
    /** ms from the raw trigger to the service confirming a person; undefined if never confirmed. */
    confirmedAfterMs?: number;
    /** ms from the raw trigger to the first frame posted to the service. */
    postedAfterMs?: number;
    /** ms from the raw trigger to the first service verdict (any reply). */
    verdictAfterMs?: number;
    /** service-reported stage timings of the confirming (else the first) reply. */
    service?: { queue_ms?: number; infer_ms?: number; total_ms?: number };
    /** frames not sent or dropped by the service as stale (one request in flight, latest wins). */
    skipped: number;
    /**
     * capture time of the confirming frame relative to the raw trigger. Negative when a pre-roll
     * frame (taken before the trigger) already confirmed the person. Shows what pre-roll bought.
     */
    confirmedFrameOffsetMs?: number;
    /** when the verifier released (gone / max hold / abort). Undefined for a veto. */
    closedAt: number;
    /** seconds the gate would hold motion beyond the raw end (0 if none). */
    heldBeyondRawS: number;
    note?: string;
    /** which group(s) confirmed, or 'unverified' when the camera's claim could not be checked. */
    confirmedBy?: string[];
    /** the same with the confirming class and best score, e.g. 'vehicle (truck 0.62)'. */
    confirmedDetail?: string[];
    /** what opened the episode: the camera's motion or the pixel gate. */
    origin?: 'camera' | 'synth';
    /** how the service confirmed the person (service 0.3.12+): 'fast_move' | 'fast_claim' | 'ticks'. */
    confirmPath?: string;
    /** service replies of the pending phase up to and including the confirming one. */
    confirmFrames?: number;
    /** confirmed after the camera's raw motion had ended (inside the grace window). */
    afterRawEnd?: boolean;
    /** a veto resting on too little evidence (fewer than 2 analysed frames, or no verdict before the raw end). */
    inconclusive?: boolean;
}

export type LogEvent =
    | { kind: 'summary'; summary: EpisodeSummary }
    | { kind: 'degraded'; reason: string }
    | { kind: 'recovered' };

export interface Effects {
    motion: boolean;
    /** whether frames should currently be pulled and sent to the service. */
    scanning: boolean;
    /** the caller should POST /reset now (fresh episode). */
    reset: boolean;
    logs: LogEvent[];
}

type State = 'idle' | 'pending' | 'active' | 'closed';

export class Episode {
    private state: State = 'idle';
    /** the camera's own raw motion, and the synthetic (pixel gate) one: only the camera's is ever passed through. */
    private camRaw = false;
    private synthRaw = false;
    private origin: 'camera' | 'synth' = 'camera';
    private get rawTrue() { return this.camRaw || this.synthRaw; }
    private rawEndedAt?: number;
    private startedAt = 0;
    private pendingSince = 0;
    private confirmedAt?: number;
    private confirmedFrameOffset?: number;
    private postedAt?: number;
    private verdictAt?: number;
    private firstService?: EpisodeSummary['service'];
    private confirmService?: EpisodeSummary['service'];
    private skipped = 0;
    private pendingReplies = 0;
    private confirmPath?: string;
    private confirmFrames?: number;
    private holdOver = false;
    private degradedReason?: string;
    private lastMotion = false;
    private lastScanning = false;
    private unverified = false;
    private confirmedBy: string[] = [];
    private confirmedDetail: string[] = [];
    /** vehicle ticks of the last 3 replies and when a vehicle was last seen (hold). */
    private vehTicks: boolean[] = [];
    private vehSeenAt = 0;

    constructor(private cfg: EpisodeConfig, initialRaw = false) {
        this.camRaw = initialRaw;
        this.lastMotion = initialRaw;
    }

    get motion() { return this.lastMotion; }
    /** the Gate value of motion (published in Gate; in Shadow only reported) */
    get wouldPublish() { return this.lastWould; }
    private lastWould = false;
    get scanning() { return this.lastScanning; }
    get degraded() { return this.degradedReason; }
    /** the CAMERA's raw motion (never the synthetic one). */
    get isRaw() { return this.camRaw; }
    /** the groups that confirmed the current episode while it is confirmed (active), else undefined */
    get confirmedGroups(): string[] | undefined { return this.state === 'active' ? this.confirmedBy.slice() : undefined; }
    get pending() { return this.state === 'pending'; }
    /** the current confirmation came after the camera's raw motion had ended (grace window). */
    get lateConfirm() { return this.state === 'active' && this.confirmedAt !== undefined && this.rawEndedAt !== undefined && this.rawEndedAt >= this.pendingSince && this.confirmedAt > this.rawEndedAt; }
    /** a pixel-gate episode is running (pending or confirmed). */
    get synthetic() { return this.origin === 'synth' && (this.state === 'pending' || this.state === 'active'); }
    get bypassed() { return this.cfg.mode === 'off' || this.cfg.killSwitch; }

    private get passthrough() { return this.bypassed || !!this.degradedReason; }

    private out(logs: LogEvent[], reset = false): Effects {
        // one state machine for Shadow and Gate: wouldPublish is the Gate value; only Gate publishes it
        const would = this.passthrough
            ? this.camRaw
            : this.state === 'active' || (this.state === 'pending' && this.holdOver);
        this.lastWould = would;
        const motion = this.cfg.mode === 'gate' ? would : this.camRaw;
        const scanning = !this.passthrough && (this.state === 'pending' || this.state === 'active');
        this.lastMotion = motion;
        this.lastScanning = scanning;
        return { motion, scanning, reset, logs };
    }

    private startPending(now: number, fresh: boolean) {
        this.state = 'pending';
        this.pendingSince = now;
        this.confirmedAt = undefined;
        this.postedAt = undefined;
        this.verdictAt = undefined;
        this.firstService = this.confirmService = undefined;
        this.skipped = 0;
        this.pendingReplies = 0;
        this.confirmPath = this.confirmFrames = undefined;
        if (fresh) {
            this.startedAt = now;
            this.origin = 'camera';
        }
        this.unverified = false;
        this.confirmedBy = [];
        this.confirmedDetail = [];
        this.vehTicks = [];
        this.vehSeenAt = 0;
    }

    private close(outcome: Outcome, now: number, logs: LogEvent[], note?: string) {
        const confirmed = this.confirmedAt !== undefined;
        const held = outcome !== 'vetoed' && confirmed && this.rawEndedAt !== undefined
            ? Math.max(0, now - this.rawEndedAt) / 1000
            : 0;
        logs.push({
            kind: 'summary',
            summary: {
                mode: this.cfg.mode,
                startedAt: this.startedAt,
                rawEndedAt: this.rawEndedAt,
                outcome,
                confirmedAfterMs: confirmed ? this.confirmedAt! - this.pendingSince : undefined,
                postedAfterMs: this.postedAt !== undefined ? this.postedAt - this.pendingSince : undefined,
                verdictAfterMs: this.verdictAt !== undefined ? this.verdictAt - this.pendingSince : undefined,
                service: this.confirmService || this.firstService,
                skipped: this.skipped,
                confirmedFrameOffsetMs: confirmed ? this.confirmedFrameOffset : undefined,
                closedAt: now,
                heldBeyondRawS: held,
                note,
                confirmedBy: confirmed && this.confirmedBy.length ? this.confirmedBy.slice() : undefined,
                confirmedDetail: confirmed && this.confirmedDetail.length ? this.confirmedDetail.slice() : undefined,
                origin: this.origin,
                confirmPath: confirmed ? this.confirmPath : undefined,
                confirmFrames: confirmed ? this.confirmFrames : undefined,
                afterRawEnd: confirmed && this.rawEndedAt !== undefined && this.rawEndedAt >= this.pendingSince && this.confirmedAt! > this.rawEndedAt ? true : undefined,
                inconclusive: outcome === 'vetoed' && (this.pendingReplies < 2 || this.verdictAt === undefined
                    || (this.rawEndedAt !== undefined && this.rawEndedAt >= this.pendingSince && this.verdictAt > this.rawEndedAt)) ? true : undefined,
            },
        });
        this.holdOver = false;
        this.unverified = false;
        this.confirmedAt = undefined;
        this.synthRaw = false; // whatever opened it: a closed episode keeps no synthetic raw input
        this.state = this.rawTrue ? 'closed' : 'idle';
    }

    /** Leave scanning states (entering pass-through). */
    private abort(now: number, logs: LogEvent[], note: string) {
        if (this.state === 'pending' || this.state === 'active')
            this.close('aborted', now, logs, note);
        this.holdOver = false;
        this.synthRaw = false;
        // A later return to service mode re-arms from the current raw value.
        this.state = 'idle';
    }

    /** Re-enter service mode: if raw is on, hold pass-through value until the first verdict. */
    private resume(now: number): boolean {
        if (this.cfg.mode === 'off' || !this.rawTrue)
            return false;
        this.startedAt = now;
        this.rawEndedAt = undefined;
        this.startPending(now, false);
        this.holdOver = true;
        return true;
    }

    setConfig(cfg: EpisodeConfig, now: number): Effects {
        const wasPass = this.passthrough;
        const wasMode = this.cfg.mode;
        this.cfg = cfg;
        const logs: LogEvent[] = [];
        let reset = false;
        if (this.passthrough && !wasPass)
            this.abort(now, logs, cfg.killSwitch ? 'kill switch' : 'mode off');
        else if (!this.passthrough && wasPass)
            reset = this.resume(now);
        else if (!this.passthrough && wasMode !== cfg.mode && (this.state === 'idle' || this.state === 'closed') && this.rawTrue)
            reset = this.resume(now);
        if (this.synthRaw && cfg.mode === 'off') {
            // Off never runs unflagged (synthetic) episodes: end it now
            if (this.state === 'pending' || this.state === 'active')
                this.close('aborted', now, logs, 'mode changed');
            this.synthRaw = false;
            if (this.state === 'closed' && !this.camRaw)
                this.state = 'idle';
        }
        return this.out(logs, reset);
    }

    /** The camera's own raw motion changed. */
    raw(value: boolean, now: number, trigger = true): Effects {
        const logs: LogEvent[] = [];
        let reset = false;
        if (value === this.camRaw)
            return this.out(logs);
        this.camRaw = value;
        if (value && !trigger) {
            // raw motion that is not a trigger (Triggers setting): known for pass-through and Shadow, starts nothing
            this.rawEndedAt = undefined;
            return this.out(logs);
        }
        if (value) {
            this.rawEndedAt = undefined;
            if (this.synthRaw && (this.state === 'pending' || this.state === 'active')) {
                this.origin = 'camera'; // merge into the running pixel-gate episode: one episode, no extra motion edge
            }
            else if (!this.passthrough && (this.state === 'idle' || this.state === 'closed')) {
                this.synthRaw = false; // a stale synthetic input must not swallow the camera's own trigger

                this.startPending(now, true);
                reset = true;
            }
            else if (this.passthrough) {
                this.startedAt = now;
            }
        }
        else {
            this.rawEndedAt = now;
            if (this.state === 'closed' && !this.synthRaw)
                this.state = 'idle';
            else if (this.state === 'active' && this.unverified)
                this.close('released', now, logs); // unverified pass-through: held by the camera's raw motion only
        }
        return this.out(logs, reset);
    }

    /** A raw edge that was held back (waiting for the camera's AI claim) becomes a trigger now. */
    trigger(now: number): Effects {
        if (!this.camRaw || this.passthrough || !(this.state === 'idle' || this.state === 'closed'))
            return this.out([]);
        this.synthRaw = false;
        this.startPending(now, true);
        return this.out([], true);
    }

    /**
     * Gate mode only: the pixel gate saw something the camera did not trigger on. Opens a fresh confirmation
     * (reset + scan) whose raw input is synthetic: it NEVER raises motion by itself and is never passed through.
     */
    synthStart(now: number): Effects {
        if (this.passthrough || this.synthRaw || this.state === 'pending' || this.state === 'active')
            return this.out([]);
        this.startPending(now, true); // from idle or from a closed (released) camera episode
        this.origin = 'synth';
        this.synthRaw = true;
        return this.out([], true);
    }

    /** The pixel gate has been quiet (or the cap was reached): the synthetic raw input ends, a confirmed episode is held by the service. */
    synthEnd(now: number): Effects {
        if (!this.synthRaw)
            return this.out([]);
        this.synthRaw = false;
        if (!this.camRaw)
            this.rawEndedAt = now;
        if (this.state === 'closed' && !this.camRaw)
            this.state = 'idle';
        return this.out([]);
    }

    /**
     * The camera claims a class this service cannot verify (e.g. a package): fail open for this episode.
     * Motion is raised at once and held by the camera's raw motion (and the max hold), not by the service.
     */
    acceptUnverified(now: number): Effects {
        if (this.passthrough || this.state !== 'pending')
            return this.out([]);
        this.state = 'active';
        this.confirmedAt = now;
        this.confirmedFrameOffset = 0;
        this.unverified = true;
        this.confirmedBy = ['unverified'];
        this.confirmedDetail = ['unverified'];
        this.holdOver = false;
        return this.out([]);
    }

    /** A frame request is about to be sent (first one of the pending phase is recorded). */
    posted(now: number) {
        if ((this.state === 'pending' || this.state === 'active') && this.postedAt === undefined)
            this.postedAt = now;
    }

    /**
     * A frame was not analysed: skipped because a request was in flight (latest wins) or dropped by
     * the service as stale (503 {reason:"stale"}). Changes nothing else and never degrades the service.
     */
    skip(now: number): Effects {
        if (this.state === 'pending' || this.state === 'active')
            this.skipped++;
        return this.out([]);
    }

    /**
     * @param now wall clock when the reply was handled (latency the user feels).
     * @param frameT capture time of the frame that produced the reply; pre-roll frames carry
     * past times, so replies for a burst arrive back to back with frameT before pendingSince.
     */
    reply(r: FrameReply, now: number, frameT: number = now): Effects {
        const logs: LogEvent[] = [];
        let reset = false;
        if (this.passthrough || (this.state !== 'pending' && this.state !== 'active'))
            return this.out(logs);
        if (this.unverified)
            return this.out(logs); // pass-through episode: the service verdict is not used
        const timings = { queue_ms: r.queue_ms, infer_ms: r.infer_ms ?? r.coral_ms, total_ms: r.total_ms };
        if (this.verdictAt === undefined) {
            this.verdictAt = now;
            this.firstService = timings;
        }
        // person.confirmed and animal.confirmed are the service's 2-of-3. A vehicle is decided here from the detections:
        // a car/truck/bus/motorcycle (never a bicycle) at >= 0.6, not parked, in 2 of the last 3 replies; it is held while one
        // at >= 0.3 keeps arriving and released after VEHICLE_HOLD_MS without. Single frames and other classes never confirm.
        const on = (g: Group) => !this.cfg.detect || this.cfg.detect.includes(g);
        const veh = vehicleView(r, this.cfg.vehicleConfirm);
        if (this.state === 'pending') {
            this.pendingReplies++;
            this.vehTicks.push(veh.tick);
            if (this.vehTicks.length > 3) this.vehTicks.shift();
        }
        if (veh.hold) this.vehSeenAt = now;
        const vehicleConfirmed = this.vehTicks.filter(Boolean).length >= 2;
        const vehiclePresent = veh.hold || (this.vehSeenAt > 0 && now - this.vehSeenAt <= VEHICLE_HOLD_MS);
        const confirmedNow: Record<Group, boolean> = { person: !!r.person?.confirmed, vehicle: vehicleConfirmed, animal: !!r.animal?.confirmed };
        const presentNow = (on('person') && !!r.person?.present) || (on('vehicle') && vehiclePresent) || (on('animal') && !!r.animal?.present);
        if (this.state === 'pending') {
            // the camera's claimed class is only a hint (see acceptUnverified): any enabled group may confirm
            const by = GROUPS.filter(g => on(g) && confirmedNow[g]);
            if (by.length) {
                this.state = 'active';
                this.confirmedAt = now;
                this.confirmedFrameOffset = frameT - this.pendingSince;
                this.confirmService = timings;
                this.confirmedBy = by;
                this.confirmFrames = this.pendingReplies;
                this.confirmPath = by.includes('person') && typeof r.person?.confirmed_by === 'string' ? r.person.confirmed_by : undefined;
                this.confirmedDetail = by.map(g => g === 'vehicle' && veh.best ? `vehicle (${veh.best.cls} ${veh.best.score.toFixed(2)})` : g === 'animal' ? animalDetail(r) : g);
                this.holdOver = false;
            }
        }
        else if (!presentNow) {
            // Released while the camera's raw motion is still on: the state is 'closed' and stays so until raw
            // goes false and rises again (no immediate re-episode that would only be vetoed). Motion stays false in Gate.
            this.close('released', now, logs);
        }
        return this.out(logs, reset);
    }

    /** confirm timeout, extended to graceMs after the raw end when the camera's motion ended while still unconfirmed. */
    private vetoAt(): number {
        const base = this.pendingSince + this.cfg.confirmTimeoutMs;
        const g = this.cfg.graceMs ?? 0;
        if (g > 0 && !this.rawTrue && this.rawEndedAt !== undefined && this.rawEndedAt >= this.pendingSince)
            return Math.max(base, this.rawEndedAt + g);
        return base;
    }

    tick(now: number): Effects {
        const logs: LogEvent[] = [];
        if (!this.passthrough) {
            if (this.state === 'pending' && now >= this.vetoAt()) {
                // A veto needs at least one verdict. No reply at all means the service or the stream
                // is not working: fail open instead of silently dropping a real trigger.
                if (this.verdictAt === undefined)
                    return this.failure('no verdict within the confirm timeout', now);
                this.close('vetoed', now, logs);
            }
            else if (this.state === 'active' && this.confirmedAt !== undefined && now - this.confirmedAt >= this.cfg.maxHoldMs) {
                this.close('maxhold', now, logs);
                if (this.rawTrue) {
                    // safety ceiling while raw is still on: start a fresh confirmation instead of closing for good;
                    // motion stays as it is until the first verdict (a still-present subject re-confirms at once)
                    this.startedAt = now;
                    this.rawEndedAt = undefined;
                    this.startPending(now, false);
                    this.holdOver = true;
                }
            }
        }
        return this.out(logs);
    }

    /** A request failed, timed out, returned non-200, or health said not ok. */
    failure(reason: string, now: number): Effects {
        const logs: LogEvent[] = [];
        if (this.degradedReason)
            return this.out(logs);
        const wasPass = this.passthrough;
        this.degradedReason = reason;
        logs.push({ kind: 'degraded', reason });
        if (!wasPass)
            this.abort(now, logs, reason);
        return this.out(logs);
    }

    /** The service answered healthily (health probe ok). Leaves degraded mode. */
    healthy(now: number): Effects {
        const logs: LogEvent[] = [];
        if (!this.degradedReason)
            return this.out(logs);
        this.degradedReason = undefined;
        logs.push({ kind: 'recovered' });
        const reset = !this.passthrough && this.resume(now);
        return this.out(logs, reset);
    }
}

/** Exponential backoff for retries: 1 s, 2 s, 4 s ... capped at 60 s. attempt starts at 0. */
export function backoffMs(attempt: number, base = 1000, cap = 60000): number {
    return Math.min(cap, base * 2 ** Math.max(0, attempt));
}

export function formatSummary(s: EpisodeSummary): string {
    const t = (ms: number) => new Date(ms).toISOString().substring(11, 19);
    const parts = [
        `${s.mode} raw ${t(s.startedAt)}-${s.rawEndedAt !== undefined ? t(s.rawEndedAt) : 'ongoing'}`,
        s.confirmedAfterMs !== undefined
            ? `confirmed +${s.confirmedAfterMs} ms` + (s.confirmPath ? ` (${s.confirmPath}${s.confirmFrames ? `, ${s.confirmFrames} frame${s.confirmFrames === 1 ? '' : 's'}` : ''})` : '')
            + (s.afterRawEnd ? ' (after raw end, grace)' : '')
            + (s.confirmedFrameOffsetMs !== undefined && s.confirmedFrameOffsetMs < 0 ? ` (frame ${(s.confirmedFrameOffsetMs / 1000).toFixed(1)}s, pre-roll)` : '')
            : 'never confirmed',
        s.outcome === 'vetoed' ? (s.inconclusive ? 'VETOED (inconclusive)' : 'VETOED') : `${s.outcome} ${t(s.closedAt)}`,
        `held beyond raw ${s.heldBeyondRawS.toFixed(0)}s`,
    ];
    if (s.postedAfterMs !== undefined)
        parts.push(`posted +${s.postedAfterMs} ms`);
    if (s.verdictAfterMs !== undefined)
        parts.push(`verdict +${s.verdictAfterMs} ms`);
    if (s.service) {
        const { queue_ms, infer_ms, total_ms } = s.service;
        const st = [['queue', queue_ms], ['infer', infer_ms], ['total', total_ms]].filter(x => x[1] !== undefined).map(x => `${x[0]} ${Math.round(x[1] as number)}`);
        if (st.length)
            parts.push(`service ${st.join('/')} ms`);
    }
    if (s.confirmedBy?.length)
        parts.push(`confirmed: ${(s.confirmedDetail?.length ? s.confirmedDetail : s.confirmedBy).join(' + ')}`);
    if (s.skipped)
        parts.push(`${s.skipped} frames skipped`);
    if (s.note)
        parts.push(s.note);
    return parts.join(' | ');
}

export function percentile(values: number[], p: number): number | undefined {
    if (!values.length)
        return undefined;
    const v = values.slice().sort((a, b) => a - b);
    return v[Math.min(v.length - 1, Math.max(0, Math.ceil(p * v.length) - 1))];
}

export interface LatencyStats {
    n: number;
    posted: { p50?: number; p95?: number };
    verdict: { p50?: number; p95?: number };
    confirmed: { p50?: number; p95?: number };
}

/** p50 / p95 (ms) of trigger->first frame posted, ->first verdict, ->confirmed over the given episodes. */
export function latencyStats(list: EpisodeSummary[]): LatencyStats {
    const real = list.filter(s => s.outcome !== 'aborted');
    const pick = (f: (s: EpisodeSummary) => number | undefined) => {
        const v = real.map(f).filter((x): x is number => x !== undefined);
        return { p50: percentile(v, 0.5), p95: percentile(v, 0.95) };
    };
    return {
        n: real.length,
        posted: pick(s => s.postedAfterMs),
        verdict: pick(s => s.verdictAfterMs),
        confirmed: pick(s => s.confirmedAfterMs),
    };
}

const ms = (v?: number) => v === undefined ? 'n/a' : `${Math.round(v)}`;

/** One-paragraph rolling summary (last 50 episodes) for the settings description. */
export function summarize(list: EpisodeSummary[]): string {
    const real = list.filter(s => s.outcome !== 'aborted');
    if (!real.length)
        return 'No episodes yet.';
    const vetoed = real.filter(s => s.outcome === 'vetoed').length;
    const held = real.reduce((a, s) => a + s.heldBeyondRawS, 0);
    const l = latencyStats(real);
    const last = formatSummary(list[list.length - 1]);
    return `Last ${real.length} episodes: ${vetoed} vetoed, ${real.length - vetoed} confirmed, ${held.toFixed(0)}s total held beyond raw. `
        + `Latency from raw trigger, p50/p95 ms: first frame posted ${ms(l.posted.p50)}/${ms(l.posted.p95)}, first verdict ${ms(l.verdict.p50)}/${ms(l.verdict.p95)}, confirmed ${ms(l.confirmed.p50)}/${ms(l.confirmed.p95)}. Latest: ${last}`;
}
