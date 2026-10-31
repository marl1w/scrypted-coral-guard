// Camera tamper detector: covered, moved (turned) or blurred (defocused, sprayed). Runs on the tiny gray frames the
// pixel gate already gets (128x72, 1 fps), sampled every few seconds: no extra ffmpeg, no Coral inference.
// Per sample: 2x2 average to 64x36, per-pixel median of the last 3 samples (drops rain, snow, headlight sweeps and
// passers-by that last less than one sample), then brightness, contrast (std), edge energy (RMS Laplacian) and a 32x18
// gradient-magnitude map, compared with a slowly updated reference (EMA, updated only by explained clean samples).
// References are kept per profile (the analysed stream, and day / night when the camera has a Night mode sensor), each with
// a long-term anchor of its sharpest stable view, so a slow spray or defocus that the EMA would follow is still caught.
// A sudden global brightness or contrast step with the structure kept (IR switch, a floodlight) re-learns the reference.
// The sensor turns on after a condition has held for onMs and off after a clean view for offMs.

import { DEFAULTS } from './defaults';

export type TamperReason = 'covered' | 'moved' | 'blurred';
/** a sample that is not tamper but not explained either (part of the view changed): the reference is frozen */
export type TamperCondition = TamperReason | 'changed';

export interface TamperOptions {
    /** gray frame size, as delivered (the gate's 128x72); width and height must be even */
    width: number;
    height: number;
    /** at most one sample per this many ms */
    sampleMs: number;
    /** a condition must hold this long before the sensor turns on, and a clean view this long before it turns off */
    onMs: number;
    offMs: number;
    /** reference time constant (EMA), ms */
    refTauMs: number;
    /** covered: std and edge energy both below these shares of the reference */
    coveredStdRatio: number;
    coveredEdgeRatio: number;
    /** covered: std below this many gray levels is flat whatever the reference (unless the reference is itself near flat) */
    coveredStdAbs: number;
    /** covered: mean below darkLevel or above brightLevel with the std below brightStdRatio of the reference */
    darkLevel: number;
    brightLevel: number;
    brightStdRatio: number;
    /** damage safeguard (never an alarm): share of the anchor's textured cells (8x6 grid) whose gradient energy collapsed ... */
    damageFraction: number;
    /** ... each below this share of the anchor's (gain-corrected) */
    damageCellRatio: number;
    /** blurred: sharpness (edge / std) below this share of the reference or the anchor */
    blurRatio: number;
    blurMaxLevelDelta: number;
    /** moved: gradient-map NCC with the reference below this, while sharpness stays above sharpKeepRatio of the reference */
    movedNcc: number;
    sharpKeepRatio: number;
    /** global shift (gradient-map cells) tolerated before comparing */
    maxShift: number;
    /** lighting change (IR switch, floodlight): mean moved by more than this many levels ... */
    relearnLevels: number;
    /** ... or std changed by more than this factor (either way) ... */
    relearnStdRatio: number;
    /** ... with the structure kept (similarity at least this, sharpness kept): re-learn. Below it a sample is not explained */
    relearnNcc: number;
    /** samples not explained for this long (sensor off, view learnable): the reference is re-learnt (the anchor is kept) */
    changedRelearnMs: number;
    /** the reference sharpness is never below this share of the anchor */
    anchorClampRatio: number;
    /** the anchor is refreshed after this long of clean samples at least anchorKeepRatio of its sharpness */
    anchorRefreshMs: number;
    anchorKeepRatio: number;
    /** on this long: the view is accepted as the reference (re-aimed camera, fog); never a flat or partly covered view; 0 = never */
    acceptMs: number;
    /** after a profile switch (day/night, stream) samples are not judged for this long */
    settleMs: number;
    /** more than flapFlips profile switches within flapWindowMs (day/night flapping under a hand, glare) count as 'covered' */
    flapFlips: number;
    flapWindowMs: number;
    /** samples further apart than this (stream restart, gap): the median history and the pending timer start again */
    gapMs: number;
}

/** the tuned values, documented, live in DEFAULTS.tamper (src/defaults.ts) */
export const defaultTamperOptions = (): TamperOptions => ({ ...DEFAULTS.tamper });

export interface TamperFeatures { mean: number; std: number; edge: number; sharp: number; ncc: number }

export interface TamperState {
    on: boolean;
    reason?: TamperReason;
    /** the condition seen in the latest sample (undefined = clean or not judged) */
    condition?: TamperCondition;
    /** no reference for the current profile yet */
    learning: boolean;
    /** learning, but the view looks flat (covered, black, blank): nothing is learnt from it */
    flat: boolean;
    /** current profile key */
    profile: string;
    /** on: since when (the sample time that turned it on); off: undefined */
    sinceMs?: number;
}

interface Ref { mean: number; std: number; edge: number; sharp: number; grad: Float32Array }
interface Anchor { sharp: number; grad: Float32Array }
interface Profile { ref: Ref; anchor: Anchor; anchorRunStart?: number }

/** the stream part of a profile key ('<stream>|day') */
const streamOf = (key: string) => key.split('|')[0];

interface SnapProfile { ref: { mean: number; std: number; edge: number; sharp: number; grad: number[] }; anchor: { sharp: number; grad: number[] } }
/** What survives a plugin restart: the references and the sensor state (camera storage, JSON). `sinceWall` is the caller's. */
export interface TamperSnapshot { v: 2; profiles: Record<string, SnapProfile>; on: boolean; reason?: TamperReason; sinceWall?: number }

/** A transition of the sensor, for the log line and the child device. */
export interface TamperChange { on: boolean; reason?: TamperReason; detail: string }

const HIST = 3;
/** damage-safeguard grid on the gradient map */
const CX = 8, CY = 6;
/** at most this many profiles are kept (oldest learnt dropped first) */
const MAX_PROFILES = 8;
const REASONS: TamperReason[] = ['covered', 'moved', 'blurred'];

export class TamperDetector {
    readonly opts: TamperOptions;
    private readonly w: number;
    private readonly h: number;
    private readonly n: number;
    private readonly gw: number;
    private readonly gh: number;
    /** last HIST downscaled samples (ring) and the per-pixel median */
    private readonly hist: Uint8Array[];
    private histLen = 0;
    private histAt = 0;
    private readonly med: Uint8Array;
    private readonly grad: Float32Array;
    private readonly small: Float32Array;
    private readonly cellCur = new Float64Array(CX * CY);
    private readonly cellRef = new Float64Array(CX * CY);
    private profiles = new Map<string, Profile>();
    private key = 'default';
    private settleUntil = -Infinity;
    private lastSampleAt = -Infinity;
    private on = false;
    private reason: TamperReason | undefined;
    private sinceMs: number | undefined;
    private condition: TamperCondition | undefined;
    private flat = false;
    /** first sample of the current run of conditions (on: of clean samples), undefined = no run */
    private runStart: number | undefined;
    /** first sample of the current run of unexplained samples */
    private changedStart: number | undefined;
    private last: TamperFeatures | undefined;
    private changed = false;
    /** times of the recent profile switches (flapping) */
    private flips: number[] = [];

    constructor(opts: Partial<TamperOptions> = {}) {
        this.opts = { ...defaultTamperOptions(), ...opts };
        this.w = this.opts.width >> 1;
        this.h = this.opts.height >> 1;
        this.n = this.w * this.h;
        this.gw = this.w >> 1;
        this.gh = this.h >> 1;
        this.hist = Array.from({ length: HIST }, () => new Uint8Array(this.n));
        this.med = new Uint8Array(this.n);
        this.grad = new Float32Array(this.gw * this.gh);
        this.small = new Float32Array(this.gw * this.gh);
    }

    get state(): TamperState {
        return { on: this.on, reason: this.on ? this.reason : undefined, condition: this.condition, learning: !this.profiles.has(this.key), flat: this.flat,
            profile: this.key, sinceMs: this.on ? this.sinceMs : undefined };
    }
    /** features of the latest judged sample (diagnostics, tests) */
    get features(): TamperFeatures | undefined { return this.last; }
    /** the current profile's reference scalars and anchor sharpness (diagnostics, tests); undefined while learning */
    get reference(): { mean: number; std: number; sharp: number; anchorSharp: number } | undefined {
        const p = this.profiles.get(this.key);
        return p && { mean: p.ref.mean, std: p.ref.std, sharp: p.ref.sharp, anchorSharp: p.anchor.sharp };
    }
    /** something worth saving changed since the last takeDirty() */
    get dirty(): boolean { return this.changed; }
    takeDirty(): boolean { const d = this.changed; this.changed = false; return d; }

    /**
     * Switch the reference set: one per analysed stream and day/night. The sensor state is kept (a tampered camera stays on);
     * the new profile is learnt only while the sensor is off. Samples are not judged for settleMs after a switch.
     */
    setProfile(key: string, nowMs: number): void {
        if (key === this.key) return;
        this.key = key;
        this.touch(key);
        // the median history and the unexplained run start again; a pending tamper run is kept (it goes on if the new profile
        // also sees a tamper condition, so day/night flapping under a hand cannot reset it forever)
        this.histLen = 0;
        this.changedStart = undefined;
        this.settleUntil = nowMs + this.opts.settleMs;
        this.flips = this.flips.filter(t => nowMs - t < this.opts.flapWindowMs);
        this.flips.push(nowMs);
    }

    /** drops the profiles matching `drop` (the '|any' ones once the camera has a Night mode sensor); never the current one */
    dropProfiles(drop: (key: string) => boolean): void {
        for (const k of [...this.profiles.keys()])
            if (k !== this.key && drop(k)) { this.profiles.delete(k); this.changed = true; }
    }

    /** LRU: a profile switched to or learnt is the most recent */
    private touch(key: string): void {
        const p = this.profiles.get(key);
        if (p) { this.profiles.delete(key); this.profiles.set(key, p); }
    }

    /** stream restart: the next samples rebuild the median history; the references and the sensor state are kept */
    reset(): void { this.histLen = 0; this.runStart = undefined; this.changedStart = undefined; this.condition = undefined; }

    /** forget the current profile's reference; the sensor goes off at once */
    relearn(): TamperChange | undefined {
        if (this.profiles.delete(this.key)) this.changed = true;
        this.reset();
        return this.setOff('reference re-learned');
    }

    /** the sensor goes off now (nothing to check any more: no frames, detection off); the references are kept */
    clear(detail: string): TamperChange | undefined {
        this.reset();
        return this.setOff(detail);
    }

    snapshot(): TamperSnapshot | undefined {
        if (!this.profiles.size) return undefined;
        const arr = (g: Float32Array) => Array.from(g, x => Math.round(x * 10) / 10);
        const profiles: Record<string, SnapProfile> = {};
        for (const [k, p] of this.profiles)
            profiles[k] = { ref: { mean: p.ref.mean, std: p.ref.std, edge: p.ref.edge, sharp: p.ref.sharp, grad: arr(p.ref.grad) }, anchor: { sharp: p.anchor.sharp, grad: arr(p.anchor.grad) } };
        return { v: 2, profiles, on: this.on, reason: this.on ? this.reason : undefined };
    }

    /** restores a snapshot (profiles that do not fit this frame size are dropped); a restored 'on' clears after offMs of a clean view */
    restore(s: TamperSnapshot | undefined, nowMs: number): boolean {
        if (!s || s.v !== 2 || !s.profiles || typeof s.profiles !== 'object') return false;
        const len = this.gw * this.gh;
        const nums = (a: unknown[]) => a.every(x => typeof x === 'number' && Number.isFinite(x));
        const grad = (g: unknown) => Array.isArray(g) && g.length === len && nums(g) ? Float32Array.from(g as number[]) : undefined;
        for (const [k, p] of Object.entries(s.profiles).slice(-MAX_PROFILES)) {
            const r = p?.ref, a = p?.anchor;
            const rg = grad(r?.grad), ag = grad(a?.grad);
            if (!rg || !ag || !nums([r.mean, r.std, r.edge, r.sharp, a.sharp])) continue;
            this.profiles.set(k, { ref: { mean: r.mean, std: r.std, edge: r.edge, sharp: r.sharp, grad: rg }, anchor: { sharp: a.sharp, grad: ag } });
        }
        if (!this.profiles.size) return false;
        this.on = !!s.on && REASONS.includes(s.reason as TamperReason);
        this.reason = this.on ? s.reason : undefined;
        this.sinceMs = this.on ? nowMs : undefined;
        return true;
    }

    /**
     * One gray frame. Returns a change when the sensor turns on or off, else undefined. Frames closer than sampleMs to the
     * last sample are skipped (cheap: nothing is computed).
     */
    push(gray: Uint8Array, nowMs: number): TamperChange | undefined {
        const o = this.opts;
        if (nowMs - this.lastSampleAt < o.sampleMs) return undefined;
        if (gray.length < o.width * o.height) throw new Error(`tamper: frame has ${gray.length} bytes, expected ${o.width * o.height}`);
        if (nowMs - this.lastSampleAt > o.gapMs) this.reset();
        this.lastSampleAt = nowMs;
        this.addSample(gray);
        this.flips = this.flips.filter(t => nowMs - t < o.flapWindowMs);
        if (this.flips.length > o.flapFlips) {
            // day/night flapping (a hand over the lens switches the IR on and off): a cover candidate, nothing else is judged
            this.condition = 'covered';
            return this.track('covered', nowMs, () => `day/night switched ${this.flips.length} times in ${Math.round(o.flapWindowMs / 1000)} s`);
        }
        if (this.histLen < HIST || nowMs < this.settleUntil) return undefined;
        this.median();
        const f = this.measure();
        const p = this.profiles.get(this.key);
        if (!p) {
            // never learn a flat, black or blown-out view (a camera covered at start, after a stream or day/night switch)
            this.last = { ...f, ncc: 1 };
            this.flat = !this.learnable(f);
            if (this.flat) {
                // no reference for this profile: a flat view is its first learn ('learning: view looks flat'), never an alarm
                // and never learnt; the alarm is only for a deviation from a learnt reference
                this.condition = undefined;
                if (!this.on) this.runStart = undefined;
                return undefined;
            }
            this.condition = undefined;
            if (!this.on) { this.runStart = undefined; this.learnProfile(f); return undefined; }
            // tampered while a new profile shows up: only the accept timeout learns it (never a flat view, above)
            return this.acceptAfter(nowMs) ? this.accept(f, false) : undefined;
        }
        this.flat = false;
        f.ncc = this.ncc(this.grad, p.ref.grad);
        this.last = f;
        const damaged = this.damaged(p);
        const cond = this.judge(f, p, damaged);
        if (cond !== undefined) p.anchorRunStart = undefined; // the anchor follows only an unbroken run of clean samples
        if (cond === 'relearn') {
            this.condition = undefined;
            this.runStart = undefined;
            this.changedStart = undefined;
            this.learnRef(p, f);
            return undefined;
        }
        if (this.on && cond && this.acceptAfter(nowMs) && this.acceptable(f, cond, damaged))
            return this.accept(f, cond === 'moved');
        this.condition = cond;
        if (cond === 'changed') {
            this.runStart = undefined; // breaks a run of tamper conditions, and a clean run while on
            if (!this.on) {
                this.changedStart ??= nowMs;
                // never re-learn a damaged view (part of the lens sprayed or taped): it stays pending, without an alarm
                if (nowMs - this.changedStart >= o.changedRelearnMs && this.learnable(f) && !damaged) {
                    this.learnRef(p, f);
                    this.changedStart = undefined;
                    this.condition = undefined;
                }
            }
            return undefined;
        }
        this.changedStart = undefined;
        if (!cond && !this.on) { this.runStart = undefined; this.update(p, f, nowMs); return undefined; }
        return this.track(cond, nowMs, () => `mean ${f.mean.toFixed(0)}/${p.ref.mean.toFixed(0)}, std ${f.std.toFixed(1)}/${p.ref.std.toFixed(1)}, sharpness ${f.sharp.toFixed(2)}/${p.ref.sharp.toFixed(2)} (anchor ${p.anchor.sharp.toFixed(2)}), similarity ${f.ncc.toFixed(2)}`);
    }

    // ---- decision ----

    private judge(f: TamperFeatures, p: Profile, damaged: boolean): TamperCondition | 'relearn' | undefined {
        const o = this.opts, r = p.ref, a = p.anchor;
        const dMean = Math.abs(f.mean - r.mean);
        const refSharp = Math.max(r.sharp, o.anchorClampRatio * a.sharp);
        const flat = (f.std < o.coveredStdRatio * r.std && f.edge < o.coveredEdgeRatio * r.edge)
            || (f.std < o.coveredStdAbs && r.std >= 2 * o.coveredStdAbs)
            || ((f.mean < o.darkLevel || f.mean > o.brightLevel) && f.std < o.brightStdRatio * r.std);
        if (flat) return 'covered';
        // blurred against the reference and the anchor (a slow spray or defocus the EMA followed); a brightness change counts only
        // when the structure went with it (spray paint), never on its own
        if (f.sharp < o.blurRatio * Math.max(refSharp, a.sharp) && (dMean <= o.blurMaxLevelDelta || f.ncc < o.relearnNcc)) return 'blurred';
        const sharpOk = f.sharp >= o.sharpKeepRatio * refSharp;
        if (sharpOk && f.ncc < o.movedNcc) return 'moved'; // before the lighting step: turned AND brightened is moved
        // part of the view collapsed against the anchor (a smooth van close by, IR out of range, part of the lens sprayed):
        // never an alarm in this version, but nothing is learnt from it either
        if (damaged) return 'changed';
        const contrast = Math.max(f.std, 1) / Math.max(r.std, 1);
        const step = dMean > o.relearnLevels || contrast > o.relearnStdRatio || contrast < 1 / o.relearnStdRatio;
        // a global step with the structure kept: lighting, not tamper (only while off: a tampered view is never learnt)
        if (step) return sharpOk && f.ncc >= o.relearnNcc && !this.on ? 'relearn' : 'changed';
        if (!sharpOk || f.ncc < o.relearnNcc) return 'changed';
        return undefined;
    }

    /**
     * Learning safeguard: part of the view lost its structure against the anchor (the anchor's textured cells of an 8x6 grid whose
     * gradient energy collapsed, corrected by the global gain from the upper quartile of the cells).
     */
    private damaged(p: Profile): boolean {
        const o = this.opts, { gw, grad, cellCur, cellRef } = this, ag = p.anchor.grad;
        const cw = Math.floor(gw / CX), ch = Math.floor(this.gh / CY);
        cellCur.fill(0); cellRef.fill(0);
        for (let y = 0; y < CY * ch; y++) for (let x = 0; x < CX * cw; x++) {
            const c = Math.floor(y / ch) * CX + Math.floor(x / cw);
            cellCur[c] += grad[y * gw + x]; cellRef[c] += ag[y * gw + x];
        }
        let refMean = 0;
        for (let c = 0; c < CX * CY; c++) refMean += cellRef[c];
        refMean /= CX * CY;
        const ratios: number[] = [];
        for (let c = 0; c < CX * CY; c++)
            if (cellRef[c] >= 0.25 * refMean && cellRef[c] > 0) ratios.push(cellCur[c] / cellRef[c]);
        if (ratios.length < 8) return false;
        const gain = [...ratios].sort((a, b) => a - b)[Math.floor(ratios.length * 0.75)];
        let low = 0;
        for (const r of ratios) if (r < o.damageCellRatio * gain) low++;
        return low >= o.damageFraction * ratios.length;
    }

    private learnable(f: TamperFeatures): boolean {
        const o = this.opts;
        return f.std >= 2 * o.coveredStdAbs && f.mean >= o.darkLevel && f.mean <= o.brightLevel;
    }

    private acceptAfter(nowMs: number): boolean {
        return this.opts.acceptMs > 0 && nowMs - (this.sinceMs ?? nowMs) >= this.opts.acceptMs;
    }

    /** a long tamper is accepted only when the view is not a cover: fog, a re-aimed camera, condensation; never a flat or partly covered lens */
    private acceptable(f: TamperFeatures, cond: TamperCondition, damaged: boolean): boolean {
        return !damaged && this.learnable(f) && (cond !== 'covered' || f.ncc >= this.opts.relearnNcc);
    }

    /** accepts the current view; a camera that was moved also forgets the other profiles of this stream (the old day or night view) */
    private accept(f: TamperFeatures, moved: boolean): TamperChange | undefined {
        if (moved) {
            const st = streamOf(this.key);
            this.dropProfiles(k => streamOf(k) === st);
        }
        this.learnProfile(f);
        this.condition = undefined;
        return this.setOff(`view accepted after ${Math.round(this.opts.acceptMs / 60000)} min`);
    }

    /** persistence: a tamper condition for onMs turns the sensor on, a clean view (cond undefined) for offMs turns it off */
    private track(cond: TamperReason | undefined, nowMs: number, detail: () => string): TamperChange | undefined {
        const o = this.opts;
        if (!this.on) {
            if (!cond) { this.runStart = undefined; return undefined; }
            this.runStart ??= nowMs;
            if (nowMs - this.runStart < o.onMs) return undefined;
            this.on = true;
            this.reason = cond;
            this.sinceMs = nowMs;
            this.runStart = undefined;
            this.changed = true;
            return { on: true, reason: cond, detail: detail() };
        }
        if (cond) {
            this.runStart = undefined;
            if (this.reason !== cond) this.changed = true;
            this.reason = cond; // the latest reason (a covered lens uncovered to a turned view reads 'moved')
            return undefined;
        }
        this.runStart ??= nowMs;
        if (nowMs - this.runStart < o.offMs) return undefined;
        return this.setOff(detail());
    }

    private setOff(detail: string): TamperChange | undefined {
        this.runStart = undefined;
        if (!this.on) return undefined;
        const reason = this.reason;
        this.on = false;
        this.reason = undefined;
        this.sinceMs = undefined;
        this.changed = true;
        return { on: false, reason, detail };
    }

    // ---- references ----

    /** a new profile (or an accepted view): reference and anchor from this view */
    private learnProfile(f: TamperFeatures): void {
        this.profiles.delete(this.key);
        while (this.profiles.size >= MAX_PROFILES) this.profiles.delete(this.profiles.keys().next().value as string);
        this.profiles.set(this.key, { ref: this.refOf(f), anchor: { sharp: f.sharp, grad: Float32Array.from(this.grad) } });
        this.flat = false;
        this.changed = true;
    }

    /** lighting step or a long unexplained change: the reference only; the anchor stays */
    private learnRef(p: Profile, f: TamperFeatures): void {
        p.ref = this.refOf(f);
        p.ref.sharp = Math.max(p.ref.sharp, this.opts.anchorClampRatio * p.anchor.sharp);
        p.anchorRunStart = undefined;
        this.touch(this.key);
        this.changed = true;
    }

    private refOf(f: TamperFeatures): Ref {
        return { mean: f.mean, std: f.std, edge: f.edge, sharp: f.sharp, grad: Float32Array.from(this.grad) };
    }

    /**
     * EMA towards the current clean view (alpha = sample interval / time constant); the reference sharpness is clamped to the
     * anchor. The anchor follows only a long run of clean samples that stay sharp and similar to it.
     */
    private update(p: Profile, f: TamperFeatures, nowMs: number): void {
        const o = this.opts, r = p.ref, a = Math.min(1, o.sampleMs / Math.max(1, o.refTauMs));
        r.mean += a * (f.mean - r.mean);
        r.std += a * (f.std - r.std);
        r.edge += a * (f.edge - r.edge);
        r.sharp += a * (f.sharp - r.sharp);
        r.sharp = Math.max(r.sharp, o.anchorClampRatio * p.anchor.sharp);
        const g = this.grad, rg = r.grad;
        for (let i = 0; i < rg.length; i++) rg[i] += a * (g[i] - rg[i]);
        this.changed = true;
        if (f.sharp >= o.anchorKeepRatio * p.anchor.sharp && this.ncc(g, p.anchor.grad) >= o.relearnNcc) {
            p.anchorRunStart ??= nowMs;
            if (nowMs - p.anchorRunStart >= o.anchorRefreshMs) {
                p.anchor = { sharp: Math.max(p.anchor.sharp, r.sharp), grad: Float32Array.from(rg) }; // the anchor sharpness never declines
                p.anchorRunStart = nowMs;
            }
        }
        else
            p.anchorRunStart = undefined;
    }

    // ---- image measures ----

    /** 2x2 average into the history ring */
    private addSample(src: Uint8Array): void {
        const { w, h } = this, sw = this.opts.width;
        const dst = this.hist[this.histAt];
        for (let y = 0; y < h; y++) {
            const r0 = 2 * y * sw, r1 = r0 + sw;
            for (let x = 0; x < w; x++) {
                const c = 2 * x;
                dst[y * w + x] = (src[r0 + c] + src[r0 + c + 1] + src[r1 + c] + src[r1 + c + 1] + 2) >> 2;
            }
        }
        this.histAt = (this.histAt + 1) % HIST;
        if (this.histLen < HIST) this.histLen++;
    }

    private median(): void {
        const [a, b, c] = this.hist, m = this.med;
        for (let i = 0; i < this.n; i++) {
            const x = a[i], y = b[i], z = c[i];
            m[i] = x > y ? (y > z ? y : x > z ? z : x) : (x > z ? x : y > z ? z : y);
        }
    }

    /** mean, std, RMS 4-neighbour Laplacian, sharpness, and the 32x18 gradient map into this.grad */
    private measure(): TamperFeatures {
        const { w, h, n, med: m, gw, gh, grad, small } = this;
        let s = 0, s2 = 0;
        for (let i = 0; i < n; i++) { s += m[i]; s2 += m[i] * m[i]; }
        const mean = s / n, std = Math.sqrt(Math.max(0, s2 / n - mean * mean));
        let l2 = 0, ln = 0;
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const i = y * w + x;
                const l = 4 * m[i] - m[i - 1] - m[i + 1] - m[i - w] - m[i + w];
                l2 += l * l; ln++;
            }
        }
        const edge = Math.sqrt(l2 / Math.max(1, ln));
        // 2x2 average to gw x gh, then |dx| + |dy| (forward differences, clamped at the border)
        for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
            const i = 2 * y * w + 2 * x;
            small[y * gw + x] = (m[i] + m[i + 1] + m[i + w] + m[i + w + 1]) / 4;
        }
        for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
            const i = y * gw + x;
            const dx = x < gw - 1 ? small[i + 1] - small[i] : small[i] - small[i - 1];
            const dy = y < gh - 1 ? small[i + gw] - small[i] : small[i] - small[i - gw];
            grad[i] = Math.abs(dx) + Math.abs(dy);
        }
        return { mean, std, edge, sharp: edge / Math.max(1, std), ncc: 0 };
    }

    /**
     * Best normalised cross-correlation (gain and offset invariant) over global shifts of up to `maxShift` cells of the
     * gradient map (1 cell = 4 gray pixels, about 3 % of the width): a camera shaking in the wind is not 'moved'. A flat map
     * correlates 0.
     */
    private ncc(a: Float32Array, b: Float32Array): number {
        const { gw, gh } = this, s = this.opts.maxShift;
        let best = -1;
        for (let dy = -s; dy <= s; dy++) for (let dx = -s; dx <= s; dx++) {
            const x0 = Math.max(0, -dx), x1 = Math.min(gw, gw - dx), y0 = Math.max(0, -dy), y1 = Math.min(gh, gh - dy);
            let k = 0, ma = 0, mb = 0;
            for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { ma += a[y * gw + x]; mb += b[(y + dy) * gw + x + dx]; k++; }
            if (!k) continue;
            ma /= k; mb /= k;
            let ab = 0, aa = 0, bb = 0;
            for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
                const u = a[y * gw + x] - ma, v = b[(y + dy) * gw + x + dx] - mb;
                ab += u * v; aa += u * u; bb += v * v;
            }
            const c = aa > 1e-6 && bb > 1e-6 ? ab / Math.sqrt(aa * bb) : 0;
            if (c > best) best = c;
        }
        return Math.max(0, best);
    }
}
