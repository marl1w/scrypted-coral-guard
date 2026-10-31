/**
 * Per-camera status panel: a pure data model plus a renderer that returns one compact,
 * script-free HTML string for a Scrypted Setting of `type: 'html'`.
 * Independent of the rest of the plugin; the mixin feeds it through the event methods.
 */

export type StatusState = 'idle' | 'analysing' | 'person confirmed' | 'service unavailable (passing motion through)' | 'disabled';

export interface Box { x0: number; y0: number; x1: number; y1: number; score: number; /** below the person score: drawn dashed and dim */ weak?: boolean;
    /** detection group and class, for colour and label; absent = person */ group?: 'person' | 'vehicle' | 'animal'; cls?: string; /** vehicle that is parked: dimmed, labelled parked */ parked?: boolean; /** vehicle on a 'still' track: dimmed, labelled still */ still?: boolean; /** inside an exclusion zone: dimmed, labelled (masked), never counted */ masked?: boolean }

export interface FrameEvent {
    atMs: number;
    personCount: number;
    bestScore?: number;
    /** best score of a below-threshold candidate (shown as 'weak candidate'). */
    candidateScore?: number;
    /** per-group counts above the display thresholds (for a future gate rule and the caption). */
    vehicles?: number;
    parkedVehicles?: number;
    weakVehicles?: number;
    animals?: number;
    totalMs?: number;
    inferMs?: number;
    queueMs?: number;
    boxes?: Box[];
}

export interface EpisodeEvent {
    kind: 'camera-triggered' | 'unflagged';
    rawStartMs: number;
    rawEndMs?: number;
    postedAfterMs?: number;
    verdictAfterMs?: number;
    confirmedAfterMs?: number;
    confirmedByPreroll?: boolean;
    releasedAtMs?: number;
    heldBeyondRawS: number;
    /** groups that confirmed the episode (person, vehicle, animal, unverified). */
    groups?: string[];
    vetoed: boolean;
    aborted: boolean;
    note?: string;
    /** still release (0.3.19): s from the start to the first release, re-raises, occupancy and motion time of a confirmed episode */
    stillReleaseAfterS?: number;
    reraises?: number;
    occupiedS?: number;
    motionS?: number;
}

export interface ServiceHealth { ok: boolean; coralMs?: number; text: string }
export interface StreamInfo { name?: string; connected: boolean; decode: 'gpu' | 'cpu' | 'unknown'; warm: boolean; /** GPU decode was requested but did not initialise */ gpuFailed?: boolean; /** GPU-to-CPU fallbacks of this camera since the plugin started */ gpuFallbacks?: number }
export interface LastFrame {
    jpegBase64?: string;
    /** raw JPEG: base64 is computed lazily, only when the panel renders. */
    jpeg?: Buffer;
    boxes: Box[];
}

export interface Counters { episodes: number; confirmed: number; vetoed: number; unflagged: number; skippedFrames: number; failures: number; /** motion released for stillness / raised again (0.3.19) */ stillReleases: number; reraises: number }

export interface Pct { p50?: number; p95?: number }
export interface Percentiles { n: number; posted: Pct; verdict: Pct; confirmed: Pct }

/** the 'Last failure' line disappears after this long. */
export const FAILURE_SHOW_MS = 120000;
export const MAX_HISTORY = 10;
export const MAX_LATENCY_SAMPLES = 50;
/** data-URI size limit of the panel thumbnail (60 KB worked live; busy 640 px scenes reach ~100 KB). */
export const MAX_DATA_URI_BYTES = 160 * 1024;

const STATES: StatusState[] = ['idle', 'analysing', 'person confirmed', 'service unavailable (passing motion through)', 'disabled'];

export function esc(s: unknown): string {
    return String(s ?? '').replace(/[&<>"'`]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '`': '&#96;' }[c] as string));
}

export function pctOf(values: number[], p: number): number | undefined {
    if (!values.length)
        return undefined;
    const v = values.slice().sort((a, b) => a - b);
    return v[Math.min(v.length - 1, Math.max(0, Math.ceil(p * v.length) - 1))];
}

/** ms with one decimal below 10, none above; undefined -> n/a. */
export function fmtMs(v?: number): string {
    if (v === undefined || !Number.isFinite(v))
        return 'n/a';
    if (v >= 1000)
        return `${(Math.round(v / 100) / 10).toFixed(1)} s`;
    return v < 10 ? `${(Math.round(v * 10) / 10).toFixed(1)} ms` : `${Math.round(v)} ms`;
}

export function relTime(thenMs: number | undefined, nowMs: number): string {
    if (thenMs === undefined || !Number.isFinite(thenMs))
        return 'never';
    const s = Math.max(0, Math.round((nowMs - thenMs) / 1000));
    if (s < 5)
        return 'just now';
    if (s < 60)
        return `${s} s ago`;
    const m = Math.floor(s / 60);
    if (m < 60)
        return `${m} min ago`;
    const h = Math.floor(m / 60);
    if (h < 48)
        return `${h} h ago`;
    return `${Math.floor(h / 24)} d ago`;
}

const dayKey = (ms: number) => { const d = new Date(ms); return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`; };
const clock = (ms: number) => { const d = new Date(ms); const p = (n: number) => String(n).padStart(2, '0'); return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`; };
const zero = (): Counters => ({ episodes: 0, confirmed: 0, vetoed: 0, unflagged: 0, skippedFrames: 0, failures: 0, stillReleases: 0, reraises: 0 });
const clamp01 = (n: number) => Math.min(1, Math.max(0, Number.isFinite(n) ? n : 0));

function outcomeOf(e: EpisodeEvent): string {
    if (e.aborted) return 'aborted';
    if (e.kind === 'unflagged') return 'person without camera motion';
    return e.vetoed ? 'vetoed' : 'confirmed';
}

export class CameraStatus {
    private state: StatusState = 'idle';
    private mode = 'unknown';
    private enabled = true;
    private frame?: FrameEvent;
    private history: EpisodeEvent[] = [];
    private samples: EpisodeEvent[] = [];
    private failure?: { msg: string; atMs: number };
    private health?: ServiceHealth;
    private stream?: StreamInfo;
    private last?: LastFrame;
    private counters = zero();
    private byGroup: Record<string, number> = {};
    private day: string;

    constructor(private now: () => number = Date.now) {
        this.day = dayKey(now());
    }

    private roll(ms: number) {
        const k = dayKey(ms);
        if (k !== this.day) {
            this.day = k;
            this.counters = zero();
            this.byGroup = {};
        }
    }

    private detecting = '';
    setMode(mode: string) { this.mode = mode; }
    /** enabled detection groups, shown as the 'Detecting' row. */
    private enabledGroups: string[] = ['person', 'vehicle', 'animal'];
    private nowInfo?: NowInfo;
    /** what the motion sensor outputs and why (top card) */
    setNow(n: NowInfo) { this.nowInfo = n; }
    private tracked: TrackedItem[] = [];
    private trackedAt = 0;
    /** objects the service is holding, from the last analysed reply (they drive motion) */
    setTracked(items: TrackedItem[], atMs: number) { this.tracked = items; this.trackedAt = atMs; }

    // ---- Activity (last 10 min): one entry per analysed frame, in memory, capped ----
    private activity: ActivityEntry[] = [];
    recordActivity(e: ActivityEntry) {
        this.activity.push(e);
        const cut = e.t - ACTIVITY_MS;
        while (this.activity.length > ACTIVITY_MAX || (this.activity.length && this.activity[0].t < cut))
            this.activity.shift();
    }

    /** 60 buckets of 10 s: frames per group (a frame counts once per group it contains), any analysis, published / would lanes. */
    activityBuckets(nowMs: number): { analysed: number; person: number; vehicle: number; animal: number; published: boolean; would: boolean }[] {
        const b = Array.from({ length: ACTIVITY_BUCKETS }, () => ({ analysed: 0, person: 0, vehicle: 0, animal: 0, published: false, would: false }));
        for (const e of this.activity) {
            const i = ACTIVITY_BUCKETS - 1 - Math.floor((nowMs - e.t) / (ACTIVITY_MS / ACTIVITY_BUCKETS));
            if (i < 0 || i >= ACTIVITY_BUCKETS) continue;
            const x = b[i];
            x.analysed++;
            for (const g of e.groups) if (g === 'person' || g === 'vehicle' || g === 'animal') x[g]++;
            x.published ||= e.published;
            x.would ||= e.would;
        }
        return b;
    }

    private activityChart(nowMs: number): string {
        const b = this.activityBuckets(nowMs);
        const shadow = this.nowInfo?.mode === 'Shadow';
        const W = 600, x0 = 56, x1 = 594, bw = (x1 - x0) / ACTIVITY_BUCKETS, top = 8, base = 70;
        const max = Math.max(1, ...b.map(x => x.person + x.vehicle + x.animal));
        const colors: Record<string, string> = { person: C.red, vehicle: C.blue, animal: C.green };
        const groups = ['person', 'vehicle', 'animal'].filter(g => this.enabledGroups.includes(g));
        let bars = '';
        b.forEach((x, i) => {
            const bx = (x0 + i * bw + 0.5).toFixed(1), w = (bw - 1).toFixed(1);
            if (!x.analysed) { bars += `<rect x="${bx}" y="${base - 3}" width="${w}" height="3" fill="currentColor" fill-opacity=".18"><title>no analysis</title></rect>`; return; }
            let y = base;
            for (const g of groups) {
                const n = (x as any)[g] as number;
                if (!n) continue;
                const h = ((base - top) * n) / max;
                y -= h;
                bars += `<rect x="${bx}" y="${y.toFixed(1)}" width="${w}" height="${h.toFixed(1)}" fill="${colors[g]}"><title>${esc(`${g}: ${n} frame${n === 1 ? '' : 's'}`)}</title></rect>`;
            }
        });
        const lane = (y: number, key: 'published' | 'would', label: string) => `<text x="${x0 - 4}" y="${y + 6}" font-size="10" fill="currentColor" text-anchor="end" opacity=".7">${esc(label)}</text>`
            + `<rect x="${x0}" y="${y}" width="${x1 - x0}" height="7" fill="currentColor" fill-opacity=".08"/>`
            + b.map((x, i) => x[key] ? `<rect x="${(x0 + i * bw).toFixed(1)}" y="${y}" width="${bw.toFixed(1)}" height="7" fill="${C.green}"/>` : '').join('');
        const H = shadow ? 118 : 106;
        const t = (x: number, y: number, s: string, a = 'start') => `<text x="${x}" y="${y}" font-size="10" fill="currentColor" text-anchor="${a}" opacity=".7">${esc(s)}</text>`;
        const legend = groups.map(g => `<span style="color:${colors[g]}">&#9679;</span> ${esc(g)}`).join(' &nbsp;');
        return `<div style="margin-top:10px"><div style="font-weight:600">Activity</div><div style="opacity:.7;font-size:85%">${legend} &nbsp;<span style="opacity:.5">&#9679;</span> gap</div>`
            + `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;max-width:${W}px;color:currentColor" role="img" aria-label="Activity, last 10 minutes">`
            + t(x0 - 4, top + 8, `${max} frame${max === 1 ? '' : 's'}`, 'end') + t(x0 - 4, base, '0', 'end')
            + `<line x1="${x0}" y1="${base}" x2="${x1}" y2="${base}" stroke="currentColor" stroke-opacity=".3"/>`
            + bars
            + lane(base + 6, 'published', 'Published') + (shadow ? lane(base + 18, 'would', 'Would') : '')
            + t(x0, H - 2, '10 min ago') + t(x1, H - 2, 'now', 'end')
            + `</svg></div>`;
    }

    private nowCard(nowMs: number): string {
        const n = this.nowInfo;
        if (!n) return '';
        const reason = (r: NowReason) => 'text' in r ? r.text
            : `${cap(r.groups.join('+'))} tracked${r.sinceMs ? ` \u00b7 since ${clock(r.sinceMs)}` : ''}${r.lastSeenMs ? ` \u00b7 ${Math.max(0, Math.round((nowMs - r.lastSeenMs) / 1000))} s` : ''}`;
        const big = (on: boolean) => `<div style="font-size:150%;font-weight:700;color:${on ? C.green : C.gray}">Motion ${on ? 'ON' : 'OFF'}</div>`;
        const line = (t: string, color?: string) => `<div style="margin-top:2px${color ? `;color:${color}` : ''}">${esc(t)}</div>`;
        const body = n.mode === 'Shadow'
            ? big(n.published) + line(`Would publish ${n.coral ? 'ON' : 'OFF'} \u00b7 ${reason(n.reason)}`, n.coral ? C.green : undefined)
            : big(n.published) + line(reason(n.reason));
        const occ = n.occupancy ? line(sensorLine(n.occupancy, n.mode === 'Shadow' ? n.coral : n.published, n.mode === 'Shadow'), n.occupancy.on ? C.green : undefined) : '';
        return `<div style="border:1px solid rgba(127,127,127,.35);border-radius:8px;padding:8px 12px;margin-top:8px"><div style="opacity:.7;font-size:85%">Now</div>${body}${occ}</div>`;
    }

    private nowLists(nowMs: number): string {
        const det = (this.last?.boxes || []).filter(b => !b.masked && !b.weak)
            .map(b => boxLabel(b));
        const trk = this.tracked.map(t => {
            const seen = t.lastSeenMs ?? (t.detected ? this.trackedAt : undefined);
            const left = t.holdLeftMs === undefined ? undefined : Math.max(0, Math.round((t.holdLeftMs - (nowMs - this.trackedAt)) / 1000));
            return t.state === 'parked' ? `${t.cls} \u00b7 parked${t.sinceMs ? ` \u00b7 since ${clock(t.sinceMs)}` : ''}`
                : `${t.cls} \u00b7 ${t.state}${left !== undefined ? ` \u00b7 ${left} s left` : seen ? ` \u00b7 ${Math.max(0, Math.round((nowMs - seen) / 1000))} s` : ''}`;
        });
        const col = (title: string, items: string[], empty: string) => `<div style="flex:1 1 160px;min-width:0"><div style="font-weight:600">${esc(title)}</div>`
            + (items.length ? items.map(i => `<div>${esc(i)}</div>`).join('') : `<div style="opacity:.7">${esc(empty)}</div>`) + `</div>`;
        return `<div style="display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:10px">`
            + col('Detected', det, 'nothing') + col('Tracked', trk, 'nothing') + `</div>`;
    }

    /** dashed boxes for tracked objects that are not detected in the shown frame */
    private trackedBoxes(nowMs: number): string {
        return this.tracked.filter(t => !t.detected && Array.isArray(t.box) && t.box.length === 4).slice(0, 10).map(t => {
            const [a, b, c, d] = t.box!;
            const x0 = clamp01(Math.min(a, c)), x1 = clamp01(Math.max(a, c)), y0 = clamp01(Math.min(b, d)), y1 = clamp01(Math.max(b, d));
            const pc = (v: number) => (Math.round(v * 1000) / 10).toString();
            const seen = t.lastSeenMs ?? this.trackedAt;
            const label = `${t.cls} tracked (${Math.max(0, Math.round((nowMs - seen) / 1000))} s)`;
            return `<div style="position:absolute;left:${pc(x0)}%;top:${pc(y0)}%;width:${pc(x1 - x0)}%;height:${pc(y1 - y0)}%;border:2px dashed currentColor;opacity:.8;box-sizing:border-box"><span style="position:absolute;left:-2px;top:-1.5em;font-size:11px;white-space:nowrap;background:rgba(127,127,127,.6);color:#fff;padding:0 4px;border-radius:2px">${esc(label)}</span></div>`;
        }).join('');
    }
    private sensorless = false;
    setSensorless(v: boolean) { this.sensorless = v; }
    private zones: { name: string; points: [number, number][] }[] = [];
    setZones(z: { name: string; points: [number, number][] }[]) { this.zones = z; }
    /** exclusion polygons as an SVG overlay (0..100 viewBox, stretched over the image), drawn under the boxes */
    zonesSvg(): string {
        const polys = this.zones.filter(z => z.points.length >= 3);
        if (!polys.length) return '';
        const n = (v: number) => (Math.round(Math.min(1, Math.max(0, v)) * 1000) / 10).toString();
        return `<svg viewBox="0 0 100 100" preserveAspectRatio="none" style="position:absolute;left:0;top:0;width:100%;height:100%;color:currentColor;pointer-events:none" aria-label="exclusion zones">`
            + polys.map(z => `<polygon points="${z.points.map(p => `${n(p[0])},${n(p[1])}`).join(' ')}" fill="currentColor" fill-opacity=".25" stroke="currentColor" stroke-width="1" stroke-dasharray="2 2" vector-effect="non-scaling-stroke"><title>${esc(z.name)}</title></polygon>`).join('')
            + `</svg>`;
    }
    private tamper?: TamperInfo;
    setTamper(t: TamperInfo) { this.tamper = t; }
    private zoomOut = 0;
    setZoomOut(v: number) { this.zoomOut = v; }
    setDetect(groups: string[]) { this.enabledGroups = groups; this.detecting = groups.length === 3 ? groups.join(', ') : groups.length === 1 ? `${groups[0]} only` : groups.join(', '); }
    setEnabled(enabled: boolean) { this.enabled = enabled; if (!enabled) this.state = 'disabled'; else if (this.state === 'disabled') this.state = 'idle'; }
    setState(state: StatusState) { this.state = state; }
    setStream(s: StreamInfo) { this.stream = s; }
    private tooLarge = false;
    /** the stored last frame is the one the next onFrameAnalysed describes (not an older one kept after an oversize frame) */
    private lastFresh = false;
    setLastFrame(f: LastFrame) { this.last = f; this.tooLarge = false; this.lastFresh = true; }
    /** the newest frame was too large for the panel: the previous thumbnail (if any) stays. */
    setThumbTooLarge() { this.tooLarge = true; this.lastFresh = false; }
    /** 'Last detection': the newest analysed frame with a detection of an enabled group at its threshold; kept until replaced. */
    private detection?: { last: LastFrame; frame: FrameEvent };
    private okAt = 0;
    onServiceHealth(h: ServiceHealth) { this.health = h; if (h.ok) this.okAt = this.now(); }

    onFrameAnalysed(f: FrameEvent) {
        this.frame = f;
        const on = (g: string) => this.enabledGroups.includes(g);
        this.recordActivity({
            t: f.atMs,
            groups: [on('person') && f.personCount > 0 ? 'person' : '', on('vehicle') && !!f.vehicles ? 'vehicle' : '', on('animal') && !!f.animals ? 'animal' : ''].filter(Boolean),
            published: !!this.nowInfo?.published, would: !!this.nowInfo?.coral,
        });
        this.okAt = this.now();
        if (f.boxes && this.last) this.last = { ...this.last, boxes: f.boxes };
        const hit = (on('person') && f.personCount > 0) || (on('vehicle') && !!f.vehicles) || (on('animal') && !!f.animals);
        if (hit && this.last && this.lastFresh)
            this.detection = { last: this.last, frame: f }; // same JPEG reference: one extra JPEG per camera at most
        this.lastFresh = false;
    }

    onFrameSkipped() { this.roll(this.now()); this.counters.skippedFrames++; }

    onFailure(msg: string) {
        const at = this.now();
        this.roll(at);
        this.counters.failures++;
        this.failure = { msg, atMs: at };
    }

    /** a still release or re-raise of the motion (counted per day). */
    onStill(event: 'release' | 'reraise') {
        this.roll(this.now());
        if (event === 'release') this.counters.stillReleases++;
        else this.counters.reraises++;
    }

    onEpisode(e: EpisodeEvent) {
        this.roll(this.now());
        const c = this.counters;
        c.episodes++;
        if (e.kind === 'unflagged') c.unflagged++;
        else if (e.vetoed) c.vetoed++;
        else if (!e.aborted) c.confirmed++;
        if (e.groups && !e.vetoed && !e.aborted) for (const g of e.groups) this.byGroup[g] = (this.byGroup[g] || 0) + 1;
        this.history.push(e);
        if (this.history.length > MAX_HISTORY) this.history.shift();
        if (!e.aborted) {
            this.samples.push(e);
            if (this.samples.length > MAX_LATENCY_SAMPLES) this.samples.shift();
        }
    }

    /** Counters for the current local day as of nowMs. */
    today(nowMs: number = this.now()): Counters {
        return dayKey(nowMs) === this.day ? { ...this.counters } : zero();
    }

    percentiles(): Percentiles {
        const pick = (f: (e: EpisodeEvent) => number | undefined): Pct => {
            const v = this.samples.map(f).filter((x): x is number => typeof x === 'number');
            return { p50: pctOf(v, 0.5), p95: pctOf(v, 0.95) };
        };
        return { n: this.samples.length, posted: pick(e => e.postedAfterMs), verdict: pick(e => e.verdictAfterMs), confirmed: pick(e => e.confirmedAfterMs) };
    }

    private frameLine(nowMs: number): string {
        const f = this.frame;
        if (!f) return 'No analysis yet';
        const person = f.personCount > 0
            ? `person yes${f.bestScore !== undefined ? ` (${f.bestScore.toFixed(2)})` : ''}`
            : 'person no';
        return `${relTime(f.atMs, nowMs)}: ${person}${f.totalMs !== undefined ? `, ${fmtMs(f.totalMs)}` : ''}`;
    }

    private episodeLine(e: EpisodeEvent, nowMs: number): string {
        const bits = [`${clock(e.rawStartMs)} (${relTime(e.rawStartMs, nowMs)})`, outcomeOf(e)];
        if (e.confirmedAfterMs !== undefined) bits.push(`confirmed after ${fmtMs(e.confirmedAfterMs)}${e.confirmedByPreroll ? ' via pre-roll' : ''}`);
        if (e.heldBeyondRawS > 0) bits.push(`held ${Math.round(e.heldBeyondRawS)} s beyond raw`);
        if (e.stillReleaseAfterS !== undefined) bits.push(`motion released (still) +${e.stillReleaseAfterS} s${e.reraises ? `, re-raised ${e.reraises}x` : ''}`);
        if (e.occupiedS !== undefined && e.motionS !== undefined && e.motionS !== e.occupiedS) bits.push(`occupied ${e.occupiedS} s, motion ${e.motionS} s`);
        if (e.note) bits.push(e.note);
        return bits.join(', ');
    }

    summaryLine(nowMs: number = this.now()): string {
        const t = this.today(nowMs);
        const p = this.percentiles();
        const lat = p.n ? `, confirm p50/p95 ${p.confirmed.p50 === undefined ? 'n/a' : fmtMs(p.confirmed.p50)}/${p.confirmed.p95 === undefined ? 'n/a' : fmtMs(p.confirmed.p95)}` : '';
        const still = t.stillReleases || t.reraises ? `, ${t.stillReleases} still releases, ${t.reraises} re-raises` : '';
        return `${this.state}; today ${t.episodes} episodes (${t.confirmed} confirmed, ${t.vetoed} vetoed)${still}${lat}`;
    }

    toJSON() {
        return {
            state: this.state, mode: this.mode, enabled: this.enabled, frame: this.frame, history: this.history.slice(),
            failure: this.failure, health: this.health, stream: this.stream, counters: { ...this.counters }, day: this.day,
            percentiles: this.percentiles(), hasThumbnail: !!this.last?.jpegBase64,
        };
    }

    // ---- rendering: inline styles only, semantic mid-tone colours readable on light and dark themes ----

    private liveFrame?: LastFrame & { atMs: number };
    /** Keep a reference (no copy, no base64) to the newest frame delivered by the frame source. */
    setLiveFrame(jpeg: Buffer, atMs: number) {
        if (this.liveFrame) { this.liveFrame.jpeg = jpeg; this.liveFrame.atMs = atMs; this.liveFrame.jpegBase64 = undefined; }
        else this.liveFrame = { jpeg, atMs, boxes: [] };
    }

    /** The panel shows the live frame when nothing was analysed yet, or the analysis is older than LIVE_AFTER_MS. */
    showsLive(): boolean {
        const live = this.liveFrame;
        if (!live) return false;
        return !this.last || this.frame === undefined; // an analysed frame (with its boxes) always wins, however old
    }

    private thumb(nowMs: number): string {
        const live = this.showsLive();
        const l = live ? this.liveFrame : this.last;
        if (l && !l.jpegBase64 && l.jpeg && l.jpeg.length * 4 / 3 + 30 <= MAX_DATA_URI_BYTES)
            l.jpegBase64 = l.jpeg.toString('base64'); // lazy: once per panel render, not once per frame
        const b64 = l?.jpegBase64;
        const placeholder = (!live && this.tooLarge) || (l && l.jpeg && !l.jpegBase64) || (l?.jpegBase64 && l.jpegBase64.length + 23 > MAX_DATA_URI_BYTES)
            ? `<div style="opacity:.7;margin-top:10px">Snapshot too large</div>` : '';
        if (!l || !b64 || !/^[A-Za-z0-9+/=]+$/.test(b64)) return placeholder;
        const uri = `data:image/jpeg;base64,${b64}`;
        if (uri.length > MAX_DATA_URI_BYTES) return placeholder;
        const boxes = live ? '' : this.boxesHtml(l.boxes) + this.trackedBoxes(nowMs);
        const f = this.frame;
        const caption = live ? `${ago(nowMs - this.liveFrame!.atMs)} \u00b7 live, not analysed` : !f ? '' : `${ago(nowMs - f.atMs)} \u00b7 ${topItem(l.boxes)}`;
        return `<div style="font-weight:600;margin-top:10px">Last analysis</div>` + this.snap(uri, boxes, caption, live ? 'Live frame' : 'Last analysed frame');
    }

    private snap(uri: string, boxes: string, caption: string, alt: string): string {
        return `<div style="position:relative;width:100%;max-width:640px;margin-top:4px"><img src="${uri}" alt="${alt}" style="display:block;width:100%;height:auto;border-radius:4px">${this.zonesSvg()}${boxes}</div>${caption ? `<div style="opacity:.7;font-size:90%;margin-top:4px">${esc(caption)}</div>` : ''}`;
    }

    /** 'Last detection' block: the newest frame with an enabled detection, or an empty state. */
    private detectionThumb(nowMs: number): string {
        const head = `<div style="font-weight:600;margin-top:12px">Last detection</div>`;
        const d = this.detection;
        if (!d) return head + `<div style="opacity:.7;font-size:90%;margin-top:4px">none yet</div>`;
        const l = d.last;
        if (!l.jpegBase64 && l.jpeg && l.jpeg.length * 4 / 3 + 30 <= MAX_DATA_URI_BYTES)
            l.jpegBase64 = l.jpeg.toString('base64');
        const b64 = l.jpegBase64;
        const caption = `${ago(nowMs - d.frame.atMs)} \u00b7 ${topItem(d.frame.boxes ?? l.boxes)}`;
        if (!b64 || !/^[A-Za-z0-9+/=]+$/.test(b64) || b64.length + 23 > MAX_DATA_URI_BYTES)
            return head + `<div style="opacity:.7;font-size:90%;margin-top:4px">${esc(caption)}</div>`;
        return head + this.snap(`data:image/jpeg;base64,${b64}`, this.boxesHtml(d.frame.boxes ?? l.boxes), caption, 'Last detection');
    }

    private boxesHtml(list: Box[] | undefined): string {
        return (list || []).slice(0, 20).map(b => {
            const x0 = clamp01(Math.min(b.x0, b.x1)), x1 = clamp01(Math.max(b.x0, b.x1));
            const y0 = clamp01(Math.min(b.y0, b.y1)), y1 = clamp01(Math.max(b.y0, b.y1));
            const pc = (n: number) => (Math.round(n * 1000) / 10).toString();
            const g = b.group || 'person';
            const col = g === 'vehicle' ? C.blue : g === 'animal' ? C.green : b.weak ? C.amber : C.red;
            const dim = b.weak || b.parked || b.still || b.masked;
            const name = g === 'person' ? (b.weak ? 'possible' : 'person') : `${b.weak ? 'possible ' : ''}${b.cls || g}`;
            const text = `${name}${b.parked ? ' parked' : b.still ? ' still' : ''} ${Number.isFinite(b.score) ? b.score.toFixed(2) : ''}${b.masked ? ' (masked)' : ''}`.trim();
            const label = `<span style="position:absolute;left:-2px;top:-1.5em;background:${col};color:#fff;font-size:11px;line-height:1.4;padding:0 4px;border-radius:2px;white-space:nowrap">${esc(text)}</span>`;
            return `<div style="position:absolute;left:${pc(x0)}%;top:${pc(y0)}%;width:${pc(x1 - x0)}%;height:${pc(y1 - y0)}%;border:2px ${dim ? 'dashed' : 'solid'} ${col};opacity:${dim ? '.65' : '1'};box-sizing:border-box">${label}</div>`;
        }).join('');
    }

    private detectionSummary(f: FrameEvent): string {
        const parts: string[] = [];
        if (f.personCount > 0)
            parts.push(`${f.personCount} ${f.personCount === 1 ? 'person' : 'persons'}${f.bestScore !== undefined ? ` (best ${f.bestScore.toFixed(2)})` : ''}`);
        if (f.vehicles)
            parts.push(`${f.vehicles} ${f.vehicles === 1 ? 'vehicle' : 'vehicles'}${f.parkedVehicles ? ` (${f.parkedVehicles} parked)` : ''}`);
        if (f.weakVehicles)
            parts.push(`${f.weakVehicles} possible ${f.weakVehicles === 1 ? 'vehicle' : 'vehicles'}`);
        if (f.animals)
            parts.push(`${f.animals} ${f.animals === 1 ? 'animal' : 'animals'}`);
        if (parts.length)
            return parts.join(' \u00b7 ');
        return f.candidateScore !== undefined ? `no person (weak candidate ${f.candidateScore.toFixed(2)})` : (f.vehicles === undefined && f.animals === undefined ? 'no person' : 'nothing detected');
    }

    private chip(text: string, color: string, dot = true): string {
        return `<span style="display:inline-block;padding:1px 9px;border-radius:11px;border:1px solid ${color};color:${color};white-space:nowrap">${dot ? '&#9679; ' : ''}${esc(text)}</span>`;
    }

    /** One row of the plugin Overview table (escaped html cells). */
    overviewCells(nowMs: number = this.now()): { mode: string; state: string; today: string; p50: string; decode: string; lastDetection: string } {
        const t = this.today(nowMs);
        const p = this.percentiles().confirmed.p50;
        return {
            mode: esc(this.mode ?? ""),
            state: this.stateChip(),
            today: esc(`${t.episodes} / ${t.confirmed} / ${t.vetoed} / ${t.unflagged} / ${t.stillReleases} / ${t.reraises}`),
            p50: p === undefined ? '\u2014' : esc(fmtMs(p)),
            decode: !this.stream?.connected ? '\u2014' : this.stream.decode === 'gpu' ? 'GPU' : this.stream.decode === 'cpu' ? (this.stream.gpuFailed ? 'CPU (GPU failed)' : 'CPU') : '?',
            lastDetection: this.detection ? esc(relTime(this.detection.frame.atMs, nowMs)) : '\u2014',
        };
    }

    private stateChip(): string {
        if (!this.enabled || this.state === 'disabled') return this.chip('Disabled', C.gray);
        switch (this.state) {
            case 'person confirmed': return this.chip('Person confirmed', C.green);
            case 'analysing': return this.chip('Analysing', C.blue);
            case 'service unavailable (passing motion through)': return this.chip('Service unavailable', C.red);
            default: return this.stream?.connected ? this.chip('Watching', C.green) : this.chip('Idle', C.gray);
        }
    }

    render(nowMs: number = this.now()): string {
        const dash = '&mdash;';
        const t = this.today(nowMs);
        const p = this.percentiles();
        const two = (x: Pct) => x.p50 === undefined ? undefined : `${fmtMs(x.p50)} / ${fmtMs(x.p95)}`;
        const grid = (k: string, v: string) => `<div style="opacity:.7">${esc(k)}</div><div>${v}</div>`;
        const grp = Object.entries(this.byGroup).filter(([, n]) => n > 0).map(([g, n]) => `${g} ${n}`).join(', ');
        const today = [`${t.episodes} episodes`, `${t.confirmed} confirmed${grp ? ` (${esc(grp)})` : ''}`, `${t.vetoed} vetoed`, `${t.unflagged} unflagged`]
            .concat(t.stillReleases || t.reraises ? [`${t.stillReleases} still releases`, `${t.reraises} re-raises`] : [], t.skippedFrames ? [`${t.skippedFrames} skipped`] : [], t.failures ? [`${t.failures} failures`] : []).join(' &middot; ');
        const lat = two(p.confirmed) ? esc(two(p.confirmed)!) : dash;
        const video = !this.stream ? dash
            : this.stream.connected ? esc(`${this.stream.decode === 'gpu' ? 'GPU' : this.stream.decode === 'cpu' ? 'CPU' : '?'}${this.stream.gpuFailed ? ` (GPU failed${this.stream.gpuFallbacks ? ` \u00d7${this.stream.gpuFallbacks}` : ''})` : ''}${this.stream.name ? ` \u00b7 ${this.stream.name}` : ''}`) : `<span style="color:${C.red}">not connected</span>`;
        const service = !this.health ? dash
            : `${this.health.ok ? `<span style="color:${C.green}">OK</span>` : `<span style="color:${C.red}">down</span>`}${this.health.coralMs !== undefined ? esc(` \u00b7 Coral ${fmtMs(this.health.coralMs)}`) : ''}${this.health.text ? ` (${esc(this.health.text)})` : ''}`;
        const result = (e: EpisodeEvent) => e.aborted ? this.chip('Aborted', C.red, false)
            : e.kind === 'unflagged' ? this.chip('Unflagged', C.blue, false)
                : e.vetoed ? this.chip('Vetoed', C.amber, false) : this.chip(e.groups?.length ? `Confirmed \u00b7 ${e.groups.join('+')}` : 'Confirmed', C.green, false);
        const cellStyle = 'style="padding:2px 8px 2px 0;text-align:left;vertical-align:middle"';
        const rows = this.history.slice(-8).reverse().map(e =>
            `<tr><td ${cellStyle}>${esc(clock(e.rawStartMs))}</td><td ${cellStyle}>${result(e)}</td><td ${cellStyle}>${e.heldBeyondRawS > 0 ? esc(`${Math.round(e.heldBeyondRawS)} s`) : dash}</td><td ${cellStyle}>${e.occupiedS !== undefined ? esc(occMotion(e)) : dash}</td><td ${cellStyle}>${e.confirmedAfterMs !== undefined ? esc(fmtMs(e.confirmedAfterMs)) : e.verdictAfterMs !== undefined && e.vetoed ? esc(fmtMs(e.verdictAfterMs)) : dash}</td></tr>`).join('');
        const episodes = rows
            ? `<table style="border-collapse:collapse;margin-top:4px"><tr style="opacity:.7">${['Time', 'Result', 'Held', 'Occupied / motion', 'Latency'].map(h => `<th ${cellStyle}>${h}</th>`).join('')}</tr>${rows}</table>`
            : `<div style="opacity:.7;margin-top:4px">none</div>`;
        const showFailure = !!this.failure && this.okAt <= this.failure.atMs && nowMs - this.failure.atMs <= FAILURE_SHOW_MS;
        const failure = this.failure && showFailure ? `<div style="margin-top:8px;color:${C.red}">Failure: ${esc(this.failure.msg)} \u00b7 ${esc(ago(nowMs - this.failure.atMs))}</div>` : '';
        const head = `<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;opacity:.7"><span>Updated ${esc(clock(nowMs))}</span><span>${esc(this.mode)}${this.sensorless ? ' \u00b7 sensorless' : ''}${this.zoomOut ? ` \u00b7 zoom-out ${Math.round(this.zoomOut * 100)} %` : ''}</span></div>`;
        const bypass = !this.enabled || this.state === 'disabled' ? `<div style="margin-top:6px">${this.chip('Disabled', C.gray)}</div>` : '';
        const tamperChip = this.tamper?.on ? `<div style="margin-top:6px">${this.chip(`Tamper: ${this.tamper.reason ?? 'on'}`, C.red)}</div>` : '';
        return `<div style="font-size:90%;line-height:1.45;max-width:680px;color:currentColor">`
            + head + bypass + tamperChip
            + this.nowCard(nowMs)
            + this.activityChart(nowMs)
            + this.nowLists(nowMs)
            + this.thumb(nowMs)
            + `<div style="max-width:420px">${this.detectionThumb(nowMs)}</div>`
            + `<div style="font-weight:600;margin-top:12px">Stats</div><div style="display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;font-size:85%">${grid('Today', today)}${grid('Latency', lat)}${grid('Decode', video)}${grid('Service', service)}${this.tamper ? grid('Tamper', tamperLine(this.tamper, nowMs)) : ''}</div>`
            + `<div style="margin-top:10px;font-weight:600">Episodes</div><div style="font-size:85%;overflow-x:auto">${episodes}</div>${failure}</div>`;
    }
}

/** Tamper detector state for the panel (wall-clock since). */
export interface TamperInfo { enabled: boolean; watching: boolean; learning: boolean; /** learning, the view looks flat */ flat?: boolean; on: boolean; reason?: string; condition?: string; sinceMs?: number; /** 'day' / 'night' with a Night mode sensor */ profile?: string }

/** the Stats 'Tamper' cell (html): on with reason and since, a pending condition, learning, OK, or why it is not checked. */
export function tamperLine(t: TamperInfo, nowMs: number): string {
    if (!t.enabled) return `<span style="opacity:.7">off (disabled for this camera)</span>`;
    const ref = t.profile ? esc(` \u00b7 ${t.profile} reference`) : '';
    if (t.on) return `<span style="color:${C.red}">ON \u00b7 ${esc(t.reason ?? '')}${t.sinceMs ? esc(` \u00b7 since ${clock(t.sinceMs)} (${relTime(t.sinceMs, nowMs)})`) : ''}</span>${ref}`;
    if (!t.watching) return `<span style="opacity:.7">not checked (needs the warm stream)</span>`;
    if (t.learning) return t.flat ? `<span style="color:${C.amber}">learning: view looks flat (covered?)</span>${ref}` : esc('learning the view') + ref;
    if (t.condition) return `<span style="color:${C.amber}">${esc(`OK \u00b7 checking: ${t.condition === 'changed' ? 'view changed' : t.condition}`)}</span>${ref}`;
    return `<span style="color:${C.green}">OK</span>${ref}`;
}

export interface ActivityEntry { t: number; groups: string[]; published: boolean; would: boolean }
const ACTIVITY_MS = 10 * 60 * 1000, ACTIVITY_BUCKETS = 60, ACTIVITY_MAX = 1200;
export type NowReason = { text: string } | { groups: string[]; sinceMs?: number; lastSeenMs?: number };
export interface NowInfo { published: boolean; coral: boolean; mode: string; reason: NowReason; /** Occupancy (0.3.19): in Shadow the would-publish values */ occupancy?: OccupancyInfo }
/** on: occupied; groups held; stillS: every held subject still this long; releasedAfterS: motion released for stillness after this still time. */
export interface OccupancyInfo { on: boolean; groups?: string[]; stillS?: number; releasedAfterS?: number }

/** 'Occupancy: on (person, still 42 s) · Motion: off (released after 170 s still)' (Shadow: 'Would: ...'). */
export function sensorLine(o: OccupancyInfo, motion: boolean, would = false): string {
    const occBits = o.on ? [...(o.groups?.length ? [o.groups.join('+')] : []), ...(o.stillS !== undefined && o.stillS >= 3 ? [`still ${Math.round(o.stillS)} s`] : [])] : [];
    const occ = `Occupancy: ${o.on ? 'on' : 'off'}${occBits.length ? ` (${occBits.join(', ')})` : ''}`;
    const mot = `Motion: ${motion ? 'on' : 'off'}${!motion && o.on && o.releasedAfterS !== undefined ? ` (released after ${Math.round(o.releasedAfterS)} s still)` : ''}`;
    return `${would ? 'Would: ' : ''}${occ} \u00b7 ${mot}`;
}

/** 'occupied 300 s / motion 170 s' of an episode row. */
export function occMotion(e: EpisodeEvent): string {
    return `${e.occupiedS} s / ${e.motionS ?? e.occupiedS} s`;
}
export interface TrackedItem { group: string; cls: string; state: string; sinceMs?: number; lastSeenMs?: number; box?: number[]; detected: boolean; /** hold remaining at analysis time (service 0.3.0+) */ holdLeftMs?: number }
const cap = (t: string) => t ? t[0].toUpperCase() + t.slice(1) : t;
function boxLabel(b: Box): string {
    const name = b.group === 'person' || !b.group ? 'person' : b.cls || b.group;
    return `${name} ${Number.isFinite(b.score) ? b.score.toFixed(2) : ''}${b.parked ? ' parked' : b.still ? ' still' : ''}`.trim();
}
/** caption item: the best valid detection of the frame, or 'nothing' */
function topItem(boxes: Box[] | undefined): string {
    const v = (boxes || []).filter(b => !b.masked && !b.weak).sort((a, b) => b.score - a.score);
    return v.length ? boxLabel(v[0]) + (v.length > 1 ? ` +${v.length - 1}` : '') : 'nothing';
}
const ago = (ms: number) => { const s = Math.max(0, Math.round(ms / 1000)); return s < 60 ? `${s} s ago` : s < 3600 ? `${Math.floor(s / 60)} min ago` : `${Math.floor(s / 3600)} h ago`; };

const C = { green: '#2f9e5b', amber: '#d9930d', red: '#e5484d', blue: '#3b82f6', gray: '#8b8b8b' };
