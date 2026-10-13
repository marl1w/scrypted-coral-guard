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
}

export interface ServiceHealth { ok: boolean; coralMs?: number; text: string }
export interface StreamInfo { name?: string; connected: boolean; decode: 'gpu' | 'cpu' | 'unknown'; warm: boolean; /** GPU decode was requested but did not initialise */ gpuFailed?: boolean }
export interface LastFrame {
    jpegBase64?: string;
    /** raw JPEG: base64 is computed lazily, only when the panel renders. */
    jpeg?: Buffer;
    boxes: Box[];
}

export interface Counters { episodes: number; confirmed: number; vetoed: number; unflagged: number; skippedFrames: number; failures: number }

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
const zero = (): Counters => ({ episodes: 0, confirmed: 0, vetoed: 0, unflagged: 0, skippedFrames: 0, failures: 0 });
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
        this.okAt = this.now();
        if (f.boxes && this.last) this.last = { ...this.last, boxes: f.boxes };
        const on = (g: string) => this.enabledGroups.includes(g);
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
        if (e.note) bits.push(e.note);
        return bits.join(', ');
    }

    summaryLine(nowMs: number = this.now()): string {
        const t = this.today(nowMs);
        const p = this.percentiles();
        const lat = p.n ? `, confirm p50/p95 ${p.confirmed.p50 === undefined ? 'n/a' : fmtMs(p.confirmed.p50)}/${p.confirmed.p95 === undefined ? 'n/a' : fmtMs(p.confirmed.p95)}` : '';
        return `${this.state}; today ${t.episodes} episodes (${t.confirmed} confirmed, ${t.vetoed} vetoed)${lat}`;
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
            ? `<div style="opacity:.7;margin-top:10px">Snapshot too large to display</div>` : '';
        if (!l || !b64 || !/^[A-Za-z0-9+/=]+$/.test(b64)) return placeholder;
        const uri = `data:image/jpeg;base64,${b64}`;
        if (uri.length > MAX_DATA_URI_BYTES) return placeholder;
        const boxes = live ? '' : this.boxesHtml(l.boxes);
        const f = this.frame;
        const caption = live ? `Live frame (not analysed), ${relTime(this.liveFrame!.atMs, nowMs)}` : !f ? '' : `Last analysed ${relTime(f.atMs, nowMs)}: ${this.detectionSummary(f)}${f.totalMs !== undefined ? `, ${fmtMs(f.totalMs)}` : ''}`;
        return `<div style="opacity:.7;margin-top:10px">Last analysis</div>` + this.snap(uri, boxes, caption, live ? 'Live frame' : 'Last analysed frame');
    }

    private snap(uri: string, boxes: string, caption: string, alt: string): string {
        return `<div style="position:relative;width:100%;max-width:640px;margin-top:4px"><img src="${uri}" alt="${alt}" style="display:block;width:100%;height:auto;border-radius:4px">${this.zonesSvg()}${boxes}</div>${caption ? `<div style="opacity:.7;font-size:90%;margin-top:4px">${esc(caption)}</div>` : ''}`;
    }

    /** 'Last detection' block: the newest frame with an enabled detection, or an empty state. */
    private detectionThumb(nowMs: number): string {
        const head = `<div style="opacity:.7;margin-top:12px">Last detection</div>`;
        const d = this.detection;
        if (!d) return head + `<div style="opacity:.7;font-size:90%;margin-top:4px">No detection yet</div>`;
        const l = d.last;
        if (!l.jpegBase64 && l.jpeg && l.jpeg.length * 4 / 3 + 30 <= MAX_DATA_URI_BYTES)
            l.jpegBase64 = l.jpeg.toString('base64');
        const b64 = l.jpegBase64;
        const caption = `Last detection ${relTime(d.frame.atMs, nowMs)}: ${this.detectionSummary(d.frame)}`;
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
            today: esc(`${t.episodes} / ${t.confirmed} / ${t.vetoed} / ${t.unflagged}`),
            p50: p === undefined ? '\u2014' : esc(fmtMs(p)),
            decode: !this.stream?.connected ? '\u2014' : this.stream.decode === 'gpu' ? 'GPU' : this.stream.decode === 'cpu' ? 'CPU' : '?',
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
            .concat(t.skippedFrames ? [`${t.skippedFrames} skipped frames`] : [], t.failures ? [`${t.failures} failures`] : []).join(' &middot; ');
        const lat = [two(p.confirmed) && `confirmed ${two(p.confirmed)}`, two(p.posted) && `first frame ${two(p.posted)}`].filter(Boolean).join(' &middot; ');
        const video = !this.stream ? dash
            : this.stream.connected ? esc(`Connected, ${this.stream.decode === 'gpu' ? 'GPU decode' : this.stream.decode === 'cpu' ? 'CPU decode' : 'decode unknown'}${this.stream.warm ? ', warm' : ''}${this.stream.gpuFailed ? ' (GPU requested, not initialised)' : ''}${this.stream.name ? `, stream ${this.stream.name}` : ''}`) : `<span style="color:${C.red}">Not connected</span>`;
        const service = !this.health ? dash
            : `${this.health.ok ? `<span style="color:${C.green}">OK</span>` : `<span style="color:${C.red}">Unavailable</span>`}${this.health.coralMs !== undefined ? esc(`, Coral ${fmtMs(this.health.coralMs)}`) : ''}${this.health.text ? ` (${esc(this.health.text)})` : ''}`;
        const result = (e: EpisodeEvent) => e.aborted ? this.chip('Aborted', C.red, false)
            : e.kind === 'unflagged' ? this.chip('Unflagged', C.blue, false)
                : e.vetoed ? this.chip('Vetoed', C.amber, false) : this.chip(e.groups?.length ? `Confirmed \u00b7 ${e.groups.join('+')}` : 'Confirmed', C.green, false);
        const cellStyle = 'style="padding:3px 10px 3px 0;text-align:left;vertical-align:middle"';
        const rows = this.history.slice(-8).reverse().map(e =>
            `<tr><td ${cellStyle}>${esc(clock(e.rawStartMs))}</td><td ${cellStyle}>${result(e)}</td><td ${cellStyle}>${e.heldBeyondRawS > 0 ? esc(`${Math.round(e.heldBeyondRawS)} s`) : dash}</td><td ${cellStyle}>${e.confirmedAfterMs !== undefined ? esc(fmtMs(e.confirmedAfterMs)) : e.verdictAfterMs !== undefined && e.vetoed ? `<span style="opacity:.7">verdict </span>${esc(fmtMs(e.verdictAfterMs))}` : dash}</td></tr>`).join('');
        const episodes = rows
            ? `<table style="border-collapse:collapse;margin-top:4px"><tr style="opacity:.7">${['Time', 'Result', 'Held', 'Latency'].map(h => `<th ${cellStyle}>${h}</th>`).join('')}</tr>${rows}</table>`
            : `<div style="opacity:.7;margin-top:4px">No episodes yet.</div>`;
        // a failure line is shown until a later success (analysis or healthy service) or 120 s, whichever comes first
        const showFailure = !!this.failure && this.okAt <= this.failure.atMs && nowMs - this.failure.atMs <= FAILURE_SHOW_MS;
        const failure = this.failure && showFailure ? `<div style="margin-top:10px;color:${C.red}">Last failure: ${esc(this.failure.msg)} (${esc(relTime(this.failure.atMs, nowMs))})</div>` : '';
        return `<div style="font-size:90%;line-height:1.45;max-width:680px">`
            + `<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">${this.stateChip()}<span style="opacity:.7">Mode: ${esc(this.mode)}</span></div>`
            + this.thumb(nowMs)
            + this.detectionThumb(nowMs)
            + `<div style="display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;margin-top:12px">${this.sensorless ? grid('Motion', esc('Sensorless: motion comes only from Coral Guard')) : ''}${this.detecting ? grid('Detecting', esc(this.detecting)) : ''}${this.zoomOut ? grid('Analysis', esc(`Zoom-out ${Math.round(this.zoomOut * 100)} %`)) : ''}${grid('Today', today)}${grid('Latency p50 / p95', lat || dash)}${grid('Video', video)}${grid('Service', service)}</div>`
            + `<div style="margin-top:12px;font-weight:600">Recent episodes</div>${episodes}${failure}</div>`;
    }
}

const C = { green: '#2f9e5b', amber: '#d9930d', red: '#e5484d', blue: '#3b82f6', gray: '#8b8b8b' };
