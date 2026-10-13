// Plugin page 'Performance' group: a snapshot of the service / Coral metrics (GET /v1/metrics, piggybacked on the 30 s health
// poll) with tiles and 60-minute sparklines from an in-memory ring (reset on plugin restart). Not live: computed on page load.

import { esc } from './status';

export interface PerfSample { at: number; busy?: number; fps?: number; inferP95?: number; hostCpu?: number }

const n = (v: unknown): number | undefined => typeof v === 'number' && Number.isFinite(v) ? v : undefined;

/** At most `max` samples, one per `minGapMs` (the poll also runs after a save). */
export class MetricsRing {
    samples: PerfSample[] = [];
    constructor(readonly max = 120, readonly minGapMs = 25000) { }
    push(at: number, m: any): boolean {
        const last = this.samples[this.samples.length - 1];
        if (last && at - last.at < this.minGapMs)
            return false;
        this.samples.push({ at, busy: n(m?.coral?.utilization_pct), fps: n(m?.coral?.invocations_per_s), inferP95: n(m?.stages?.infer_ms?.p95), hostCpu: n(m?.host?.cpu_pct) });
        while (this.samples.length > this.max)
            this.samples.shift();
        return true;
    }
}

/** Polyline path ('M x y L ...') inside the box for the values (missing ones skipped); '' with fewer than 2 points. A constant series (no range) is a flat mid line. */
export function scaledPath(values: (number | undefined)[], box: { x0: number; x1: number; y0: number; y1: number }, range?: { min: number; max: number }): string {
    const pts = values.map((v, i) => ({ v, i })).filter((p): p is { v: number; i: number } => typeof p.v === 'number' && Number.isFinite(p.v));
    if (pts.length < 2 || values.length < 2)
        return '';
    const min = range?.min ?? Math.min(...pts.map(p => p.v)), max = range?.max ?? Math.max(...pts.map(p => p.v));
    const span = max - min;
    const x = (i: number) => box.x0 + (i / (values.length - 1)) * (box.x1 - box.x0);
    const y = (v: number) => span === 0 ? (box.y0 + box.y1) / 2 : box.y1 - ((Math.min(max, Math.max(min, v)) - min) / span) * (box.y1 - box.y0);
    return pts.map((p, k) => `${k ? 'L' : 'M'}${x(p.i).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' ');
}

/** Compact path (160x32), kept for callers that need only the line. */
export function sparkPath(values: (number | undefined)[], w = 160, h = 32): string {
    return scaledPath(values, { x0: 1, x1: w - 1, y0: 1, y1: h - 1 });
}

/** Axis range: percentages fixed 0-100, others auto-scaled with ~10 % padding (never below 0). */
export function axisRange(vals: number[], percent: boolean): { min: number; max: number } {
    if (percent) return { min: 0, max: 100 };
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.1 || Math.max(1, Math.abs(hi) * 0.1);
    return { min: Math.max(0, lo - pad), max: hi + pad };
}

/** 240x60 chart: y labels (top/bottom) with units, dashed labelled midline, latest point dot with value, x labels (oldest age, now). */
export function sparkChart(title: string, samples: { at: number; v?: number }[], nowMs: number, unit: string, digits: number, percent: boolean): string {
    const W = 240, H = 60, box = { x0: 44, x1: 214, y0: 6, y1: 44 };
    const vals = samples.map(s => s.v).filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const path = vals.length >= 2 ? scaledPath(samples.map(s => s.v), box, axisRange(vals, percent)) : '';
    if (!path)
        return '';
    const r = axisRange(vals, percent);
    const f = (v: number) => `${v.toFixed(digits)}${unit}`;
    const mid = (r.min + r.max) / 2, midY = (box.y0 + box.y1) / 2;
    const lastIdx = samples.map(s => s.v).map((v, i) => (typeof v === 'number' && Number.isFinite(v) ? i : -1)).filter(i => i >= 0).pop()!;
    const lastV = samples[lastIdx].v!;
    const lx = box.x0 + (lastIdx / (samples.length - 1)) * (box.x1 - box.x0);
    const ly = box.y1 - ((Math.min(r.max, Math.max(r.min, lastV)) - r.min) / (r.max - r.min || 1)) * (box.y1 - box.y0);
    const oldest = nowMs - samples[0].at;
    const left = oldest >= 59 * 60000 ? '60 min ago' : oldest >= 60000 ? `${Math.round(oldest / 60000)} min ago` : `${Math.round(oldest / 1000)} s ago`;
    const t = (x: number, y: number, s: string, anchor = 'start', extra = '') => `<text x="${x}" y="${y}" font-size="10" fill="currentColor" text-anchor="${anchor}"${extra}>${esc(s)}</text>`;
    return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block;color:currentColor" role="img" aria-label="${esc(title)}">`
        + t(box.x0 - 4, box.y0 + 4, f(r.max), 'end', ' opacity=".7"') + t(box.x0 - 4, box.y1, f(r.min), 'end', ' opacity=".7"')
        + `<line x1="${box.x0}" y1="${midY}" x2="${box.x1}" y2="${midY}" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="3 3"/>` + t(box.x0 - 4, midY + 3, f(mid), 'end', ' opacity=".5"')
        + `<path d="${path}" fill="none" stroke="currentColor" stroke-width="1.5"/>`
        + `<circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="2.5" fill="currentColor"/>` + t(lx + 4, ly + 3, f(lastV))
        + t(box.x0, H - 3, left, 'start', ' opacity=".7"') + t(box.x1, H - 3, 'now', 'end', ' opacity=".7"')
        + `</svg>`;
}

const clock = (ms: number) => { const d = new Date(ms); const p = (x: number) => String(x).padStart(2, '0'); return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`; };
const ago = (ms: number) => { const s = Math.max(0, Math.round(ms / 1000)); return s < 60 ? `${s} s ago` : s < 3600 ? `${Math.floor(s / 60)} min ago` : `${Math.floor(s / 3600)} h ago`; };
const fmt = (v: unknown, d = 0, u = '') => typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(d)}${u}` : 'n/a';
const pair = (o: any) => o && (n(o.median) !== undefined || n(o.p95) !== undefined) ? `${fmt(o.median)} / ${fmt(o.p95)} ms` : 'n/a';
const uptime = (s: unknown) => typeof s !== 'number' ? 'n/a' : s >= 86400 ? `${Math.floor(s / 86400)} d ${Math.floor((s % 86400) / 3600)} h` : s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min` : `${Math.floor(s / 60)} min`;
const C = { green: '#2f9e5b', amber: '#d9930d', red: '#e5484d' };

export function headerText(nowMs: number, at: number | undefined): string {
    return at === undefined ? 'No snapshot yet: the service has not answered since the plugin started. Press Refresh or reload the page.'
        : `Snapshot taken at ${clock(at)} (${ago(nowMs - at)}), not live: press Refresh or reload the page.`;
}

export function renderPerformance(nowMs: number, latest: { at: number; metrics: any } | undefined, ring: MetricsRing, unavailableSince?: number): string {
    const head = `<div style="opacity:.7">${esc(headerText(nowMs, latest?.at))}</div>`;
    const down = unavailableSince !== undefined ? `<div style="color:${C.red};margin-top:6px">${esc(`Service unavailable since ${clock(unavailableSince).slice(0, 5)}`)}</div>` : '';
    if (!latest)
        return `<div style="color:currentColor;line-height:1.5">${head}${down}</div>`;
    const m = latest.metrics || {}, c = m.coral || {}, st = m.stages || {}, l = m.load || {}, g = m.gpu || {};
    const okState = ['ok', 'ready', 'running'].includes(String(c.state));
    const drops = (n(l.dropped_stale_60s) ?? 0) + (n(l.rejected_429_60s) ?? 0);
    const tile = (k: string, v: string, color?: string, hint?: string) => `<div style="border:1px solid currentColor;border-color:rgba(127,127,127,.35);border-radius:6px;padding:4px 8px;min-width:110px"><div style="opacity:.7;font-size:85%">${esc(k)}</div><div${color ? ` style="color:${color}"` : ''}>${esc(v)}</div>${hint ? `<div style="opacity:.55;font-size:75%;max-width:260px">${esc(hint)}</div>` : ''}</div>`;
    const tiles = `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">`
        + tile('Coral', `${c.state ?? 'unknown'}, ${fmt(c.utilization_pct, 0, ' %')} busy`, okState ? C.green : C.red)
        + tile('Frames/s', fmt(c.invocations_per_s, 1))
        + tile('Inference p50 / p95', pair(st.infer_ms))
        + tile('End-to-end p50 / p95', pair(st.total_ms))
        + tile('Queue p95', st.queue_ms ? fmt(st.queue_ms.p95, 0, ' ms') : 'n/a')
        + tile('Frames skipped (last minute)', `${fmt(l.dropped_stale_60s)} too late \u00b7 ${fmt(l.rejected_429_60s)} rate-limited`, drops > 0 ? C.amber : undefined, 'too late: waited too long in the queue \u00b7 rate-limited: a camera sent frames too fast')
        + tile('Host CPU', fmt(m.host?.cpu_pct, 0, ' %'))
        + tile('Service CPU', fmt(m.process?.cpu_pct, 0, ' %'))
        + tile('GPU busy', g.available === false ? 'n/a' : fmt(g.busy_pct, 0, ' %'))
        + tile('Uptime', uptime(m.uptime_s))
        + `</div>`;
    const s = ring.samples;
    const spark = (k: string, pick: (x: PerfSample) => number | undefined, d: number, u: string, percent: boolean) => {
        const svg = sparkChart(k, s.map(x => ({ at: x.at, v: pick(x) })), nowMs, u, d, percent);
        const vals = s.map(pick).filter((v): v is number => v !== undefined);
        const label = vals.length ? `min ${Math.min(...vals).toFixed(d)}${u}, max ${Math.max(...vals).toFixed(d)}${u}, now ${vals[vals.length - 1].toFixed(d)}${u}` : '';
        const body = svg
            ? `${svg}<div style="opacity:.7;font-size:85%">${esc(label)}</div>`
            : `<div style="opacity:.7;font-size:85%">${esc(`collecting\u2026 (${s.length} ${s.length === 1 ? 'sample' : 'samples'})`)}</div>`;
        return `<div style="min-width:250px"><div style="font-size:85%">${esc(k)}</div>${body}</div>`;
    };
    const sparks = `<div style="opacity:.7;margin-top:10px;font-size:90%">Last 60 minutes (one sample every 30 s, since the plugin started)</div><div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:4px">`
        + spark('Coral busy', x => x.busy, 0, ' %', true) + spark('Frames/s', x => x.fps, 1, '/s', false) + spark('Inference p95', x => x.inferP95, 0, ' ms', false) + spark('Host CPU', x => x.hostCpu, 0, ' %', true)
        + `</div>`;
    return `<div style="color:currentColor;line-height:1.5">${head}${down}${tiles}${sparks}</div>`;
}
