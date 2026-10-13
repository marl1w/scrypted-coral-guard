import test from 'node:test';
import assert from 'node:assert/strict';
import { renderOverview } from '../src/overview';
import { MetricsRing, axisRange, headerText, renderPerformance, sparkChart, sparkPath } from '../src/performance';
import { loadProvider } from './harness';

const cam = { name: 'Kitchen <b>&"', mode: 'Gate', state: '<span>chip</span>', today: '3 / 1 / 2 / 0', p50: '420 ms', decode: 'GPU', lastDetection: '3 min ago' };
const metrics = {
    coral: { state: 'ok', utilization_pct: 42.4, invocations_per_s: 7.25 },
    stages: { infer_ms: { median: 11, p95: 19 }, total_ms: { median: 30, p95: 61 }, queue_ms: { median: 1, p95: 9 } },
    load: { dropped_stale_60s: 2, rejected_429_60s: 0, cameras_active: 13 },
    host: { cpu_pct: 35, load1: 2.5, mem_used_pct: 61 }, process: { cpu_pct: 48, rss_mb: 210 }, gpu: { available: false, busy_pct: null }, uptime_s: 7300,
};

test('overview: camera table only, names escaped, empty state', () => {
    const h = renderOverview([cam]);
    assert.ok(h.includes('Kitchen &lt;b&gt;&amp;&quot;') && !h.includes('<b>&"'));
    assert.match(h, /<span>chip<\/span>/); assert.match(h, /3 min ago/);
    assert.ok(!/Coral|invocations/.test(h));
    assert.match(renderOverview([]), /No camera has Coral Guard enabled/);
});

test('metrics ring: one sample per 25 s, capped at 120', () => {
    const r = new MetricsRing();
    assert.equal(r.push(0, metrics), true);
    assert.equal(r.push(10000, metrics), false);
    for (let i = 1; i <= 200; i++) r.push(i * 30000, metrics);
    assert.equal(r.samples.length, 120);
    assert.equal(r.samples[0].at, 81 * 30000);
    assert.equal(r.samples[119].busy, 42.4);
});

test('sparkline path: empty / single -> none, constant -> flat mid line, varying -> scaled, gaps skipped', () => {
    assert.equal(sparkPath([]), ''); assert.equal(sparkPath([5]), ''); assert.equal(sparkPath([undefined, undefined]), '');
    assert.equal(sparkPath([3, 3, 3]), 'M1.0 16.0 L80.0 16.0 L159.0 16.0');
    assert.equal(sparkPath([0, 10]), 'M1.0 31.0 L159.0 1.0');
    assert.equal(sparkPath([0, undefined, 10]), 'M1.0 31.0 L159.0 1.0');
});

test('performance: header says snapshot and reload, tiles, colour only for Coral state and drops, sparklines after 2 samples', () => {
    const at = new Date(2026, 9, 13, 21, 52, 10).getTime();
    assert.equal(headerText(at + 12000, at), 'Snapshot taken at 21:52:10 (12 s ago), not live: press Refresh or reload the page.');
    const r = new MetricsRing();
    r.push(at, metrics);
    let h = renderPerformance(at + 12000, { at, metrics }, r);
    assert.match(h, /Snapshot taken at 21:52:10 \(12 s ago\)/);
    for (const re of [/ok, 42 % busy/, />7\.3</, /11 \/ 19 ms/, /30 \/ 61 ms/, />9 ms</, /2 too late \u00b7 0 rate-limited/, />35 %</, />48 %</, /GPU busy<\/div><div>n\/a/, /2 h 1 min/])
        assert.match(h, re);
    assert.match(h, /color:#d9930d">2 too late/);
    assert.match(h, /Frames skipped \(last minute\)/); assert.match(h, /rate-limited: a camera sent frames too fast/);
    assert.ok(!/\b429\b|\b503\b/.test(h), 'no raw HTTP codes');
    assert.match(h, /collecting… \(1 sample\)/);
    r.push(at + 30000, { ...metrics, coral: { ...metrics.coral, utilization_pct: 60 } });
    h = renderPerformance(at + 31000, { at: at + 30000, metrics }, r);
    assert.match(h, /<svg width="240" height="60"/);
    assert.match(h, />100 %<\/text>/); assert.match(h, />0 %<\/text>/); assert.match(h, />50 %<\/text>/, 'percent axis fixed 0-100 with midline value');
    assert.match(h, /<circle [^>]*\/><text[^>]*>60 %<\/text>/, 'dot and value on the latest point');
    assert.match(h, />31 s ago<\/text>/); assert.match(h, />now<\/text>/);
    assert.match(h, /min 42 %, max 60 %, now 60 %/);
});

test('performance: unreachable service shows the last snapshot time and since when; escaped state', () => {
    const at = new Date(2026, 9, 13, 21, 0, 0).getTime();
    const down = new Date(2026, 9, 13, 21, 5, 0).getTime();
    const h = renderPerformance(at + 600000, { at, metrics: { ...metrics, coral: { state: '<x>' } } }, new MetricsRing(), down);
    assert.match(h, /Snapshot taken at 21:00:00 \(10 min ago\)/);
    assert.match(h, /Service unavailable since 21:05/);
    assert.ok(h.includes('&lt;x&gt;') && !h.includes('<x>'));
    assert.match(renderPerformance(at, undefined, new MetricsRing(), down), /No snapshot yet[\s\S]*Service unavailable since 21:05/);
});

test('provider: Overview first, Performance group from the poll; a failed poll marks the service down', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    p.console = { log() { }, warn() { } };
    p.client = { metrics: async () => metrics, check: async () => ({ ok: true }) };
    await p.checkHealth();
    let st = await p.getSettings();
    assert.equal(st[0].key, 'overview');
    const groups: string[] = [];
    for (const x of st) { const g = x.group || 'General'; if (!groups.includes(g)) groups.push(g); }
    assert.deepEqual(groups, ['Overview', 'Performance', 'General', 'Detection thresholds']);
    const perf = st.find((x: any) => x.key === 'performance');
    assert.equal(perf.group, 'Performance');
    assert.match(perf.value, /42 % busy/);
    p.client = { metrics: async () => { throw new Error('ECONNREFUSED'); }, check: async () => { throw new Error('down'); } };
    await p.checkHealth();
    st = await p.getSettings();
    assert.match(st.find((x: any) => x.key === 'performance').value, /Service unavailable since/);
    clearInterval(p.healthTimer);
});

test('Refresh button: fetches health and metrics now, updates the snapshot even inside the ring spacing, one Settings event', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    p.console = { log() { }, warn() { } };
    let fetched = 0, checks = 0, events = 0, busy = 10;
    p.onDeviceEvent = () => { events++; };
    p.client = { metrics: async () => { fetched++; return { ...metrics, coral: { ...metrics.coral, utilization_pct: busy } }; }, check: async () => { checks++; return { ok: true }; } };
    await p.checkHealth();
    const st = await p.getSettings();
    const i = st.findIndex((x: any) => x.key === 'refreshPerformance');
    assert.equal(st[i].type, 'button'); assert.equal(st[i].group, 'Performance'); assert.equal(st[i + 1].key, 'performance');
    busy = 77;
    const f0 = fetched, c0 = checks;
    await p.putSetting('refreshPerformance', undefined);
    assert.equal(fetched, f0 + 1); assert.equal(checks, c0 + 1);
    assert.equal(events, 1);
    assert.equal(p.perfLatest.metrics.coral.utilization_pct, 77);
    assert.equal(p.perfRing.samples.length, 1, 'ring spacing kept');
    assert.match((await p.getSettings()).find((x: any) => x.key === 'performance').value, /press Refresh or reload the page/);
    clearInterval(p.healthTimer);
});

test('sparkline chart: auto-scaled ms axis with padding, 60 min label, escaped title, none below 2 points', () => {
    const now = 4_000_000;
    assert.deepEqual(axisRange([10, 20], true), { min: 0, max: 100 });
    assert.deepEqual(axisRange([10, 20], false), { min: 9, max: 21 });
    assert.deepEqual(axisRange([5, 5], false), { min: 4, max: 6 });
    const samples = Array.from({ length: 120 }, (_, i) => ({ at: now - (119 - i) * 30000, v: 10 + (i % 10) }));
    const svg = sparkChart('Inference <p95>', samples, now, ' ms', 0, false);
    assert.match(svg, /aria-label="Inference &lt;p95&gt;"/);
    assert.match(svg, />60 min ago<\/text>/);
    assert.match(svg, />9 ms<\/text>/); assert.match(svg, />20 ms<\/text>/, 'padded range 9..20 (19+~1)');
    assert.match(svg, /stroke-dasharray="3 3"/);
    assert.equal(sparkChart('x', [{ at: now, v: 1 }], now, '', 0, false), '');
});
