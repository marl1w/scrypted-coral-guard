import test from 'node:test';
import assert from 'node:assert/strict';
import { CameraStatus, EpisodeEvent, MAX_DATA_URI_BYTES, fmtMs, pctOf, relTime, esc } from '../src/status';

const T0 = new Date(2026, 9, 13, 12, 0, 0).getTime();
const mk = () => { const c = { t: T0 }; const s = new CameraStatus(() => c.t); return { s, c }; };
const ep = (o: Partial<EpisodeEvent> = {}): EpisodeEvent => ({ kind: 'camera-triggered', rawStartMs: T0, heldBeyondRawS: 0, vetoed: false, aborted: false, ...o });

test('empty state: dashes instead of clutter, empty episode line, no image', () => {
    const { s } = mk();
    const h = s.render(T0);
    assert.match(h, /Updated 12:00:00/);
    assert.match(h, /Episodes<\/div><div[^>]*><div[^>]*>none/);
    assert.match(h, /0 episodes/);
    assert.ok(!/<img/.test(h));
    assert.ok(!/Failure:/.test(h));
    assert.ok(!/Analysed/.test(h));
});

test('each state renders its chip; disabled via setEnabled; Watching when the stream is connected', () => {
    const { s } = mk();
    const want: [any, RegExp][] = [['idle', /Idle/], ['analysing', /Analysing/], ['person confirmed', /Person confirmed/], ['service unavailable (passing motion through)', /Service unavailable/], ['disabled', /Disabled/]];
    for (const [st, re] of want) { s.setState(st); assert.match(s.overviewCells(T0).state, re, st); }
    s.setState('idle'); s.setStream({ connected: true, decode: 'cpu', warm: true });
    assert.match(s.overviewCells(T0).state, /Watching/);
    s.setEnabled(false);
    assert.match(s.render(T0), /Disabled/);
    s.setEnabled(true);
    assert.match(s.summaryLine(T0), /^idle;/);
});

test('caption, video, service, mode', () => {
    const { s } = mk();
    s.setMode('Gate');
    s.onFrameAnalysed({ atMs: T0 - 12000, personCount: 2, bestScore: 0.72, totalMs: 39, inferMs: 10, queueMs: 1 });
    s.setLastFrame({ jpegBase64: 'AAAA', boxes: [] });
    s.setStream({ connected: true, decode: 'gpu', warm: true });
    s.onServiceHealth({ ok: true, coralMs: 29, text: '' });
    const h = s.render(T0);
    assert.match(h, /12 s ago \u00b7 nothing/);
    assert.match(h, /Decode<\/div><div>GPU</);
    assert.match(h, /OK<\/span> \u00b7 Coral 29 ms/);
    assert.match(h, /<span>Gate<\/span>/);
    s.onFrameAnalysed({ atMs: T0, personCount: 0, totalMs: 4.26 });
    assert.match(s.render(T0), /0 s ago \u00b7 nothing/);
    s.setStream({ connected: false, decode: 'unknown', warm: false });
    assert.match(s.render(T0), /not connected/);
});

test('episodes table: result chips, humanised numbers, dashes, newest first, max 8', () => {
    const { s } = mk();
    s.onEpisode(ep({ confirmedAfterMs: 1234, heldBeyondRawS: 27, rawStartMs: T0 }));
    s.onEpisode(ep({ vetoed: true, rawStartMs: T0 + 1000 }));
    s.onEpisode(ep({ aborted: true, rawStartMs: T0 + 2000 }));
    s.onEpisode(ep({ kind: 'unflagged', rawStartMs: T0 + 3000 }));
    const h = s.render(T0 + 5000);
    for (const w of ['Confirmed', 'Vetoed', 'Aborted', 'Unflagged']) assert.match(h, new RegExp(w));
    assert.match(h, /1\.2 s/);
    assert.match(h, /27 s/);
    assert.match(h, /&mdash;/);
    assert.ok(h.indexOf('12:00:03') < h.indexOf('12:00:00'));
    for (let i = 0; i < 12; i++) s.onEpisode(ep({ rawStartMs: T0 + 100000 + i * 1000 }));
    assert.equal((s.render(T0 + 200000).match(/<td [^>]*>12:\d\d:\d\d<\/td>/g) || []).length, 8);
});

test('counters and midnight rollover', () => {
    const { s, c } = mk();
    c.t = new Date(2026, 9, 13, 23, 59, 0).getTime();
    s.onEpisode(ep());
    s.onEpisode(ep({ vetoed: true }));
    s.onEpisode(ep({ kind: 'unflagged' }));
    s.onEpisode(ep({ aborted: true }));
    s.onFrameSkipped(); s.onFailure('boom');
    assert.deepEqual(s.today(), { episodes: 4, confirmed: 1, vetoed: 1, unflagged: 1, skippedFrames: 1, failures: 1 });
    c.t = new Date(2026, 9, 14, 0, 0, 5).getTime();
    assert.equal(s.today().episodes, 0);
    assert.match(s.render(c.t), /0 episodes &middot; 0 confirmed/);
    s.onEpisode(ep());
    assert.deepEqual(s.today(), { episodes: 1, confirmed: 1, vetoed: 0, unflagged: 0, skippedFrames: 0, failures: 0 });
});

test('history ring limited to 10, newest first', () => {
    const { s } = mk();
    for (let i = 0; i < 15; i++) s.onEpisode(ep({ rawStartMs: T0 + i * 1000, note: `n${i}` }));
    const j = s.toJSON();
    assert.equal(j.history.length, 10);
    assert.equal(j.history[0].note, 'n5');
});

test('percentile maths 0/1/2/50+ samples', () => {
    const { s } = mk();
    assert.equal(s.percentiles().n, 0);
    assert.equal(s.percentiles().confirmed.p50, undefined);
    s.onEpisode(ep({ confirmedAfterMs: 100, postedAfterMs: 10 }));
    assert.deepEqual(s.percentiles().confirmed, { p50: 100, p95: 100 });
    s.onEpisode(ep({ confirmedAfterMs: 300 }));
    assert.deepEqual(s.percentiles().confirmed, { p50: 100, p95: 300 });
    assert.deepEqual(s.percentiles().posted, { p50: 10, p95: 10 });
    const b = mk().s;
    for (let i = 1; i <= 60; i++) b.onEpisode(ep({ confirmedAfterMs: i }));
    const p = b.percentiles();
    assert.equal(p.n, 50);
    assert.equal(p.confirmed.p50, 35);
    assert.equal(p.confirmed.p95, 58);
    b.onEpisode(ep({ aborted: true, confirmedAfterMs: 99999 }));
    assert.equal(b.percentiles().n, 50);
    assert.equal(pctOf([], 0.5), undefined);
});

test('escaping of every dynamic string', () => {
    const { s } = mk();
    const x = '<script>alert(1)</script>';
    s.setMode(x);
    s.onFailure(x);
    s.onServiceHealth({ ok: false, text: x });
    s.onEpisode(ep({ note: x }));
    const h = s.render(T0);
    assert.ok(!h.includes('<script'));
    assert.ok(h.includes('&lt;script&gt;'));
    s.setLastFrame({ jpegBase64: 'AAAA', boxes: [{ x0: 0, y0: 0, x1: 1, y1: 1, score: NaN }] });
    assert.ok(!s.render(T0).includes('NaN'));
    assert.equal(esc(`"'&`), '&quot;&#39;&amp;');
    assert.ok(!/\son\w+=/i.test(h));
});

test('thumbnail with boxes, malformed base64 and size cap', () => {
    const { s } = mk();
    s.setLastFrame({ jpegBase64: 'AAAA', boxes: [{ x0: 0.1, y0: 0.2, x1: 0.5, y1: 0.9, score: 0.8 }, { x0: -1, y0: 0, x1: 2, y1: 1, score: 0.5 }] });
    let h = s.render(T0);
    assert.match(h, /<img src="data:image\/jpeg;base64,AAAA"/);
    assert.match(h, /left:10%;top:20%;width:40%;height:70%/);
    assert.match(h, /left:0%;top:0%;width:100%;height:100%/);
    assert.match(h, />person 0\.80<\/span>/);
    assert.match(h, /max-width:640px/);
    assert.match(h, /width:100%;height:auto/);
    assert.ok(!/<canvas|<script/.test(h));
    s.setLastFrame({ jpegBase64: 'A'.repeat(MAX_DATA_URI_BYTES), boxes: [] });
    assert.ok(!s.render(T0).includes('<img'));
    s.setLastFrame({ jpegBase64: '"><script>x</script>', boxes: [] });
    h = s.render(T0);
    assert.ok(!h.includes('<img') && !h.includes('<script'));
    s.setLastFrame({ jpegBase64: 'AAAA', boxes: [] });
    assert.ok(s.render(T0).length < 100 * 1024);
});

test('relative time and number formatting', () => {
    assert.equal(relTime(T0, T0 + 2000), 'just now');
    assert.equal(relTime(T0, T0 + 12000), '12 s ago');
    assert.equal(relTime(T0, T0 + 3 * 60000), '3 min ago');
    assert.equal(relTime(T0, T0 + 2 * 3600000), '2 h ago');
    assert.equal(relTime(T0, T0 + 3 * 86400000), '3 d ago');
    assert.equal(relTime(undefined, T0), 'never');
    assert.equal(relTime(T0 + 5000, T0), 'just now');
    assert.equal(fmtMs(4.26), '4.3 ms');
    assert.equal(fmtMs(31.4), '31 ms');
    assert.equal(fmtMs(undefined), 'n/a');
    assert.equal(fmtMs(1234), '1.2 s');
});

test('summaryLine and toJSON', () => {
    const { s } = mk();
    assert.equal(s.summaryLine(T0), 'idle; today 0 episodes (0 confirmed, 0 vetoed)');
    s.onEpisode(ep({ confirmedAfterMs: 420 }));
    assert.match(s.summaryLine(T0), /1 episodes \(1 confirmed, 0 vetoed\), confirm p50\/p95 420 ms\/420 ms/);
    const j = s.toJSON();
    assert.equal(j.counters.episodes, 1);
    assert.doesNotThrow(() => JSON.stringify(j));
});

test('weak candidates: dashed dim boxes labelled possible, caption says no person (weak candidate)', () => {
    const { s } = mk();
    s.setLastFrame({ jpegBase64: 'AAAA', boxes: [{ x0: 0.1, y0: 0.1, x1: 0.4, y1: 0.8, score: 0.26, weak: true }] });
    s.onFrameAnalysed({ atMs: T0, personCount: 0, candidateScore: 0.26, totalMs: 33 });
    const h = s.render(T0);
    assert.match(h, /border:2px dashed/);
    assert.match(h, /possible 0\.26/);
    assert.match(h, /0 s ago \u00b7 nothing/);
    s.setStream({ connected: true, decode: 'cpu', warm: true, gpuFailed: true });
    assert.match(s.render(T0), /CPU \(GPU failed\)/);
});

test('vetoed episodes show the verdict latency; the thumbnail base64 is computed lazily from the raw JPEG', () => {
    const { s } = mk();
    s.onEpisode(ep({ vetoed: true, verdictAfterMs: 61, postedAfterMs: 20 }));
    assert.match(s.render(T0), />61 ms</);
    s.setLastFrame({ jpeg: Buffer.from([0xff, 0xd8, 0xff, 0xd9]), boxes: [] });
    assert.match(s.render(T0), /<img src="data:image\/jpeg;base64,\/9j\/2Q=="/);
    s.setLastFrame({ jpeg: Buffer.alloc(130 * 1024), boxes: [] });
    assert.ok(!s.render(T0).includes('<img'));
    assert.match(s.render(T0), /Snapshot too large/);
    s.setLastFrame({ jpeg: Buffer.from([0xff, 0xd8, 0xff, 0xd9]), boxes: [] });
    s.setThumbTooLarge();
    assert.ok(s.render(T0).includes('<img'), 'the previous small thumbnail stays');
});

test('live frame only while nothing was analysed; an analysed frame keeps its boxes however old', () => {
    const { s } = mk();
    const jpg = Buffer.from([0xff, 0xd8, 0xff, 0xd9]);
    s.setLiveFrame(jpg, T0 - 5000);
    let h = s.render(T0);
    assert.ok(h.includes('<img'));
    assert.match(h, /5 s ago \u00b7 live, not analysed/);
    assert.ok(!/border:2px/.test(h));
    s.setLastFrame({ jpeg: jpg, boxes: [{ x0: 0.1, y0: 0.1, x1: 0.5, y1: 0.5, score: 0.9 }] });
    s.onFrameAnalysed({ atMs: T0 - 12000, personCount: 1, bestScore: 0.9, totalMs: 30 });
    s.setLiveFrame(jpg, T0);
    h = s.render(T0);
    assert.ok(!/live, not analysed/.test(h));
    assert.match(h, /12 s ago \u00b7 person 0\.90/);
    assert.match(h, /border:2px solid/);
    s.setLiveFrame(jpg, T0 + 3 * 3600000);
    h = s.render(T0 + 3 * 3600000);
    assert.match(h, /3 h ago \u00b7 person 0\.90/, 'old analysis still shown with its boxes');
    assert.match(h, /border:2px solid/);
    s.setLastFrame({ jpeg: Buffer.alloc(130 * 1024), boxes: [] });
    assert.match(s.render(T0), /Snapshot too large/);
});

test('vehicles and animals: blue solid / dimmed dashed parked / green boxes, group caption', () => {
    const { s } = mk();
    s.setLastFrame({ jpegBase64: 'AAAA', boxes: [
        { x0: 0.1, y0: 0.1, x1: 0.4, y1: 0.5, score: 0.92, group: 'vehicle', cls: 'car', parked: false },
        { x0: 0.5, y0: 0.1, x1: 0.8, y1: 0.5, score: 0.88, group: 'vehicle', cls: 'truck', parked: true },
        { x0: 0.1, y0: 0.6, x1: 0.2, y1: 0.8, score: 0.7, group: 'animal', cls: 'dog' },
    ] });
    s.onFrameAnalysed({ atMs: T0, personCount: 1, bestScore: 0.8, vehicles: 2, parkedVehicles: 1, animals: 1, totalMs: 33 });
    const h = s.render(T0);
    assert.match(h, /border:2px solid #3b82f6[^>]*>[^<]*<span[^>]*>car 0\.92</);
    assert.match(h, /border:2px dashed #3b82f6[^>]*>[^<]*<span[^>]*>truck parked 0\.88</);
    assert.match(h, /border:2px solid #2f9e5b[^>]*>[^<]*<span[^>]*>dog 0\.70</);
    assert.match(h, /0 s ago \u00b7 car 0\.92 \+2/, 'caption: age and top item only');
    const live = mk().s;
    live.setLiveFrame(Buffer.from([0xff, 0xd8, 0xff, 0xd9]), T0);
    assert.ok(!/border:2px/.test(live.render(T0)), 'a live frame has no boxes');
});

test('episode chip names the confirming group; Today lists per-group confirmed counts', () => {
    const { s } = mk();
    s.onEpisode(ep({ confirmedAfterMs: 400, groups: ['vehicle'] }));
    s.onEpisode(ep({ confirmedAfterMs: 500, groups: ['person'] }));
    const h = s.render(T0);
    assert.match(h, /Confirmed \u00b7 vehicle/);
    assert.match(h, /2 confirmed \((person 1, vehicle 1|vehicle 1, person 1)\)/);
});

test('weak vehicle: dashed blue "possible car 0.38", caption "1 possible vehicle"', () => {
    const { s } = mk();
    s.setLastFrame({ jpegBase64: 'AAAA', boxes: [{ x0: 0.1, y0: 0.1, x1: 0.4, y1: 0.5, score: 0.38, group: 'vehicle', cls: 'car', weak: true }] });
    s.onFrameAnalysed({ atMs: T0, personCount: 0, vehicles: 0, weakVehicles: 1, animals: 0, totalMs: 20 });
    const h = s.render(T0);
    assert.match(h, /border:2px dashed #3b82f6[^>]*>[^<]*<span[^>]*>possible car 0\.38</);
    assert.match(h, /0 s ago \u00b7 nothing/, 'a weak candidate is not an item');
});

test('Last failure hides after a later success or after 120 s', () => {
    const { s, c } = mk();
    s.onFailure('health failed: ECONNREFUSED');
    assert.match(s.render(c.t), /Failure:/);
    c.t += 5000;
    s.onServiceHealth({ ok: true, text: '' });
    assert.ok(!/Failure:/.test(s.render(c.t)), 'healthy service hides it');
    const x = mk();
    x.s.onFailure('boom');
    x.c.t += 5000;
    x.s.onFrameAnalysed({ atMs: x.c.t, personCount: 0 });
    assert.ok(!/Failure:/.test(x.s.render(x.c.t)), 'a later analysis hides it');
    const y = mk();
    y.s.onFailure('boom');
    assert.match(y.s.render(y.c.t + 119000), /Failure:/);
    assert.ok(!/Failure:/.test(y.s.render(y.c.t + 121000)), '120 s');
});

test('a vehicle on a still track is dimmed and labelled still', () => {
    const { s } = mk();
    s.setLastFrame({ jpegBase64: 'AAAA', boxes: [{ x0: 0.1, y0: 0.1, x1: 0.4, y1: 0.5, score: 0.62, group: 'vehicle', cls: 'car', still: true }] });
    s.onFrameAnalysed({ atMs: T0, personCount: 0, vehicles: 1, parkedVehicles: 1, animals: 0 });
    assert.match(s.render(T0), /border:2px dashed #3b82f6[^>]*opacity:\.65[^>]*>[^<]*<span[^>]*>car still 0\.62</);
});

test('Last detection: empty state, kept across empty frames, only enabled groups, never an older JPEG after an oversize frame', () => {
    const { s } = mk();
    const jA = Buffer.from([0xff, 0xd8, 1, 0xff, 0xd9]), jB = Buffer.from([0xff, 0xd8, 2, 0xff, 0xd9]), jC = Buffer.from([0xff, 0xd8, 3, 0xff, 0xd9]);
    const det = (h: string) => /Last detection<\/div>([\s\S]*)$/.exec(h)![1];
    assert.match(s.render(T0), /Last detection<\/div><div[^>]*>none yet/);
    s.setLastFrame({ jpeg: jA, boxes: [] });
    s.onFrameAnalysed({ atMs: T0, personCount: 2, bestScore: 0.91, vehicles: 1, parkedVehicles: 1, animals: 0, boxes: [{ x0: 0, y0: 0, x1: 0.1, y1: 0.1, score: 0.91 }] });
    s.setLastFrame({ jpeg: jB, boxes: [] });
    s.onFrameAnalysed({ atMs: T0, personCount: 0, vehicles: 0, animals: 0, boxes: [] });
    let h = det(s.render(T0));
    assert.match(h, new RegExp(jA.toString('base64').replace(/[+/=]/g, '.')));
    assert.match(h, /0 s ago \u00b7 person 0\.91/);
    assert.match(s.render(T0), /Last analysis/);
    s.setDetect(['vehicle']);
    s.setLastFrame({ jpeg: jC, boxes: [] });
    s.onFrameAnalysed({ atMs: T0, personCount: 1, bestScore: 0.8, vehicles: 0, animals: 0 });
    assert.match(det(s.render(T0)), new RegExp(jA.toString('base64').replace(/[+/=]/g, '.')), 'person disabled: not a detection');
    s.setThumbTooLarge();
    s.onFrameAnalysed({ atMs: T0, personCount: 0, vehicles: 1, animals: 0 });
    h = det(s.render(T0));
    assert.ok(!h.includes(jC.toString('base64')), 'an oversize frame never pairs with the previous JPEG');
});

test('activity: 10 s buckets over 10 min, frames counted per group (not objects), enabled groups only, published / would lanes, gaps visible', () => {
    const { s } = mk();
    s.setNow({ published: true, coral: false, mode: 'Shadow', reason: { text: 'x' } });
    const T = T0;
    s.onFrameAnalysed({ atMs: T - 5000, personCount: 3, vehicles: 1, animals: 0 });      // bucket 59: 1 person frame, 1 vehicle frame
    s.onFrameAnalysed({ atMs: T - 6000, personCount: 1, vehicles: 0, animals: 1 });      // bucket 59
    s.setNow({ published: false, coral: true, mode: 'Shadow', reason: { text: 'x' } });
    s.onFrameAnalysed({ atMs: T - 95000, personCount: 0, vehicles: 0, animals: 0 });     // bucket 50: analysed, nothing
    s.onFrameAnalysed({ atMs: T - 700000, personCount: 1, vehicles: 0, animals: 0 });    // older than 10 min: dropped
    const b = s.activityBuckets(T);
    assert.equal(b.length, 60);
    assert.deepEqual([b[59].person, b[59].vehicle, b[59].animal, b[59].analysed], [2, 1, 1, 2]);
    assert.equal(b[59].published, true); assert.equal(b[59].would, false);
    assert.deepEqual([b[50].analysed, b[50].person, b[50].would], [1, 0, true]);
    assert.equal(b[0].analysed, 0);
    let h = s.render(T);
    assert.match(h, />Activity</);
    assert.match(h, />4 frames<\/text>/, "stacked max"); assert.match(h, />10 min ago<\/text>/); assert.match(h, />now<\/text>/);
    assert.match(h, /<title>no analysis<\/title>/);
    assert.match(h, />Would<\/text>/, 'Shadow shows the second lane');
    assert.match(h, /<title>person: 2 frames<\/title>/);
    s.setDetect(['vehicle']);
    s.onFrameAnalysed({ atMs: T - 1000, personCount: 1, vehicles: 0, animals: 0 });
    assert.equal(s.activityBuckets(T)[59].person, 2, 'a disabled group is not counted');
    s.setNow({ published: false, coral: false, mode: 'Gate', reason: { text: '<b>' } });
    h = s.render(T);
    assert.ok(!/>Would</.test(h), 'Gate: one lane');
    assert.ok(!/<title>person/.test(h), 'disabled group not drawn');
});

test('activity: empty state draws only no-analysis ticks', () => {
    const { s } = mk();
    const h = s.render(T0);
    assert.equal((h.match(/<title>no analysis<\/title>/g) || []).length, 60);
    assert.match(h, />1 frame<\/text>/);
});
