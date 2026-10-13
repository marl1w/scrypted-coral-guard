import test from 'node:test';
import assert from 'node:assert/strict';
import { PixelGate } from '../src/gate';

const W = 128, H = 72;
let seed = 12345;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

function scene(w = W, h = H): Uint8Array {
    const f = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) f[y * w + x] = 80 + ((x * 2 + y) % 40);
    return f;
}
function rect(f: Uint8Array, x: number, y: number, bw: number, bh: number, v: number, w = W) {
    for (let j = y; j < y + bh; j++) for (let i = x; i < x + bw; i++) f[j * w + i] = v;
}
// a "clock" that toggles a block of digits every second (changes in every consecutive pair)
function clock(f: Uint8Array, t: number, y = 3) {
    for (let k = 0; k < 6; k++) rect(f, 8 + k * 4, y, 3, 6, (t + k) % 2 ? 230 : 20);
}
/** warmed-up gate on a static scene with an optional ticking clock */
function warm(g: PixelGate, withClock: boolean, from = 0) {
    for (let t = from; t < from + 30; t++) { const f = scene(); if (withClock) clock(f, t); g.push(f, t * 1000); }
}

test('first frame opens; static scene then closes after learning', () => {
    const g = new PixelGate();
    const r = g.push(scene(), 0);
    assert.deepEqual([r.open, r.reason], [true, 'first']);
    warm(g, false, 1);
    for (let t = 40; t < 60; t++) { const r2 = g.push(scene(), t * 1000); assert.deepEqual([r2.open, r2.reason], [false, 'closed']); }
    assert.equal(g.masked, 0);
});

test('learning frames are open with reason learning', () => {
    const g = new PixelGate();
    g.push(scene(), 0);
    const r = g.push(scene(), 1000);
    assert.deepEqual([r.open, r.reason], [true, 'learning']);
});

test('a moving 6x10 blob opens', () => {
    const g = new PixelGate(); warm(g, false);
    g.push(scene(), 0);
    const f = scene(); rect(f, 50, 30, 6, 10, 220);
    const r = g.push(f, 1000);
    assert.equal(r.open, true); assert.equal(r.reason, 'blob'); assert.ok(r.blob > 24);
    const f2 = scene(); rect(f2, 54, 30, 6, 10, 220);
    assert.equal(g.push(f2, 2000).open, true);
});

test('slow drift stays closed', () => {
    const g = new PixelGate(); warm(g, false);
    for (let t = 0; t < 40; t++) {
        const f = scene(); for (let i = 0; i < f.length; i++) f[i] += Math.floor(t / 2); // +0.5 level per second
        assert.equal(g.push(f, t * 1000).open, false, `t=${t}`);
    }
});

test('sensor noise of +-3 stays closed', () => {
    const g = new PixelGate(); warm(g, false);
    for (let t = 0; t < 60; t++) {
        const f = scene(); for (let i = 0; i < f.length; i++) f[i] += Math.floor(rnd() * 7) - 3;
        assert.equal(g.push(f, t * 1000).open, false, `t=${t}`);
    }
});

test('ticking clock in the top band is learned and ignored; a real blob in the middle still opens', () => {
    const g = new PixelGate();
    warm(g, true);
    assert.ok(g.masked > 0);
    for (let t = 30; t < 50; t++) { const f = scene(); clock(f, t); assert.equal(g.push(f, t * 1000).open, false, `t=${t}`); }
    const f = scene(); clock(f, 50); rect(f, 60, 30, 6, 10, 230);
    const r = g.push(f, 50000);
    assert.equal(r.open, true); assert.equal(r.reason, 'blob');
});

test('clock in the bottom band is learned too', () => {
    const g = new PixelGate();
    for (let t = 0; t < 30; t++) { const f = scene(); clock(f, t, H - 10); g.push(f, t); }
    assert.ok(g.masked > 0);
    for (let t = 30; t < 40; t++) { const f = scene(); clock(f, t, H - 10); assert.equal(g.push(f, t).open, false); }
});

test('a blob in the top band where the clock is NOT ticking still opens', () => {
    const g = new PixelGate();
    warm(g, true);
    g.push((() => { const f = scene(); clock(f, 30); return f; })(), 0);
    const f = scene(); clock(f, 31); rect(f, 90, 4, 6, 10, 235); // far from the clock digits
    const r = g.push(f, 1000);
    assert.equal(r.open, true); assert.equal(r.reason, 'blob');
});

test('without learning (no mask) a ticking clock keeps the gate open', () => {
    const g = new PixelGate({ warmupFrames: 0 });
    g.push(scene(), 0);
    let open = 0;
    for (let t = 1; t < 10; t++) { const f = scene(); clock(f, t); if (g.push(f, t).open) open++; }
    assert.equal(open, 9);
});

test('global brightness jump returns global; small jumps do not', () => {
    const g = new PixelGate(); warm(g, false);
    g.push(scene(), 0);
    const f = scene(); for (let i = 0; i < f.length; i++) f[i] += 40;
    const r = g.push(f, 1000);
    assert.deepEqual([r.open, r.reason], [true, 'global']);
    // jump of exactly 24 is not "more than 24 grey levels": it is a fraction trigger, not global
    const g2 = new PixelGate(); warm(g2, false);
    g2.push(scene(), 0);
    const f2 = scene(); for (let i = 0; i < f2.length; i++) f2[i] += 24;
    const r2 = g2.push(f2, 1000); assert.equal(r2.open, true); assert.notEqual(r2.reason, 'global');
});

test('40% / 24-level rule', () => {
    const run = (rows: number, level: number) => {
        const g = new PixelGate(); warm(g, false); g.push(scene(), 0);
        const f = scene();
        for (let y = 10; y < 10 + rows; y++) for (let x = 0; x < W; x++) f[y * W + x] += level;
        return g.push(f, 1000);
    };
    assert.equal(run(36, 60).reason, 'global'); // 50% of the frame, +60
    assert.equal(run(32, 30).reason, 'global'); // ~41.7% after blur edges, +30
    assert.notEqual(run(28, 30).reason, 'global'); // below 40%: below 40%
    assert.notEqual(run(36, 24).reason, 'global'); // not MORE than 24 levels
    assert.equal(run(36, 24).open, true);
});

test('reset: next frame is first, mask is kept', () => {
    const g = new PixelGate(); warm(g, true);
    const m = g.masked; assert.ok(m > 0);
    g.reset();
    const f = scene(); clock(f, 99);
    assert.equal(g.push(f, 0).reason, 'first');
    const f2 = scene(); clock(f2, 100);
    assert.equal(g.push(f2, 1000).open, false);
    assert.equal(g.masked, m);
    g.relearn(); assert.equal(g.masked, 0);
});

test('other frame sizes work and short frames throw', () => {
    const w = 320, h = 180;
    const g = new PixelGate({ width: w, height: h });
    warm2(g);
    const f = scene(w, h); rect(f, 100, 80, 12, 20, 230, w);
    assert.equal(g.push(f, 0).reason, 'blob');
    assert.throws(() => g.push(new Uint8Array(10), 0));
    function warm2(gg: PixelGate) { for (let t = 0; t < 31; t++) gg.push(scene(w, h), t); }
});

test('benchmark (informational)', () => {
    const g = new PixelGate(); warm(g, false);
    const a = scene(), b = scene(); rect(b, 50, 30, 30, 20, 220);
    const N = 2000; const t0 = process.hrtime.bigint();
    for (let i = 0; i < N; i++) g.push(i % 2 ? a : b, i);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;
    console.log(`gate: ${ms.toFixed(4)} ms/frame`);
    assert.ok(ms < 5);
});
