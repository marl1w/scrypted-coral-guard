import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TamperChange, TamperDetector } from '../src/tamper';
import { DEFAULTS } from '../src/defaults';

const W = 128, H = 72;
/** a wider synthetic world, so a shifted view brings in new content instead of wrapping */
const WW = 192;

function rng(seed: number) {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

/** textured outdoor-like scene: a vertical gradient (sky to ground), 4x4 block noise and a few bright rectangles */
function world(seed = 1): Float32Array {
    const r = rng(seed), out = new Float32Array(WW * H);
    const blocks = new Float32Array((WW / 4) * (H / 4)).map(() => r() * 80 - 40);
    for (let y = 0; y < H; y++) for (let x = 0; x < WW; x++)
        out[y * WW + x] = 70 + y * 1.2 + blocks[(y >> 2) * (WW / 4) + (x >> 2)];
    for (let k = 0; k < 8; k++) {
        const x0 = Math.floor(r() * (WW - 20)), y0 = Math.floor(r() * (H - 12)), v = r() * 120 - 60;
        for (let y = y0; y < y0 + 10; y++) for (let x = x0; x < x0 + 16; x++) out[y * WW + x] += v;
    }
    return out;
}
const WORLD = world();

function view(dx = 0, gain = 1, offset = 0, src = WORLD): Uint8Array {
    const f = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        f[y * W + x] = Math.max(0, Math.min(255, Math.round(src[y * WW + x + dx] * gain + offset)));
    return f;
}

function boxBlur(f: Uint8Array, r: number): Uint8Array {
    const out = new Uint8Array(f.length);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        let s = 0, n = 0;
        for (let yy = Math.max(0, y - r); yy <= Math.min(H - 1, y + r); yy++)
            for (let xx = Math.max(0, x - r); xx <= Math.min(W - 1, x + r); xx++) { s += f[yy * W + xx]; n++; }
        out[y * W + x] = Math.round(s / n);
    }
    return out;
}

/** an IR-like picture: darker, the texture contrasts partly different (a mix with another scene) */
function irView(a: number, b: number): Uint8Array {
    const other = world(5), f = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) f[y * W + x] = Math.max(0, Math.min(255, Math.round(a * WORLD[y * WW + x] + b * other[y * WW + x])));
    return f;
}

/** similarity of a steady frame to the plain view's reference */
function similarity(f: Uint8Array): number {
    const d = new TamperDetector();
    feed(d, () => view(), 0, 30_000);
    feed(d, () => f, 30_000, 4000);                  // the median switches at the second sample: the first judged one
    return d.features!.ncc;
}

const flat = (v: number) => new Uint8Array(W * H).fill(v);

/** gray frames at 1 fps (as the ffmpeg branch delivers them) from `from` for `ms`; returns every change with its time */
function feed(d: TamperDetector, frame: (t: number) => Uint8Array, from: number, ms: number): { t: number; c: TamperChange }[] {
    const out: { t: number; c: TamperChange }[] = [];
    for (let t = from; t < from + ms; t += 1000) {
        const c = d.push(frame(t), t);
        if (c) out.push({ t, c });
    }
    return out;
}

function learned(): TamperDetector {
    const d = new TamperDetector();
    assert.deepEqual(feed(d, () => view(), 0, 60_000), []);
    assert.equal(d.state.learning, false);
    assert.equal(d.state.on, false);
    return d;
}

test('tamper: the reference is learnt from the first three samples, nothing is judged before', () => {
    const d = new TamperDetector();
    d.push(view(), 0); d.push(view(), 2000);
    assert.equal(d.state.learning, true);
    d.push(view(), 4000);
    assert.equal(d.state.learning, false);
});

test('tamper: a flat, black or blown-out view is never learnt (camera covered at start)', () => {
    const d = new TamperDetector();
    feed(d, () => flat(0), 0, 60_000);
    assert.equal(d.state.learning, true);
    assert.equal(d.state.flat, true, 'learning: view looks flat');
    feed(d, () => flat(250), 60_000, 20_000);
    assert.equal(d.state.flat, true);
    feed(d, () => view(), 80_000, 10_000);
    assert.equal(d.state.learning, false, 'the uncovered view is learnt');
    assert.equal(d.state.flat, false);
});

test('tamper: the thresholds come from DEFAULTS.tamper', () => {
    const d = new TamperDetector();
    assert.equal(d.opts.onMs, DEFAULTS.tamper.onMs);
    assert.equal(d.opts.movedNcc, DEFAULTS.tamper.movedNcc);
    assert.equal(new TamperDetector({ onMs: 3000 }).opts.onMs, 3000);
});

test('tamper: the same scene with sensor noise stays clean', () => {
    const d = learned();
    const r = rng(7);
    assert.deepEqual(feed(d, () => view().map(v => Math.max(0, Math.min(255, v + Math.round(r() * 10 - 5)))), 60_000, 300_000), []);
    assert.ok(d.features!.ncc > 0.8, `ncc ${d.features!.ncc}`);
});

test('tamper: covered (black) turns on after the persistence and off again once uncovered', () => {
    const d = learned();
    const on = feed(d, () => flat(0), 60_000, 30_000);
    assert.equal(on.length, 1);
    assert.equal(on[0].c.on, true);
    assert.equal(on[0].c.reason, 'covered');
    // the median sees the change at the second sample (t+2 s); then onMs (10 s) of persistence
    assert.ok(on[0].t >= 60_000 + DEFAULTS.tamper.onMs, `on at +${on[0].t - 60_000} ms`);
    assert.ok(on[0].t <= 60_000 + DEFAULTS.tamper.onMs + 6000, `on at +${on[0].t - 60_000} ms`);
    assert.equal(d.state.reason, 'covered');
    const off = feed(d, () => view(), 90_000, 30_000);
    assert.equal(off.length, 1);
    assert.equal(off[0].c.on, false);
    assert.ok(off[0].t >= 90_000 + DEFAULTS.tamper.offMs && off[0].t <= 90_000 + DEFAULTS.tamper.offMs + 6000, `off at +${off[0].t - 90_000} ms`);
    assert.equal(d.state.on, false);
});

test('tamper: covered by a bright flat picture (IR glare off something on the lens)', () => {
    const d = learned();
    const on = feed(d, () => flat(250), 60_000, 30_000);
    assert.equal(on[0]?.c.reason, 'covered');
});

test('tamper: blurred (box blur) is reported as blurred', () => {
    const d = learned();
    const blurred = boxBlur(view(), 4);
    const on = feed(d, () => blurred, 60_000, 30_000);
    assert.equal(on.length, 1, JSON.stringify(d.features));
    assert.equal(on[0].c.reason, 'blurred');
});

test('tamper: moved (view shifted by 20 %) is reported as moved', () => {
    const d = learned();
    const shifted = view(Math.round(W * 0.2));
    const on = feed(d, () => shifted, 60_000, 30_000);
    assert.equal(on.length, 1, JSON.stringify(d.features));
    assert.equal(on[0].c.reason, 'moved');
    assert.ok(d.features!.ncc < DEFAULTS.tamper.movedNcc);
});

test('tamper: a day/night (IR) brightness step with the structure kept is not tamper, the reference is re-learnt', () => {
    const d = learned();
    const night = view(0, 0.5, -10);                  // darker and lower contrast, same structure
    assert.deepEqual(feed(d, () => night, 60_000, 120_000), []);
    assert.equal(d.state.on, false);
    assert.ok(Math.abs(d.reference!.mean - d.features!.mean) < 2, `reference mean ${d.reference!.mean}, night mean ${d.features!.mean}`);
    // re-learnt: the night view is the reference now, so covering it is still caught
    assert.equal(feed(d, () => flat(0), 180_000, 30_000)[0]?.c.reason, 'covered');
});

test('tamper: a gain-only change (contrast) is not tamper', () => {
    const d = learned();
    assert.deepEqual(feed(d, () => view(0, 1.3, -30), 60_000, 120_000), []);
});

test('tamper: slow dusk (brightness fading over 20 min) is followed by the reference', () => {
    const d = learned();
    const g = (t: number) => Math.max(0.35, 1 - (t - 60_000) / 1_200_000 * 0.65);
    assert.deepEqual(feed(d, t => view(0, g(t), 0), 60_000, 1_500_000), []);
});

test('tamper: wind shake (the view jitters by a couple of pixels) is not moved', () => {
    const d = learned();
    assert.deepEqual(feed(d, t => view(((t / 1000) % 5) - 2 + 2), 60_000, 300_000), []);
});

test('tamper: a brief occluder (someone close for 6 s) is not tamper', () => {
    const d = learned();
    const occluded = view();
    for (let y = 0; y < H; y++) for (let x = 20; x < 100; x++) occluded[y * W + x] = 40;
    assert.deepEqual(feed(d, t => (t >= 60_000 && t < 66_000 ? occluded : view()), 60_000, 120_000), []);
    // a full black frame for 6 s as well (a hand passing over the lens)
    assert.deepEqual(feed(d, t => (t >= 180_000 && t < 186_000 ? flat(0) : view()), 180_000, 120_000), []);
});

test('tamper: headlights and a flash (brief saturation) are not tamper', () => {
    const d = learned();
    assert.deepEqual(feed(d, t => (t % 20_000 < 3000 ? flat(255) : view()), 60_000, 200_000), []);
    assert.deepEqual(feed(d, t => (t % 30_000 < 4000 ? view(0, 1.6, 60) : view()), 260_000, 200_000), []);
});

test('tamper: rain or snow (random bright specks every frame) is not tamper', () => {
    const d = learned();
    const r = rng(3);
    const snowy = () => { const f = view(); for (let i = 0; i < f.length * 0.08; i++) f[Math.floor(r() * f.length)] = 230; return f; };
    assert.deepEqual(feed(d, snowy, 60_000, 600_000), []);
});

test('tamper: a condition that comes and goes inside the persistence never turns the sensor on', () => {
    const d = learned();
    // 8 s covered, 8 s clear, repeated: each run is shorter than onMs
    assert.deepEqual(feed(d, t => ((t - 60_000) % 16_000 < 8000 ? flat(0) : view()), 60_000, 160_000), []);
});

test('tamper: the reference is frozen while tampered, so a covered camera stays on (a real cover is never accepted)', () => {
    const d = learned();
    const changes = feed(d, () => flat(0), 60_000, 3 * 3_600_000);
    assert.equal(changes.length, 1);
    assert.equal(d.state.on, true);
    assert.equal(d.state.reason, 'covered');
});

test('tamper: turned AND brightened is moved, never a lighting re-learn', () => {
    const d = learned();
    const on = feed(d, () => view(Math.round(W * 0.2), 1.3, 40), 60_000, 30_000);
    assert.equal(on[0]?.c.reason, 'moved', JSON.stringify(d.features));
    assert.equal(d.state.on, true);
});

/** the plain view with columns [x0, x1) replaced by fill(x, y) */
function patched(x0: number, x1: number, fill: (x: number, y: number, v: number) => number, base = view()): Uint8Array {
    const f = Uint8Array.from(base);
    for (let y = 0; y < H; y++) for (let x = x0; x < x1; x++) f[y * W + x] = Math.max(0, Math.min(255, Math.round(fill(x, y, f[y * W + x]))));
    return f;
}

test('tamper: part of the lens covered (left half taped black, left third white) raises nothing and is never learnt', () => {
    for (const covered of [patched(0, W / 2, () => 0), patched(0, Math.round(W / 3), () => 255)]) {
        const d = learned();
        const before = d.reference;
        assert.deepEqual(feed(d, () => covered, 60_000, 1_800_000), [], JSON.stringify(d.features));
        assert.equal(d.state.condition, 'changed');
        assert.deepEqual(d.reference, before, 'the damaged view is not learnt');
    }
});

test('tamper: a smooth panel filling 30 % of the view (a van close by) for 30 min raises nothing', () => {
    const d = learned();
    const van = patched(Math.round(W * 0.7), W, (x, y) => 150 + x * 0.1 + y * 0.05);
    assert.deepEqual(feed(d, () => van, 60_000, 1_800_000), [], JSON.stringify(d.features));
    assert.equal(d.state.on, false);
});

test('tamper: half of the lens sprayed for 30 min raises nothing and the damaged view is not learnt', () => {
    const d = learned();
    const before = d.reference;
    const blurred = boxBlur(view(), 6);
    const sprayed = patched(0, W / 2, (x, y) => 0.4 * blurred[y * W + x] + 0.6 * 170);
    assert.deepEqual(feed(d, () => sprayed, 60_000, 1_800_000), [], JSON.stringify(d.features));
    assert.deepEqual(d.reference, before, 'the reference is kept');
    // cleaned: back to normal against the same reference
    assert.deepEqual(feed(d, () => view(), 1_860_000, 60_000), []);
    assert.equal(d.state.condition, undefined);
});

test('tamper: IR on with 30 % of the view black (out of IR range) raises nothing, on the night profile', () => {
    const d = new TamperDetector();
    d.setProfile('default|day', 0);
    feed(d, () => view(), 0, 60_000);
    const night = patched(Math.round(W * 0.7), W, () => 0, view(0, 0.6, 10));
    d.setProfile('default|night', 60_000);
    assert.deepEqual(feed(d, () => night, 60_000, 1_200_000), [], JSON.stringify(d.features));
    assert.equal(d.state.profile, 'default|night');
    assert.equal(d.state.learning, false);
    d.setProfile('default|day', 1_260_000);
    assert.deepEqual(feed(d, () => view(), 1_260_000, 120_000), []);
});

test('tamper: day/night flapping (a hand over the lens toggles the IR) counts as covered', () => {
    const d = new TamperDetector();
    d.setProfile('s|day', 0);
    feed(d, () => view(), 0, 30_000);
    d.setProfile('s|night', 30_000);
    feed(d, () => view(0, 0.5, 0), 30_000, 30_000);
    const changes: { t: number; c: TamperChange }[] = [];
    for (let t = 60_000; t < 180_000; t += 1000) {
        if (t % 8000 === 0) d.setProfile(t % 16_000 ? 's|day' : 's|night', t);
        const c = d.push(t % 16_000 < 8000 ? flat(0) : flat(250), t);
        if (c) changes.push({ t, c });
    }
    assert.equal(changes[0]?.c.reason, 'covered');
    assert.match(changes[0].c.detail, /day\/night switched/);
});

test('tamper: a pending cover survives a day/night switch (the run goes on in the new profile)', () => {
    const d = new TamperDetector();
    d.setProfile('s|day', 0);
    feed(d, () => view(), 0, 30_000);
    d.setProfile('s|night', 30_000);
    feed(d, () => view(0, 0.5, 0), 30_000, 30_000);
    d.setProfile('s|day', 60_000);
    feed(d, () => view(), 60_000, 30_000);
    const a = feed(d, () => flat(0), 90_000, 8000);     // pending, not yet on
    assert.deepEqual(a, []);
    d.setProfile('s|night', 98_000);                   // the IR switches on under the hand
    const b = feed(d, () => flat(0), 98_000, 30_000);
    assert.equal(b[0]?.c.reason, 'covered');
    assert.ok(b[0].t <= 98_000 + DEFAULTS.tamper.settleMs + 6000, `on at ${b[0].t}: the run was not restarted`);
});

test('tamper: a flat view on a profile with no reference is that profile\'s first learn: no alarm, nothing learnt', () => {
    const d = new TamperDetector();
    d.setProfile('s|day', 0);
    feed(d, () => view(), 0, 30_000);
    d.setProfile('s|night', 30_000);                   // first night after deploy: black, no IR
    assert.deepEqual(feed(d, () => flat(0), 30_000, 600_000), []);
    assert.equal(d.state.on, false);
    assert.equal(d.state.learning, true, 'nothing learnt');
    assert.equal(d.state.flat, true, 'learning: view looks flat');
    feed(d, () => view(0, 0.5, 0), 630_000, 30_000);  // the IR comes on: learnt
    assert.equal(d.state.learning, false);
});

test('tamper: accepting a moved camera drops the other profiles of that stream', () => {
    const d = new TamperDetector({ acceptMs: 300_000 });
    d.setProfile('s|night', 0);
    feed(d, () => view(0, 0.5, 0), 0, 30_000);
    d.setProfile('t|day', 30_000);
    feed(d, () => view(), 30_000, 30_000);
    d.setProfile('s|day', 60_000);
    feed(d, () => view(), 60_000, 30_000);
    const shifted = view(Math.round(W * 0.2));
    const c = feed(d, () => shifted, 90_000, 400_000);
    assert.deepEqual(c.map(x => [x.c.on, x.c.reason]), [[true, 'moved'], [false, 'moved']]);
    assert.deepEqual(Object.keys(d.snapshot()!.profiles).sort(), ['s|day', 't|day']);
});

test('tamper: profiles are kept LRU (a switch counts as use), and dropProfiles never drops the current one', () => {
    const d = new TamperDetector({ settleMs: 0, flapFlips: 100 }); // many switches on purpose
    let t = 0;
    for (let i = 0; i < 8; i++) { d.setProfile(`p${i}|any`, t); feed(d, () => view(), t, 10_000); t += 20_000; }
    d.setProfile('p0|any', t); feed(d, () => view(), t, 10_000); t += 20_000;
    d.setProfile('p8|any', t); feed(d, () => view(), t, 10_000);
    const keys = Object.keys(d.snapshot()!.profiles);
    assert.equal(keys.length, 8);
    assert.ok(keys.includes('p0|any') && !keys.includes('p1|any'), keys.join());
    d.dropProfiles(k => k.endsWith('|any'));
    assert.deepEqual(Object.keys(d.snapshot()!.profiles), ['p8|any']);
});

test('tamper: the anchor sharpness never declines when its map is refreshed', () => {
    const d = new TamperDetector({ anchorRefreshMs: 60_000, anchorKeepRatio: 0.3 });
    feed(d, () => view(), 0, 30_000);
    const a0 = d.reference!.anchorSharp;
    const soft = boxBlur(view(), 1);
    feed(d, () => soft, 30_000, 1_800_000);
    assert.ok(d.reference!.anchorSharp >= a0, `${d.reference!.anchorSharp} < ${a0}`);
});

test('tamper: unexplained samples freeze the reference, which is re-learnt only after changedRelearnMs', () => {
    const d = learned();
    const before = d.reference!;
    // a different scene in the right half (something big parked in front): not tamper, not explained either
    const other = world(9);
    const changed = view();
    for (let y = 0; y < H; y++) for (let x = W / 2; x < W; x++) changed[y * W + x] = Math.max(0, Math.min(255, Math.round(other[y * WW + x])));
    assert.deepEqual(feed(d, () => changed, 60_000, 300_000), []);
    assert.equal(d.state.condition, 'changed', JSON.stringify(d.features));
    assert.deepEqual(d.reference, before, 'frozen');
    assert.deepEqual(feed(d, () => changed, 360_000, DEFAULTS.tamper.changedRelearnMs), []);
    assert.notDeepEqual(d.reference, before, 're-learnt after a long unexplained change');
    assert.equal(d.state.condition, undefined);
});

test('tamper: a slow blur (radius 0 to 4 over 5 h) is caught as blurred against the anchor', () => {
    const d = learned();
    const steps = [0, 1, 2, 3, 4].map(r => (r ? boxBlur(view(), r) : view()));
    const at = (t: number) => {
        const r = Math.min(4, (t - 60_000) / (5 * 3_600_000) * 4), i = Math.floor(r), k = r - i;
        const a = steps[i], b = steps[Math.min(4, i + 1)], f = new Uint8Array(a.length);
        for (let j = 0; j < f.length; j++) f[j] = Math.round(a[j] * (1 - k) + b[j] * k);
        return f;
    };
    const changes: { t: number; c: TamperChange }[] = [];
    for (let t = 60_000; t < 60_000 + 5 * 3_600_000 + 60_000; t += 2000) {
        const c = d.push(at(t), t);
        if (c) changes.push({ t, c });
    }
    assert.equal(changes[0]?.c.reason, 'blurred', JSON.stringify({ f: d.features, r: d.reference }));
});

test('tamper: fog (contrast gone, structure kept) that holds for acceptMs is accepted', () => {
    const d = new TamperDetector({ acceptMs: 600_000 });
    feed(d, () => view(), 0, 60_000);
    const fog = view(0, 0.25, 90);
    const changes = feed(d, () => fog, 60_000, 700_000);
    assert.deepEqual(changes.map(c => c.c.on), [true, false], JSON.stringify(d.features));
    assert.match(changes[1].c.detail, /view accepted/);
});

test('tamper: profiles: day and night keep their own reference, the IR switch never alarms', () => {
    const d = new TamperDetector();
    d.setProfile('default|day', 0);
    feed(d, () => view(), 0, 60_000);
    const day = d.reference!;
    // the IR picture: darker, other texture contrasts (a mix with another scene); without profiles this is not explained
    const ir = irView(0.15, 0.45);                    // a hard switch: on its own it would read as moved
    assert.ok(similarity(ir) < DEFAULTS.tamper.movedNcc, `similarity ${similarity(ir)}`);
    let t = 60_000;
    for (let k = 0; k < 4; k++) {
        d.setProfile('default|night', t);
        assert.deepEqual(feed(d, () => ir, t, 600_000), []);
        t += 600_000;
        assert.equal(d.state.profile, 'default|night');
        assert.ok(Math.abs(d.reference!.mean - d.features!.mean) < 3, 'night has its own reference');
        d.setProfile('default|day', t);
        assert.deepEqual(feed(d, () => view(), t, 600_000), []);
        t += 600_000;
        assert.ok(Math.abs(d.reference!.mean - day.mean) < 3, 'day kept its reference');
    }
    // covering at night is still caught against the night reference
    d.setProfile('default|night', t);
    feed(d, () => ir, t, 60_000);
    assert.equal(feed(d, () => flat(0), t + 60_000, 30_000)[0]?.c.reason, 'covered');
});

test('tamper: without profiles the same IR switch is held as not explained, then re-learnt, never an alarm', () => {
    const d = learned();
    const ir = irView(0.33, 0.27);                     // a moderate switch: neither moved nor a clean lighting step
    const sim = similarity(ir);
    assert.ok(sim >= DEFAULTS.tamper.movedNcc + 0.05 && sim < DEFAULTS.tamper.relearnNcc, `similarity ${sim}`);
    assert.deepEqual(feed(d, () => ir, 60_000, 300_000), []);
    assert.equal(d.state.condition, 'changed');
    assert.deepEqual(feed(d, () => ir, 360_000, 900_000), []);
    assert.equal(d.state.condition, undefined, 're-learnt');
});

test('tamper: a profile switch while tampered keeps the sensor on and learns nothing until the accept timeout', () => {
    const d = new TamperDetector({ acceptMs: 600_000 });
    d.setProfile('a|any', 0);
    feed(d, () => view(), 0, 60_000);
    const shifted = view(Math.round(W * 0.2));
    feed(d, () => shifted, 60_000, 30_000);
    assert.equal(d.state.on, true);
    d.setProfile('b|any', 90_000);                    // another stream
    assert.deepEqual(feed(d, () => shifted, 90_000, 300_000), []);
    assert.equal(d.state.on, true);
    assert.equal(d.state.learning, true);
    const c = feed(d, () => shifted, 390_000, 400_000);
    assert.deepEqual(c.map(x => x.c.on), [false]);
    assert.equal(d.state.learning, false);
});

test('tamper: snapshot and restore keep every profile and the state; dirty only after a change', () => {
    const d = new TamperDetector();
    d.setProfile('s|day', 0);
    feed(d, () => view(), 0, 30_000);
    d.setProfile('s|night', 30_000);
    feed(d, () => view(0, 0.5, 0), 30_000, 30_000);
    feed(d, () => flat(0), 60_000, 30_000);
    assert.equal(d.state.on, true);
    assert.equal(d.takeDirty(), true);
    feed(d, () => flat(0), 90_000, 30_000);
    assert.equal(d.dirty, false, 'nothing changes while a covered camera stays on');
    const snap = JSON.parse(JSON.stringify({ ...d.snapshot(), sinceWall: 1234 }));
    assert.deepEqual(Object.keys(snap.profiles), ['s|day', 's|night']);
    const e = new TamperDetector();
    assert.equal(e.restore(snap, 0), true);
    e.setProfile('s|night', 0);
    assert.equal(e.state.on, true);
    assert.equal(e.state.reason, 'covered');
    assert.equal(e.state.learning, false);
    assert.equal(new TamperDetector().restore({ v: 2, profiles: { x: { ref: { mean: 1 } } } } as any, 0), false);
});

test('tamper: a re-aimed camera is accepted as the new view after movedAcceptMs', () => {
    const d = new TamperDetector({ acceptMs: 600_000 });
    feed(d, () => view(), 0, 60_000);
    const shifted = view(Math.round(W * 0.2));
    const changes = feed(d, () => shifted, 60_000, 700_000);
    assert.deepEqual(changes.map(c => [c.c.on, c.c.reason]), [[true, 'moved'], [false, 'moved']]);
    assert.match(changes[1].c.detail, /view accepted/);
    assert.deepEqual(feed(d, () => shifted, 760_000, 120_000), []); // the new view is the reference
});

test('tamper: relearn() forgets the reference and turns the sensor off', () => {
    const d = learned();
    feed(d, () => flat(0), 60_000, 30_000);
    assert.equal(d.state.on, true);
    const c = d.relearn();
    assert.equal(c?.on, false);
    assert.equal(d.state.learning, true);
});

test('tamper: a gap in the frames (stream restart) restarts a pending condition', () => {
    const d = learned();
    feed(d, () => flat(0), 60_000, 8000);             // pending, not yet on
    assert.equal(d.state.on, false);
    // 30 s without frames, then the scene again: nothing turns on
    assert.deepEqual(feed(d, () => view(), 98_000, 60_000), []);
});

test('tamper: frames closer than sampleMs are not sampled', () => {
    const d = new TamperDetector();
    for (let t = 0; t < 4000; t += 500) d.push(view(), t);
    assert.equal(d.state.learning, true);              // samples at 0 and 2000 only
    d.push(view(), 4000);
    assert.equal(d.state.learning, false);
});
