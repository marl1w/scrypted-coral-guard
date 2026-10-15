import test from 'node:test';
import assert from 'node:assert/strict';
import { loadProvider } from './harness';

const MARKER = 'scrypted-coral-guard:mixin';
const camIfaces = ['VideoCamera', 'MotionSensor'];

test('canMixin: camera and doorbell with VideoCamera + MotionSensor get MotionSensor, Settings and our unique marker', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    for (const type of ['Camera', 'Doorbell']) {
        const r = await p.canMixin(type, camIfaces);
        assert.deepEqual(r, ['MotionSensor', 'OccupancySensor', 'Settings', `${MARKER}:prov1`]);
    }
    clearInterval((p as any).healthTimer);
});

test('M25: other device types and cameras without MotionSensor or VideoCamera are refused', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    assert.equal(await p.canMixin('Light', camIfaces), undefined);
    assert.ok(await p.canMixin('Camera', ['VideoCamera']), 'a camera without a MotionSensor is accepted (sensorless)');
    assert.ok((await p.canMixin('Camera', ['VideoCamera']) as string[]).includes('MotionSensor'));
    assert.ok((await p.canMixin('Camera', ['VideoCamera']) as string[]).includes('OccupancySensor'), 'sensorless cameras gain OccupancySensor too');
    assert.equal(await p.canMixin('Camera', ['MotionSensor']), undefined);
    clearInterval(p.healthTimer);
});

test('H4: our own marker does not hide us (an enabled extension stays listed), another provider\'s marker does', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    assert.ok(await p.canMixin('Camera', [...camIfaces, `${MARKER}:prov1`]));
    assert.equal(await p.canMixin('Camera', [...camIfaces, `${MARKER}:other`]), undefined);
    clearInterval(p.healthTimer);
});

test('L10: findDuplicate reports another mixin using the same service camera id', () => {
    const P = loadProvider();
    const p: any = new P('n');
    const f = { onHealth() { }, applyConfig() { } };
    const a = { ...f, id: '1', verifierId: 'cam', name: 'A' }, b = { ...f, id: '2', verifierId: 'cam', name: 'B' }, c = { ...f, id: '3', verifierId: 'other', name: 'C' };
    const a2 = { ...f, id: '2', verifierId: 'cam', name: 'B' };
    p.register(a); p.register(c);
    assert.equal(p.findDuplicate(b), 'A (device 1)');
    assert.equal(p.findDuplicate(c), undefined);
    p.active.delete(a); p.register(a2);
    assert.equal(p.findDuplicate(b), undefined, 'two instances of the same device are not a duplicate camera');
    clearInterval(p.healthTimer);
});

test('VAAPI slots: by camera id, idempotent, at most N, released by the owner only (restart, fallback, unregister)', () => {
    const P = loadProvider();
    const p: any = new P('n');
    p.checkHealth = async () => { };
    const cam = (id: string) => ({ id });
    const a = cam('a'), b = cam('b'), c = cam('c'), a2 = cam('a');
    assert.equal(p.acquireHw(a, 2), true);
    assert.equal(p.acquireHw(a, 2), true, 'a second start of the same camera takes no second slot');
    assert.equal(p.hwSlotsUsed, 1);
    assert.equal(p.acquireHw(b, 2), true);
    assert.equal(p.acquireHw(c, 2), false, 'full');
    assert.equal(p.acquireHw(a2, 2), true, 'a new instance of camera a takes over its slot');
    p.releaseHw(a); // the stale instance cannot free its successor's slot
    assert.equal(p.hwSlotsUsed, 2);
    p.releaseHw(a2); // fallback to software or stop
    assert.equal(p.acquireHw(c, 2), true);
    p.unregister(b);
    assert.equal(p.hwSlotsUsed, 1);
    clearInterval(p.healthTimer);
});

test('provider: the health poll pushes no Settings event; a save does, once', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    let n = 0;
    p.onDeviceEvent = () => { n++; };
    p.client = { check: async () => ({ ok: true }) };
    await p.checkHealth();
    await p.checkHealth();
    assert.equal(n, 0);
    await p.putSetting('baseUrl', 'http://127.0.0.1:1');
    assert.equal(n, 1);
    clearInterval(p.healthTimer);
});

test('plain http with a token to a non-loopback host warns once and never blocks', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    const warns: string[] = [];
    p.console = { warn: (m: string) => warns.push(m), log() { } };
    p.storageSettings = { values: { token: 't', baseUrl: 'http://192.0.2.7:8750', killSwitch: false } };
    p.client = { check: async () => ({ ok: true }) };
    await p.checkHealth();
    await p.checkHealth();
    assert.equal(warns.filter(w => /plain http/.test(w)).length, 1);
    clearInterval(p.healthTimer);
});

test('detection thresholds: defaults, clamped to 0.05-0.95, an inverted pair falls back to the defaults with a one-time warning', () => {
    const { resolveThresholds, DEFAULT_THRESHOLDS } = require('../src/thresholds') as typeof import('../src/thresholds');
    assert.deepEqual(resolveThresholds({}).t, { personConfirm: 0.65, personKeep: 0.2, vehicleDetect: 0.4, vehicleConfirm: 0.6, animalDetect: 0.3, animalConfirm: 0.6 });
    const c = resolveThresholds({ personConfirm: 1.5, personKeep: 0, vehicleConfirm: 'x' });
    assert.equal(c.t.personConfirm, 0.95); assert.equal(c.t.personKeep, 0.05); assert.equal(c.t.vehicleConfirm, 0.6);
    const bad = resolveThresholds({ animalConfirm: 0.2, animalDetect: 0.5, personConfirm: 0.7 });
    assert.equal(bad.t.animalConfirm, DEFAULT_THRESHOLDS.animalConfirm); assert.equal(bad.t.animalDetect, DEFAULT_THRESHOLDS.animalDetect);
    assert.equal(bad.t.personConfirm, 0.7); assert.equal(bad.invalid.length, 1);
    const P = loadProvider();
    const p: any = new P('n');
    const warns: string[] = [];
    p.console = { warn: (m: string) => warns.push(m), log() { } };
    p.storageSettings = { values: { vehicleConfirm: 0.3, vehicleDetect: 0.5 } };
    assert.equal(p.thresholds.vehicleConfirm, 0.6);
    assert.equal(p.thresholds.vehicleDetect, 0.4);
    assert.equal(warns.length, 1);
    clearInterval(p.healthTimer);
});

test('a service restart (uptime went down) pushes every camera configuration again', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    p.console = { warn() { }, log() { } };
    let pushed = 0;
    p.register({ postConfig: () => { pushed++; }, onHealth() { } });
    let up = 100;
    p.client = { check: async () => ({ ok: true, uptime_s: up }) };
    await p.checkHealth(); up = 130; await p.checkHealth();
    assert.equal(pushed, 0);
    up = 5; await p.checkHealth();
    assert.equal(pushed, 1);
    clearInterval(p.healthTimer);
});

test('Second look: Detection group, on by default, sent with every config, a change re-posts all cameras', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    clearInterval(p.healthTimer);
    p.console = { log() { }, warn() { } };
    assert.equal(p.secondLook, true);
    const st = await p.getSettings();
    const s = st.find((x: any) => x.key === 'secondLook');
    assert.equal(s.group, 'Detection');
    assert.ok(!st.some((x: any) => x.group === 'Detection thresholds'));
    let pushed = 0;
    p.register({ postConfig: () => { pushed++; }, applyConfig() { }, onHealth() { } });
    p.register({ postConfig: () => { pushed++; }, applyConfig() { }, onHealth() { } });
    p.thresholdsChanged(); // what the setting's onPut runs
    p.storageSettings = { values: { secondLook: false } };
    assert.equal(pushed, 2);
    assert.equal(p.secondLook, false);
});

test('one live mixin per device: a re-created chain replaces (and fully releases) the previous instance; a late releaseMixin of the old one is harmless', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    clearInterval(p.healthTimer);
    const logs: string[] = [];
    p.console = { log: (m: string) => logs.push(m), warn: (m: string) => logs.push(m) };
    p.client = { check: async () => ({ ok: true }), config: async () => ({}), frame: async () => null, reset: async () => ({}) };
    const a = await p.getMixin({}, ['VideoCamera', 'MotionSensor'], { id: 'cam1' });
    assert.equal(p.active.size, 1);
    const b = await p.getMixin({}, ['VideoCamera', 'MotionSensor'], { id: 'cam1' });
    assert.notEqual(a, b);
    assert.equal(a.released, true, 'the previous instance is released first');
    assert.ok(logs.some(l => /replacing previous instance/.test(l)));
    assert.equal(p.active.size, 1); assert.ok(p.active.has(b));
    assert.equal(p.isLive(b), true); assert.equal(p.isLive(a), false);
    await p.releaseMixin('cam1', a); // Scrypted releasing the old chain later
    assert.equal(p.isLive(b), true, 'the live one stays');
    assert.equal(p.findDuplicate(b), undefined, 'same device id is never a duplicate');
    await p.releaseMixin('cam1', b);
    assert.equal(p.active.size, 0);
    await b.release(); // idempotent
});

test('Release motion after still: default 170 s, clamped 0..180', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    assert.equal(p.stillReleaseS, 170);
    const real = p.storageSettings;
    for (const [v, want] of [[0, 0], [60, 60], [500, 180], [-5, 0], ['', 170]] as const) {
        p.storageSettings = { values: { stillReleaseS: v } };
        assert.equal(p.stillReleaseS, want, String(v));
    }
    p.storageSettings = real;
    clearInterval(p.healthTimer);
});
