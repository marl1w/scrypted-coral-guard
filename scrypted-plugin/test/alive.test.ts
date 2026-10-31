import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { ctl, J, make, releaseAll, sleep } from './harness';

afterEach(() => releaseAll());

const AI = ['VideoCamera', 'MotionSensor', 'ObjectDetector'];
const FAST = { frameProcess: 'Keep warm', aliveBeatMs: 60, aliveBeatJitterMs: 0, aliveFrameMs: 200 };
const beats = () => ctl.detections.filter(d => d.sourceId === 'prov-test' && Array.isArray(d.detections) && d.detections.length === 0);
async function eventually(cond: () => boolean, ms = 3000, what = 'condition') {
    const t0 = Date.now();
    while (!cond()) {
        if (Date.now() - t0 > ms) assert.fail(`timeout waiting for ${what}`);
        await sleep(10);
    }
}
function cam(settings: Record<string, any> = {}, interfaces = AI) {
    const h = make({ settings: { mode: 'Gate', ...FAST, ...settings }, interfaces });
    h.plugin.id = 'prov-test';
    return h;
}
/** keeps the warm stream delivering frames until stopped */
async function flowing() {
    await eventually(() => ctl.starts.length >= 1, 3000, 'warm stream');
    const s = ctl.starts[ctl.starts.length - 1];
    s.o.onFrame(J, Date.now());
    const t = setInterval(() => s.o.onFrame(J, Date.now()), 30);
    return () => clearInterval(t);
}

test('alive: beat delay is the interval offset per camera within ±jitter, stable for a camera', () => {
    make(); // loads the mixin module with the stubbed SDK
    const { aliveBeatDelay } = require('../src/mixin') as typeof import('../src/mixin');
    for (const id of ['1', '2', '17', '123', 'cam-front']) {
        const d = aliveBeatDelay(id, 60000, 10000);
        assert.ok(d >= 50000 && d <= 70000, `${id}: ${d}`);
        assert.equal(aliveBeatDelay(id, 60000, 10000), d);
    }
    assert.ok(new Set(['1', '2', '3', '4', '5', '6'].map(id => aliveBeatDelay(id, 60000, 10000))).size > 1, 'cameras spread');
    assert.equal(aliveBeatDelay('1', 60000, 0), 60000);
});

test('alive: Gate with frames flowing and the service up emits empty ObjectsDetected beats with a timestamp and our sourceId', async () => {
    cam();
    const stop = await flowing();
    try {
        await eventually(() => beats().length >= 2, 3000, 'two beats');
        const b = beats()[0];
        assert.deepEqual(b.detections, []);
        assert.equal(b.sourceId, 'prov-test');
        assert.ok(Math.abs(b.timestamp - Date.now()) < 5000);
        assert.equal(ctl.detections.filter(d => d.detections?.length).length, 0, 'no verified detection published by a beat');
    }
    finally { stop(); }
});

test('alive: no beat without frames (stream down), and the beats stop when the frames stop', async () => {
    cam();
    await eventually(() => ctl.starts.length >= 1);
    await sleep(250);
    assert.equal(beats().length, 0, 'no frame yet: silent');
    const stop = await flowing();
    await eventually(() => beats().length >= 1, 3000, 'beat once frames flow');
    stop();
    await sleep(300); // aliveFrameMs passes
    const n = beats().length;
    await sleep(250);
    assert.equal(beats().length, n, 'silent once the stream stopped');
});

test('alive: the service down or the kill switch silences the beat; it comes back with health', async () => {
    const h = cam();
    const stop = await flowing();
    try {
        await eventually(() => beats().length >= 1, 3000, 'first beat');
        h.mixin.onHealth(false, 'down');
        await sleep(20);
        let n = beats().length;
        await sleep(250);
        assert.equal(beats().length, n, 'silent while the service is down');
        h.mixin.onHealth(true, '');
        await eventually(() => beats().length > n, 3000, 'beats again once healthy');
        h.plugin.config.killSwitch = true;
        await sleep(20);
        n = beats().length;
        await sleep(250);
        assert.equal(beats().length, n, 'silent with the kill switch');
    }
    finally { stop(); }
});

test('alive: Off and Shadow never beat (Shadow publishes no verified detections, so it must not look live)', async () => {
    for (const mode of ['Off', 'Shadow']) {
        cam({ mode });
        const stop = await flowing().catch(() => () => { });
        await sleep(300);
        stop();
        assert.equal(beats().length, 0, mode);
        await releaseAll();
    }
});

test('alive: publishing off, or a camera without ObjectDetector, never beats', async () => {
    const h = cam();
    h.plugin.publishVerified = false;
    let stop = await flowing();
    await sleep(300);
    stop();
    assert.equal(beats().length, 0, 'publishing off');
    await releaseAll();
    cam({}, ['VideoCamera', 'MotionSensor']);
    stop = await flowing();
    await sleep(300);
    stop();
    assert.equal(beats().length, 0, 'no ObjectDetector');
});

test('alive: our own beat is not taken as a camera claim, and an empty payload never is', async () => {
    const h = cam();
    const stop = await flowing();
    try {
        await eventually(() => beats().length >= 1, 3000, 'beat');
        h.claimRaw(beats()[0]);
        h.claimRaw({ detections: [], timestamp: Date.now() });
        assert.equal((h.mixin as any).claimEvents.length, 0);
        assert.equal(h.mixin['episode'].scanning, false, 'nothing started');
    }
    finally { stop(); }
});

test('alive: release stops the beat timer', async () => {
    const h = cam();
    const stop = await flowing();
    await eventually(() => beats().length >= 1, 3000, 'beat');
    await h.mixin.release();
    const n = beats().length;
    await sleep(200);
    stop();
    assert.equal(beats().length, n);
    assert.equal((h.mixin as any).aliveTimer, undefined);
});

test('alive: a synchronous throw from onMixinEvent does not end the beat chain', async () => {
    const { sdkStub } = require('./harness') as typeof import('./harness');
    const orig = sdkStub.deviceManager.onMixinEvent;
    let thrown = 0;
    sdkStub.deviceManager.onMixinEvent = (id: any, m: any, iface: string, data: any) => {
        if (iface === 'ObjectDetector' && data?.detections?.length === 0 && thrown < 2) { thrown++; throw new Error('mid-rebuild'); }
        return orig(id, m, iface, data);
    };
    cam();
    const stop = await flowing();
    try {
        await eventually(() => beats().length >= 1, 3000, 'beat after two throws');
        assert.equal(thrown, 2);
    }
    finally { stop(); sdkStub.deviceManager.onMixinEvent = orig; }
});
