import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { confirmedReply, ctl, idleReply, J, make, releaseAll, sleep } from './harness';

afterEach(() => { ctl.cameraClasses = undefined; return releaseAll(); });

const AI = ['VideoCamera', 'MotionSensor', 'ObjectDetector'];
const frame = (s: any, t = Date.now()) => s.o.onFrame(J, t);
const verified = () => ctl.detections.filter(d => d.sourceId === 'prov-test');
async function eventually(cond: () => boolean, ms = 3000, what = 'condition') {
    const t0 = Date.now();
    while (!cond()) {
        if (Date.now() - t0 > ms) assert.fail(`timeout waiting for ${what}`);
        await sleep(10);
    }
}
function gate(settings: Record<string, any> = {}) {
    const h = make({ settings: { mode: 'Gate', ...settings }, interfaces: AI });
    h.plugin.id = 'prov-test';
    return h;
}
async function confirm(h: ReturnType<typeof make>, reply: any = confirmedReply) {
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(reply);
    await eventually(() => ctl.state.motionDetected === true, 3000, 'confirmed');
}

test('detections: confirm emits one verified event with the mapped class', async () => {
    const h = gate();
    await confirm(h);
    await sleep(50);
    const v = verified();
    assert.equal(v.length, 1);
    assert.deepEqual(v[0].detections, [{ className: 'verified_person', score: 1 }]);
    assert.ok(Math.abs(v[0].timestamp - Date.now()) < 5000);
});

test('detections: verified classes have their own names, whatever the camera calls its classes', async () => {
    ctl.cameraClasses = ['person', 'car'];
    const h = gate();
    await sleep(10);
    assert.equal(h.mixin.publishClass('person'), 'verified_person');
    assert.equal(h.mixin.publishClass('vehicle'), 'verified_vehicle');
    assert.equal(h.mixin.publishClass('animal'), 'verified_animal');
});

test('detections: held confirmation repeats at the configured interval', async () => {
    const h = gate({ detectionRepeatMs: 900 });
    await confirm(h);
    await eventually(() => verified().length >= 2, 4000, 'repeat while held');
    assert.ok(verified().every(d => d.detections[0].className === 'verified_person'));
});

test('detections: a vetoed episode emits nothing', async () => {
    const h = gate({ confirmTimeoutS: 1 });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    const t0 = Date.now();
    while (!ctl.logs.some(l => /VETOED/.test(l[1]))) {
        if (Date.now() - t0 > 4000) assert.fail('timeout waiting for veto');
        h.calls.frame.forEach(c => c.d.res(idleReply));
        frame(ctl.starts[ctl.starts.length - 1], Date.now());
        await sleep(100);
    }
    assert.equal(verified().length, 0);
});

test('detections: raw camera AI events are never re-published (Scrypted delivers them) but reach the trigger logic', async () => {
    const h = gate();
    await sleep(10);
    h.claimRaw({ detections: [{ className: 'people', score: 0.9 }], timestamp: 1 });
    h.claimRaw({ detections: [{ className: 'package', score: 0.9 }], timestamp: 2 });
    assert.equal(ctl.detections.length, 0, 'nothing re-published');
    assert.equal((h.mixin as any).claimEvents.length, 2, 'the claims reached onObjectsDetected');
    const types = await h.mixin.getObjectTypes();
    for (const c of ['verified_person', 'verified_vehicle', 'verified_animal'])
        assert.ok(types.classes!.includes(c), `${c} listed`);
});

test('detections: Shadow emits no verified events and re-publishes nothing', async () => {
    const h = make({ settings: { mode: 'Shadow' }, interfaces: AI });
    h.plugin.id = 'prov-test';
    const raw = { detections: [{ className: 'people', score: 0.9 }], timestamp: 1 };
    h.claimRaw(raw);
    assert.deepEqual(ctl.detections, []);
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(confirmedReply);
    await sleep(200);
    assert.equal(verified().length, 0);
});

test('detections: Off and the setting off publish nothing of their own', async () => {
    const h = make({ settings: { mode: 'Off' }, interfaces: AI });
    const raw = { detections: [{ className: 'vehicle', score: 0.7 }], timestamp: 3 };
    h.claimRaw(raw);
    assert.equal(ctl.detections.length, 0);
    await h.mixin.release();
    const g = gate();
    g.plugin.publishVerified = false;
    g.claimRaw(raw);
    assert.equal(ctl.detections.length, 0);
    await confirm(g);
    await sleep(50);
    assert.equal(verified().length, 0);
});

test('detections: a camera without ObjectDetector gets nothing published', async () => {
    const h = make({ settings: { mode: 'Gate' }, interfaces: ['VideoCamera', 'MotionSensor'] });
    h.plugin.id = 'prov-test';
    await confirm(h);
    await sleep(50);
    assert.equal(ctl.detections.length, 0);
});

test('detections: our own verified events are not taken as camera claims', async () => {
    const h = gate();
    await sleep(10);
    h.claimRaw({ detections: [{ className: 'verified_person', score: 1 }], sourceId: 'prov-test', timestamp: 1 });
    h.claimRaw({ detections: [{ className: 'verified_person', score: 1 }], timestamp: 2 });
    assert.equal((h.mixin as any).claimEvents.length, 0);
    h.claimRaw({ detections: [{ className: 'people', score: 0.9 }], sourceId: 'camera-plugin', timestamp: 3 });
    assert.equal((h.mixin as any).claimEvents.length, 1);
});
