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
    assert.deepEqual(v[0].detections, [{ className: 'people', score: 1 }]);
    assert.ok(Math.abs(v[0].timestamp - Date.now()) < 5000);
});

test('detections: the camera\'s own class name is preferred when it lists a different one', async () => {
    ctl.cameraClasses = ['person', 'car'];
    const h = gate();
    await sleep(10);
    assert.equal(h.mixin.publishClass('person'), 'person');
    assert.equal(h.mixin.publishClass('vehicle'), 'car');
    assert.equal(h.mixin.publishClass('animal'), 'dog_cat');
});

test('detections: held confirmation repeats at the configured interval', async () => {
    const h = gate({ detectionRepeatMs: 900 });
    await confirm(h);
    await eventually(() => verified().length >= 2, 4000, 'repeat while held');
    assert.ok(verified().every(d => d.detections[0].className === 'people'));
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

test('detections: raw camera AI events are suppressed in Gate but still reach the trigger logic', async () => {
    const h = gate();
    await sleep(10);
    h.claimRaw({ detections: [{ className: 'people', score: 0.9 }], timestamp: 1 });
    assert.equal(ctl.detections.length, 0, 'raw person class not re-published');
    assert.equal((h.mixin as any).claimEvents.length, 1, 'the claim reached onObjectsDetected');
    h.claimRaw({ detections: [{ className: 'package', score: 0.9 }, { className: 'people', score: 0.9 }], timestamp: 2 });
    assert.deepEqual(ctl.detections.map(d => d.detections.map((x: any) => x.className)), [['package']], 'unverifiable classes pass through');
    assert.deepEqual(await h.mixin.getObjectTypes(), { classes: ['people', 'vehicle', 'dog_cat', 'package'] });
});

test('detections: Shadow emits no verified events and passes raw ones through', async () => {
    const h = make({ settings: { mode: 'Shadow' }, interfaces: AI });
    h.plugin.id = 'prov-test';
    const raw = { detections: [{ className: 'people', score: 0.9 }], timestamp: 1 };
    h.claimRaw(raw);
    assert.deepEqual(ctl.detections, [raw]);
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(confirmedReply);
    await sleep(200);
    assert.equal(verified().length, 0);
});

test('detections: Off passes raw events through untouched, setting off too', async () => {
    const h = make({ settings: { mode: 'Off' }, interfaces: AI });
    const raw = { detections: [{ className: 'vehicle', score: 0.7 }], timestamp: 3 };
    h.claimRaw(raw);
    assert.equal(ctl.detections[0], raw);
    await h.mixin.release();
    const g = gate();
    g.plugin.publishVerified = false;
    g.claimRaw(raw);
    assert.equal(ctl.detections[0], raw);
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
