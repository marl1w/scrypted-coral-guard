import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { confirmedReply, ctl, idleReply, J, make, releaseAll, sleep } from './harness';

afterEach(() => releaseAll());

async function eventually(cond: () => boolean, ms = 3000, what = 'condition') {
    const t0 = Date.now();
    while (!cond()) {
        if (Date.now() - t0 > ms) assert.fail(`timeout waiting for ${what}`);
        await sleep(10);
    }
}

const personBox = { cls: 'person', group: 'person', score: 0.9, box: [0.1, 0.2, 0.3, 0.8] };
const carBox = { cls: 'car', group: 'vehicle', score: 0.9, box: [0.5, 0.5, 0.9, 0.9] };
const weakBox = { cls: 'person', group: 'person', score: 0.3, box: [0.6, 0.1, 0.7, 0.4] };

function withShots(settings: Record<string, any> = {}) {
    const h = make({ settings: { mode: 'Gate', ...settings } });
    const shots: { id: string; jpeg: Buffer; boxes: any[] }[] = [];
    h.plugin.publishLastDetection = (id: string, jpeg: Buffer, boxes: any[]) => { shots.push({ id, jpeg, boxes }); return Promise.resolve(true); };
    return { h, shots };
}

/** one camera trigger answered with `reply`; resolves once the episode is confirmed (or after the reply when it is not) */
async function episode(h: ReturnType<typeof make>, reply: any) {
    h.raw(true);
    await eventually(() => ctl.starts.length >= 1);
    const n = h.calls.frame.length;
    ctl.starts[ctl.starts.length - 1].o.onFrame(J, Date.now());
    await eventually(() => h.calls.frame.length > n);
    h.calls.frame[n].d.res(reply);
    await sleep(50);
}

test('last detection: a confirmed episode publishes one picture of its frame with the confirmed boxes only', async () => {
    const { h, shots } = withShots();
    await episode(h, { ...confirmedReply, detections: [personBox, carBox, weakBox] });
    assert.equal(ctl.state.motionDetected, true);
    assert.equal(shots.length, 1);
    assert.equal(shots[0].id, h.mixin.id);
    assert.equal(shots[0].jpeg, J, 'the analysed frame itself, not a copy');
    assert.deepEqual(shots[0].boxes, [{ x0: 0.1, y0: 0.2, x1: 0.3, y1: 0.8, group: 'person' }]);
    // held: more frames of the same episode publish nothing
    ctl.starts[ctl.starts.length - 1].o.onFrame(J, Date.now());
    await eventually(() => h.calls.frame.length === 2);
    h.calls.frame[1].d.res({ ...confirmedReply, detections: [personBox] });
    await sleep(50);
    assert.equal(shots.length, 1);
});

test('last detection: a camera trigger that is not confirmed publishes nothing', async () => {
    const { h, shots } = withShots();
    await episode(h, { ...idleReply, detections: [weakBox] });
    assert.equal(shots.length, 0);
});

test('last detection: a confirmation without a box waits for a frame that shows one', async () => {
    const { h, shots } = withShots();
    await episode(h, { ...confirmedReply, detections: [] });
    assert.equal(shots.length, 0);
    ctl.starts[ctl.starts.length - 1].o.onFrame(J, Date.now());
    await eventually(() => h.calls.frame.length === 2);
    h.calls.frame[1].d.res({ ...confirmedReply, detections: [personBox] });
    await eventually(() => shots.length === 1, 3000, 'picture');
});

test('last detection: a second episode within the minimum interval waits, one after it publishes again', async () => {
    const { h, shots } = withShots({ lastDetectionMinMs: 400 });
    const m: any = h.mixin;
    await episode(h, { ...confirmedReply, detections: [personBox] });
    assert.equal(shots.length, 1);
    // a new episode: the next confirmation is too soon, then allowed once the interval has passed
    m.shotTaken = false;
    m.noteHit(J, [{ x0: 0.1, y0: 0.2, x1: 0.3, y1: 0.8, score: 0.9, group: 'person' }]);
    m.lastDetectionStep();
    assert.equal(shots.length, 1, 'too soon');
    await sleep(450);
    m.lastDetectionStep();
    assert.equal(shots.length, 2);
});

const box = (group: string, x0 = 0.1) => ({ x0, y0: 0.2, x1: x0 + 0.2, y1: 0.8, score: 0.9, group });

test('last detection: the newest frame of a confirmed group, not a newer one of another group; frames freed once taken', async () => {
    const { h, shots } = withShots();
    const m: any = h.mixin;
    await episode(h, { ...confirmedReply, detections: [] }); // person confirmed, no box yet
    assert.equal(shots.length, 0);
    const P = Buffer.from([0xff, 0xd8, 0xff, 1]), C = Buffer.from([0xff, 0xd8, 0xff, 2]);
    m.noteHit(P, [box('person')]);
    await sleep(5);
    m.noteHit(C, [box('vehicle', 0.5)]);
    m.lastDetectionStep();
    assert.equal(shots.length, 1);
    assert.equal(shots[0].jpeg, P, 'the person frame, not the newer vehicle one');
    assert.deepEqual(shots[0].boxes.map(b => b.group), ['person']);
    assert.equal(m.lastHits.size, 0, 'freed once the picture is taken');
    m.noteHit(P, [box('person')]);
    assert.equal(m.lastHits.size, 0, 'nothing kept once the episode has its picture');
});

test('last detection: frames of an episode that ends unconfirmed are freed', async () => {
    const { h, shots } = withShots();
    const m: any = h.mixin;
    await episode(h, { ...idleReply, detections: [personBox] });
    assert.equal(m.lastHits.size, 1);
    const ep = m.episode;
    m.episode = { scanning: false };
    try { m.lastDetectionStep(); }
    finally { m.episode = ep; }
    assert.equal(m.lastHits.size, 0);
    assert.equal(shots.length, 0);
});
