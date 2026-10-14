import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { confirmedReply, ctl, J, make, releaseAll, sdkStub, sleep } from './harness';
import { fetchSnapshot, FastPathTimer, SNAP_MAX_FAILS } from '../src/fastpath';

afterEach(() => releaseAll());

async function eventually(cond: () => boolean, ms = 3000, what = 'condition') {
    const t0 = Date.now();
    while (!cond()) {
        if (Date.now() - t0 > ms) assert.fail(`timeout waiting for ${what}`);
        await sleep(5);
    }
}
const media = { convertMediaObjectToBuffer: async (mo: any) => mo };
const motion = () => ctl.state.motionDetected;

test('fetchSnapshot: returns the JPEG, rejects a non-JPEG, a camera without snapshots and a slow snapshot', async () => {
    assert.deepEqual(await fetchSnapshot({ takePicture: async () => J }, media, 'periodic'), J);
    await assert.rejects(fetchSnapshot({ takePicture: async () => Buffer.from('nope') }, media, 'periodic'), /not a JPEG/);
    await assert.rejects(fetchSnapshot({}, media, 'periodic'), /no snapshots/);
    await assert.rejects(fetchSnapshot({ takePicture: () => new Promise(() => { }) }, media, 'event', 30), /timed out/);
});

test('FastPathTimer: one line per claim, relative to the camera push; disarms at the verdict', () => {
    const f = new FastPathTimer();
    assert.equal(f.verdict('confirmed', 5), undefined);
    f.claim('people', 1000);
    f.claim('vehicle', 1010); // the first claim owns the episode
    f.scan(1002); f.posted(1150); f.edge(1400);
    assert.equal(f.verdict('confirmed person', 1300), 'fast path: camera AI people -> confirmed person +300 ms (scan +2 ms, first frame +150 ms, camera motion +400 ms)');
    assert.equal(f.armed, false);
    f.claim('dog_cat', 0); f.expire(40000);
    assert.equal(f.armed, false);
});

function snapCam(h: ReturnType<typeof make>, pic: () => Promise<any>) {
    sdkStub.mediaManager.convertMediaObjectToBuffer = media.convertMediaObjectToBuffer;
    (h.mixin as any).mixinDevice.takePicture = pic;
    h.plugin.pacedSnapshots = true;
    h.mixin.applyConfig();
}

test('paced snapshots: the warm decode stops and heartbeats post takePicture frames at low priority', async () => {
    const h = make({ settings: { mode: 'Shadow', frameProcess: 'Keep warm', heartbeatMs: 20 } });
    await eventually(() => ctl.starts.length === 1);
    let pics = 0;
    snapCam(h, async () => { pics++; return J; });
    assert.equal(ctl.starts[0].stopped, true, 'continuous decode stopped');
    await eventually(() => h.calls.frame.length === 1, 3000, 'heartbeat post');
    assert.ok(pics >= 1);
    assert.equal(h.calls.frame[0].a[3], true, 'prio=low');
    assert.equal(ctl.starts.length, 1, 'no new ffmpeg while idle');
    await h.mixin.release();
});

test('paced snapshots: repeated failures fall back to the warm stream', async () => {
    const h = make({ settings: { mode: 'Shadow', frameProcess: 'Keep warm', heartbeatMs: 10 } });
    await eventually(() => ctl.starts.length === 1);
    snapCam(h, async () => { throw new Error('boom'); });
    await eventually(() => ctl.starts.length === 2, 3000, 'warm restart');
    assert.ok(ctl.logs.some(([k, m]) => k === 'warn' && m.includes(`failed ${SNAP_MAX_FAILS} times`)));
    await h.mixin.release();
});

test('paced snapshots: a scan opens with the camera snapshot before the stream delivers', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    snapCam(h, async () => J);
    h.raw(true);
    await eventually(() => h.calls.frame.length === 1, 3000, 'bridge frame');
    assert.equal(h.calls.frame[0].a[3], false, 'camera episode: normal priority');
    await h.mixin.release();
});

test('camera AI fast path: a people claim starts verification before the motion edge, at normal priority, and logs the latency', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.plugin.aiFastPath = true;
    h.claim('people');
    await eventually(() => ctl.starts.length === 1, 3000, 'scan started');
    assert.equal(h.calls.reset.length, 1);
    ctl.starts[0].o.onFrame(J, Date.now());
    await eventually(() => h.calls.frame.length === 1);
    assert.equal(h.calls.frame[0].a[3], false, 'not low priority');
    h.raw(true); // the camera's edge merges into the running scan: no second reset
    assert.equal(h.calls.reset.length, 1);
    h.calls.frame[0].d.res(confirmedReply);
    await eventually(() => motion() === true);
    assert.ok(ctl.logs.some(([, m]) => /fast path: camera AI people -> confirmed person \+\d+ ms \(scan \+\d+ ms, first frame \+\d+ ms, camera motion \+\d+ ms\)/.test(m)), 'latency line');
    await h.mixin.release();
});

test('camera AI fast path off (or a non-verifiable class): a claim alone starts nothing', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.plugin.aiFastPath = false;
    h.claim('people');
    h.plugin.aiFastPath = true;
    h.claim('package');
    await sleep(50);
    assert.equal(ctl.starts.length, 0);
    assert.equal(h.calls.reset.length, 0);
    await h.mixin.release();
});
