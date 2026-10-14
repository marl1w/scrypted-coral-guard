import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { confirmedReply, ctl, holdingReply, idleReply, J, loadProvider, make, releaseAll, sleep } from './harness';

afterEach(() => releaseAll());
import { MockServer } from './mock-server';
import { VerifierClient } from '../src/client';

const last = (h: ReturnType<typeof make>) => ctl.starts[ctl.starts.length - 1];
const motion = () => ctl.state.motionDetected;
const frame = (s: any, t = Date.now()) => s.o.onFrame(J, t);

async function eventually(cond: () => boolean, ms = 3000, what = 'condition') {
    const t0 = Date.now();
    while (!cond()) {
        if (Date.now() - t0 > ms) assert.fail(`timeout waiting for ${what}`);
        await sleep(10);
    }
}

// ---- H1 ---------------------------------------------------------------------------------------

test('H1: Gate with a hanging stream fails open instead of vetoing (raw passes through)', async () => {
    const h = make({ settings: { mode: 'Gate', confirmTimeoutS: 1 }, hangStart: true });
    h.raw(true);
    assert.equal(motion(), false); // gated while waiting
    await eventually(() => motion() === true, 3000, 'pass-through');
    assert.ok(ctl.logs.some(l => /unavailable/.test(l[1])));
    h.raw(false);
    assert.equal(motion(), false);
    await h.mixin.release();
});

// ---- H2 ---------------------------------------------------------------------------------------

test('H2/t1: toggling the mode while a /frame request is in flight does not wedge frame posting', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    const s1 = ctl.starts[0];
    frame(s1);
    await eventually(() => h.calls.frame.length === 1);
    await h.mixin.putMixinSetting('mode', 'Off');
    await h.mixin.putMixinSetting('mode', 'Gate');
    await eventually(() => ctl.starts.length === 2);
    assert.equal(s1.stopped, true);
    const s2 = ctl.starts[1];
    for (let i = 0; i < 3; i++) frame(s2);
    await eventually(() => h.calls.frame.length === 2, 3000, 'new generation posts while the old request hangs');
    h.calls.frame[1].d.res(idleReply);
    await eventually(() => h.calls.frame.length === 3, 3000, 'burst continues in order');
    h.calls.frame[0].d.res(confirmedReply); // the old request completes late: must be ignored
    await sleep(30);
    assert.ok(!ctl.logs.some(l => /episode:.*confirmed \+\d/.test(l[1])), 'the late reply confirmed nothing');
    await h.mixin.release();
});

test('H2/t7: editing a setting while confirmed and a request is in flight keeps frames flowing and motion not stuck', async () => {
    const h = make({ settings: { mode: 'Gate', maxHoldS: 2, preRollS: 4 } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    let s = ctl.starts[0];
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(confirmedReply);
    await eventually(() => motion() === true);
    await sleep(1100);
    frame(s, Date.now() + 5000);
    await eventually(() => h.calls.frame.length === 2);
    h.raw(false);
    (h.mixin as any).stopSource(); // what a frame-setting edit used to do: restart the frame process
    (h.mixin as any).syncSource(true);
    await eventually(() => ctl.starts.length === 2);
    s = last(h);
    for (let i = 0; i < 3; i++) frame(s);
    await eventually(() => h.calls.frame.length === 3, 3000, 'posting after the edit');
    for (let i = 0; i < 10; i++) { // answer the whole burst
        await sleep(10);
        h.calls.frame.slice(2).forEach(c => c.d.res(holdingReply));
    }
    const n0 = h.calls.frame.length;
    await sleep(1100);
    frame(s, Date.now() + 9000);
    await eventually(() => h.calls.frame.length > n0, 3000, 'live frames after the edit');
    h.calls.frame[1].d.res(holdingReply); // old, late
    await eventually(() => motion() === false, 5000, 'max hold release');
    await h.mixin.release();
});

test('H2: veto then retrigger within the request time posts the new episode\'s frames', async () => {
    const h = make({ settings: { mode: 'Gate', confirmTimeoutS: 2, frameProcess: 'Keep warm' } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    h.raw(true);
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(idleReply); // a verdict, so the timeout is a veto and not a failure
    await sleep(500);
    frame(s, Date.now() + 100); // request in flight when the veto fires
    await eventually(() => h.calls.frame.length === 2);
    await eventually(() => ctl.logs.some(l => /VETOED/.test(l[1])), 3000, 'veto');
    h.raw(false);
    h.raw(true); // retrigger within the old request's lifetime
    frame(s, Date.now() + 200);
    await eventually(() => h.calls.frame.length === 3, 3000, 'new episode frame posted');
    h.calls.frame[1].d.res(confirmedReply); // L9: the old episode's reply must not confirm the new one
    await sleep(30);
    assert.equal(motion(), false);
    await h.mixin.release();
});

// ---- H3 ---------------------------------------------------------------------------------------

test('H3: a skipped frame (429/stale -> null) is not an outage', async () => {
    const h = make({ settings: { mode: 'Gate' }, client: {
        frame: async () => null, reset: async () => ({}), health: async () => ({ ok: true }), check: async () => ({ ok: true }), config: async () => ({}),
    } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await sleep(50);
    assert.ok(!ctl.logs.some(l => /unavailable/.test(l[1])));
    assert.equal(motion(), false);
    await h.mixin.release();
});

test('H3/t4: wrong token does not cause a degrade/recover loop with an ffmpeg start per second', async () => {
    const m = new MockServer();
    await m.start();
    m.script.token = 'secret'; // health needs no token, everything else does
    const client = new VerifierClient(() => ({ baseUrl: m.url, token: 'WRONG', timeoutMs: 1000, frameTimeoutMs: 500 }));
    const h = make({ settings: { mode: 'Gate' }, client });
    h.raw(true);
    const t0 = Date.now();
    while (Date.now() - t0 < 3600) {
        const s = ctl.starts[ctl.starts.length - 1];
        if (s && !s.stopped) frame(s);
        await sleep(100);
    }
    const rec = ctl.logs.filter(l => /recovered/.test(l[1])).length;
    const deg = ctl.logs.filter(l => /unavailable/.test(l[1])).length;
    assert.equal(rec, 0, 'a wrong token must not look recovered (authenticated probe)');
    assert.equal(deg, 1);
    assert.ok(ctl.starts.length <= 2, `ffmpeg starts ${ctl.starts.length}`);
    assert.equal(motion(), true); // raw passes through
    await h.mixin.release();
    await m.stop();
});

test('H3: frames the service keeps rejecting (400) back off instead of looping at 1 Hz', async () => {
    const m = new MockServer();
    await m.start();
    m.script.frames = [400];
    const client = new VerifierClient(() => ({ baseUrl: m.url, timeoutMs: 1000, frameTimeoutMs: 500 }));
    const h = make({ settings: { mode: 'Gate' }, client });
    h.raw(true);
    const t0 = Date.now();
    while (Date.now() - t0 < 4600) {
        const s = ctl.starts[ctl.starts.length - 1];
        if (s && !s.stopped) frame(s);
        await sleep(100);
    }
    assert.ok(ctl.starts.length <= 3, `ffmpeg starts ${ctl.starts.length} (backoff 1 s, 2 s, 4 s)`);
    await h.mixin.release();
    await m.stop();
});

// ---- M1 / M3 / M5 -----------------------------------------------------------------------------

test('startup is transparent: a persisted motionDetected=true is neither replayed, nor republished, nor scanned (Gate and Shadow)', async () => {
    for (const mode of ['Gate', 'Shadow']) {
        const h = make({ settings: { mode, frameProcess: 'Keep warm' }, initial: true });
        await eventually(() => ctl.starts.length === 1);
        frame(ctl.starts[0]);
        await sleep(150);
        assert.equal(ctl.writes.length, 0, `${mode}: no motionDetected write at startup`);
        assert.equal(h.calls.reset.length, 0, `${mode}: no scan at startup`);
        await h.mixin.release();
        assert.equal(ctl.writes.length, 0, `${mode}: nor at release`);
    }
});

test('only a genuine rising edge after load emits: Gate publishes on confirmation, Shadow mirrors raw', async () => {
    let h = make({ settings: { mode: 'Shadow' }, initial: false });
    h.raw(true);
    assert.deepEqual(ctl.writes.map(w => w[1]), [true]);
    h.raw(false);
    assert.deepEqual(ctl.writes.map(w => w[1]), [true, false]);
    await h.mixin.release();
});

test('M3: frames wait for the reset (per-trigger burst)', async () => {
    const h = make({ settings: { mode: 'Gate' }, holdReset: true });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await sleep(60);
    assert.equal(h.calls.reset.length, 1);
    assert.equal(h.calls.frame.length, 0, 'no frame before the reset finished');
    h.calls.reset[0].d.res({});
    await eventually(() => h.calls.frame.length === 1);
    await h.mixin.release();
});

test('M3: frames wait for the reset (Keep warm ring hand-off)', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm' }, holdReset: true });
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0], Date.now() - 1500);
    frame(ctl.starts[0], Date.now() - 500);
    h.raw(true);
    await sleep(60);
    assert.equal(h.calls.frame.length, 0);
    h.calls.reset[0].d.res({});
    await eventually(() => h.calls.frame.length === 1);
    await h.mixin.release();
});

test('M3: a failed reset fails open and no frame is posted', async () => {
    const h = make({ settings: { mode: 'Gate' }, client: {
        frame: async () => { throw new Error('should not be called'); }, reset: async () => { throw new Error('HTTP 500'); },
        health: async () => ({ ok: true }), check: async () => ({ ok: true }), config: async () => ({}),
    } });
    h.raw(true);
    await eventually(() => motion() === true, 1000, 'pass-through after reset failure');
    await h.mixin.release();
});

test('M5: not being the outermost MotionSensor mixin is detected, warned once and shown in the status', async () => {
    const h = make({ settings: { mode: 'Gate' }, persistWrites: false });
    h.raw(true); // gating publishes false; another mixin outside us shows true
    ctl.state.motionDetected = true;
    await eventually(() => ctl.logs.some(l => /NOT the outermost/.test(l[1])), 2000, 'warning');
    await sleep(700);
    assert.equal(ctl.logs.filter(l => /NOT the outermost/.test(l[1])).length, 1);
    const st = (await h.mixin.getMixinSettings()).find(s => s.key === 'mode');
    assert.match(String(st?.description), /outermost/);
    await h.mixin.release();
});

test('M5: no warning when the write is persisted', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await sleep(500);
    assert.ok(!ctl.logs.some(l => /outermost/.test(l[1])));
    await h.mixin.release();
});

// ---- release / lifecycle (M22, M24 at mixin level) ---------------------------------------------

test('release: listener removed, ffmpeg stopped, late events inert, best-effort raw write', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    await h.mixin.release();
    assert.equal(ctl.removed, 2); // motion and object-detector listeners
    assert.equal(ctl.starts[0].stopped, true);
    const writes = ctl.writes.length;
    h.raw(false);
    h.raw(true);
    await sleep(50);
    assert.equal(ctl.writes.length, writes);
    assert.equal(ctl.starts.length, 1);
    assert.equal(motion(), true); // released with raw on: raw value, not a forced false
});

test('per-trigger: the ffmpeg stops when the scan ends', async () => {
    const h = make({ settings: { mode: 'Gate', confirmTimeoutS: 1 } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(idleReply);
    await eventually(() => ctl.starts[0].stopped, 3000, 'stop after veto');
    await h.mixin.release();
});

// ---- M26 / M27: fail-open of the runtime ------------------------------------------------------

test('M27: a frame request error fails open (raw passes through)', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.rej(new Error('HTTP 503'));
    await eventually(() => motion() === true);
    await h.mixin.release();
});

test('M26: a stream that never delivers frames fails open before the confirm timeout', async () => {
    const h = make({ settings: { mode: 'Gate', confirmTimeoutS: 4 } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    await eventually(() => motion() === true, 4000, 'no-frames watchdog');
    assert.ok(ctl.logs.some(l => /no frames/.test(l[1])));
    await h.mixin.release();
});

test('frame source ended while scanning keeps the episode alive and restarts the source (no immediate fail-open)', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    await sleep(20);
    ctl.starts[0].exit('ffmpeg exited 1');
    await eventually(() => ctl.starts.length === 2, 4000, 'source restarted');
    assert.equal(motion(), false, 'episode not aborted by the restart');
    assert.ok(!ctl.logs.some(l => /verifier unavailable/.test(String(l[1]))));
    await h.mixin.release();
});

test('a runtime VAAPI failure restarts in software at once and keeps the episode', async () => {
    (ctl as any).vaapi = true;
    try {
        const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', hwDecode: 'Auto' } });
        await eventually(() => ctl.starts.length === 1);
        frame(ctl.starts[0]);
        h.raw(true);
        ctl.starts[0].exit('hwaccel failed: ffmpeg exited 251; [hwdownload] Failed to download frame');
        await eventually(() => ctl.starts.length === 2, 3000, 'immediate software restart');
        assert.equal(ctl.starts[1].o.hwaccel, false);
        assert.ok(!ctl.logs.some(l => /verifier unavailable/.test(String(l[1]))));
        await h.mixin.release();
    }
    finally { (ctl as any).vaapi = false; }
});

// ---- L7 / L8 / L3 / L10 ------------------------------------------------------------------------

test('L7: the burst queue is bounded (contract burst of 10)', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    for (let i = 0; i < 40; i++) frame(s, Date.now());
    let done = 0;
    for (let i = 0; i < 60; i++) {
        await sleep(5);
        while (done < h.calls.frame.length) h.calls.frame[done++].d.res(idleReply);
    }
    assert.ok(h.calls.frame.length <= 11, `posted ${h.calls.frame.length}`);
    await h.mixin.release();
});

test('L8: a prototype-key mode value is rejected and falls back to shadow', async () => {
    const h = make({ settings: { mode: '__proto__' } });
    h.raw(true);
    assert.equal(motion(), true); // shadow: pass-through
    const st = (await h.mixin.getMixinSettings()).find(s => s.key === 'mode');
    assert.equal(st?.description, 'Off, Shadow (measure only) or Gate.');
    assert.match(String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value), /<span>Shadow<\/span>/);
    await h.mixin.release();
});

test('verifier id is the name slug plus the Scrypted device id, so duplicates cannot collide', async () => {
    const h = make({});
    assert.equal(h.mixin.verifierId, 'cam_1_cam1');
    await h.mixin.release();
});

test('L10: a duplicate service camera id is warned about', async () => {
    const h = make({ dup: 'Other Cam' });
    assert.ok(ctl.logs.some(l => /is also used by camera Other Cam/.test(l[1])));
    await h.mixin.release();
});

test('L5: an invalid stream override fails open and never starts ffmpeg', async () => {
    const h = make({ settings: { mode: 'Gate', streamUrl: 'file:///etc/passwd' } });
    h.raw(true);
    await sleep(50);
    assert.equal(motion(), true);
    assert.ok(ctl.logs.some(l => /only rtsp/.test(l[1])));
    await h.mixin.release();
});

test('Keep warm: a dead ffmpeg is restarted with a NEW startFrames (new getVideoStream), never reusing the old one', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm' } });
    await eventually(() => ctl.starts.length === 1);
    ctl.starts[0].exit('ffmpeg exited 1: boom');
    await eventually(() => ctl.starts.length === 2, 6000, 'warm restart');
    assert.notEqual(ctl.starts[0], ctl.starts[1]);
    await h.mixin.release();
});

test('Hardware decode: a hwaccel failure restarts once in software and stays in software', async () => {
    (ctl as any).vaapi = true;
    try {
        const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', hwDecode: 'Auto' } });
        await eventually(() => ctl.starts.length === 1);
        assert.equal(ctl.starts[0].o.hwaccel, true);
        ctl.starts[0].exit('hwaccel failed: ffmpeg exited 1');
        await eventually(() => ctl.starts.length === 2, 3000, 'software restart');
        assert.equal(ctl.starts[1].o.hwaccel, false);
        assert.ok(ctl.logs.some(l => /decode GPU failed, falling back to CPU/.test(String(l[1]))));
        await h.mixin.release();
    }
    finally { (ctl as any).vaapi = false; }
});

test('settings: Refresh and Status on top, then Mode with a short description (no repeated summary)', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    const st = await h.mixin.getMixinSettings();
    assert.deepEqual(st.map(x => x.key), ['refresh', 'status', 'mode', 'detect', 'stream', 'zoomOut', 'zoneNames']);
    assert.equal(st[1].type, 'html');
    assert.equal(st[1].readonly, true);
    assert.ok(!st.some(x => x.group), 'one group only (the mixin default)');
    assert.match(String(st[1].value), /Episodes/);
    assert.equal(st[2].description, 'Off, Shadow (measure only) or Gate.');
    await h.mixin.release();
});

test('status panel: a person reply shows analysing/person confirmed and is throttled to the settings page', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res({ ...confirmedReply, detections: [{ cls: 'person', score: 0.9, box: [0.1, 0.1, 0.4, 0.8] }], total_ms: 12 });
    await sleep(30);
    const html = String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value);
    assert.match(html, /Detected<\/div><div>person 0\.90/);
    assert.match(html, /Motion ON/);
    await h.mixin.release();
});

const flat = () => new Uint8Array(128 * 72);
const blobFrame = () => { const g = new Uint8Array(128 * 72); for (let y = 20; y < 40; y++) for (let x = 40; x < 60; x++) g[y * 128 + x] = 200; return g; };

test('Shadow: a pixel-gate hit runs the same verified unflagged episode as Gate; wouldPublish ON, motion never written', async () => {
    const h = make({ settings: { mode: 'Shadow', frameProcess: 'Keep warm', gate: { warmupFrames: 0 } as any } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    s.o.gray.onGray(flat());
    s.o.gray.onGray(flat());
    await sleep(30);
    assert.equal(h.calls.frame.length, 0);
    s.o.gray.onGray(blobFrame());
    await eventually(() => h.calls.frame.length === 1);
    assert.equal(h.calls.frame[0].a[3], true, 'prio=low');
    h.calls.frame[0].d.res(confirmedReply);
    await eventually(() => ctl.logs.some(l => /person seen but camera did not trigger: would raise motion \(Shadow\)/.test(String(l[1]))));
    assert.equal(ctl.writes.length, 0, 'Shadow never publishes the unflagged raise');
    assert.equal(h.mixin['episode'].wouldPublish, true, 'but it would (same decision as Gate)');
    assert.match(String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value), /Would publish ON/);
    await h.mixin.release();
});

test('Gate: a pixel change alone never raises motion; it only asks the service (synthetic episode, vetoed on a person-less verdict)', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', confirmTimeoutS: 1, gate: { warmupFrames: 0 } as any } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    s.o.gray.onGray(flat());
    s.o.gray.onGray(blobFrame());
    await eventually(() => h.calls.frame.length === 1);
    assert.equal(h.calls.frame[0].a[3], true, 'low priority while no camera episode');
    h.calls.frame[0].d.res(idleReply);
    await sleep(1300);
    assert.equal(ctl.writes.filter(w => w[1] === true).length, 0);
    assert.equal(motion(), false);
    assert.ok(!ctl.logs.some(l => /raised motion/.test(String(l[1]))));
    // a person-less synthetic scan leaves no trace in the history, and a cooldown blocks an immediate re-scan
    assert.ok(!/Unflagged/.test(String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value).replace(/Unflagged persons/g, '')));
    s.o.gray.onGray(blobFrame());
    await sleep(50);
    assert.equal(h.calls.reset.length, 1, 'cooldown after a person-less scan');
    await h.mixin.release();
});

test('Gate: a confirmed person in an unflagged scan raises motion once, holds while present, releases when gone; camera raw mid-scan merges', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', gate: { warmupFrames: 0 } as any } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    s.o.gray.onGray(flat());
    s.o.gray.onGray(blobFrame());
    await eventually(() => h.calls.frame.length === 1);
    assert.equal(motion(), false);
    h.calls.frame[0].d.res(confirmedReply);
    await eventually(() => motion() === true);
    assert.ok(ctl.logs.some(l => /person seen but camera did not trigger: raised motion \(gate/.test(String(l[1]))));
    const trues = () => ctl.writes.filter(w => w[1] === true).length;
    assert.equal(trues(), 1);
    h.raw(true); // the camera catches up: merged into the same episode, no second motion edge
    await sleep(50);
    assert.equal(trues(), 1);
    assert.equal(motion(), true);
    h.raw(false);
    await sleep(50);
    assert.equal(motion(), true, 'still held by the service');
    frame(s, Date.now() + 10);
    await eventually(() => h.calls.frame.length === 2);
    assert.equal(h.calls.frame[1].a[3], false, 'normal priority once a camera episode owns it');
    h.calls.frame[1].d.res(idleReply); // the service says gone
    await eventually(() => motion() === false);
    assert.equal(trues(), 1);
    await h.mixin.release();
});

test('gate monitoring: not posted while a camera episode is active', async () => {
    const h = make({ settings: { mode: 'Shadow', frameProcess: 'Keep warm', gate: { warmupFrames: 0 } as any } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    const flat = new Uint8Array(128 * 72), blob = new Uint8Array(128 * 72).fill(200);
    s.o.gray.onGray(flat);
    h.raw(true); // camera episode in progress
    await eventually(() => h.calls.frame.length === 1); // the episode's own pre-roll frame
    h.calls.frame[0].d.res(idleReply);
    await sleep(30);
    s.o.gray.onGray(blob);
    await sleep(30);
    assert.equal(h.calls.frame.length, 1, 'the gate path added no request');
    await h.mixin.release();
});

test('the warm stream request asks the rebroadcast for enough history to contain a sync frame', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm' } });
    await eventually(() => ctl.starts.length === 1);
    assert.ok(ctl.starts[0].o.prebufferMs >= 8000);
    await h.mixin.release();
});

test('panel: a 0.26 detection is a weak candidate, not a person; 0.9 is a person', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res({ ...idleReply, detections: [{ cls: 'person', score: 0.26, box: [0.1, 0.1, 0.3, 0.5] }], total_ms: 33 });
    await sleep(30);
    const html = String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value);
    assert.match(html, /Detected<\/div><div style="opacity:\.7">nothing/);
    assert.match(html, /possible 0\.26/);
    await h.mixin.release();
});

test('an oversized snapshot is logged once with its size and the panel says so instead of showing nothing', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    ctl.starts[0].o.onFrame(Buffer.alloc(130 * 1024), Date.now());
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(idleReply);
    ctl.starts[0].o.onFrame(Buffer.alloc(130 * 1024), Date.now() + 1);
    await sleep(60);
    assert.equal(ctl.logs.filter(l => /too large for the panel thumbnail/.test(String(l[1]))).length, 1);
    assert.ok(ctl.logs.some(l => /130 KB/.test(String(l[1]))));
    assert.match(String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value), /Snapshot too large/);
    await h.mixin.release();
});

test('panel: the latest delivered frame is shown as a live frame before anything was analysed', async () => {
    const h = make({ settings: { mode: 'Shadow', frameProcess: 'Keep warm' } });
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    const html = String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value);
    assert.ok(html.includes('<img'));
    assert.match(html, /live, not analysed/);
    await h.mixin.release();
});

const vehicleMoving = { person: { state: 'idle', confirmed: false, present: false }, vehicle: { moving: true }, detections: [{ cls: 'car', score: 0.9, box: [0.1, 0.1, 0.5, 0.5] }] };
const parkedReply = { person: { state: 'idle', confirmed: false, present: false }, vehicle: { moving: false, tracks: [{ id: 1, cls: 'car', parked: true, since_ms: 99999 }] } };

test('decisions: a moving vehicle confirms and holds in Gate; a parked one never does; the log says which group', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(parkedReply);
    await sleep(40);
    assert.equal(motion(), false);
    frame(ctl.starts[0], Date.now() + 5);
    await eventually(() => h.calls.frame.length === 2);
    h.calls.frame[1].d.res(vehicleMoving);
    await sleep(40);
    assert.equal(motion(), false, 'one vehicle frame never confirms');
    frame(ctl.starts[0], Date.now() + 10);
    await eventually(() => h.calls.frame.length === 3);
    h.calls.frame[2].d.res(vehicleMoving);
    await eventually(() => motion() === true);
    await h.mixin.release();
});

test('claim: the camera says people, only a person confirms; vehicle alone does not raise motion', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    h.claim('people');
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(vehicleMoving);
    await sleep(40);
    assert.equal(motion(), false);
    frame(ctl.starts[0], Date.now() + 5);
    await eventually(() => h.calls.frame.length === 2);
    h.calls.frame[1].d.res(confirmedReply);
    await eventually(() => motion() === true);
    await h.mixin.release();
});

test('claim a package: not verifiable, logged and passed through (Gate raises motion at once, released with raw)', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    h.claim('package');
    await eventually(() => motion() === true);
    assert.ok(ctl.logs.some(l => /camera claims package: not verifiable, passing through/.test(String(l[1]))));
    h.raw(false);
    await eventually(() => motion() === false);
    await h.mixin.release();
});

test('Shadow judges claims like Gate (would pass an unverifiable claim through) while publishing raw', async () => {
    const h = make({ settings: { mode: 'Shadow' } });
    h.raw(true);
    h.claim('package');
    await sleep(40);
    assert.ok(ctl.logs.some(l => /not verifiable/.test(String(l[1]))));
    assert.equal(motion(), true);
    assert.equal(h.mixin['episode'].wouldPublish, true);
    await h.mixin.release();
});

test('unflagged monitoring raises motion for a moving vehicle in Gate (any group), parked never', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', gate: { warmupFrames: 0 } as any } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    s.o.gray.onGray(new Uint8Array(128 * 72));
    const b = new Uint8Array(128 * 72); for (let y = 20; y < 40; y++) for (let x = 40; x < 60; x++) b[y * 128 + x] = 200;
    s.o.gray.onGray(b);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(vehicleMoving);
    await sleep(40);
    assert.equal(motion(), false, 'a single frame never confirms');
    frame(s, Date.now() + 700);
    await eventually(() => h.calls.frame.length === 2);
    h.calls.frame[1].d.res(vehicleMoving);
    await eventually(() => motion() === true);
    assert.ok(ctl.logs.some(l => /vehicle seen but camera did not trigger: raised motion/.test(String(l[1]))));
    await h.mixin.release();
});

test('Detect setting: Person only ignores a sustained vehicle; the description names the enabled set; empty means all', async () => {
    const h = make({ settings: { mode: 'Gate', detect: ['Person'] } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    for (let i = 0; i < 2; i++) {
        frame(ctl.starts[0], Date.now() + i * 5);
        await eventually(() => h.calls.frame.length === i + 1);
        h.calls.frame[i].d.res(vehicleMoving);
    }
    await sleep(60);
    assert.equal(motion(), false);
    const st = await h.mixin.getMixinSettings();
    assert.deepEqual(h.mixin['detectGroups'](), ['person']);
    await h.mixin.release();
    const all = make({ settings: { mode: 'Gate', detect: [] } });
    assert.ok(!/Detect:/.test(String((await all.mixin.getMixinSettings()).find(x => x.key === 'mode')?.description)));
    await all.mixin.release();
});

test('GPU slot lifecycle at the mixin: held while a hardware process runs, freed on fallback, a software start holds none', async () => {
    (ctl as any).vaapi = true;
    try {
        const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', hwDecode: 'Auto' } });
        await eventually(() => ctl.starts.length === 1);
        assert.equal(ctl.starts[0].o.hwaccel, true);
        assert.equal(h.plugin.hw.size, 1);
        assert.ok(ctl.logs.some(l => /GPU decode slots: 1\/64 used/.test(String(l[1]))));
        ctl.starts[0].exit('hwaccel failed: ffmpeg exited 1');
        await eventually(() => ctl.starts.length === 2, 3000);
        assert.equal(ctl.starts[1].o.hwaccel, false);
        assert.equal(h.plugin.hw.size, 0, 'the failed camera freed its slot and the software start holds none');
        await h.mixin.release();
    }
    finally { (ctl as any).vaapi = false; }
});

test('Detect change applies to the running episode without a restart, is logged and shown as Detecting', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    await h.mixin.putMixinSetting('detect', ['Animal']);
    assert.ok(ctl.logs.some(l => /Detect: animal$/.test(String(l[1]))));
    h.calls.frame[0].d.res(confirmedReply);
    await sleep(40);
    assert.equal(motion(), false, 'person is off now: it no longer confirms');
    assert.deepEqual(h.mixin['detectGroups'](), ['animal']);
    await h.mixin.putMixinSetting('detect', ['Person', 'Vehicle', 'Animal']);
    assert.deepEqual(h.mixin['detectGroups'](), ['person', 'vehicle', 'animal']);
    frame(ctl.starts[0], Date.now() + 5);
    await eventually(() => h.calls.frame.length === 2);
    h.calls.frame[1].d.res(confirmedReply);
    await eventually(() => motion() === true);
    await h.mixin.putMixinSetting('mode', 'Shadow');
    assert.ok(ctl.logs.some(l => /Mode: Shadow/.test(String(l[1]))));
    await h.mixin.release();
});

test('Detect value is parsed whatever the UI sends: array, comma string, JSON string, single string, junk, empty', async () => {
    for (const [v, want] of [[['Vehicle', 'Animal'], 'vehicle, animal'], ['Person,Vehicle', 'person, vehicle'], ['["Animal"]', 'animal'], ['animal', 'animal'], ['nonsense', 'person, vehicle, animal'], ['', 'person, vehicle, animal'], [[], 'person, vehicle, animal']] as [any, string][]) {
        const h = make({ settings: { mode: 'Gate', detect: v } });
        assert.equal(h.mixin['detectGroups']().join(', '), want, `${JSON.stringify(v)} -> ${want}`);
        await h.mixin.release();
    }
});

test('Settings notifications: none from frames, episodes, replies, timers or health; exactly one after the user\'s own save', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(confirmedReply);
    await eventually(() => motion() === true);
    h.mixin.onHealth(true, '');
    h.mixin.onHealth(false, 'down');
    h.mixin.onHealth(true, '');
    h.raw(false);
    await sleep(1300); // timers (tick) run
    assert.equal((ctl as any).notified, 0, 'nothing may re-render the page under the user\'s hands');
    const r = await h.mixin.putMixinSetting('detect', ['Person']);
    assert.equal((ctl as any).notified, 1, 'exactly one, right after the user\'s save');
    assert.equal(r, true, 'handled: the base class does not notify again');
    await h.mixin.release();
});

test('claim is a hint only: a person under a dog_cat claim still confirms; a bicycle claim is ignored', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    h.claim('dog_cat');
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(confirmedReply);
    await eventually(() => motion() === true);
    await h.mixin.release();
    const b = make({ settings: { mode: 'Gate' } });
    b.raw(true);
    b.claim('bicycle');
    await sleep(40);
    assert.ok(!ctl.logs.some(l => /not verifiable/.test(String(l[1]))), 'a bicycle claim is not a claim');
    await b.mixin.release();
});

const blobG = () => { const g = new Uint8Array(128 * 72); for (let y = 20; y < 40; y++) for (let x = 40; x < 60; x++) g[y * 128 + x] = 200; return g; };

test('pixel-gate episode: degrade never publishes motion; after a veto the camera\'s own trigger starts a fresh episode in the same second', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', confirmTimeoutS: 1, gate: { warmupFrames: 0 } as any } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    s.o.gray.onGray(new Uint8Array(128 * 72));
    s.o.gray.onGray(blobG());
    await eventually(() => h.calls.frame.length === 1);
    h.mixin.onHealth(false, 'down');
    await sleep(40);
    assert.equal(ctl.writes.filter(w => w[1] === true).length, 0, 'degraded pass-through is the camera\'s raw only');
    h.mixin.onHealth(true, '');
    await sleep(40);
    const resets = h.calls.reset.length;
    h.raw(true);
    await eventually(() => h.calls.reset.length === resets + 1, 3000, 'camera trigger starts its own episode');
    await h.mixin.release();
});

test('a synthetic confirmation publishes true before any camera event, and the release publishes false; startup alone never writes', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', gate: { warmupFrames: 0 } as any } });
    await eventually(() => ctl.starts.length === 1);
    assert.equal(ctl.writes.length, 0);
    const s = ctl.starts[0];
    frame(s);
    s.o.gray.onGray(new Uint8Array(128 * 72));
    s.o.gray.onGray(blobG());
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(confirmedReply);
    await eventually(() => motion() === true);
    frame(s, Date.now() + 5);
    await eventually(() => h.calls.frame.length === 2);
    h.calls.frame[1].d.res(idleReply);
    await eventually(() => motion() === false);
    assert.deepEqual(ctl.writes.map(w => w[1]), [true, false]);
    await h.mixin.release();
});

test('a failing source restart is not respawned per event: the backoff timer owns the next spawn', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    ctl.starts[0].exit('ffmpeg exited 1');
    await sleep(100);
    h.mixin.onHealth(true, '');
    h.mixin.applyConfig();
    await sleep(200);
    assert.equal(ctl.starts.length, 1, 'no spawn before the backoff delay');
    await eventually(() => ctl.starts.length === 2, 3000);
    await h.mixin.release();
});

test('a malformed reply field is a skipped frame, never an outage', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res({ ...idleReply, detections: [{ cls: 'car', score: 0.9, box: null }], vehicle: { moving: true, tracks: [null] } });
    await sleep(60);
    assert.ok(!ctl.logs.some(l => /verifier unavailable/.test(String(l[1]))));
    await h.mixin.release();
});

test('release: a pixel-gate confirmation published before any camera event is released correctly', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', gate: { warmupFrames: 0 } as any } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    s.o.gray.onGray(new Uint8Array(128 * 72));
    s.o.gray.onGray(blobG());
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(confirmedReply);
    await eventually(() => motion() === true);
    await h.mixin.release();
    assert.equal(motion(), false, 'our own true is not left behind');
});

// ---- heartbeat (continuous tracking) -------------------------------------------------------------

const hbP = (extra: object = {}) => ({ person: { state: 'idle', confirmed: false, present: false }, ...extra });
const personHold = { person: { state: 'holding', confirmed: true, present: true }, detections: [{ cls: 'person', score: 0.9, box: [0.1, 0.1, 0.4, 0.8] }] };
const carMoving = hbP({ vehicle: { moving: true }, detections: [{ cls: 'car', score: 0.9, box: [0.1, 0.1, 0.5, 0.5] }] });
const carParked = hbP({ vehicle: { moving: false, tracks: [{ id: 1, cls: 'car', parked: true, since_ms: 99999, box: [0.1, 0.1, 0.5, 0.5] }] }, detections: [{ cls: 'car', score: 0.9, box: [0.1, 0.1, 0.5, 0.5] }] });
const hb = (mode: string, extra: Record<string, any> = {}) => make({ settings: { mode, frameProcess: 'Keep warm', heartbeatMs: 1, ...extra } });

test('heartbeat: an idle camera posts the fresh frame at low priority with no reset and the panel shows the analysis', async () => {
    const h = hb('Shadow');
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    assert.equal(h.calls.frame[0].a[3], true, 'prio=low');
    assert.equal(h.calls.reset.length, 0);
    h.calls.frame[0].d.res(hbP({ detections: [{ cls: 'person', score: 0.91, box: [0.1, 0.1, 0.4, 0.8] }], total_ms: 30 }));
    await sleep(40);
    assert.match(String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value), /0 s ago \u00b7 person 0\.91/);
    await h.mixin.release();
});

test('heartbeat Shadow: same verified unflagged episode as Gate (would raise, released by the tracker, counted), motion never touched', async () => {
    const h = hb('Shadow');
    await eventually(() => ctl.starts.length === 1);
    const st = ctl.starts[0];
    frame(st);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(personHold);
    await eventually(() => h.calls.frame.length === 2, 3000, 'the verified scan posts');
    h.calls.frame[1].d.res(personHold);
    await eventually(() => ctl.logs.some(l => /person seen but camera did not trigger: would raise motion \(Shadow\)/.test(String(l[1]))));
    assert.equal(h.mixin['episode'].wouldPublish, true);
    frame(st, Date.now() + 5);
    await eventually(() => h.calls.frame.length === 3);
    h.calls.frame[2].d.res(idleReply);
    await eventually(() => h.mixin['episode'].wouldPublish === false, 3000, 'released by the tracker');
    assert.equal(ctl.writes.length, 0);
    assert.match(String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value), /1 unflagged/);
    await h.mixin.release();
});

test('heartbeat Gate: a static person confirmed over the heartbeat raises motion once through the verified path (no reset), holds, releases when gone', async () => {
    const h = hb('Gate');
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(personHold);
    await eventually(() => h.calls.frame.length === 2, 3000, 'the verified scan posts its first frame');
    assert.equal(h.calls.reset.length, 0, 'no reset: the session already tracks the subject');
    assert.equal(motion(), false, 'the heartbeat alone never raises motion');
    h.calls.frame[1].d.res(personHold);
    await eventually(() => motion() === true);
    frame(s, Date.now() + 5);
    await eventually(() => h.calls.frame.length === 3);
    h.calls.frame[2].d.res(personHold);
    await sleep(40);
    assert.equal(motion(), true);
    frame(s, Date.now() + 10);
    await eventually(() => h.calls.frame.length === 4);
    h.calls.frame[3].d.res(idleReply);
    await eventually(() => motion() === false);
    assert.equal(ctl.writes.filter(w => w[1] === true).length, 1);
    await h.mixin.release();
});

test('heartbeat Gate: a parked vehicle never confirms; a moving one does after 2 consecutive replies', async () => {
    const h = hb('Gate');
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    for (let i = 0; i < 3; i++) {
        frame(s, Date.now() + i);
        await eventually(() => h.calls.frame.length === i + 1);
        h.calls.frame[i].d.res(carParked);
        await sleep(30);
    }
    assert.equal(ctl.writes.length, 0);
    assert.equal(h.calls.frame.length, 3, 'no scan was started');
    const n = h.calls.frame.length;
    frame(s, Date.now() + 10);
    await eventually(() => h.calls.frame.length === n + 1);
    h.calls.frame[n].d.res(carMoving);
    await sleep(30);
    frame(s, Date.now() + 20);
    await eventually(() => h.calls.frame.length === n + 2);
    h.calls.frame[n + 1].d.res(carMoving);
    await eventually(() => h.calls.frame.length === n + 3, 3000, 'second tick starts the verified scan');
    assert.equal(h.calls.reset.length, 0);
    await h.mixin.release();
});

test('heartbeat is skipped during an episode, when Off, with the kill switch, when degraded; stale replies are silent', async () => {
    const h = hb('Gate');
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    h.raw(true);
    await eventually(() => h.calls.reset.length === 1);
    h.calls.reset[0].d.res({});
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    assert.equal(h.calls.frame[0].a[3], false, 'a camera episode posts at normal priority, not as a heartbeat');
    await h.mixin.release();
    const off = hb('Off');
    await sleep(60);
    if (ctl.starts[0]) frame(ctl.starts[0]);
    await sleep(60);
    assert.equal(off.calls.frame.length, 0);
    await off.mixin.release();
    const k = hb('Shadow');
    k.plugin.config.killSwitch = true;
    await eventually(() => ctl.starts.length >= 1);
    frame(ctl.starts[0]);
    await sleep(60);
    assert.equal(k.calls.frame.length, 0, 'kill switch');
    await k.mixin.release();
    const d = hb('Shadow');
    await eventually(() => ctl.starts.length === 1);
    d.mixin.onHealth(false, 'down');
    frame(ctl.starts[0]);
    await sleep(60);
    assert.equal(d.calls.frame.length, 0, 'degraded');
    await d.mixin.release();
    const stale = make({ settings: { mode: 'Shadow', frameProcess: 'Keep warm', heartbeatMs: 1 }, client: {
        frame: async () => null, reset: async () => ({}), health: async () => ({ ok: true }), check: async () => ({ ok: true }), config: async () => ({}),
    } });
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await sleep(60);
    assert.ok(!ctl.logs.some(l => /verifier unavailable/.test(String(l[1]))));
    await stale.mixin.release();
});

test('heartbeat jitter spreads cameras by device id', () => {
    const { heartbeatJitter } = require('../src/mixin') as typeof import('../src/mixin');
    const vals = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5'].map(id => heartbeatJitter(id, 1000));
    assert.ok(vals.every(v => v >= 0 && v < 1000));
    assert.ok(new Set(vals).size > 1);
});

test('heartbeat vehicle: ticks separated by a camera episode do not add up to a confirmation', async () => {
    const h = hb('Gate');
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(carMoving);
    await sleep(30);
    h.raw(true);
    await sleep(30);
    h.raw(false);
    await sleep(30);
    const n = h.calls.frame.length;
    for (const f of h.calls.frame.slice(1)) f.d.res(idleReply);
    await eventually(() => !h.mixin['episode'].scanning, 8000, 'episode over');
    await sleep(30);
    const m = h.calls.frame.length;
    frame(s, Date.now() + 50);
    await eventually(() => h.calls.frame.length === m + 1);
    h.calls.frame[m].d.res(carMoving);
    await sleep(60);
    assert.equal(h.calls.frame.length, m + 1, 'one fresh tick alone starts no verified scan');
    assert.ok(n >= 1);
    await h.mixin.release();
});

test('heartbeat: a reply from before a camera episode is void after it', async () => {
    const h = hb('Shadow');
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.raw(true);
    await eventually(() => h.calls.reset.length === 1);
    h.calls.reset[0].d.res({});
    h.raw(false);
    await eventually(() => !h.mixin['episode'].scanning && !h.mixin['scanActive'], 8000);
    h.calls.frame[0].d.res(personHold);
    await sleep(60);
    assert.ok(!ctl.logs.some(l => /\(heartbeat\)/.test(String(l[1]))), 'stale heartbeat reply ignored');
    await h.mixin.release();
});

test('heartbeat: once the subject is gone, the next camera episode resets the session again', async () => {
    const h = hb('Shadow');
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(hbP({ animal: { present: true, confirmed: false } })); // present (tracked), not confirmed: no scan
    await sleep(30);
    frame(s, Date.now() + 5);
    await eventually(() => h.calls.frame.length === 2);
    h.calls.frame[1].d.res(hbP());
    await sleep(30);
    h.raw(true);
    await eventually(() => h.calls.reset.length === 1, 2000, 'reset not skipped for an absent subject');
    await h.mixin.release();
});

test('a codec-probe failure switches the camera to the long-probe default stream for good', async () => {
    const h = hb('Shadow');
    await eventually(() => ctl.starts.length === 1);
    assert.ok(!ctl.starts[0].o.longProbe);
    ctl.starts[0].exit('codec probe failed: ffmpeg exited 183; Could not find codec parameters');
    await eventually(() => ctl.starts.length === 2);
    assert.equal(ctl.starts[1].o.longProbe, true);
    assert.ok(ctl.logs.some(l => /longer probe/.test(String(l[1]))));
    await h.mixin.release();
});

test('Stream setting: default unchanged; choosing a stream restarts the source on it; no first frame falls back to the default for good', async () => {
    const h = hb('Shadow');
    await eventually(() => ctl.starts.length === 1);
    assert.equal(ctl.starts[0].o.streamId, undefined, 'default: no id');
    (h.mixin as any).mixinDevice.getVideoStreamOptions = async () => [{ id: 'sub', name: 'sub' }];
    const st = (await h.mixin.getMixinSettings()).find(x => x.key === 'stream')!;
    assert.deepEqual(st.choices, ['Default (already open stream)', 'sub']);
    assert.equal(st.value, 'Default (already open stream)');
    await h.mixin.putMixinSetting('stream', 'sub');
    await eventually(() => ctl.starts.length === 2, 2000, 'restarted at once');
    assert.equal(ctl.starts[0].stopped, true);
    assert.equal(ctl.starts[1].o.streamId, 'sub');
    assert.equal((await h.mixin.getMixinSettings()).find(x => x.key === 'stream')!.value, 'sub');
    ctl.starts[1].exit('codec probe failed: ffmpeg exited 183');
    await eventually(() => ctl.starts.length === 3);
    assert.equal(ctl.starts[2].o.streamId, undefined);
    assert.ok(ctl.logs.some(l => /stream sub failed, using the default stream/.test(String(l[1]))));
    await h.mixin.release();
});

test('Stream setting: an unknown stored id uses the default with a one-time warning', async () => {
    const h = hb('Shadow');
    await eventually(() => ctl.starts.length === 1);
    (h.mixin as any).storageSettings.values; // stored directly
    await (h.mixin as any).storageSettings.putSetting('stream', 'gone');
    ctl.starts[0].o.onStream({ name: 'default', explicit: false, unknown: true });
    ctl.starts[0].o.onStream({ name: 'default', explicit: false, unknown: true });
    assert.equal(ctl.logs.filter(l => /not among the camera's streams, using the default stream/.test(String(l[1]))).length, 1);
    assert.equal((h.mixin as any).chosenStream(), undefined);
    await h.mixin.release();
});

test('thresholds are posted to the service at start; an old service rejecting the new keys gets the person keys only', async () => {
    const posted: any[] = [];
    const h = make({ settings: { mode: 'Shadow' }, client: {
        frame: async () => null, reset: async () => ({}), health: async () => ({ ok: true }), check: async () => ({ ok: true }),
        config: async (_id: string, cfg: any) => { posted.push(cfg); if ('vehicle_thr' in cfg) { const e: any = new Error('bad'); e.status = 400; throw e; } return {}; },
    } });
    await eventually(() => posted.length === 2);
    assert.deepEqual(posted[0], { start_thr: 0.65, cont_thr: 0.2, vehicle_thr: 0.4, animal_start_thr: 0.6, animal_cont_thr: 0.3, zoom_out: 0, second_look: true, name: 'Cam 1' });
    assert.deepEqual(posted[1], { start_thr: 0.65, cont_thr: 0.2 });
    assert.ok(!ctl.logs.some(l => /tracker config not applied/.test(String(l[1]))));
    await h.mixin.release();
});

// ---- sensorless cameras (no MotionSensor of their own) ------------------------------------------

test('sensorless: no motion listener, false written once at start, panel and overview say so; normal cameras write nothing at start', async () => {
    const n = make({ settings: { mode: 'Gate' } });
    await sleep(20);
    assert.equal(ctl.writes.length, 0, 'normal camera: startup transparent');
    await n.mixin.release();
    const h = make({ settings: { mode: 'Gate' }, sensorless: true });
    await sleep(20);
    assert.equal(ctl.listeners, 1, 'only the ObjectDetector listener');
    assert.deepEqual(ctl.writes.map(w => w[1]), [false]);
    assert.match(String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value), /sensorless/);
    assert.match(h.mixin.overviewRow().name, /\(sensorless\)/);
    await h.mixin.release();
});

test('sensorless Shadow: a confirmed heartbeat person makes wouldPublish true, motion is never written true', async () => {
    const h = make({ settings: { mode: 'Shadow', frameProcess: 'Keep warm', heartbeatMs: 1 }, sensorless: true });
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(personHold);
    await eventually(() => h.calls.frame.length === 2, 3000);
    h.calls.frame[1].d.res(personHold);
    await eventually(() => h.mixin['episode'].wouldPublish === true, 3000, 'would publish');
    assert.ok(!ctl.writes.some(w => w[1] === true));
    await h.mixin.release();
});

test('sensorless Gate: a confirmed heartbeat person raises motion through the verified path and releases when gone', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', heartbeatMs: 1 }, sensorless: true });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(personHold);
    await eventually(() => h.calls.frame.length === 2, 3000);
    h.calls.frame[1].d.res(personHold);
    await eventually(() => motion() === true);
    frame(s, Date.now() + 10);
    await eventually(() => h.calls.frame.length === 3);
    h.calls.frame[2].d.res(idleReply);
    await eventually(() => motion() === false);
    assert.equal(ctl.writes.filter(w => w[1] === true).length, 1);
    assert.ok(!ctl.logs.some(l => /NOT the outermost/.test(String(l[1]))));
    await h.mixin.release();
});

test('sensorless: failing open keeps motion false (nothing to pass through)', async () => {
    const h = make({ settings: { mode: 'Gate' }, sensorless: true });
    await sleep(20);
    h.mixin.onHealth(false, 'down');
    await sleep(20);
    assert.ok(!ctl.writes.some(w => w[1] === true));
    await h.mixin.release();
});

test('Zoom-out: Off by default, a choice is stored as a share, re-posts the config at once, shown on the panel', async () => {
    const posted: any[] = [];
    const h = make({ settings: { mode: 'Shadow' }, client: {
        frame: async () => null, reset: async () => ({}), health: async () => ({ ok: true }), check: async () => ({ ok: true }),
        config: async (_id: string, cfg: any) => { posted.push(cfg); return {}; },
    } });
    await eventually(() => posted.length === 1);
    assert.equal(posted[0].zoom_out, 0);
    let st = await h.mixin.getMixinSettings();
    const z = st.find(x => x.key === 'zoomOut')!;
    assert.deepEqual(z.choices, ['Off', '20 %', '35 %', '50 %']); assert.equal(z.value, 'Off');
    assert.ok(!/zoom-out/i.test(String(st.find(x => x.key === 'status')!.value)));
    await h.mixin.putMixinSetting('zoomOut', '35 %');
    await eventually(() => posted.length === 2);
    assert.equal(posted[1].zoom_out, 0.35);
    assert.equal((h.mixin as any).storageSettings.values.zoomOut, '0.35');
    st = await h.mixin.getMixinSettings();
    assert.equal(st.find(x => x.key === 'zoomOut')!.value, '35 %');
    assert.match(String(st.find(x => x.key === 'status')!.value), /zoom-out 35 %/);
    await h.mixin.release();
});

// ---- exclusion zones -------------------------------------------------------------------------

const personIn = { person: { state: 'confirmed', confirmed: true, present: true }, detections: [{ cls: 'person', score: 0.9, box: [0.1, 0.1, 0.3, 0.6] }] };

test('zones: add by name, draw with the clippath editor, stored as JSON; settings show the editor', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    await h.mixin.putMixinSetting('zoneNames', ['Street']);
    await h.mixin.putMixinSetting('zone:Street', JSON.stringify([[0, 0], [0.5, 0], [0.5, 1], [0, 1]]));
    assert.deepEqual(JSON.parse((h.mixin as any).storage.getItem('zones')), [{ name: 'Street', points: [[0, 0], [0.5, 0], [0.5, 1], [0, 1]] }]);
    const st = await h.mixin.getMixinSettings();
    const ed = st.find(x => x.key === 'zone:Street')!;
    assert.equal(ed.type, 'clippath'); assert.equal(ed.subgroup, 'Zones');
    assert.deepEqual(st.find(x => x.key === 'zoneNames')!.value, ['Street']);
    await h.mixin.putMixinSetting('zoneNames', []);
    assert.deepEqual(JSON.parse((h.mixin as any).storage.getItem('zones')), []);
    await h.mixin.putMixinSetting('zones', 'not json');
    assert.ok(ctl.logs.some(l => l[0] === 'warn' && /not valid JSON/.test(String(l[1]))));
    await h.mixin.release();
});

test('zones: a confirmed person inside an exclusion zone never raises motion in Gate; the panel shows it masked', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    await h.mixin.putMixinSetting('zones', JSON.stringify([{ name: 'Street', points: [[0, 0], [0.5, 0], [0.5, 1], [0, 1]] }]));
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    for (let i = 0; i < 3; i++) {
        frame(s, Date.now() + i * 10);
        await eventually(() => h.calls.frame.length === i + 1);
        h.calls.frame[i].d.res(personIn);
        await sleep(30);
    }
    assert.equal(motion(), false, 'masked person never confirms');
    const html = String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value);
    assert.match(html, /person 0\.90 \(masked\)/);
    assert.match(html, /<polygon points="0,0 50,0 50,100 0,100" fill="currentColor" fill-opacity="\.25"[^>]*stroke-dasharray="2 2"/);
    await h.mixin.release();
});

test('camera settings tabs: Overview (Status only) first, then Setup (Mode, Detect, Stream, Zoom-out), then Zones', async () => {
    const h = make({ settings: { mode: 'Shadow' } });
    await h.mixin.putMixinSetting('zoneNames', ['Street']);
    const st = await h.mixin.getMixinSettings();
    assert.ok(st.every(x => !!x.subgroup), 'no ungrouped setting');
    assert.deepEqual([...new Set(st.map(x => x.subgroup))], ['Overview', 'Setup', 'Zones'], 'exactly three tabs');
    assert.equal(st.find(x => x.key === 'zone:Street')!.title, 'Edit zone: Street');
    assert.deepEqual(st.map(x => [x.key, x.subgroup]), [
        ['refresh', 'Overview'], ['status', 'Overview'], ['mode', 'Setup'], ['detect', 'Setup'], ['stream', 'Setup'], ['zoomOut', 'Setup'],
        ['zoneNames', 'Zones'], ['zone:Street', 'Zones'],
    ]);
    await h.mixin.release();
});

test('GPU decode: on by default for every camera (cap 64, default VA driver, no LIBVA_DRIVER_NAME); a fallback or a failed start releases the slot at once', async () => {
    const { DEFAULTS } = require('../src/defaults') as typeof import('../src/defaults');
    (ctl as any).vaapi = true;
    try {
        const src = require('fs').readFileSync(require('path').join(__dirname, '..', '..', 'src', 'defaults.ts'), 'utf8');
        assert.match(src, /hwDecode: 'Auto'/); assert.match(src, /maxHwCameras: 64,/); assert.match(src, /vaDriver: '',/);
        const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', hwDecode: 'Auto' } });
        assert.equal(DEFAULTS.maxHwCameras, 64); assert.equal(DEFAULTS.vaDriver, '');
        await eventually(() => ctl.starts.length === 1);
        assert.equal(ctl.starts[0].o.hwaccel, true); assert.equal(ctl.starts[0].o.vaDriver, undefined, 'LIBVA_DRIVER_NAME left unset');
        assert.equal(h.plugin.hwSlotsUsed, 1);
        ctl.starts[0].exit('hwaccel failed: ffmpeg exited 1; Failed to transfer data to output frame');
        await eventually(() => ctl.starts.length === 2);
        assert.equal(ctl.starts[1].o.hwaccel, false);
        assert.equal(h.plugin.hwSlotsUsed, 0, 'software fallback holds no slot');
        await h.mixin.release();
        (ctl as any).failStart = true;
        const k = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', hwDecode: 'Auto', maxHwCameras: 64 } });
        await eventually(() => ctl.starts.length >= 1);
        await sleep(30);
        assert.equal(k.plugin.hwSlotsUsed, 0, 'a failed start holds no slot');
        await k.mixin.release();
    }
    finally { (ctl as any).vaapi = false; (ctl as any).failStart = false; }
});

test('a GPU start that exits 0 before any frame falls back to software on the next start and never loops on the GPU', async () => {
    (ctl as any).vaapi = true;
    try {
        const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', hwDecode: 'Auto' } });
        await eventually(() => ctl.starts.length === 1);
        assert.equal(ctl.starts[0].o.hwaccel, true);
        ctl.starts[0].exit('hwaccel failed: ffmpeg exited 0; Failed to read image from surface');
        await eventually(() => ctl.starts.length === 2);
        assert.equal(ctl.starts[1].o.hwaccel, false);
        ctl.starts[1].exit('ffmpeg exited 1: x');
        await eventually(() => ctl.starts.length === 3, 4000);
        assert.equal(ctl.starts[2].o.hwaccel, false, 'stays in software');
        await h.mixin.release();
    }
    finally { (ctl as any).vaapi = false; }
});

// ---- continuous analysis and the redesigned panel ------------------------------------------------

async function closedEpisode(h: any) {
    const s = ctl.starts[0];
    h.raw(true);
    await eventually(() => h.calls.reset.length >= 1 || h.calls.frame.length >= 1);
    frame(s, Date.now() + 1);
    await eventually(() => h.calls.frame.some((f: any) => f.a[3] === false));
    let i = h.calls.frame.findIndex((f: any) => f.a[3] === false);
    h.calls.frame[i].d.res(confirmedReply);
    await eventually(() => motion() === true);
    frame(s, Date.now() + 2);
    await eventually(() => h.calls.frame.length > i + 1);
    h.calls.frame[h.calls.frame.length - 1].d.res(idleReply);
    await eventually(() => motion() === false, 3000, 'released while raw stays on (closed)');
    assert.equal(h.mixin['episode'].scanning, false);
}

test('heartbeat keeps analysing while camera raw motion stays on after a release (closed), at low priority, and re-opens on a person', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', heartbeatMs: 1 } });
    await eventually(() => ctl.starts.length === 1);
    await closedEpisode(h);
    const n = h.calls.frame.length;
    frame(ctl.starts[0], Date.now() + 10);
    await eventually(() => h.calls.frame.length === n + 1, 3000, 'heartbeat posts while raw is on');
    assert.equal(h.calls.frame[n].a[3], true, 'low priority');
    h.calls.frame[n].d.res(personHold);
    await eventually(() => h.calls.frame.length === n + 2, 3000, 'a new verified scan starts');
    h.calls.frame[n + 1].d.res(personHold);
    await eventually(() => motion() === true, 3000, 're-opened');
    await h.mixin.release();
});

test('one request in flight per camera: an episode frame waits for an in-flight heartbeat', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', heartbeatMs: 1 } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    h.raw(true);
    await sleep(60);
    assert.equal(h.calls.frame.length, 1, 'no second request while the heartbeat is in flight');
    h.calls.frame[0].d.res(hbP());
    await eventually(() => h.calls.frame.length === 2, 3000, 'the episode frame goes once the heartbeat answered');
    assert.equal(h.calls.frame[1].a[3], false);
    await h.mixin.release();
});

test('panel: Now card (Gate: motion and reason; Shadow: published and would-publish), Detected now vs Tracked, dashed tracked box', async () => {
    const h = make({ settings: { mode: 'Shadow', frameProcess: 'Keep warm', heartbeatMs: 1 } });
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    h.calls.frame[0].d.res(personHold);
    await sleep(40);
    frame(ctl.starts[0], Date.now() + 5);
    await eventually(() => h.calls.frame.length === 2);
    const now = Date.now();
    h.calls.frame[1].d.res({ person: { state: 'holding', confirmed: true, present: true, since_ms: now - 60000, last_seen_ms: now - 3000 }, detections: [{ cls: 'car', score: 0.7, box: [0.6, 0.6, 0.9, 0.9] }], vehicle: { moving: false, tracks: [{ id: 1, cls: 'car', parked: true, state: 'parked', since_ms: now - 3600000, box: [0.6, 0.6, 0.9, 0.9] }] } });
    await sleep(40);
    const html = String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value);
    assert.match(html, />Now</);
    assert.match(html, /Motion OFF/);
    assert.match(html, /Would publish ON \u00b7 Person tracked \u00b7 since \d\d:\d\d:\d\d \u00b7 3 s/);
    assert.match(html, /Detected<\/div><div>car 0\.70 parked/);
    assert.match(html, /Tracked<\/div><div>person \u00b7 held \u00b7 3 s/);
    assert.match(html, /car \u00b7 parked \u00b7 since/);
    assert.ok(!/what Home|drives motion|press Refresh/.test(html), 'no explanatory sentences');
    assert.match(html, /border:2px dashed currentColor[^>]*><span[^>]*>person tracked \(3 s\)/, 'tracked-but-not-detected person at its last box');
    await h.mixin.release();
    const g = make({ settings: { mode: 'Gate' } });
    const gh = String((await g.mixin.getMixinSettings()).find(x => x.key === 'status')!.value);
    assert.match(gh, /Motion OFF<\/div><div[^>]*>Nothing tracked/);
    await g.mixin.release();
});

// ---- episode journal ------------------------------------------------------------------------------

test('journal: a camera episode posts start, confirmed update and end with one id, in Shadow as in Gate; never affects motion', async () => {
    for (const mode of ['Gate', 'Shadow']) {
        const h = make({ settings: { mode } });
        h.raw(true);
        await eventually(() => ctl.starts.length === 1);
        const s = ctl.starts[0];
        frame(s);
        await eventually(() => h.calls.frame.length === 1);
        h.calls.frame[0].d.res(confirmedReply);
        await eventually(() => h.calls.episodes.some((e: any) => e.action === 'update'));
        h.raw(false);
        frame(s, Date.now() + 5);
        await eventually(() => h.calls.frame.length === 2);
        h.calls.frame[1].d.res(idleReply);
        await eventually(() => h.calls.episodes.some((e: any) => e.action === 'end'));
        const ev = h.calls.episodes;
        assert.deepEqual(ev.map((e: any) => e.action), ['start', 'update', 'end'], mode);
        assert.equal(new Set(ev.map((e: any) => e.id)).size, 1);
        assert.match(ev[0].id, /^[0-9a-f-]{36}$/);
        assert.equal(ev[0].kind, 'camera'); assert.equal(ev[0].mode, mode.toLowerCase());
        assert.equal(ev[1].outcome, 'confirmed'); assert.match(ev[1].timeline_event.text, /^confirmed person/);
        assert.equal(ev[2].outcome, 'released'); assert.deepEqual(ev[2].groups, ['person']);
        await h.mixin.release();
        await releaseAll();
        ctl.starts = [];
    }
});

test('journal: an old service (404) is tolerated silently and no longer called; a vetoed episode ends vetoed', async () => {
    (ctl as any).journal404 = true;
    try {
        const h = make({ settings: { mode: 'Gate', confirmTimeoutS: 1 } });
        h.raw(true);
        await eventually(() => h.calls.episodes.length === 1);
        await sleep(30);
        h.raw(false);
        await sleep(1500);
        assert.equal(h.calls.episodes.length, 1, 'no further journal calls after the 404');
        assert.ok(!ctl.logs.some(l => l[0] === 'warn' && /episodes|journal/.test(String(l[1]))));
        await h.mixin.release();
    } finally { (ctl as any).journal404 = false; }
    const v = make({ settings: { mode: 'Gate', confirmTimeoutS: 1 } });
    v.raw(true);
    await eventually(() => ctl.starts.length >= 1);
    frame(ctl.starts[ctl.starts.length - 1]);
    await eventually(() => v.calls.frame.length === 1);
    v.calls.frame[0].d.res(idleReply);
    for (let i = 0; i < 20 && !v.calls.episodes.some((e: any) => e.action === 'end'); i++) {
        frame(ctl.starts[ctl.starts.length - 1], Date.now() + i); // keep frames flowing: a veto, not a dead stream
        await sleep(100);
        for (const f of v.calls.frame) f.d.res(idleReply);
    }
    await eventually(() => v.calls.episodes.some((e: any) => e.action === 'end'), 4000);
    assert.equal(v.calls.episodes.find((e: any) => e.action === 'end').outcome, 'vetoed');
    await v.mixin.release();
});

test('camera Refresh button: top of the Overview tab; a press sends exactly one Settings refresh and stores nothing', async () => {
    const h = make({ settings: { mode: 'Shadow' } });
    const st = await h.mixin.getMixinSettings();
    assert.deepEqual([st[0].key, st[0].type, st[0].subgroup], ['refresh', 'button', 'Overview']);
    const before = JSON.stringify((h.mixin as any).store);
    const n0 = (ctl as any).notified || 0;
    await h.mixin.putMixinSetting('refresh', undefined as any);
    assert.equal(((ctl as any).notified || 0) - n0, 1);
    assert.equal(JSON.stringify((h.mixin as any).store), before, 'no storage writes');
    assert.match(String((await h.mixin.getMixinSettings())[1].value), /Updated \d\d:\d\d:\d\d/);
    await h.mixin.release();
});

test('service 0.3.0 fields: tracked person boxes from tracks_detail, hold remaining shown; older replies keep the fallback', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => ctl.starts.length === 1);
    frame(ctl.starts[0]);
    await eventually(() => h.calls.frame.length === 1);
    const t = Date.now();
    h.calls.frame[0].d.res({ t, person: { state: 'holding', confirmed: true, present: true, since_ms: t - 20000, last_seen_ms: t - 2000, tracks: 1, hold_until_ms: t + 6000,
        tracks_detail: [{ id: 4, box: [0.2, 0.2, 0.4, 0.9], score: 0.8, last_seen_ms: t - 2000, since_ms: t - 20000, stationary: false }] }, detections: [] });
    await sleep(40);
    const html = String((await h.mixin.getMixinSettings()).find(x => x.key === 'status')!.value);
    assert.match(html, /Tracked<\/div><div>person · held · 6 s left/);
    assert.match(html, /left:20%;top:20%;width:20%;height:70%;border:2px dashed currentColor/, 'box from tracks_detail');
    await h.mixin.release();
});

test('journal: an unflagged scan is journalled only when it confirms (start back-dated to its trigger); one that finds nothing is never journalled', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', heartbeatMs: 1, confirmTimeoutS: 1 } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    frame(s);
    await eventually(() => h.calls.frame.length === 1);
    const t0 = Date.now();
    h.calls.frame[0].d.res(personHold);
    await eventually(() => h.calls.frame.length === 2, 3000, 'unflagged scan posts');
    await sleep(50);
    assert.equal(h.calls.episodes.length, 0, 'not journalled before it confirms');
    h.calls.frame[1].d.res(personHold);
    await eventually(() => h.calls.episodes.length === 2);
    const [st, up] = h.calls.episodes;
    assert.deepEqual([st.action, st.kind, up.action, up.outcome], ['start', 'unflagged', 'update', 'confirmed']);
    assert.ok(st.t_ms >= t0 && st.t_ms <= up.t_ms, 'start back-dated to the trigger');
    await h.mixin.release();
    const v = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', heartbeatMs: 1, confirmTimeoutS: 1 } });
    await eventually(() => ctl.starts.length >= 1);
    const sv = ctl.starts[ctl.starts.length - 1];
    frame(sv);
    await eventually(() => v.calls.frame.length === 1);
    v.calls.frame[0].d.res(personHold);
    for (let i = 0; i < 25 && !v.mixin['episode'].scanning; i++) await sleep(20);
    for (let i = 0; i < 20 && v.mixin['episode'].scanning; i++) {
        frame(sv, Date.now() + i);
        await sleep(100);
        for (const f of v.calls.frame) f.d.res(idleReply);
    }
    await sleep(100);
    assert.equal(v.calls.episodes.length, 0, 'a vetoed unflagged scan is never journalled');
    await v.mixin.release();
});

test('release closes an open journal episode as aborted (plugin restart); config carries the camera name', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => h.calls.episodes.length === 1);
    await h.mixin.release();
    const end = h.calls.episodes[h.calls.episodes.length - 1];
    assert.deepEqual([end.action, end.outcome, end.note], ['end', 'aborted', 'plugin restart']);
    assert.equal(end.id, h.calls.episodes[0].id);
    const n = h.calls.episodes.length;
    const q = make({ settings: { mode: 'Gate' } });
    await q.mixin.release();
    assert.equal(q.calls.episodes.length, 0, 'nothing open: nothing sent');
    assert.equal(n, 2);
});

test('posting rate: 2 fps while verifying, 1 fps once confirmed (holding), back to every frame at a max-hold re-pend; no heartbeat during the episode', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', heartbeatMs: 1, maxHoldS: 2, holdIntervalMs: 1000 } });
    await eventually(() => ctl.starts.length === 1);
    const s = ctl.starts[0];
    h.raw(true);
    await eventually(() => h.mixin['episode'].scanning === true);
    const feed = async (n: number, reply: any) => {
        const before = h.calls.frame.length;
        for (let i = 0; i < n; i++) {
            frame(s, Date.now());
            await sleep(250);
            for (const f of h.calls.frame) f.d.res(reply);
            await sleep(10);
        }
        return h.calls.frame.length - before;
    };
    const pending = await feed(4, idleReply);
    assert.ok(pending >= 3, `verifying: every frame posted (${pending})`);
    await feed(1, confirmedReply);
    await eventually(() => motion() === true);
    const holding = await feed(6, confirmedReply);
    assert.ok(holding >= 1 && holding <= 3, `holding: about 1 fps (${holding} of 6 frames at ~4 fps)`);
    assert.ok(h.calls.frame.every((f: any) => f.a[3] === false), 'no low-priority heartbeat during the episode');
    await eventually(() => !h.mixin['episode'].confirmedGroups, 4000, 'max hold re-pends');
    const repend = await feed(4, idleReply);
    assert.ok(repend >= 3, `re-pend: back to every frame (${repend})`);
    await h.mixin.release();
});

test('defaults: 2 fps while verifying, 1 fps once confirmed', () => {
    const src = require('fs').readFileSync(require('path').join(__dirname, '..', '..', 'src', 'defaults.ts'), 'utf8');
    assert.match(src, /frameIntervalMs: 500,/); assert.match(src, /holdIntervalMs: 1000,/);
});

// ---- Triggers (plugin-wide) ---------------------------------------------------------------------

test('Triggers: default all; Camera motion starts on any raw edge', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    assert.equal(h.mixin['episode'].scanning, true);
    await h.mixin.release();
});

test('Triggers: Camera AI only ignores raw-only edges, starts when a claim arrives within 1.5 s, or with a claim just before', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.plugin.triggers = ['Camera AI', 'Paced'];
    h.raw(true);
    assert.equal(h.mixin['episode'].scanning, false, 'held back, waiting for a claim');
    await sleep(300);
    h.claim('person');
    assert.equal(h.mixin['episode'].scanning, true, 'a claim within the window triggers');
    h.raw(false);
    await h.mixin.release();
    const n = make({ settings: { mode: 'Gate' } });
    n.plugin.triggers = ['Camera AI'];
    n.raw(true);
    await sleep(1600);
    n.claim('person');
    assert.equal(n.mixin['episode'].scanning, false, 'a claim after the window does not trigger');
    n.raw(false);
    n.claim('car');
    n.raw(true);
    assert.equal(n.mixin['episode'].scanning, true, 'a claim just before the edge triggers at once');
    await n.mixin.release();
});

test('Triggers: Shadow with Camera AI only still publishes raw for an ignored edge', async () => {
    const h = make({ settings: { mode: 'Shadow' } });
    h.plugin.triggers = ['Camera AI'];
    h.raw(true);
    await sleep(20);
    assert.equal(motion(), true, 'Shadow publishes raw');
    assert.equal(h.mixin['episode'].scanning, false);
    h.raw(false);
    await sleep(20);
    assert.equal(motion(), false);
    await h.mixin.release();
});

test('Triggers: a camera without ObjectDetector treats Camera AI as Camera motion (logged once)', async () => {
    const h = make({ settings: { mode: 'Gate' }, interfaces: ['VideoCamera', 'MotionSensor', 'Camera'] });
    h.plugin.triggers = ['Camera AI'];
    h.raw(true);
    assert.equal(h.mixin['episode'].scanning, true);
    h.raw(false); h.raw(true);
    assert.equal(ctl.logs.filter(l => /Camera AI acts as Camera motion/.test(String(l[1]))).length, 1);
    await h.mixin.release();
});

test('Triggers: Paced only: camera edges and claims start nothing; Paced off: no heartbeat', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.plugin.triggers = ['Paced'];
    h.raw(true); h.claim('person');
    assert.equal(h.mixin['episode'].scanning, false);
    await h.mixin.release();
    const k = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', heartbeatMs: 1 } });
    k.plugin.triggers = ['Camera motion'];
    await eventually(() => ctl.starts.length >= 1);
    frame(ctl.starts[ctl.starts.length - 1]);
    await sleep(80);
    assert.equal(k.calls.frame.length, 0, 'no heartbeat without Paced');
    await k.mixin.release();
});

test('Triggers setting: Detection group, all by default, empty or junk means all', async () => {
    const P = loadProvider();
    const p: any = new P('n');
    clearInterval(p.healthTimer);
    const st = await p.getSettings();
    const t = st.find((x: any) => x.key === 'triggers');
    assert.equal(t.group, 'Detection');
    assert.deepEqual(p.triggers, ['Camera motion', 'Camera AI', 'Paced']);
    p.storageSettings = { values: { triggers: [] } }; assert.equal(p.triggers.length, 3);
    p.storageSettings = { values: { triggers: 'junk' } }; assert.equal(p.triggers.length, 3);
    p.storageSettings = { values: { triggers: ['Camera AI'] } }; assert.deepEqual(p.triggers, ['Camera AI']);
});

test('outermost check reads the device\'s current mixin list when available (last = outermost)', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    const sdkStub = require('./harness').sdkStub;
    const orig = sdkStub.systemManager.getDeviceById;
    try {
        h.plugin.id = 'prov1';
        sdkStub.systemManager.getDeviceById = (id: string) => ({ ...orig(id), mixins: ['a', 'b', 'prov1'] });
        h.mixin['verifyPublished']();
        assert.equal(h.mixin['notOutermost'], false);
        sdkStub.systemManager.getDeviceById = (id: string) => ({ ...orig(id), mixins: ['prov1', 'z'] });
        h.mixin['verifyPublished']();
        assert.equal(h.mixin['notOutermost'], true);
        assert.ok(ctl.logs.some(l => /NOT the outermost/.test(String(l[1]))));
    } finally { sdkStub.systemManager.getDeviceById = orig; }
    await h.mixin.release();
});

/** every journalled id: exactly one start first, exactly one end last, updates in between */
function assertPaired(evs: any[], what: string) {
    const ids = [...new Set(evs.map(e => e.id))];
    for (const id of ids) {
        const seq = evs.filter(e => e.id === id).map(e => e.action);
        assert.equal(seq[0], 'start', `${what}: ${id} starts first (${seq})`);
        assert.equal(seq.filter(a => a === 'start').length, 1, `${what}: one start`);
        assert.equal(seq.filter(a => a === 'end').length, 1, `${what}: one end (${seq})`);
        assert.equal(seq[seq.length - 1], 'end', `${what}: end last`);
    }
    return ids.length;
}

test('journal pairing: a slow start never lets its update or end overtake it; unflagged confirm -> start, update, end in order', async () => {
    (ctl as any).journalDelay = 80;
    try {
        const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', heartbeatMs: 1 } });
        await eventually(() => ctl.starts.length === 1);
        const s = ctl.starts[0];
        frame(s);
        await eventually(() => h.calls.frame.length === 1);
        h.calls.frame[0].d.res(personHold);
        await eventually(() => h.calls.frame.length === 2, 3000);
        h.calls.frame[1].d.res(personHold);
        await eventually(() => motion() === true);
        frame(s, Date.now() + 5);
        await eventually(() => h.calls.frame.length === 3);
        h.calls.frame[2].d.res(idleReply);
        await eventually(() => h.calls.episodes.some((e: any) => e.action === 'end'), 3000);
        assert.deepEqual(h.calls.episodes.map((e: any) => e.action), ['start', 'update', 'end']);
        assertPaired(h.calls.episodes, 'unflagged');
        await h.mixin.release();
    } finally { (ctl as any).journalDelay = 0; }
});

test('journal pairing: mode change mid-episode, instance release mid-episode and a camera episode after a closed one each end exactly once', async () => {
    const h = make({ settings: { mode: 'Gate' } });
    h.raw(true);
    await eventually(() => h.calls.episodes.length === 1);
    await h.mixin.putMixinSetting('mode', 'Off');          // aborts the running episode
    await sleep(30);
    await h.mixin.putMixinSetting('mode', 'Gate');         // resumes: raw still on -> a new episode
    await sleep(30);
    h.raw(false);
    await h.mixin.release();                                // ends whatever is still open
    await sleep(50);
    assert.ok(assertPaired(h.calls.episodes, 'mode change + release') >= 1);
});
