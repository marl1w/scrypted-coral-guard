// 0.3.12: Fast confirm (burst + the service's fast path), grace after the raw end, inconclusive vetoes.
import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { formatSummary, Episode, EpisodeConfig, Effects, EpisodeSummary } from '../src/episode';
import type { FrameReply } from '../src/client';
import { confirmedReply, ctl, idleReply, J, make, releaseAll, sleep } from './harness';

const R = (x: object) => x as FrameReply;
const base: EpisodeConfig = { mode: 'gate', confirmTimeoutMs: 6000, maxHoldMs: 600000, killSwitch: false };
const mk = (over: Partial<EpisodeConfig> = {}) => new Episode({ ...base, ...over }, false);
const summaries = (fx: Effects[]) => fx.flatMap(f => f.logs).filter(l => l.kind === 'summary').map(l => (l as any).summary as EpisodeSummary);
const idle = { person: { state: 'idle', confirmed: false, present: false } };
const fastConfirmed = { person: { state: 'confirmed', confirmed: true, present: true, confirmed_by: 'fast_move' } };

afterEach(async () => { await releaseAll(); });

async function eventually(cond: () => boolean, ms = 3000, what = 'condition') {
    const t0 = Date.now();
    while (!cond()) {
        if (Date.now() - t0 > ms) assert.fail(`timeout waiting for ${what}`);
        await sleep(5);
    }
}

// ---- episode: confirm path, grace, inconclusive ----

test('the log line names the confirm path and the frames it took', () => {
    const e = mk();
    const fx = [e.raw(true, 0), e.reply(R(idle), 150), e.reply(R(idle), 300), e.reply(R(fastConfirmed), 420), e.raw(false, 2000), e.reply(R(idle), 2100)];
    const [s] = summaries(fx);
    assert.equal(s.confirmPath, 'fast_move');
    assert.equal(s.confirmFrames, 3);
    assert.match(formatSummary(s), /confirmed \+420 ms \(fast_move, 3 frames\)/);
});

test('grace: raw ends unconfirmed, the episode keeps verifying and a late confirmation is the SAME episode', () => {
    const e = mk({ graceMs: 3000 });
    const fx: Effects[] = [e.raw(true, 0), e.reply(R(idle), 500), e.raw(false, 5000)];
    fx.push(e.tick(6000));                                            // the plain confirm timeout: no veto inside the grace
    assert.equal(summaries(fx).length, 0);
    assert.equal(e.scanning, true);
    const c = e.reply(R(confirmedReply), 7400);
    assert.equal(c.motion, true, 'Gate publishes the late confirmation');
    assert.equal(e.lateConfirm, true);
    fx.push(c, e.reply(R(idle), 9000));
    const [s] = summaries(fx);
    assert.equal(s.outcome, 'released');
    assert.equal(s.afterRawEnd, true);
    assert.match(formatSummary(s), /confirmed \+7400 ms \(after raw end, grace\)/);
});

test('grace: nothing confirms by raw end + grace -> veto; graceMs 0 keeps the plain timeout', () => {
    const e = mk({ graceMs: 3000 });
    const fx: Effects[] = [e.raw(true, 0), e.reply(R(idle), 500), e.reply(R(idle), 1000), e.raw(false, 5000), e.tick(6000), e.tick(7900)];
    assert.equal(summaries(fx).length, 0);
    fx.push(e.tick(8000));
    assert.equal(summaries(fx)[0].outcome, 'vetoed');
    const o = mk({ graceMs: 0 });
    const g = [o.raw(true, 0), o.reply(R(idle), 500), o.raw(false, 5000), o.tick(6000)];
    assert.equal(summaries(g)[0].outcome, 'vetoed');
    const r = mk({ graceMs: 3000 });                                   // raw still on at the timeout: vetoed as before
    assert.equal(summaries([r.raw(true, 0), r.reply(R(idle), 500), r.tick(6000)])[0].outcome, 'vetoed');
});

test('inconclusive: a veto on fewer than 2 analysed frames or whose first verdict came after the raw end', () => {
    const one = mk();
    const [a] = summaries([one.raw(true, 0), one.reply(R(idle), 500), one.tick(6000)]);
    assert.equal(a.inconclusive, true);
    assert.match(formatSummary(a), /VETOED \(inconclusive\)/);
    const late = mk({ graceMs: 3000 });
    const [b] = summaries([late.raw(true, 0), late.raw(false, 300), late.reply(R(idle), 400), late.reply(R(idle), 900), late.tick(6000)]);
    assert.equal(b.inconclusive, true);
    const solid = mk();
    const [c] = summaries([solid.raw(true, 0), solid.reply(R(idle), 300), solid.reply(R(idle), 800), solid.reply(R(idle), 1300), solid.tick(6000)]);
    assert.equal(c.outcome, 'vetoed');
    assert.equal(c.inconclusive, undefined);
});

// ---- mixin: burst posting ----

test('burst: frames go back to back while unconfirmed (latest wins, deadline, claim), the ffmpeg burst stops at confirm', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', preRollS: 0 }, fast: true });
    await eventually(() => ctl.starts.length === 1);
    const s: any = ctl.starts[0];
    assert.equal(s.o.burstFps, 6, 'warm stream is started with the burst filter');
    h.claim('people');
    h.raw(true);
    await eventually(() => h.mixin['episode'].scanning === true);
    assert.deepEqual(s.burst, [true], 'burst switched on at the scan start');
    s.o.onFrame(J, Date.now());
    await eventually(() => h.calls.frame.length === 1);
    const opt = h.calls.frame[0].a[4];
    assert.equal(opt.deadlineMs, 300);
    assert.equal(opt.claim, 'person');
    s.o.onFrame(J, Date.now() + 1);                                   // in flight: kept as the next frame
    s.o.onFrame(J, Date.now() + 2);                                   // replaces it (latest wins)
    await sleep(20);
    assert.equal(h.calls.frame.length, 1, 'one request in flight');
    h.calls.frame[0].d.res(idleReply);
    await eventually(() => h.calls.frame.length === 2, 1000, 'next frame sent as soon as the reply arrived');
    h.calls.frame[1].d.res({ ...confirmedReply, person: { ...confirmedReply.person, confirmed_by: 'fast_claim' } });
    await eventually(() => s.burst.length === 2);
    assert.deepEqual(s.burst, [true, false], 'burst off once confirmed');
});

test('burst off (Fast confirm off): today\'s behaviour, no deadline, no claim, a frame arriving in flight is skipped', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', preRollS: 0 } });
    await eventually(() => ctl.starts.length === 1);
    const s: any = ctl.starts[0];
    h.claim('people');
    h.raw(true);
    await eventually(() => h.mixin['episode'].scanning === true);
    assert.deepEqual(s.burst, []);
    s.o.onFrame(J, Date.now());
    await eventually(() => h.calls.frame.length === 1);
    assert.equal(h.calls.frame[0].a[4].deadlineMs, undefined);
    assert.equal(h.calls.frame[0].a[4].claim, undefined);
    s.o.onFrame(J, Date.now() + 1);
    h.calls.frame[0].d.res(idleReply);
    await sleep(30);
    assert.equal(h.calls.frame.length, 1, 'the in-flight frame was skipped, not kept');
});

test('burst ends after burstMaxMs even when nothing confirms', async () => {
    const h = make({ settings: { mode: 'Gate', frameProcess: 'Keep warm', preRollS: 0, burstMaxMs: 100 }, fast: true });
    await eventually(() => ctl.starts.length === 1);
    const s: any = ctl.starts[0];
    h.raw(true);
    await eventually(() => s.burst.length === 1);
    await sleep(120);
    s.o.onFrame(J, Date.now());
    await eventually(() => s.burst.length === 2);
    assert.deepEqual(s.burst, [true, false]);
});
