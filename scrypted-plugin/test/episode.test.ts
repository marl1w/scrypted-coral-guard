import test from 'node:test';
import assert from 'node:assert/strict';
import { backoffMs, formatSummary, Episode, EpisodeConfig, Effects, EpisodeSummary, latencyStats, percentile, summarize } from '../src/episode';
import type { FrameReply } from '../src/client';
import { candidate, confirmed, holding, idle, vehicle } from './mock-server';

const R = (x: object) => x as FrameReply;
const base: EpisodeConfig = { mode: 'gate', confirmTimeoutMs: 6000, maxHoldMs: 600000, killSwitch: false };
const mk = (over: Partial<EpisodeConfig> = {}, raw = false) => new Episode({ ...base, ...over }, raw);
const summaries = (fx: Effects[]) => fx.flatMap(f => f.logs).filter(l => l.kind === 'summary').map(l => (l as any).summary as EpisodeSummary);

test('shadow mirrors raw exactly, whatever the service says', () => {
    const e = mk({ mode: 'shadow' });
    assert.equal(e.raw(true, 0).motion, true);
    assert.equal(e.reply(R(idle), 1000).motion, true);
    assert.equal(e.tick(7000).motion, true); // would be vetoed
    assert.equal(e.raw(false, 8000).motion, false);
    assert.equal(e.raw(true, 9000).motion, true);
    assert.equal(e.reply(R(confirmed), 10000).motion, true);
    assert.equal(e.raw(false, 11000).motion, false); // gate would hold, shadow does not
});

test('shadow logs what gate would have done: veto', () => {
    const e = mk({ mode: 'shadow' });
    const fx = [e.raw(true, 0), e.reply(R(idle), 1000), e.tick(6000), e.raw(false, 6500)];
    const [s] = summaries(fx);
    assert.equal(s.outcome, 'vetoed');
    assert.equal(s.mode, 'shadow');
    assert.equal(s.confirmedAfterMs, undefined);
});

test('shadow logs held-beyond-raw', () => {
    const e = mk({ mode: 'shadow' });
    const fx = [e.raw(true, 0), e.reply(R(confirmed), 1000), e.raw(false, 5000), e.reply(R(holding), 8000), e.reply(R(idle), 15000)];
    const [s] = summaries(fx);
    assert.equal(s.outcome, 'released');
    assert.equal(s.confirmedAfterMs, 1000);
    assert.equal(s.heldBeyondRawS, 10);
    assert.equal(fx[4].motion, false);
});

test('gate: raw rising edge requests reset and starts scanning, motion stays false', () => {
    const e = mk();
    const fx = e.raw(true, 0);
    assert.equal(fx.motion, false);
    assert.equal(fx.reset, true);
    assert.equal(fx.scanning, true);
});

test('gate: candidate does not open the gate, confirmed does', () => {
    const e = mk();
    e.raw(true, 0);
    assert.equal(e.reply(R(candidate), 1000).motion, false);
    assert.equal(e.reply(R(confirmed), 2000).motion, true);
});

test('gate: veto when never confirmed, scanning stops, stays false until next rising edge', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(idle), 1000);
    assert.equal(e.tick(5999).scanning, true);
    const fx = e.tick(6000);
    assert.equal(fx.motion, false);
    assert.equal(fx.scanning, false);
    assert.equal(summaries([fx])[0].outcome, 'vetoed');
    // late confirmation is ignored for this trigger
    assert.equal(e.reply(R(confirmed), 7000).motion, false);
    // raw falls, new rising edge starts afresh
    e.raw(false, 8000);
    const again = e.raw(true, 9000);
    assert.equal(again.reset, true);
    assert.equal(again.scanning, true);
});

test('gate: confirm latency is measured from the raw trigger', () => {
    const e = mk();
    e.raw(true, 1000);
    e.reply(R(candidate), 2000);
    const fx = [e.reply(R(confirmed), 3500), e.reply(R(idle), 4500)];
    assert.equal(summaries(fx)[0].confirmedAfterMs, 2500);
});

test('gate: motion held beyond raw end while service says present, released when gone', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    const f1 = e.raw(false, 5000);
    assert.equal(f1.motion, true);
    assert.equal(f1.scanning, true);
    assert.equal(e.reply(R(holding), 9000).motion, true);
    const f2 = e.reply(R(idle), 14000);
    assert.equal(f2.motion, false);
    assert.equal(f2.scanning, false);
    const s = summaries([f2])[0];
    assert.equal(s.outcome, 'released');
    assert.equal(s.heldBeyondRawS, 9);
});

test('gate: released while raw is still on does NOT open a new episode; only a new rising edge does', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    const f = e.reply(R(idle), 20000);
    assert.equal(f.motion, false);
    assert.equal(f.scanning, false, 'no re-scan, no veto of a phantom episode');
    assert.equal(summaries([f]).length, 1);
    assert.equal(e.tick(40000).logs.length, 0);
    assert.equal(e.raw(false, 41000).motion, false);
    const up = e.raw(true, 42000);
    assert.equal(up.scanning, true);
    assert.equal(up.reset, true);
});

test('synthetic (pixel gate): synthStart opens a scan while a released camera episode still has raw on, and it can confirm', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    e.reply(R(idle), 20000);
    const fx = e.synthStart(21000);
    assert.equal(fx.reset, true);
    assert.equal(fx.motion, false, 'a pixel change alone never raises motion');
    assert.equal(e.reply(R(confirmed), 21500).motion, true);
    assert.equal(mk({ mode: 'shadow' }).synthStart(0).reset, false, 'Shadow never runs one');
});

test('synthetic episode: never passed through, ends on degrade and on mode change, closed one does not block the camera\'s own trigger', () => {
    const e = mk();
    assert.equal(e.synthStart(0).motion, false);
    const d = e.failure('boom', 100);
    assert.equal(d.motion, false, 'degraded pass-through publishes the camera\'s raw only, never the synthetic one');
    assert.equal(e.synthetic, false);
    const m = mk();
    m.synthStart(0);
    assert.equal(m.setConfig({ ...base, mode: 'shadow' }, 100).motion, false);
    assert.equal(m.synthetic, false);
    const v = mk();
    v.synthStart(0);
    v.reply(R(idle), 500);
    assert.equal(v.tick(7000).scanning, false, 'vetoed after the confirm timeout');
    const up = v.raw(true, 7500); // the camera rises right after the gate episode closed
    assert.equal(up.scanning, true);
    assert.equal(up.reset, true);
});

test('synthetic then camera edge merges: one episode, no duplicate motion edge, camera origin', () => {
    const e = mk();
    e.synthStart(0);
    e.reply(R(confirmed), 500);
    const up = e.raw(true, 800);
    assert.equal(up.motion, true);
    assert.equal(up.reset, false);
    assert.equal(e.synthetic, false, 'now a camera episode');
});

test('max hold with raw still on re-pends a fresh confirmation (motion held until the verdict)', () => {
    const e = mk({ maxHoldMs: 10000 });
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    const fx = e.tick(11000);
    assert.equal(summaries([fx])[0].outcome, 'maxhold');
    assert.equal(fx.scanning, true);
    assert.equal(fx.motion, true, 'held until the first verdict');
    assert.equal(e.reply(R(confirmed), 11500).motion, true);
    assert.equal(e.reply(R(idle), 12000).motion, false);
});

test('gate: raw rising during a held episode does not reset the session', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    e.raw(false, 3000);
    const fx = e.raw(true, 4000);
    assert.equal(fx.reset, false);
    assert.equal(fx.motion, true);
});

test('gate: pending survives raw falling until confirm timeout (service may still confirm)', () => {
    const e = mk();
    e.raw(true, 0);
    assert.equal(e.raw(false, 500).scanning, true);
    assert.equal(e.reply(R(confirmed), 1500).motion, true);
});

test('gate: max hold releases even if the service keeps saying present', () => {
    const e = mk({ maxHoldMs: 10000 });
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    e.raw(false, 2000);
    assert.equal(e.tick(10999).motion, true);
    const fx = e.tick(11000);
    assert.equal(fx.motion, false);
    assert.equal(summaries([fx])[0].outcome, 'maxhold');
    // not re-opened by the next present reply
    assert.equal(e.reply(R(holding), 12000).motion, false);
});

const idleP = { state: 'idle', confirmed: false, present: false };
const parkedOnly = { person: idleP, vehicle: { moving: false, tracks: [{ id: 1, cls: 'car', parked: true, since_ms: 999999 }] } };
const animalConfirmed = { person: idleP, animal: { confirmed: true, present: true } };
const animalGone = { person: idleP, animal: { confirmed: false, present: false } };

const det = (cls: string, score: number) => ({ cls, score, box: [0.1, 0.1, 0.5, 0.5] });
const withDets = (dets: object[], extra: object = {}) => ({ person: idleP, detections: dets, ...extra });

test('gate: a vehicle needs a car/truck/bus/motorcycle >= 0.6, not parked, in 2 of the last 3 replies; a single frame never confirms', () => {
    const e = mk();
    e.raw(true, 0);
    assert.equal(e.reply(R(withDets([det('car', 0.9)])), 1000).motion, false, 'one frame');
    assert.equal(e.reply(R(withDets([])), 2000).motion, false);
    assert.equal(e.reply(R(withDets([det('truck', 0.7)])), 3000).motion, true, 'second tick within the last 3');
    const fx = e.reply(R(idle), 4000);
    assert.equal(fx.motion, true, 'held for ~6 s after the last vehicle');
    const g = e.reply(R(idle), 10000);
    assert.equal(g.motion, false);
    const sm = summaries([g])[0];
    assert.deepEqual(sm.confirmedBy, ['vehicle']);
    assert.match(formatSummary(sm), /confirmed: vehicle \(truck 0\.70\)/);
});

test('gate: bicycles and weak or parked vehicles never confirm', () => {
    const e = mk();
    e.raw(true, 0);
    for (let i = 1; i <= 4; i++) assert.equal(e.reply(R(withDets([det('bicycle', 0.95)])), i * 500).motion, false);
    for (let i = 5; i <= 8; i++) assert.equal(e.reply(R(withDets([det('car', 0.55)])), i * 500).motion, false, 'below 0.6');
    const p = mk();
    p.raw(true, 0);
    const parked = { vehicle: { moving: false, tracks: [{ id: 1, cls: 'car', parked: true, since_ms: 9, box: [0.1, 0.1, 0.5, 0.5] }] } };
    for (let i = 1; i <= 4; i++) assert.equal(p.reply(R(withDets([det('car', 0.95)], parked)), i * 500).motion, false);
});

test('Detect: a disabled group never confirms nor holds; a claim of a disabled group vetoes', () => {
    const e = mk({ detect: ['person'] });
    e.raw(true, 0);
    e.reply(R(withDets([det('car', 0.9)])), 500);
    assert.equal(e.reply(R(withDets([det('car', 0.9)])), 1000).motion, false, 'vehicle off');
    assert.equal(e.reply(R(animalConfirmed), 1500).motion, false, 'animal off');
    assert.equal(e.reply(R(confirmed), 2000).motion, true);
    const f = mk({ detect: ['vehicle', 'animal'] });
    f.raw(true, 0);
    assert.equal(f.reply(R(confirmed), 1000).motion, false, 'person off');
    assert.equal(f.tick(6000).scanning, false, 'vetoed');
    const g = mk({ detect: ['animal'] });
    g.raw(true, 0);
    assert.equal(g.reply(R(confirmed), 500).motion, false, 'person is a disabled group');
    assert.equal(g.reply(R(animalConfirmed), 1000).motion, true);
    const h = mk({ detect: ['animal'] });
    h.raw(true, 0);
    assert.equal(h.acceptUnverified(10).motion, true, 'package still passes through');
});

test('gate: an animal confirms (2-of-3 is the service\'s), holds while present, releases when gone', () => {
    const e = mk();
    e.raw(true, 0);
    assert.equal(e.reply(R(animalConfirmed), 1000).motion, true);
    assert.equal(e.reply(R(animalGone), 2000).motion, false);
});

test('a claim is only a hint: whatever the service confirms confirms (person under a dog_cat or car claim)', () => {
    const e = mk();
    e.raw(true, 0);
    assert.equal(e.reply(R(confirmed), 1000).motion, true);
    const a = mk();
    a.raw(true, 0);
    assert.equal(a.reply(R(animalConfirmed), 1000).motion, true);
});

test('claim the model cannot verify (package): passes through unverified, held by raw motion only', () => {
    const e = mk();
    assert.equal(e.raw(true, 0).motion, false);
    assert.equal(e.acceptUnverified(100).motion, true);
    assert.equal(e.reply(R(idle), 1000).motion, true, 'the service verdict is not used');
    const fx = e.raw(false, 5000);
    assert.equal(fx.motion, false);
    assert.deepEqual(summaries([fx])[0].confirmedBy, ['unverified']);
    assert.ok(!mk({ mode: 'shadow' }).acceptUnverified(0).motion);
});

// ---- fail open -----------------------------------------------------------------------------

test('fail open: an error while raw is on passes raw through immediately, without flapping', () => {
    const e = mk();
    e.raw(true, 0);
    const seen: boolean[] = [];
    seen.push(e.failure('HTTP 503', 1000).motion);
    seen.push(e.failure('timeout', 1500).motion); // repeated failure: no change
    assert.deepEqual(seen, [true, true]);
    assert.equal(e.degraded, 'HTTP 503');
});

test('fail open: confirmed gate drops to raw when the service fails', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    e.raw(false, 2000);
    assert.equal(e.motion, true); // held
    assert.equal(e.failure('timeout', 3000).motion, false); // raw is false -> pass-through false
});

test('fail open: raw edges follow exactly while degraded, no service use', () => {
    const e = mk();
    e.failure('x', 0);
    const a = e.raw(true, 1000);
    assert.equal(a.motion, true);
    assert.equal(a.scanning, false);
    assert.equal(a.reset, false);
    assert.equal(e.raw(false, 2000).motion, false);
});

test('fail open: failure is logged once', () => {
    const e = mk();
    const fx = [e.failure('a', 0), e.failure('b', 1), e.failure('c', 2)];
    assert.equal(fx.flatMap(f => f.logs).filter(l => l.kind === 'degraded').length, 1);
});

test('fail open: health not ok behaves like an error', () => {
    const e = mk();
    e.raw(true, 0);
    const fx = e.failure('health not ok (coral stalled)', 500);
    assert.equal(fx.motion, true);
    assert.equal(fx.scanning, false);
});

test('recovery while raw is on: motion does not flap, resets the service, verdict decides', () => {
    const e = mk();
    e.raw(true, 0);
    e.failure('x', 1000);
    const rec = e.healthy(5000);
    assert.equal(rec.motion, true); // hold-over, no drop
    assert.equal(rec.reset, true);
    assert.equal(rec.scanning, true);
    assert.ok(rec.logs.some(l => l.kind === 'recovered'));
    assert.equal(e.reply(R(idle), 6000).motion, true);
    assert.equal(e.reply(R(confirmed), 7000).motion, true); // stays true
});

test('recovery while raw is on: unconfirmed hold-over ends after the confirm timeout', () => {
    const e = mk();
    e.raw(true, 0);
    e.failure('x', 1000);
    e.healthy(5000);
    e.reply(R(idle), 5500);
    assert.equal(e.tick(10999).motion, true);
    assert.equal(e.tick(11000).motion, false);
});

test('recovery while raw is off: nothing happens', () => {
    const e = mk();
    e.failure('x', 0);
    const rec = e.healthy(1000);
    assert.equal(rec.motion, false);
    assert.equal(rec.scanning, false);
    assert.equal(rec.reset, false);
});

test('healthy() without a failure is a no-op', () => {
    const e = mk();
    const fx = e.healthy(0);
    assert.equal(fx.logs.length, 0);
});

test('replies arriving while degraded are ignored', () => {
    const e = mk();
    e.raw(true, 0);
    e.failure('x', 100);
    assert.equal(e.reply(R(confirmed), 200).motion, true);
    e.raw(false, 300);
    assert.equal(e.reply(R(confirmed), 400).motion, false);
});

test('kill switch: pass-through everywhere, abort in-flight episode, no scanning', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    const fx = e.setConfig({ ...base, killSwitch: true }, 2000);
    assert.equal(fx.motion, true); // raw still true
    assert.equal(fx.scanning, false);
    assert.equal(summaries([fx])[0].outcome, 'aborted');
    assert.equal(e.raw(false, 3000).motion, false);
    assert.equal(e.raw(true, 4000).scanning, false);
});

test('kill switch released with raw on: hold-over then verdict', () => {
    const e = mk({ killSwitch: true });
    e.raw(true, 0);
    const fx = e.setConfig(base, 1000);
    assert.equal(fx.motion, true);
    assert.equal(fx.scanning, true);
    assert.equal(fx.reset, true);
});

test('mode off: pass-through, never scans', () => {
    const e = mk({ mode: 'off' });
    const fx = e.raw(true, 0);
    assert.equal(fx.motion, true);
    assert.equal(fx.scanning, false);
    assert.equal(fx.reset, false);
});

test('switching gate -> shadow mid-episode publishes raw', () => {
    const e = mk();
    e.raw(true, 0);
    assert.equal(e.motion, false);
    assert.equal(e.setConfig({ ...base, mode: 'shadow' }, 500).motion, true);
});

test('raw initially true is honoured as pass-through before any edge', () => {
    const e = mk({}, true);
    assert.equal(e.motion, true);
});

test('backoff doubles and caps at 60 s', () => {
    assert.deepEqual([0, 1, 2, 3, 4, 5, 6, 7, 20].map(a => backoffMs(a)), [1000, 2000, 4000, 8000, 16000, 32000, 60000, 60000, 60000]);
});

test('summarize: empty and aggregate', () => {
    assert.equal(summarize([]), 'No episodes yet.');
    const e = mk();
    const fx = [e.raw(true, 0), e.reply(R(confirmed), 1000), e.raw(false, 2000), e.reply(R(idle), 5000)];
    fx.push(e.raw(true, 10000), e.reply(R(idle), 10500), e.tick(16000));
    const text = summarize(summaries(fx));
    assert.match(text, /2 episodes: 1 vetoed, 1 confirmed/);
});

// ---- pre-roll bursts -----------------------------------------------------------------------

test('pre-roll burst: replies for past frames confirm within ms of the trigger and report a negative frame offset', () => {
    const e = mk();
    e.raw(true, 10000);
    // burst of 4 past frames (t = 7.0 .. 10.0 s) replied back to back at wall ~10.4 s
    assert.equal(e.reply(R(idle), 10400, 7000).motion, false);
    assert.equal(e.reply(R(candidate), 10410, 8000).motion, false);
    const fx = e.reply(R(confirmed), 10420, 9000);
    assert.equal(fx.motion, true);
    const done = e.reply(R(idle), 20000, 20000);
    const s = summaries([done])[0];
    assert.equal(s.confirmedAfterMs, 420);
    assert.equal(s.confirmedFrameOffsetMs, -1000);
});

test('pre-roll burst: a person already present in the burst does not stay vetoed by the later live frames', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(confirmed), 300, -2000);
    assert.equal(e.reply(R(holding), 1300, 1000).motion, true);
});

// ---- stale / one-in-flight / latency metrics -----------------------------------------------

test('skip (stale or busy) changes nothing: no degrade, no motion change, no state change', () => {
    const e = mk();
    e.raw(true, 0);
    for (let i = 1; i <= 5; i++) {
        const fx = e.skip(i * 100);
        assert.equal(fx.motion, false);
        assert.equal(fx.scanning, true);
        assert.equal(fx.logs.length, 0);
    }
    assert.equal(e.degraded, undefined);
    assert.equal(e.reply(R(confirmed), 600).motion, true);
    const s = summaries([e.reply(R(idle), 700)])[0];
    assert.equal(s.skipped, 5);
});

test('skip in shadow keeps pass-through and does not count outside an episode', () => {
    const e = mk({ mode: 'shadow' });
    assert.equal(e.skip(0).motion, false);
    e.raw(true, 100);
    assert.equal(e.skip(200).motion, true);
});

test('latency metrics: trigger -> first post -> first verdict -> confirmed, with service timings', () => {
    const e = mk();
    e.raw(true, 1000);
    e.posted(1180);
    e.posted(1300); // only the first counts
    e.reply(R({ ...candidate, queue_ms: 3, infer_ms: 30, total_ms: 41 }), 1230, 1100);
    e.reply(R({ ...confirmed, queue_ms: 5, infer_ms: 31, total_ms: 44 }), 1330, 1200);
    const s = summaries([e.reply(R(idle), 5000)])[0];
    assert.equal(s.postedAfterMs, 180);
    assert.equal(s.verdictAfterMs, 230);
    assert.equal(s.confirmedAfterMs, 330);
    assert.deepEqual(s.service, { queue_ms: 5, infer_ms: 31, total_ms: 44 }); // confirming reply
});

test('latency metrics: veto keeps posted/verdict and the first reply timings', () => {
    const e = mk();
    e.raw(true, 0);
    e.posted(100);
    e.reply(R({ ...idle, total_ms: 40 }), 160);
    const s = summaries([e.tick(6000)])[0];
    assert.equal(s.confirmedAfterMs, undefined);
    assert.equal(s.verdictAfterMs, 160);
    assert.equal(s.service?.total_ms, 40);
});

test('latency stats: p50/p95 over episodes, aborted ones excluded', () => {
    const list = [100, 200, 300, 400, 1000].map((c, i) => ({ mode: 'gate', startedAt: i, outcome: 'released', closedAt: i, heldBeyondRawS: 0, skipped: 0, postedAfterMs: c / 10, verdictAfterMs: c / 2, confirmedAfterMs: c }) as EpisodeSummary);
    list.push({ ...list[0], outcome: 'aborted', confirmedAfterMs: 99999 });
    const l = latencyStats(list);
    assert.equal(l.n, 5);
    assert.equal(l.confirmed.p50, 300);
    assert.equal(l.confirmed.p95, 1000);
    assert.equal(l.verdict.p50, 150);
    assert.equal(l.posted.p50, 30);
    assert.match(summarize(list), /confirmed 300\/1000/);
    assert.equal(percentile([], 0.5), undefined);
});

// ---- review fixes ------------------------------------------------------------------------------

test('H1: no verdict within the confirm timeout fails open instead of vetoing', () => {
    const e = mk();
    e.raw(true, 0);
    e.posted(100); // a frame went out, but no reply ever came
    const fx = e.tick(6000);
    assert.equal(fx.motion, true); // raw passes through
    assert.equal(fx.scanning, false);
    assert.equal(e.degraded, 'no verdict within the confirm timeout');
    assert.equal(summaries([fx]).filter(s => s.outcome === 'vetoed').length, 0);
});

test('H1: with a verdict the same timeout is a normal veto', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(idle), 500);
    const fx = e.tick(6000);
    assert.equal(summaries([fx])[0].outcome, 'vetoed');
    assert.equal(e.degraded, undefined);
});

test('M02: a failure aborts the episode (summary) and leaves no stale state after recovery', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(R(confirmed), 1000);
    e.raw(false, 2000); // held beyond raw
    const fx = e.failure('timeout', 3000);
    assert.equal(summaries([fx])[0].outcome, 'aborted');
    const rec = e.healthy(4000);
    assert.equal(rec.scanning, false);
    assert.equal(rec.motion, false);
    assert.equal(rec.reset, false);
});

test('M05: present without confirmed (holding) does not confirm a pending episode', () => {
    const e = mk();
    e.raw(true, 0);
    assert.equal(e.reply(R(holding), 1000).motion, false);
    assert.equal(e.reply(R({ person: { state: 'candidate', confirmed: false, present: true } }), 2000).motion, false);
    assert.equal(e.reply(R(confirmed), 3000).motion, true);
});

test('a reply with no episode in progress is ignored', () => {
    const e = mk();
    assert.equal(e.reply(R(confirmed), 0).motion, false);
    assert.equal(e.raw(true, 10).motion, false); // the earlier reply did not pre-confirm anything
});

test('a camera trigger after a merged pixel-gate episode was vetoed starts a fresh episode (stale synthetic input cannot swallow it)', () => {
    const e = mk();
    e.synthStart(0);
    e.raw(true, 500); // merge
    e.reply(R(idle), 1000);
    assert.equal(e.tick(7000).scanning, false, 'vetoed');
    e.raw(false, 8000);
    const up = e.raw(true, 9000);
    assert.equal(up.scanning, true);
    assert.equal(up.reset, true);
    assert.equal(e.reply(R(confirmed), 9500).motion, true, 'verdict flow runs');
});

test('Shadow to Gate while the camera is still moving: a closed episode is re-armed so Gate verifies the ongoing trigger (held until the verdict)', () => {
    const e = mk({ mode: 'shadow' });
    e.raw(true, 0);
    e.reply(R(idle), 1000);
    e.tick(7000); // shadow vetoes: closed
    const fx = e.setConfig({ ...base, mode: 'gate' }, 8000);
    assert.equal(fx.reset, true);
    assert.equal(fx.scanning, true);
    assert.equal(fx.motion, true, 'held until the first verdict instead of dropping real motion');
});

test('vehicle guard: a still track neither ticks nor holds; a moving one does', () => {
    const { vehicleView } = require('../src/episode') as typeof import('../src/episode');
    const box = [0.1, 0.1, 0.5, 0.5];
    const r = (state: string) => ({ person: { state: 'idle', confirmed: false, present: false }, vehicle: { tracks: [{ id: 1, cls: 'car', parked: false, state, since_ms: 0, box }] }, detections: [{ cls: 'car', score: 0.9, box }] }) as any;
    assert.deepEqual([vehicleView(r('still')).tick, vehicleView(r('still')).hold], [false, false]);
    assert.deepEqual([vehicleView(r('moving')).tick, vehicleView(r('moving')).hold], [true, true]);
});

test('vehicle guard: the confirm score is the Vehicle confirm (moving) threshold', () => {
    const { vehicleView } = require('../src/episode') as typeof import('../src/episode');
    const r = { person: { state: 'idle', confirmed: false, present: false }, detections: [{ cls: 'car', score: 0.55, box: [0.1, 0.1, 0.5, 0.5] }] } as any;
    assert.equal(vehicleView(r).tick, false, 'default 0.6');
    assert.equal(vehicleView(r, 0.5).tick, true);
});
