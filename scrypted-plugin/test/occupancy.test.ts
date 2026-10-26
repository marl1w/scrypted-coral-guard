import test from 'node:test';
import assert from 'node:assert/strict';
import { Episode, EpisodeConfig, EpisodeSummary, Effects, formatSummary, LogEvent, MIN_MOTION_OFF_MS } from '../src/episode';
import type { FrameReply } from '../src/client';
import { idle } from './mock-server';
import { CameraStatus, occMotion, sensorLine } from '../src/status';

const base: EpisodeConfig = { mode: 'gate', confirmTimeoutMs: 6000, maxHoldMs: 600000, killSwitch: false, stillReleaseMs: 60000 };
const mk = (over: Partial<EpisodeConfig> = {}, raw = false) => new Episode({ ...base, ...over }, raw);
const R = (x: object) => x as FrameReply;
/** person reply with the given tracks: [id, stationary] */
const person = (...tracks: [number, boolean][]) => R({ person: { state: 'confirmed', confirmed: true, present: true, tracks_detail: tracks.map(([id, stationary]) => ({ id, stationary })) } });
const stillLogs = (fx: Effects[]) => fx.flatMap(f => f.logs).filter((l): l is Extract<LogEvent, { kind: 'still' }> => l.kind === 'still');
const summaries = (fx: Effects[]) => fx.flatMap(f => f.logs).filter(l => l.kind === 'summary').map(l => (l as any).summary as EpisodeSummary);

test('still person: occupancy stays on while motion releases after stillReleaseS', () => {
    const e = mk();
    const fx: Effects[] = [e.raw(true, 0)];
    fx.push(e.reply(person([1, false]), 1000));
    assert.equal(fx[1].motion, true);
    assert.equal(fx[1].occupied, true);
    fx.push(e.raw(false, 5000));
    fx.push(e.reply(person([1, true]), 10000));
    assert.equal(e.motion, true, 'still, but not yet for stillReleaseS');
    fx.push(e.reply(person([1, true]), 60999));
    assert.equal(e.motion, true);
    fx.push(e.reply(person([1, true]), 61000));
    assert.equal(e.motion, false, 'released: still since the last movement at 1 s');
    assert.equal(e.occupied, true);
    assert.equal(e.stillReleased, true);
    fx.push(e.tick(120000));
    assert.equal(e.motion, false);
    assert.equal(e.occupied, true);
    const [rel] = stillLogs(fx);
    assert.equal(rel.event, 'release');
    assert.equal(rel.afterS, 61);
    fx.push(e.reply(R(idle), 130000));
    assert.equal(e.occupied, false);
    const [s] = summaries(fx);
    assert.equal(s.stillReleases, 1);
    assert.equal(s.stillReleaseAfterS, 61);
    assert.equal(s.occupiedS, 129);
    assert.equal(s.motionS, 60);
    assert.match(formatSummary(s), /motion released \(still\) \+61 s/);
    assert.match(formatSummary(s), /occupied 129 s, motion 60 s/);
});

test('the still timer also releases from tick (no reply needed)', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(person([1, true]), 1000);
    assert.equal(e.tick(61000).motion, false);
    assert.equal(e.occupied, true);
});

test('a new moving track re-raises motion with an off gap of at least MIN_MOTION_OFF_MS', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(person([1, false]), 1000);
    e.reply(person([1, true]), 61000);
    assert.equal(e.motion, false);
    // a second person appears 500 ms after the release: motion waits for the off gap
    const fx = [e.reply(person([1, true], [2, false]), 61500)];
    assert.equal(e.motion, false, 'off gap not over');
    fx.push(e.tick(61000 + MIN_MOTION_OFF_MS - 1));
    assert.equal(e.motion, false);
    fx.push(e.tick(61000 + MIN_MOTION_OFF_MS));
    assert.equal(e.motion, true, 'natural off->on edge');
    assert.equal(e.occupied, true);
    const [r] = stillLogs(fx);
    assert.equal(r.event, 'reraise');
    assert.equal(r.reason, 'new subject');
    // both still again: released again after stillReleaseS from the last movement
    e.reply(person([1, true], [2, true]), 70000);
    assert.equal(e.motion, true);
    e.reply(person([1, true], [2, true]), 121500);
    assert.equal(e.motion, false);
});

test('a held subject that moves again re-raises motion', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(person([1, true]), 1000);
    e.reply(person([1, true]), 61000);
    assert.equal(e.motion, false);
    const fx = e.reply(person([1, false]), 70000);
    assert.equal(fx.motion, true);
    assert.equal(stillLogs([fx])[0].reason, 'moving');
});

test('stillReleaseS 0 keeps the old behaviour: motion held while anyone is present', () => {
    const e = mk({ stillReleaseMs: 0 });
    e.raw(true, 0);
    e.reply(person([1, true]), 1000);
    const fx = [e.reply(person([1, true]), 500000), e.tick(550000)];
    assert.equal(e.motion, true);
    assert.equal(e.occupied, true);
    assert.equal(stillLogs(fx).length, 0);
});

test('subject disappears while the still timer runs: motion and occupancy false at once', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(person([1, true]), 1000);
    e.reply(person([1, true]), 30000);
    const fx = e.reply(R(idle), 31000);
    assert.equal(fx.motion, false);
    assert.equal(fx.occupied, false);
    // and after a still release, too
    const f = mk();
    f.raw(true, 0);
    f.reply(person([1, true]), 1000);
    f.reply(person([1, true]), 61000);
    const gx = f.reply(R(idle), 62000);
    assert.equal(gx.motion, false);
    assert.equal(gx.occupied, false);
});

test('vehicle that stops (still/parked) behaves like still: a still person keeps occupancy, motion releases', () => {
    const e = mk();
    e.raw(true, 0);
    const car = (state: string) => R({ person: { state: 'confirmed', confirmed: true, present: true, tracks_detail: [{ id: 1, stationary: true }] }, vehicle: { moving: state === 'moving', tracks: [{ id: 7, cls: 'car', parked: state === 'parked', state, since_ms: 0, box: [0.1, 0.1, 0.5, 0.5] }] }, detections: [{ cls: 'car', score: 0.9, box: [0.1, 0.1, 0.5, 0.5], group: 'vehicle' }] });
    e.reply(car('moving'), 1000);
    assert.equal(e.motion, true);
    e.reply(car('moving'), 20000);
    e.reply(car('parked'), 50000);
    assert.equal(e.motion, true, 'moving vehicle 30 s ago');
    e.reply(car('parked'), 80000);
    assert.equal(e.motion, false, 'parked vehicle + still person for 60 s');
    assert.equal(e.occupied, true);
});

test('shadow: motion and occupancy mirror raw; the still release is only reported (wouldPublish)', () => {
    const e = mk({ mode: 'shadow' });
    e.raw(true, 0);
    e.reply(person([1, true]), 1000);
    const fx = e.reply(person([1, true]), 61000);
    assert.equal(fx.motion, true);
    assert.equal(fx.occupied, true);
    assert.equal(e.wouldPublish, false);
    assert.equal(e.wouldOccupied, true);
    assert.equal(stillLogs([fx]).length, 1);
    assert.equal(e.raw(false, 62000).occupied, false, 'shadow occupancy is the raw motion');
});

test('off: occupancy is the raw motion', () => {
    const e = mk({ mode: 'off' });
    assert.equal(e.raw(true, 0).occupied, true);
    assert.equal(e.raw(false, 1000).occupied, false);
});

test('fail open: motion and occupancy follow raw, a still release is dropped', () => {
    const e = mk();
    e.raw(true, 0);
    e.reply(person([1, true]), 1000);
    e.reply(person([1, true]), 61000);
    assert.equal(e.motion, false);
    const fx = e.failure('down', 62000);
    assert.equal(fx.motion, true);
    assert.equal(fx.occupied, true);
    assert.equal(e.raw(false, 63000).occupied, false);
});

test('max hold re-pend keeps a still release (no new edge for the same still subject)', () => {
    const e = mk({ maxHoldMs: 100000 });
    e.raw(true, 0);
    e.reply(person([1, true]), 1000);
    e.reply(person([1, true]), 61000);
    assert.equal(e.motion, false);
    e.tick(101000);
    assert.equal(e.pending, true);
    assert.equal(e.motion, false);
    assert.equal(e.occupied, true);
    e.reply(person([1, true]), 102000);
    assert.equal(e.motion, false);
    assert.equal(e.occupied, true);
});

test('status: sensor line, occupied/motion cell, still counters', () => {
    assert.equal(sensorLine({ on: true, groups: ['person'], stillS: 42, releasedAfterS: 60 }, false), 'Occupancy: on (person, still 42 s) · Motion: off (released after 60 s still)');
    assert.equal(sensorLine({ on: true, groups: ['person'], stillS: 1 }, true), 'Occupancy: on (person) · Motion: on');
    assert.equal(sensorLine({ on: false }, false, true), 'Would: Occupancy: off · Motion: off');
    assert.equal(occMotion({ occupiedS: 300, motionS: 170 } as any), '300 s / 170 s');
    const s = new CameraStatus();
    s.onStill('release'); s.onStill('release'); s.onStill('reraise');
    assert.equal(s.today().stillReleases, 2);
    assert.equal(s.today().reraises, 1);
    assert.match(s.summaryLine(), /2 still releases, 1 re-raises/);
    assert.match(s.overviewCells().today, / \/ 2 \/ 1$/);
    s.onEpisode({ kind: 'camera-triggered', rawStartMs: Date.now(), heldBeyondRawS: 0, vetoed: false, aborted: false, groups: ['person'], stillReleaseAfterS: 61, reraises: 1, occupiedS: 300, motionS: 170 });
    const html = s.render();
    assert.match(html, /300 s \/ 170 s/);
    assert.match(html, /2 still releases/);
});
