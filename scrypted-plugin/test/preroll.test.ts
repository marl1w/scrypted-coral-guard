import test from 'node:test';
import assert from 'node:assert/strict';
import { FrameClock } from '../src/preroll';

test('burst frames are labelled on an interval grid ending in the past, never in the future', () => {
    const c = new FrameClock(1000, 4000, true);
    const t = [10000, 10010, 10020, 10030].map(a => c.stamp(a));
    assert.deepEqual(t, [6000, 7000, 8000, 9000]);
    assert.ok(t.every((x, i) => x <= 10000 + i * 10));
    assert.equal(c.inBurst, true);
});

test('a frame arriving half an interval after the previous one is live and uses arrival time', () => {
    const c = new FrameClock(1000, 4000, true);
    c.stamp(10000); c.stamp(10020);
    assert.equal(c.stamp(11020), 11020);
    assert.equal(c.inBurst, false);
    assert.equal(c.stamp(12020), 12020);
});

test('burst labels are monotonic and capped at arrival time (short real prebuffer)', () => {
    const c = new FrameClock(1000, 1500, true);
    const t = [100, 105, 110, 115, 120].map(a => c.stamp(a));
    assert.deepEqual(t, [-1400, -400, 110, 115, 120].map((x, i) => Math.min(x, [100, 105, 110, 115, 120][i])));
    for (let i = 1; i < t.length; i++) assert.ok(t[i] >= t[i - 1]);
});

test('disabled or zero pre-roll: all frames live', () => {
    assert.equal(new FrameClock(1000, 4000, false).stamp(5), 5);
    assert.equal(new FrameClock(1000, 0, true).stamp(5), 5);
});
