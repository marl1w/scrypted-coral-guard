import test from 'node:test';
import assert from 'node:assert/strict';
import { fixLegacyPath, isMasked, maskReply, parseZones, polygonArea, visibleFraction } from '../src/zones';
import { groupOf } from '../src/client';

const left = { name: 'left', points: [[0, 0], [0.5, 0], [0.5, 1], [0, 1]] as [number, number][] };

test('polygon area and point sampling', () => {
    assert.equal(polygonArea(left.points), 0.5);
    assert.equal(polygonArea([[0, 0], [1, 0], [0, 1]]), 0.5);
    assert.equal(visibleFraction([0.6, 0.1, 0.9, 0.4], [left]), 1, 'outside');
    assert.equal(visibleFraction([0.1, 0.1, 0.4, 0.4], [left]), 0, 'inside');
    assert.equal(visibleFraction([0.3, 0.1, 0.7, 0.4], [left]), 0.5, 'straddling half');
    assert.equal(visibleFraction([0.1, 0.1, 0.4, 0.4], []), 1, 'no zones');
});

test('standing point rule: excluded when the bottom centre is in a zone or the box lies almost entirely inside', () => {
    assert.equal(isMasked([0.1, 0.1, 0.4, 0.4], [left], 0.6), true, 'inside');
    assert.equal(isMasked([0.3, 0.1, 0.6, 0.4], [left], 0.6), true, 'feet at x 0.45 inside the zone');
    assert.equal(isMasked([0.4, 0.1, 0.9, 0.4], [left], 0.6), false, 'feet at x 0.65 outside');
    assert.equal(isMasked([0.1, 0.1, 0.4, 0.4], [{ name: 'x', points: [[0, 0], [1, 1]] }], 0.6), false);
});

test('a person whose head and shoulders overlap a background zone but who stands outside it is valid', () => {
    const top = { name: 'top', points: [[0, 0], [1, 0], [1, 0.6], [0, 0.6]] as [number, number][] };   // background: upper 60 %
    assert.equal(isMasked([0.25, 0.32, 0.42, 1.0], [top], 0.6), false, 'feet on the path');
    assert.equal(isMasked([0.25, 0.1, 0.42, 0.5], [top], 0.6), true, 'standing inside the background zone');
});

test('storage: parse round-trip, legacy 0..100 paths, malformed entries ignored with warnings', () => {
    const json = JSON.stringify([left, { name: 'legacy', points: [[0, 0], [50, 0], [50, 100]] }]);
    const p = parseZones(json);
    assert.deepEqual(p.zones[0], left);
    assert.deepEqual(p.zones[1].points, [[0, 0], [0.5, 0], [0.5, 1]]);
    assert.deepEqual(fixLegacyPath([[0.1, 0.2]]), [[0.1, 0.2]]);
    assert.equal(parseZones('{bad').warnings.length, 1);
    assert.equal(parseZones('{}').warnings.length, 1);
    const m = parseZones([{ name: 'ok', points: [] }, { points: [[0, 0]] }, { name: 'ok', points: [] }, { name: 'b', points: [['x', 1]] }]);
    assert.deepEqual(m.zones.map(z => z.name), ['ok']);
    assert.equal(m.warnings.length, 3);
});

test('maskReply: a masked confirming person cannot confirm; a valid one needs 2 of 3 valid replies; no zones = unchanged', () => {
    const th = { personConfirm: 0.65, personKeep: 0.2, animalDetect: 0.3 };
    const r = (box: number[]) => ({ person: { state: 'confirmed', confirmed: true, present: true }, detections: [{ cls: 'person', score: 0.9, box }] }) as any;
    const ticks: boolean[] = [];
    for (let i = 0; i < 3; i++) {
        const m = maskReply(r([0.1, 0.1, 0.3, 0.6]), [left], 0.6, th, ticks, groupOf);
        assert.equal(m.person!.confirmed, false); assert.equal(m.person!.present, false); assert.equal(m.detections!.length, 0);
    }
    const t2: boolean[] = [];
    assert.equal(maskReply(r([0.6, 0.1, 0.8, 0.6]), [left], 0.6, th, t2, groupOf).person!.confirmed, false, 'first valid reply');
    assert.equal(maskReply(r([0.6, 0.1, 0.8, 0.6]), [left], 0.6, th, t2, groupOf).person!.confirmed, true, 'second valid reply');
    const raw = r([0.1, 0.1, 0.3, 0.6]);
    assert.equal(maskReply(raw, [], 0.6, th, [], groupOf), raw);
});
