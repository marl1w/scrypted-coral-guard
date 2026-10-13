import test from 'node:test';
import assert from 'node:assert/strict';
import { isParked, viewOfReply } from '../src/reply-view';
import { groupOf } from '../src/client';

const th = { person: 0.65, vehicle: 0.5, animal: 0.5 };
const P = { state: 'idle', confirmed: false, present: false };

test('group comes from the service or is derived from the class (old service)', () => {
    assert.equal(groupOf({ cls: 'dog' }), 'animal');
    assert.equal(groupOf({ cls: 'truck' }), 'vehicle');
    assert.equal(groupOf({ cls: 'person' }), 'person');
    assert.equal(groupOf({ cls: 'chair' }), undefined);
    assert.equal(groupOf({ cls: 'car', group: 'animal' }), 'animal');
});

test('view: persons above 0.65, weak candidate kept apart, vehicles and animals from 0.5, unknown classes ignored', () => {
    const v = viewOfReply({
        person: P as any,
        detections: [
            { cls: 'person', score: 0.9, box: [0, 0, 1, 1] }, { cls: 'person', score: 0.26, box: [0, 0, 1, 1] },
            { cls: 'car', score: 0.92, box: [0, 0, 0.5, 0.5] }, { cls: 'bus', score: 0.4, box: [0, 0, 0.5, 0.5] },
            { cls: 'dog', score: 0.7, box: [0, 0, 0.1, 0.1] }, { cls: 'chair', score: 0.9, box: [0, 0, 0.1, 0.1] },
        ],
    } as any, th);
    assert.equal(v.frame.personCount, 1);
    assert.equal(v.frame.bestScore, 0.9);
    assert.equal(v.frame.candidateScore, 0.26);
    assert.equal(v.frame.vehicles, 1);
    assert.equal(v.frame.animals, 1);
    assert.equal(v.boxes.length, 4);
});

test('parked: per-detection flag, else overlapping track, else only an unambiguous track list', () => {
    const tracks = [{ id: 1, cls: 'car', parked: true, since_ms: 1, box: [0.1, 0.1, 0.5, 0.5] }, { id: 2, cls: 'car', parked: false, since_ms: 1, box: [0.6, 0.1, 0.9, 0.5] }];
    assert.equal(isParked({ box: [0.12, 0.1, 0.5, 0.5], cls: 'car' }, tracks), true);
    assert.equal(isParked({ box: [0.62, 0.1, 0.9, 0.5], cls: 'car' }, tracks), false);
    assert.equal(isParked({ box: [0, 0, 1, 1], cls: 'car', parked: true }, undefined), true);
    assert.equal(isParked({ box: [0, 0, 1, 1], cls: 'car' }, [{ id: 1, cls: 'car', parked: true, since_ms: 1 }]), true);
    assert.equal(isParked({ box: [0, 0, 1, 1], cls: 'car' }, [{ id: 1, cls: 'car', parked: true, since_ms: 1 }, { id: 2, cls: 'car', parked: false, since_ms: 1 }]), false);
    assert.equal(isParked({ box: [0, 0, 1, 1], cls: 'car' }, undefined), false);
});

test('weak vehicles (service flag): drawn dashed as possible, counted apart, never in the vehicle count; old services without the flag are unchanged', () => {
    const v = viewOfReply({ person: P as any, detections: [
        { cls: 'car', score: 0.38, box: [0, 0, 0.5, 0.5], weak: true }, { cls: 'truck', score: 0.9, box: [0.5, 0, 1, 0.5] }, { cls: 'bus', score: 0.4, box: [0, 0, 0.1, 0.1] },
    ] } as any, th);
    assert.equal(v.frame.vehicles, 1);
    assert.equal(v.frame.weakVehicles, 1);
    assert.equal(v.boxes.length, 2, 'an unflagged 0.4 from an old service is not drawn');
    assert.equal(v.boxes[0].weak, true);
});

test('track states: only a vehicle on a moving track is eligible; still/parked/unmatched are not and are labelled; no state keeps the old rule', () => {
    const box = [0.1, 0.1, 0.5, 0.5];
    const tr = (state: string, parked = false) => [{ id: 1, cls: 'car', parked, state, since_ms: 0, box }];
    assert.equal(isParked({ box, cls: 'car' }, tr('moving')), false);
    assert.equal(isParked({ box, cls: 'car' }, tr('still')), true, 'a still car never confirms');
    assert.equal(isParked({ box, cls: 'car', parked: false }, tr('still')), true, 'state wins over the detection flag');
    assert.equal(isParked({ box, cls: 'car' }, tr('parked', true)), true);
    assert.equal(isParked({ box: [0.6, 0.6, 0.9, 0.9], cls: 'car' }, tr('moving')), true, 'no matching moving track');
    assert.equal(isParked({ box, cls: 'car' }, [{ id: 1, cls: 'car', parked: false, since_ms: 0, box }]), false, 'old service: non-parked stays eligible');
    const v = viewOfReply({ person: { state: 'idle', confirmed: false, present: false }, vehicle: { tracks: tr('still') }, detections: [{ cls: 'car', score: 0.62, box }] } as any, { person: 0.65, vehicle: 0.5, animal: 0.5 });
    assert.equal(v.boxes[0].still, true);
    assert.ok(!v.boxes[0].parked);
});

test('panel thresholds: persons solid from confirm, dashed from keep, hidden below; vehicles from detect (0.40-0.50 shown); animals from detect (0.30 shown)', () => {
    const v = viewOfReply({ person: { state: 'idle', confirmed: false, present: false }, detections: [
        { cls: 'person', score: 0.7, box: [0, 0, 0.1, 0.1] }, { cls: 'person', score: 0.3, box: [0, 0, 0.1, 0.1] }, { cls: 'person', score: 0.1, box: [0, 0, 0.1, 0.1] },
        { cls: 'car', score: 0.45, box: [0, 0, 0.1, 0.1] }, { cls: 'car', score: 0.35, box: [0, 0, 0.1, 0.1] },
        { cls: 'dog', score: 0.32, box: [0, 0, 0.1, 0.1] },
    ] } as any, { person: 0.65, personKeep: 0.2, vehicle: 0.4, animal: 0.3 });
    assert.deepEqual(v.boxes.map(b => `${b.cls}:${b.weak ? 'weak' : 'solid'}`), ['person:solid', 'person:weak', 'car:solid', 'dog:solid']);
    assert.equal(v.frame.vehicles, 1); assert.equal(v.frame.animals, 1);
});

test('panel: categories disabled in Detect are neither drawn nor counted (weak ones too)', () => {
    const v = viewOfReply({ person: { state: 'idle', confirmed: false, present: false }, detections: [
        { cls: 'person', score: 0.9, box: [0, 0, 0.1, 0.1] }, { cls: 'person', score: 0.3, box: [0, 0, 0.1, 0.1] },
        { cls: 'car', score: 0.8, box: [0, 0, 0.1, 0.1] }, { cls: 'car', score: 0.35, weak: true, box: [0, 0, 0.1, 0.1] },
        { cls: 'dog', score: 0.7, box: [0, 0, 0.1, 0.1] },
    ] } as any, { person: 0.65, personKeep: 0.2, vehicle: 0.4, animal: 0.3, enabled: ['animal'] });
    assert.deepEqual(v.boxes.map(b => b.cls), ['dog']);
    assert.equal(v.frame.personCount, 0); assert.equal(v.frame.vehicles, 0); assert.equal(v.frame.weakVehicles, 0);
    assert.equal(v.frame.candidateScore, undefined); assert.equal(v.frame.animals, 1);
});
