import test from 'node:test';
import assert from 'node:assert/strict';
import { parsePresenceNativeId, presenceNativeId, presenceOf } from '../src/presence';

const none = () => false;

test('presence: still and parked vehicles count, as do held people and animals', () => {
    const items = [
        { group: 'vehicle', cls: 'car', state: 'still', box: [0.1, 0.5, 0.3, 0.9] },
        { group: 'vehicle', cls: 'car', state: 'parked', box: [0.4, 0.5, 0.6, 0.9] },
    ];
    assert.deepEqual(presenceOf(items, ['person', 'vehicle', 'animal'], none), { person: false, vehicle: true, animal: false });
    assert.deepEqual(presenceOf([{ group: 'person' }, { group: 'animal' }], ['person', 'vehicle', 'animal'], none), { person: true, vehicle: false, animal: true });
});

test('presence: a group the camera does not detect is never reported', () => {
    assert.deepEqual(presenceOf([{ group: 'vehicle', box: [0, 0, 1, 1] }, { group: 'animal' }], ['person'], none), { person: false, vehicle: false, animal: false });
});

test('presence: an object whose box lies in an exclusion zone is ignored', () => {
    const inZone = (b: number[]) => b[0] >= 0.5;
    assert.equal(presenceOf([{ group: 'vehicle', box: [0.6, 0.5, 0.9, 0.9] }], ['vehicle'], inZone).vehicle, false);
    assert.equal(presenceOf([{ group: 'vehicle', box: [0.1, 0.5, 0.3, 0.9] }], ['vehicle'], inZone).vehicle, true);
    assert.equal(presenceOf([{ group: 'person' }], ['person'], () => true).person, true, 'no box: kept');
});

test('presence: native ids round-trip and reject anything else', () => {
    assert.deepEqual(parsePresenceNativeId(presenceNativeId('cam1', 'vehicle')), { cameraId: 'cam1', group: 'vehicle' });
    assert.equal(parsePresenceNativeId('presence:cam1:package'), undefined);
    assert.equal(parsePresenceNativeId('something-else'), undefined);
});
