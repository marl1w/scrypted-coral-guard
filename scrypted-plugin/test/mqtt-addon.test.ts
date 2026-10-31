import test from 'node:test';
import assert from 'node:assert/strict';
import { loadProvider } from './harness';

test('MQTT add-on: one occupancy entity per enabled group, named as Home Assistant ids them', () => {
    loadProvider();
    const { mqttPresenceEntities } = require('../src/main');
    const e = mqttPresenceEntities('Front Door', ['person', 'animal']);
    assert.deepEqual(e.map((x: any) => [x.key, x.name, x.object_id]), [
        ['person', 'Person presence', 'front_door_person_occupancy'],
        ['animal', 'Animal presence', 'front_door_animal_occupancy'],
    ]);
    assert.ok(e.every((x: any) => x.platform === 'binary_sensor' && x.device_class === 'occupancy'));
});

test('MQTT add-on: presence changes reach the add-on only for enabled groups', () => {
    const P = loadProvider();
    const p: any = new P('n');
    clearInterval(p.healthTimer);
    p.mqtt.stop();
    const sent: any[] = [];
    p.mqtt.state = (device: string, key: string, value: any) => sent.push([device, key, value]);
    p.announced.set('presence:1:person', 'x');
    p.announced.set('presence:1:vehicle', 'x');
    p.presenceDevice = () => ({ set() { } });
    p.setPresence('1', 'vehicle', true);
    p.setPresence('1', 'person', true);
    p.setPresence('1', 'animal', true); // not enabled
    assert.deepEqual(sent, [['1', 'vehicle', true], ['1', 'person', true]]);
});
