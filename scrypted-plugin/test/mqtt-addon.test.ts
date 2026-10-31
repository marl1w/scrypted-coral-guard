import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { loadProvider, sdkStub } from './harness';

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

test('MQTT add-on: a Last detection image entity next to presence, named as Home Assistant ids it', () => {
    loadProvider();
    const { mqttLastDetectionEntity } = require('../src/main');
    assert.deepEqual(mqttLastDetectionEntity('Front Door'), { key: 'last_detection', name: 'Last detection', platform: 'image', object_id: 'front_door_last_detection' });
    assert.equal(mqttLastDetectionEntity('***').object_id, undefined);
});

test('MQTT add-on: images are kept, sent as base64 once configured, sent again on hello, and only JPEGs', () => {
    const { MqttAddon } = require('../src/mqtt-addon');
    const logs: string[] = [];
    const a = new MqttAddon('coral-guard', { warn: (m: string) => logs.push(m) } as any);
    const sent: any[] = [];
    a.emit = (m: any) => sent.push(m);
    const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1, 2]);
    a.image('42', 'last_detection', jpeg);
    assert.equal(sent.length, 0, 'not configured yet');
    a.config('42', [{ key: 'last_detection', name: 'Last detection', platform: 'image' }]);
    assert.deepEqual(sent.map(m => [m.type, m.jpeg]), [['config', undefined], ['image', jpeg.toString('base64')]]);
    sent.length = 0;
    a.image('42', 'last_detection', Buffer.from('GIF89a'));
    assert.equal(sent.length, 0);
    assert.equal(logs.length, 1);
    a.resend();
    assert.deepEqual(sent.map(m => m.type), ['alive', 'config', 'image']);
    sent.length = 0;
    a.image('42', 'last_detection', null);
    a.image('42', 'last_detection', null);
    assert.deepEqual(sent, [{ type: 'image', device: '42', key: 'last_detection', jpeg: null }]);
});

test('MQTT add-on: platforms come from the bridge hello; none before it or from a bridge too old to list them', () => {
    const { MqttAddon } = require('../src/mqtt-addon');
    let changed = 0;
    const a = new MqttAddon('coral-guard', console, undefined, () => changed++);
    const sent: any[] = [];
    a.emit = (m: any) => sent.push(m);
    assert.equal(a.connected, false);
    assert.equal(a.supports('image'), false);
    a.onBridge({ v: 1, type: 'hello' });
    assert.equal(a.supports('image'), false, 'old bridge');
    assert.equal(changed, 0);
    a.onBridge({ v: 1, type: 'hello', platforms: ['binary_sensor', 'image', 7] });
    assert.deepEqual(a.platforms, ['binary_sensor', 'image']);
    assert.equal(a.supports('image'), true);
    assert.equal(changed, 1);
    assert.deepEqual(sent.map(m => m.type), ['alive', 'alive'], 'everything sent again on every hello');
    a.onBridge({ v: 1, type: 'hello', platforms: ['binary_sensor', 'image'] });
    assert.equal(changed, 1, 'unchanged');
    a.onBridge({ v: 2, type: 'hello' });
    assert.equal(a.supports('image'), true, 'another contract version is ignored');
});

test('MQTT add-on: finding the bridge asks it for a hello, answering a hello does not (no loop)', () => {
    const { MqttAddon } = require('../src/mqtt-addon');
    const a = new MqttAddon('coral-guard', console);
    const sent: any[] = [];
    a.emit = (m: any) => sent.push(m);
    a.resend(true);
    a.onBridge({ v: 1, type: 'hello', platforms: ['image'] });
    assert.deepEqual(sent, [{ type: 'alive', hello: true }, { type: 'alive' }]);
});

function quietProvider() {
    const P = loadProvider();
    const p: any = new P('n');
    clearInterval(p.healthTimer);
    p.mqtt.stop();
    const sent: any[] = [];
    p.mqtt.emit = (m: any) => sent.push(m);
    return { p, sent };
}

test('MQTT add-on: the Last detection entity is declared only while the bridge takes images, and re-sent when a hello adds them', () => {
    const { p, sent } = quietProvider();
    p.announced.set('presence:1:person', 'Front Door Person Presence');
    p.ensurePresence('1', 'Front Door', ['person']);
    p.ensurePresence('1', 'Front Door', ['person']);
    assert.deepEqual(sent.filter(m => m.type === 'config').map(m => m.entities.map((e: any) => e.key)), [['person']]);
    sent.length = 0;
    p.mqtt.onBridge({ v: 1, type: 'hello', platforms: ['binary_sensor', 'image'] });
    assert.deepEqual(sent.filter(m => m.type === 'config').map(m => m.entities.map((e: any) => e.key)), [['person', 'last_detection'], ['person', 'last_detection']]);
    sent.length = 0;
    p.mqtt.onBridge({ v: 1, type: 'hello' });
    assert.deepEqual(sent.filter(m => m.type === 'config').map(m => m.entities.map((e: any) => e.key)), [['person'], ['person']]);
});

test('Last detection: nothing is drawn without the bridge or its image support', async () => {
    const { p } = quietProvider();
    let lookups = 0;
    sdkStub.mediaManager.getFFmpegPath = async () => { lookups++; return 'ffmpeg'; };
    const J = Buffer.from([0xff, 0xd8, 0xff, 0xd9]);
    assert.equal(await p.publishLastDetection('1', J, []), false);
    p.mqtt.bridge = { id: 'b', listener: { removeListener() { } } };
    assert.equal(await p.publishLastDetection('1', J, []), false, 'connected, no image support');
    assert.equal(lookups, 0);
});

test('Last detection: ffmpeg is looked up once, a hung lookup times out and is retried; failures logged once, too large retried smaller', async () => {
    const { p } = quietProvider();
    p.mqtt.bridge = { id: 'b', listener: { removeListener() { } } };
    p.mqtt.onBridge({ v: 1, type: 'hello', platforms: ['image'] });
    const fake = path.join(os.tmpdir(), `coral-guard-ld-${process.pid}.sh`);
    const frame = Buffer.from([0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x01, 0x68, 0x02, 0x80, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xd9]);
    const warns: string[] = [];
    p.console = { ...p.console, warn: (m: string) => warns.push(m) };
    const images: Buffer[] = [];
    p.mqtt.image = (_d: string, _k: string, j: Buffer) => images.push(j);
    let lookups = 0;
    try {
        // hung lookup: times out, the next picture looks again
        mock.timers.enable({ apis: ['setTimeout'] });
        sdkStub.mediaManager.getFFmpegPath = () => { lookups++; return new Promise(() => { }); };
        const hung = p.publishLastDetection('1', frame, []);
        await new Promise(r => setImmediate(r)); // setTimeout is mocked here
        mock.timers.tick(10_000);
        mock.timers.reset();
        assert.equal(await hung, false);
        assert.match(warns[0], /ffmpeg path lookup timed out/);
        // a picture over 1 MB at the first quality, fine at the second; the lookup is not repeated
        fs.writeFileSync(fake, `#!/bin/sh\ncat >/dev/null\ncase "$*" in *"-q:v 5 "*) head -c 1048600 /dev/zero;; *) printf '\\377\\330\\377\\331';; esac\n`, { mode: 0o755 });
        sdkStub.mediaManager.getFFmpegPath = async () => { lookups++; return fake; };
        assert.equal(await p.publishLastDetection('1', frame, []), true);
        assert.equal(await p.publishLastDetection('1', frame, []), true);
        assert.equal(lookups, 2);
        assert.deepEqual(images.map(i => [...i]), [[0xff, 0xd8, 0xff, 0xd9], [0xff, 0xd8, 0xff, 0xd9]]);
        // failures: the reason is logged, at most once every few minutes
        fs.writeFileSync(fake, `#!/bin/sh\ncat >/dev/null\necho 'Invalid data found' >&2\nexit 1\n`, { mode: 0o755 });
        p.annotateWarnedAt = -Infinity;
        warns.length = 0;
        assert.equal(await p.publishLastDetection('1', frame, []), false);
        assert.equal(await p.publishLastDetection('1', frame, []), false);
        assert.equal(warns.length, 1);
        assert.match(warns[0], /exit 1: Invalid data found/);
    }
    finally {
        mock.timers.reset();
        fs.rmSync(fake, { force: true });
    }
});
