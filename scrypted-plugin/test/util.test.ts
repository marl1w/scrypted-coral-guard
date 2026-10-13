import test from 'node:test';
import assert from 'node:assert/strict';
import { maskSecrets, monoNow, validCameraId, validStreamUrl } from '../src/util';
import { JpegSplitter } from '../src/jpeg';

test('L4: maskSecrets removes credentials and query strings', () => {
    const t = maskSecrets('Error opening rtsp://admin:hunter2@192.0.2.5:554/h264?token=abc&x=1: Connection refused; also https://u@h/p?k=v');
    assert.ok(!/hunter2|admin|token=abc|k=v/.test(t), t);
    assert.match(t, /rtsp:\/\/\*\*\*@192\.0\.2\.5/);
});

test('L5: only rtsp/rtsps stream overrides are valid', () => {
    for (const ok of ['rtsp://h/p', 'rtsps://u:p@h:322/p?x=1']) assert.equal(validStreamUrl(ok), true);
    for (const bad of ['file:///etc/passwd', 'http://h/p', 'concat:a|b', '/etc/passwd', 'rtsp://h/p -f x', 'pipe:0', '']) assert.equal(validStreamUrl(bad), false, bad);
});

test('L3: camera id validation', () => {
    for (const ok of ['front_door', 'a.b-c', 'x']) assert.equal(validCameraId(ok), true);
    for (const bad of ['', '.', '..', 'a/b', 'a b', 'x'.repeat(65), 'é']) assert.equal(validCameraId(bad), false, bad);
});

test('L6: monoNow is monotonic and close to the wall clock', async () => {
    const a = monoNow();
    await new Promise(r => setTimeout(r, 20));
    const b = monoNow();
    assert.ok(b >= a + 15);
    assert.ok(Math.abs(b - Date.now()) < 1000);
});

test('L2: the splitter accumulator is capped and resyncs after an overflow', () => {
    const s = new JpegSplitter();
    const out: Buffer[] = [];
    s.push(Buffer.from([0xff, 0xd8, 1, 2]), f => out.push(f)); // an open frame that never ends
    const junk = Buffer.alloc(1 << 20, 0x11);
    for (let i = 0; i < 12; i++) s.push(junk, f => out.push(f)); // > 8 MB: dropped, resync
    const good = Buffer.from([0xff, 0xd8, 5, 6, 0xff, 0xd9]);
    s.push(good, f => out.push(f));
    assert.equal(out.length, 1);
    assert.ok(out[0].equals(good));
});
