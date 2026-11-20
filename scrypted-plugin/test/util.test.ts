import test from 'node:test';
import assert from 'node:assert/strict';
import { boxMoved, maskSecrets, monoNow, validCameraId, validStreamUrl } from '../src/util';
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

test('boxMoved: two sides travelling move a box; one side redrawn (laundry blown wider) or a wobble does not; at the border one inner side does', () => {
    const L = 0.03 * Math.SQRT2;
    const rack = [0.67, 0.156, 0.705, 0.321];
    assert.equal(boxMoved([0.671, 0.159, 0.720, 0.32], rack, L), false, 'the 13:20 laundry: one side 0.015 wider');
    assert.equal(boxMoved([0.60, 0.156, 0.80, 0.321], rack, L), true, 'both x sides far apart: moved (or a much larger box)');
    assert.equal(boxMoved([0.25, 0.1, 0.55, 0.8], [0.1, 0.1, 0.4, 0.8], L), true, 'walking sideways: both x sides moved');
    assert.equal(boxMoved([0.1, 0.1, 0.4, 0.95], [0.1, 0.1, 0.4, 0.8], L), false, 'only the bottom redrawn');
    assert.equal(boxMoved([0.0, 0.1, 0.25, 0.8], [0.0, 0.1, 0.4, 0.8], L), true, 'cut by the left border: the inner (right) side moved');
});
