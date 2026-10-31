import test from 'node:test';
import assert from 'node:assert/strict';
import child_process from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { annotateFilter, annotateJpeg, isJpeg, MAX_IMAGE_BYTES, TOO_LARGE } from '../src/annotate';
import { jpegDimensions } from '../src/jpeg';

test('annotateFilter: boxes in pixels with the group colour, clamped; wide frames scaled to 1280 first', () => {
    assert.equal(annotateFilter([{ x0: 0.1, y0: 0.2, x1: 0.3, y1: 0.7, group: 'person' }], 640, 360),
        'drawbox=x=64:y=72:w=128:h=180:color=0xe5484d:t=2');
    assert.equal(annotateFilter([{ x0: 0.5, y0: 0.5, x1: -1, y1: 2, group: 'vehicle' }], 2560, 1440),
        'scale=1280:720,drawbox=x=0:y=360:w=640:h=360:color=0x3b82f6:t=4');
    assert.equal(annotateFilter([], 640, 360), 'null');
    assert.equal(annotateFilter(Array.from({ length: 30 }, () => ({ x0: 0, y0: 0, x1: 1, y1: 1, group: 'animal' as const })), 640, 360).split(',').length, 20);
});

test('annotateJpeg: refuses what is not a JPEG and says why ffmpeg failed: exit code and stderr tail, timeout, too large', async () => {
    assert.deepEqual(await annotateJpeg('ffmpeg', Buffer.from('GIF89a'), []), { error: 'not a JPEG frame' });
    const fake = path.join(os.tmpdir(), `coral-guard-annotate-${process.pid}.sh`);
    fs.writeFileSync(fake, `#!/bin/sh\ncat >/dev/null\nprintf '%0300d' 0 >&2\necho ' Invalid data found' >&2\nexit 1\n`, { mode: 0o755 });
    const sof = Buffer.from([0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x01, 0x68, 0x02, 0x80, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xd9]);
    try {
        const failed = await annotateJpeg(fake, sof, []);
        assert.match(failed.error!, /^exit 1: 0+ Invalid data found$/);
        assert.ok(failed.error!.length <= 'exit 1: '.length + 200, 'only the end of stderr');
        fs.writeFileSync(fake, '#!/bin/sh\nsleep 5\n', { mode: 0o755 });
        const t0 = Date.now();
        assert.deepEqual(await annotateJpeg(fake, sof, [], 200), { error: 'timeout' });
        assert.ok(Date.now() - t0 < 2000, 'killed at the timeout');
        fs.writeFileSync(fake, `#!/bin/sh\ncat >/dev/null\nhead -c ${MAX_IMAGE_BYTES + 10} /dev/zero\n`, { mode: 0o755 });
        assert.deepEqual(await annotateJpeg(fake, sof, []), { error: TOO_LARGE });
        fs.writeFileSync(fake, `#!/bin/sh\ncat >/dev/null\necho "$@" >&2\nexit 2\n`, { mode: 0o755 });
        assert.match((await annotateJpeg(fake, sof, [], undefined, 12)).error!, /-q:v 12 /, 'the quality asked for');
    }
    finally { fs.unlinkSync(fake); }
});

const hasFfmpeg = child_process.spawnSync('ffmpeg', ['-version']).status === 0;
test('annotateJpeg: a real frame comes back as a JPEG of the same size with the box drawn', { skip: !hasFfmpeg && 'no ffmpeg on PATH' }, async () => {
    const src = child_process.spawnSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=640x360', '-frames:v', '1', '-f', 'image2pipe', '-c:v', 'mjpeg', 'pipe:1']).stdout;
    const out = (await annotateJpeg('ffmpeg', src, [{ x0: 0.25, y0: 0.25, x1: 0.75, y1: 0.75, group: 'person' }])).jpeg;
    assert.ok(out && isJpeg(out));
    assert.deepEqual(jpegDimensions(out!), { w: 640, h: 360 });
    const rgb = child_process.spawnSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-i', 'pipe:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], { input: out }).stdout;
    const px = (x: number, y: number) => [...rgb.subarray((y * 640 + x) * 3, (y * 640 + x) * 3 + 3)];
    assert.ok(px(161, 180)[0] > 150, 'the box edge is red');
    assert.ok(px(320, 180)[0] < 40, 'inside stays black');
});
