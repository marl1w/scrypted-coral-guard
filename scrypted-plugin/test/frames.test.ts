// startFrames against a fake ffmpeg executable (a node script), no real ffmpeg or Scrypted needed.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import path from 'path';
import { loadFrames, sdkStub, sleep } from './harness';

const dir = path.join(__dirname, 'fixtures');
const fake = path.join(dir, 'fake-ffmpeg.js');

function setup(convert?: () => Promise<any>) {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(fake, `#!/usr/bin/env node
const fs = require('fs'), path = require('path');
if (process.argv[2] === '-version') { console.log('ffmpeg version ' + (process.env.FAKE_VERSION || '7.0.1') + ' Copyright (c) test'); process.exit(0); }
fs.writeFileSync(path.join(__dirname, 'args.json'), JSON.stringify(process.argv.slice(2)));
fs.writeFileSync(path.join(__dirname, 'env.txt'), process.env.LIBVA_DRIVER_NAME || '');
fs.writeFileSync(path.join(__dirname, 'pid'), String(process.pid));
const mode = process.env.FAKE_MODE || process.argv.join(' ').match(/MODE=(\\w+)/)?.[1] || 'frames';
const J = n => Buffer.from([0xff, 0xd8, n, 0xff, 0x00, 0xff, 0xd9]);
// one-shot listener: every -i is one connection, the second is refused (the live failure)
if (process.argv.filter(a => a === '-i').length > 1) { process.stderr.write('[tcp @ 0x1] Connection to tcp://127.0.0.1:33893?token=SECRET failed: Connection refused\\nError opening input files: Connection refused'); process.exit(145); }
if (mode === 'hwfail' && process.argv.includes('-hwaccel')) { process.stderr.write('Failed to initialise VAAPI'); process.exit(1); }
if (mode === 'gray') { const g = Buffer.alloc(8 * 4, 7); try { require('fs').writeSync(3, g); require('fs').writeSync(3, g); } catch (e) {} }
if (mode === 'hwruntime' && process.argv.includes('-hwaccel')) { process.stdout.write(Buffer.concat([Buffer.from([0xff, 0xd8, 1, 0xff, 0x00, 0xff, 0xd9])])); setTimeout(() => { process.stderr.write('[hwdownload] Failed to download frame: -5'); process.exit(251); }, 150); return; }
if (mode === 'probe') { process.stderr.write('[rtsp @ 0x1] Could not find codec parameters for stream 0 (Video: h264): unspecified size'); process.exit(183); }
if (mode === 'hwzero' && process.argv.includes('-hwaccel')) { process.stderr.write('[vaapi @ 0x1] Failed to read image from surface 0x4: 20 (the requested function is not implemented).'); process.exit(0); }
if (mode === 'die') { process.stderr.write('Failed rtsp://admin:hunter2@192.0.2.1/live?token=SECRET'); process.exit(3); }
process.stdout.write(Buffer.concat([J(1), J(2), J(3)]));
setInterval(() => {}, 1000);
`);
    fs.chmodSync(fake, 0o755);
    // frames.ts keeps the first mediaManager object it saw: mutate it instead of replacing it
    sdkStub.mediaManager = Object.assign(sdkStub.mediaManager || {}, {
        getFFmpegPath: async () => fake,
        convertMediaObjectToJSON: convert || (async () => ({ url: 'rtsp://x/live', inputArguments: ['-rtsp_transport', 'tcp'], ffmpegPath: fake })),
    });
}
const args = () => JSON.parse(fs.readFileSync(path.join(dir, 'args.json'), 'utf8')) as string[];
const alive = (pid: number) => { try { process.kill(pid, 0); return true; } catch { return false; } };

test('startFrames splits the ffmpeg stdout into JPEGs, builds the width/quality/fps arguments', async () => {
    setup();
    const { startFrames } = loadFrames();
    const got: Buffer[] = [];
    const fsx = await startFrames({ camera: { getVideoStream: async () => ({}) } as any, intervalMs: 500, prebufferMs: 4000, maxWidth: 640, quality: 85, console, onFrame: j => got.push(j) });
    for (let i = 0; i < 100 && got.length < 3; i++) await sleep(20);
    assert.equal(got.length, 3);
    assert.deepEqual([...got[1]], [0xff, 0xd8, 2, 0xff, 0x00, 0xff, 0xd9]);
    const a = args().join(' ');
    assert.match(a, /-vf fps=2\.0000,scale='min\(640,iw\)':-2 /);
    assert.match(a, /-q:v 6/);
    fsx.stop();
    await fsx.exited;
});

test('M24: stop() really kills the ffmpeg process', async () => {
    setup();
    const { startFrames } = loadFrames();
    const fsx = await startFrames({ camera: { getVideoStream: async () => ({}) } as any, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: () => { } });
    await sleep(300);
    const pid = Number(fs.readFileSync(path.join(dir, 'pid'), 'utf8'));
    assert.ok(alive(pid));
    fsx.stop();
    for (let i = 0; i < 50 && alive(pid); i++) await sleep(20);
    assert.equal(alive(pid), false);
});

test('L5: stream override gets a protocol whitelist, non-rtsp schemes are rejected', async () => {
    setup();
    const { startFrames } = loadFrames();
    const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live', intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: () => { } });
    await sleep(300);
    const a = args();
    assert.equal(a[a.indexOf('-protocol_whitelist') + 1], 'rtsp,rtsps,rtp,udp,tcp,tls,crypto');
    assert.ok(a.indexOf('-protocol_whitelist') < a.indexOf('-i'));
    fsx.stop();
    await assert.rejects(startFrames({ camera: {} as any, inputUrl: 'file:///etc/passwd', intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: () => { } }), /rtsp/);
});

test('L4: the exit message masks credentials and query strings from ffmpeg stderr', async () => {
    setup();
    const { startFrames } = loadFrames();
    const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live?MODE=die', intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: () => { } });
    const msg = await fsx.exited;
    assert.match(String(msg), /ffmpeg exited 3/);
    assert.ok(!/hunter2|SECRET|admin/.test(String(msg)), String(msg));
});

test('producer inputArguments already carry -i <url>: exactly one -i, no second connection (live regression)', async () => {
    const url = 'tcp://127.0.0.1:33893/abc/def';
    setup(async () => ({ url, inputArguments: ['-analyzeduration', '0', '-f', 'rtsp', '-i', url], ffmpegPath: fake }));
    const { startFrames } = loadFrames();
    const logs: string[] = [];
    const con = { log: (m: string) => logs.push(m), warn: (m: string) => logs.push(m) } as any;
    const got: Buffer[] = [];
    const fsx = await startFrames({ camera: { getVideoStream: async () => ({}) } as any, intervalMs: 500, prebufferMs: 4000, maxWidth: 640, quality: 85, console: con, onFrame: j => got.push(j) });
    for (let i = 0; i < 100 && got.length < 3; i++) await sleep(20);
    assert.equal(got.length, 3);
    const a = args();
    assert.equal(a.filter(x => x === '-i').length, 1);
    assert.ok(a.indexOf('-i') < a.indexOf('-an'));
    assert.ok(logs.some(l => /tcp:\/\/127\.0\.0\.1:33893 \(query masked\).*url carried in inputArguments/.test(l)), logs.join('|'));
    assert.ok(logs.some(l => /first jpeg \+\d+ ms/.test(l)));
    fsx.stop();
    await fsx.exited;
});

test('inputArguments without -i still get the url appended; failure logs a masked stderr tail', async () => {
    setup(async () => ({ url: 'tcp://127.0.0.1:1/x', inputArguments: ['-i', 'tcp://127.0.0.1:1/x', '-i', 'tcp://127.0.0.1:1/x'], ffmpegPath: fake }));
    const { buildInputArgs, describeUrl } = loadFrames() as any;
    assert.deepEqual(buildInputArgs(['-f', 'rtsp'], 'rtsp://a/b'), ['-f', 'rtsp', '-i', 'rtsp://a/b']);
    assert.equal(describeUrl('rtsp://u:p@h:554/x?t=1'), 'rtsp://h:554');
    // a double -i (old behaviour) is refused by the fake one-shot listener and the tail is masked
    const { startFrames } = loadFrames();
    const logs: string[] = [];
    const con = { log: () => { }, warn: (m: string) => logs.push(m) } as any;
    const f2 = await startFrames({ camera: { getVideoStream: async () => ({}) } as any, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console: con, onFrame: () => { } });
    const m2 = String(await f2.exited);
    assert.match(m2, /ffmpeg exited 145/);
    assert.ok(!/SECRET/.test(m2) && logs.some(l => /stderr tail: .*Connection refused/.test(l) && !/SECRET/.test(l)), logs.join('|'));
});

test('all frames are decoded (no -skip_frame), the fps filter samples first, software chain has no hwdownload', async () => {
    const url = 'tcp://127.0.0.1:33893/abc';
    setup(async () => ({ url, inputArguments: ['-analyzeduration', '0', '-f', 'rtsp', '-i', url], ffmpegPath: fake }));
    const { startFrames } = loadFrames();
    const fsx = await startFrames({ camera: { getVideoStream: async () => ({}) } as any, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: () => { } });
    await sleep(300);
    const a = args();
    assert.ok(!a.includes('-skip_frame'));
    assert.ok(!a.includes('-fps_mode') && !a.includes('-vsync'), 'dense frames: default CFR sampling by the fps filter');
    assert.ok(!a.includes('-hwaccel') && !/hwdownload/.test(a.join(' ')));
    fsx.stop();
    await fsx.exited;
});

test('VAAPI: hwaccel args before -i with nv12 output, no hwdownload step, never scale_vaapi', async () => {
    setup();
    const { startFrames } = loadFrames();
    const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live', hwaccel: true, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: () => { } });
    await sleep(300);
    const a = args();
    assert.equal(a[a.indexOf('-hwaccel') + 1], 'vaapi');
    assert.equal(a[a.indexOf('-hwaccel_output_format') + 1], 'nv12');
    assert.equal(a[a.indexOf('-threads') + 1], '1');
    assert.ok(a.indexOf('-hwaccel') < a.indexOf('-i'));
    const vf = a[a.indexOf('-vf') + 1];
    assert.match(vf, /^fps=1\.0000,scale=/);
    assert.ok(!/hwdownload/.test(a.join(' ')));
    assert.ok(!/scale_vaapi/.test(a.join(' ')));
    fsx.stop();
    await fsx.exited;
});

test('VAAPI failure before the first frame resolves as hwaccel failed (caller falls back to software)', async () => {
    setup();
    const { startFrames } = loadFrames();
    const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live?MODE=hwfail', hwaccel: true, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console: { log() { }, warn() { } } as any, onFrame: () => { } });
    assert.match(String(await fsx.exited), /^hwaccel failed: ffmpeg exited 1/);
});

test('gray output: filter_complex with two outputs, rawvideo gray on pipe:3, frames delivered whole', async () => {
    setup();
    const { startFrames } = loadFrames();
    const gray: Uint8Array[] = [];
    const got: Buffer[] = [];
    const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live?MODE=gray', intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: j => got.push(j), gray: { width: 8, height: 4, onGray: f => gray.push(f) } });
    for (let i = 0; i < 100 && gray.length < 2; i++) await sleep(20);
    assert.equal(gray.length, 2);
    assert.equal(gray[0].length, 32);
    assert.equal(gray[0][0], 7);
    const a = args();
    const g = a[a.indexOf('-filter_complex') + 1];
    assert.match(g, /^fps=1\.0000,scale='min\(640,iw\)':-2,split=2\[v1\]\[b\];\[b\]fps=1,scale=8:4:flags=area,format=gray\[v2\]$/);
    assert.ok(a.includes('pipe:3') && a.includes('rawvideo') && !a.includes('-vf'));
    fsx.stop();
    await fsx.exited;
});

test('minGapMs drops keyframes arriving closer than the frame interval; unexpected exit logs a masked 15-line stderr tail', async () => {
    setup();
    const { startFrames, stderrTail } = loadFrames();
    const got: Buffer[] = [];
    const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live', minGapMs: 5000, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: j => got.push(j) });
    await sleep(300);
    assert.equal(got.length, 1);
    fsx.stop();
    await fsx.exited;
    const tail = stderrTail(Array.from({ length: 20 }, (_, i) => `line ${i} rtsp://u:p@192.0.2.2:554/path?token=SECRET`).join('\n'));
    assert.equal(tail.split(' | ').length, 15);
    assert.ok(!/SECRET|u:p|\/path/.test(tail) && /rtsp:\/\/192\.0\.2\.2:554/.test(tail), tail);
});

test('first jpeg logs the verified decode path (GPU only when frames came through the VAAPI chain)', async () => {
    setup();
    const { startFrames } = loadFrames();
    for (const [o, want] of [[{ hwaccel: true }, /decode GPU\(vaapi\) verified/], [{ hwaccel: true, vaDriver: 'i965' }, /decode GPU\(vaapi, driver i965\) verified/], [{ hwFailedBefore: true }, /decode CPU\(requested GPU, not initialised\)/], [{}, /decode CPU$/]] as [any, RegExp][]) {
        const logs: string[] = [];
        const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live', ...o, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console: { log: (m: string) => logs.push(m), warn() { } } as any, onFrame: () => { } });
        for (let i = 0; i < 100 && !logs.some(l => /decode /.test(l)); i++) await sleep(20);
        assert.ok(logs.some(l => want.test(l)), logs.join('|'));
        fsx.stop();
        await fsx.exited;
    }
});

test('decoder runs with one thread, before the input', async () => {
    setup();
    const { startFrames } = loadFrames();
    const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live', intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: () => { } });
    await sleep(300);
    const a = args();
    assert.equal(a[a.indexOf('-threads') + 1], '1');
    assert.ok(a.indexOf('-threads') < a.indexOf('-i'));
    fsx.stop();
    await fsx.exited;
});

test('ffmpegVersion picks the per-output passthrough option by version (-fps_mode from 5.1, -vsync 0 before)', async () => {
    setup();
    const { ffmpegVersion } = loadFrames();
    const oldFake = path.join(dir, 'fake-old-ffmpeg.js');
    fs.writeFileSync(oldFake, `#!/usr/bin/env node\nconsole.log('ffmpeg version 4.4.2 Copyright');`);
    fs.chmodSync(oldFake, 0o755);
    assert.deepEqual((await ffmpegVersion(oldFake)).passthrough, ['-vsync', '0']);
    assert.deepEqual((await ffmpegVersion(fake)).passthrough, ['-fps_mode', 'passthrough']);
});

test('a VAAPI failure after the first frame (stderr names hwdownload) also resolves as hwaccel failed', async () => {
    setup();
    const { startFrames } = loadFrames();
    const fsx = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live?MODE=hwruntime', hwaccel: true, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console: { log() { }, warn() { } } as any, onFrame: () => { } });
    assert.match(String(await fsx.exited), /^hwaccel failed: ffmpeg exited 251/);
});

test('a hung getVideoStream times out (the caller retries with backoff) and exit reasons carry no URL paths or credentials', async () => {
    setup();
    const { startFrames } = loadFrames();
    await assert.rejects(startFrames({ camera: { getVideoStream: () => new Promise(() => { }) } as any, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console, onFrame: () => { }, resolveTimeoutMs: 60 }), /getVideoStream timed out/);
    const { maskUrls } = require('../src/util') as typeof import('../src/util');
    const m = maskUrls('Connection to rtsp://admin:pw@192.0.2.9:554/live/main?token=abc refused');
    assert.equal(m, 'Connection to rtsp://192.0.2.9:554 refused');
});

const opt = (id: string, w: number, h: number, container = 'rtsp') => ({ id, name: id, container, video: { width: w, height: h, codec: 'h264', bitrate: 1000000 } });

test('stream choice by resolution: wide cameras need >= 1280 px, others >= 896, else the largest below 1920, never above ~2 Mpx', () => {
    const { chooseStream } = loadFrames();
    const wide = chooseStream([opt('main', 3840, 1080), opt('mid', 1920, 576), opt('sub', 1536, 576), opt('tiny', 640, 240)]);
    assert.equal(wide.wide, true);
    assert.equal(wide.choice?.id, 'sub', 'smallest at >= 1280 wide');
    const normal = chooseStream([opt('main', 2560, 1440), opt('mid', 1920, 1080), opt('sub', 1280, 720), opt('tiny', 640, 360)]);
    assert.equal(normal.wide, false);
    assert.equal(normal.choice?.id, 'sub');
    const none = chooseStream([opt('big', 3840, 2160), opt('a', 800, 450), opt('b', 704, 576)]);
    assert.equal(none.choice?.id, 'b', 'nothing reaches 896: the largest (by area) below 1920 wide; big is above ~2 Mpx');
    assert.equal(chooseStream([opt('only', 3840, 2160)]).choice, undefined, 'only an oversized stream: fall back to the default destination');
    assert.equal(chooseStream(undefined).choice, undefined);
    assert.equal(chooseStream([{ id: 'x' }]).all.length, 0, 'streams without dimensions are ignored');
});

test('the chosen stream id is requested, the width cap follows the aspect, and the log names stream and posted size', async () => {
    setup();
    const { startFrames } = loadFrames();
    const asked: any[] = [];
    const logs: string[] = [];
    const camera = { getVideoStreamOptions: async () => [opt('main', 3840, 1080), opt('sub', 1536, 576)], getVideoStream: async (o: any) => { asked.push(o); return {}; } };
    const fsx = await startFrames({ bySize: true, camera: camera as any, intervalMs: 500, prebufferMs: 8000, maxWidth: 960, maxWidthWide: 1280, quality: 85, console: { log: (m: string) => logs.push(m), warn() { } } as any, onFrame: () => { } });
    await sleep(300);
    assert.equal(asked[0].id, 'sub');
    assert.ok(!('destination' in asked[0]));
    assert.match(args().join(' '), /scale='min\(1280,iw\)':-2/);
    assert.ok(logs.some(l => /stream options: main 3840x1080 h264 1000k, sub 1536x576/.test(l)), logs.join('|'));
    assert.ok(logs.some(l => /analysis stream sub 1536x576, posting/.test(l)), logs.join('|'));
    fsx.stop();
    await fsx.exited;
    asked.length = 0;
    const cam2 = { getVideoStreamOptions: async () => [opt('sub', 1280, 720)], getVideoStream: async (o: any) => { asked.push(o); return {}; } };
    const f2 = await startFrames({ bySize: true, camera: cam2 as any, intervalMs: 500, prebufferMs: 0, maxWidth: 960, maxWidthWide: 1280, quality: 85, console, onFrame: () => { } });
    await sleep(300);
    assert.match(args().join(' '), /scale='min\(960,iw\)':-2/);
    f2.stop();
    await f2.exited;
});

test('default: the camera default stream (no destination, no id) with prebuffer; low-resolution on request; wide cap still from the options; RTMP never chosen by size', async () => {
    setup();
    const { startFrames } = loadFrames();
    const asked: any[] = [];
    const logs: string[] = [];
    const camera = { getVideoStreamOptions: async () => [opt('main', 3840, 1080), opt('sub', 1536, 576)], getVideoStream: async (o: any) => { asked.push(o); return {}; } };
    const f = await startFrames({ camera: camera as any, intervalMs: 500, prebufferMs: 8000, maxWidth: 960, maxWidthWide: 1280, quality: 85, console: { log: (m: string) => logs.push(m), warn() { } } as any, onFrame: () => { } });
    await sleep(300);
    assert.ok(!('destination' in asked[0]), 'attach to the already open default rebroadcast');
    assert.equal(asked[0].prebuffer, 8000);
    assert.ok(!('id' in asked[0]));
    assert.match(args().join(' '), /scale='min\(1280,iw\)':-2/);
    assert.ok(logs.some(l => /stream options: main 3840x1080/.test(l)));
    f.stop(); await f.exited;
    asked.length = 0;
    const lr = await startFrames({ destination: 'low-resolution', camera: camera as any, intervalMs: 500, prebufferMs: 8000, maxWidth: 960, quality: 85, console, onFrame: () => { } });
    await sleep(300);
    assert.equal(asked[0].destination, 'low-resolution');
    lr.stop(); await lr.exited;
    asked.length = 0;
    const rtmp = { getVideoStreamOptions: async () => [opt('main.bcs', 2560, 1920, 'rtmp'), opt('ext.bcs', 896, 672, 'rtmp')], getVideoStream: async (o: any) => { asked.push(o); return {}; } };
    const g = await startFrames({ bySize: true, camera: rtmp as any, intervalMs: 500, prebufferMs: 8000, maxWidth: 960, quality: 85, console, onFrame: () => { } });
    await sleep(300);
    assert.ok(!('id' in asked[0]) && !('destination' in asked[0]), 'RTMP ids are never requested');
    g.stop(); await g.exited;
});

test('a codec-probe failure (183, no jpeg) is reported as such; the long-probe fallback uses the destination and replaces the probe options before -i', async () => {
    setup(async () => ({ url: 'rtsp://x/live', inputArguments: ['-analyzeduration', '0', '-probesize', '512', '-rtsp_transport', 'tcp', '-i', 'rtsp://x/live'], ffmpegPath: fake }));
    const { startFrames, PROBE_FAILED_PREFIX } = loadFrames();
    process.env.FAKE_MODE = 'probe';
    try {
        const camera = { getVideoStreamOptions: async () => [opt('sub', 1280, 720)], getVideoStream: async () => ({}) };
        const f = await startFrames({ bySize: true, camera: camera as any, intervalMs: 500, prebufferMs: 0, maxWidth: 960, quality: 85, console: { log() { }, warn() { } } as any, onFrame: () => { } });
        assert.match(String(await f.exited), new RegExp('^' + PROBE_FAILED_PREFIX));
    } finally { delete process.env.FAKE_MODE; }
    const asked: any[] = [];
    const cam = { getVideoStreamOptions: async () => [opt('sub', 1280, 720)], getVideoStream: async (o: any) => { asked.push(o); return {}; } };
    const g = await startFrames({ longProbe: true, bySize: true, destination: 'low-resolution', camera: cam as any, intervalMs: 500, prebufferMs: 0, maxWidth: 960, quality: 85, console: { log() { }, warn() { } } as any, onFrame: () => { } });
    await sleep(300);
    assert.ok(!('destination' in asked[0]) && !('id' in asked[0]), 'the fallback is the default stream');
    const a = args();
    assert.deepEqual(a.slice(a.indexOf('-i') - 4, a.indexOf('-i')), ['-analyzeduration', '1000000', '-probesize', '500000']);
    assert.equal(a.filter(x => x === '-probesize').length, 1);
    g.stop(); await g.exited;
});

test('Stream choices: labelled name WxH, open (prebuffered) ones marked and first, then by resolution', () => {
    const { streamLabels } = loadFrames();
    const l = streamLabels([
        { id: 'main', name: 'RTSP main', video: { width: 2560, height: 1920 }, prebuffer: 10000 },
        { id: 'ext', name: 'RTMP ext.bcs', video: { width: 896, height: 672 } },
        { id: 'sub', name: 'RTMP sub.bcs', video: { width: 640, height: 480 } },
        { id: 'x' }, { name: 'no id' },
    ]);
    assert.deepEqual(l.map(x => x.label), ['RTSP main 2560x1920 (open, prebuffered)', 'RTMP sub.bcs 640x480', 'RTMP ext.bcs 896x672', 'x']);
    assert.deepEqual(l.map(x => x.open), [true, false, false, false]);
});

test('Stream: default resolves to the first open stream, else no id; an explicit id is requested; an unknown one uses the default', async () => {
    setup();
    const { startFrames } = loadFrames();
    const run = async (opts: any[], extra: any) => {
        const asked: any[] = []; const seen: any[] = [];
        const camera = { getVideoStreamOptions: async () => opts, getVideoStream: async (o: any) => { asked.push(o); return {}; } };
        const f = await startFrames({ camera: camera as any, intervalMs: 500, prebufferMs: 8000, maxWidth: 960, quality: 85, console: { log() { }, warn() { } } as any, onFrame: () => { }, onStream: s => seen.push(s), ...extra });
        await sleep(200); f.stop(); await f.exited;
        return { a: asked[0], s: seen[0] };
    };
    const opts = [{ id: 'sub', name: 'sub', video: { width: 640, height: 480 } }, { id: 'main', name: 'main', video: { width: 2560, height: 1920 }, prebuffer: 8000 }];
    let r = await run(opts, {});
    assert.equal(r.a.id, 'main', 'the open stream'); assert.equal(r.a.prebuffer, 8000);
    r = await run([opts[0]], {});
    assert.ok(!('id' in r.a) && !('destination' in r.a), 'nothing open: camera default');
    r = await run(opts, { streamId: 'sub' });
    assert.equal(r.a.id, 'sub'); assert.equal(r.s.explicit, true); assert.equal(r.s.name, 'sub 640x480');
    r = await run([opts[0]], { streamId: 'gone' });
    assert.ok(!('id' in r.a)); assert.equal(r.s.unknown, true);
});

test('VAAPI env: LIBVA_DRIVER_NAME set only for a hardware decode with a driver; the failure matcher covers the decoder-side errors', async () => {
    setup();
    const { startFrames, HW_RUNTIME_RE } = loadFrames();
    const env = () => fs.readFileSync(path.join(dir, 'env.txt'), 'utf8');
    const run = async (o: any) => { const f = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live', ...o, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console: { log() { }, warn() { } } as any, onFrame: () => { } }); await sleep(300); f.stop(); await f.exited; };
    await run({ hwaccel: true, vaDriver: 'i965' }); assert.equal(env(), 'i965');
    await run({ hwaccel: false, vaDriver: 'i965' }); assert.equal(env(), '');
    await run({ hwaccel: true }); assert.equal(env(), '');
    for (const t of ['Failed to transfer data to output frame: -5', '[hwaccel] error', 'va_sync failed', 'Failed to sync surface 0x3', 'AVHWFramesContext init failed', 'vaapi: unsupported'])
        assert.match(t, HW_RUNTIME_RE, t);
    assert.doesNotMatch('Connection refused', HW_RUNTIME_RE);
});

test('hardware decode: an exit with code 0 before the first JPEG is a hardware failure (never a silent restart on the GPU)', async () => {
    setup();
    const { startFrames, HW_FAILED_PREFIX, HW_RUNTIME_RE } = loadFrames();
    assert.match('Failed to read image from surface 0x4: 20 (the requested function is not implemented)', HW_RUNTIME_RE);
    process.env.FAKE_MODE = 'hwzero';
    try {
        const f = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live', hwaccel: true, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console: { log() { }, warn() { } } as any, onFrame: () => { } });
        assert.match(String(await f.exited), new RegExp('^' + HW_FAILED_PREFIX + ': ffmpeg exited 0'));
        const g = await startFrames({ camera: {} as any, inputUrl: 'rtsp://cam/live', hwaccel: false, intervalMs: 1000, prebufferMs: 0, maxWidth: 640, quality: 85, console: { log() { }, warn() { } } as any, onFrame: () => { } });
        await sleep(200); g.stop();
        assert.equal(await g.exited, undefined, 'software start is unaffected');
    } finally { delete process.env.FAKE_MODE; }
});

// ---- frames: the runtime burst switch ----

test('frames: the burst filter keeps idle cost (every Nth frame before scale/encode) and switches through a stdin command', () => {
    const F = loadFrames();
    assert.equal(F.burstStep(2, 6), 3);
    assert.equal(F.burstStep(2, undefined), 1);
    assert.equal(F.burstStep(4, 3), 1);
    assert.equal(F.filterChain(false, 2, 960, 6), "fps=6.0000,framestep@burst=step=3,scale='min(960,iw)':-2");
    assert.equal(F.filterChain(false, 2, 960), "fps=2.0000,scale='min(960,iw)':-2");
    assert.match(F.grayGraph(false, 2, 960, 128, 72, 6), /^fps=6\.0000,framestep@burst=step=3,scale=.*split=2\[v1\]\[b\];\[b\]fps=1,/);
    assert.equal(F.burstCommand(true), 'cframestep@burst -1 enable 0\n');
    assert.equal(F.burstCommand(false), 'cframestep@burst -1 enable 1\n');
});
