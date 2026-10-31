// 'Last detection' picture: the frame of a confirmed detection with its boxes drawn, for the Scrypted MQTT add-on.
// One short ffmpeg (the one Scrypted ships, drawbox filter) per confirmed episode, never on the frame path: the
// frame is already a JPEG the service analysed, so no stream is opened and nothing is decoded in Node.

import child_process from 'child_process';
import os from 'os';
import { jpegDimensions } from './jpeg';

/** The contract's limit for an image topic (docs/contract.md of scrypted-mqtt-ha). */
export const MAX_IMAGE_BYTES = 1024 * 1024;
/** wider frames (camera snapshots) are scaled down to this: plenty for a dashboard card, small on the broker. */
export const ANNOTATE_MAX_WIDTH = 1280;
const ANNOTATE_TIMEOUT_MS = 10_000;
/** JPEG quality (-q:v, 2 best .. 31 worst): the default, and the retry for a picture that came out too large */
export const ANNOTATE_QUALITY = 5;
export const ANNOTATE_QUALITY_SMALLER = 12;
/** the failure reason of a picture over MAX_IMAGE_BYTES (worth one retry at a lower quality) */
export const TOO_LARGE = 'too large';
const STDERR_TAIL = 200;
const MAX_BOXES = 20;
/** colours of the status panel's boxes */
const COLOR = { person: '0xe5484d', vehicle: '0x3b82f6', animal: '0x2f9e5b' } as const;

export interface DrawBox { x0: number; y0: number; x1: number; y1: number; group: keyof typeof COLOR }

/** the picture, or why there is none: 'timeout', TOO_LARGE, or ffmpeg's exit code and the end of its stderr */
export type Annotated = { jpeg: Buffer; error?: undefined } | { jpeg?: undefined; error: string };

export function isJpeg(b: Buffer): boolean {
    return b.length >= 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff;
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0));

/** ffmpeg filter for a w x h frame: scaled to at most ANNOTATE_MAX_WIDTH (even sizes), then one drawbox per box in pixels. */
export function annotateFilter(boxes: DrawBox[], w: number, h: number): string {
    const ow = Math.min(w, ANNOTATE_MAX_WIDTH);
    const oh = ow === w ? h : Math.max(2, Math.round(h * ow / w / 2) * 2);
    const t = Math.max(2, Math.round(ow / 320));
    const parts = ow === w ? [] : [`scale=${ow}:${oh}`];
    for (const b of boxes.slice(0, MAX_BOXES)) {
        const x0 = Math.round(clamp01(Math.min(b.x0, b.x1)) * ow), x1 = Math.round(clamp01(Math.max(b.x0, b.x1)) * ow);
        const y0 = Math.round(clamp01(Math.min(b.y0, b.y1)) * oh), y1 = Math.round(clamp01(Math.max(b.y0, b.y1)) * oh);
        parts.push(`drawbox=x=${x0}:y=${y0}:w=${Math.max(1, x1 - x0)}:h=${Math.max(1, y1 - y0)}:color=${COLOR[b.group] ?? COLOR.person}:t=${t}`);
    }
    return parts.join(',') || 'null';
}

/**
 * The frame with its boxes drawn, as a JPEG of at most MAX_IMAGE_BYTES, or why not (ffmpeg failed, was too slow, or the
 * result was too large). One thread at low priority: it shares the box with recording.
 */
export function annotateJpeg(ffmpegPath: string, jpeg: Buffer, boxes: DrawBox[], timeoutMs = ANNOTATE_TIMEOUT_MS, quality = ANNOTATE_QUALITY): Promise<Annotated> {
    const dim = jpegDimensions(jpeg);
    if (!dim || !isJpeg(jpeg)) return Promise.resolve({ error: 'not a JPEG frame' });
    const args = ['-hide_banner', '-loglevel', 'error', '-threads', '1',
        '-f', 'image2pipe', '-c:v', 'mjpeg', '-i', 'pipe:0',
        '-vf', annotateFilter(boxes, dim.w, dim.h), '-frames:v', '1',
        '-c:v', 'mjpeg', '-q:v', String(quality), '-f', 'image2pipe', 'pipe:1'];
    return new Promise(resolve => {
        let cp: child_process.ChildProcessWithoutNullStreams;
        try { cp = child_process.spawn(ffmpegPath, args, { stdio: 'pipe' }); }
        catch (e) { resolve({ error: (e as Error).message }); return; }
        try { if (cp.pid) os.setPriority(cp.pid, 10); } catch { /* not permitted: normal priority */ }
        const chunks: Buffer[] = [];
        let size = 0, done = false, stderr = Buffer.alloc(0);
        const finish = (r: Annotated) => {
            if (done) return;
            done = true;
            clearTimeout(timer);
            if (cp.exitCode === null) cp.kill('SIGKILL');
            resolve(r);
        };
        const timer = setTimeout(() => finish({ error: 'timeout' }), timeoutMs);
        cp.stdout.on('data', (c: Buffer) => {
            size += c.length;
            if (size > MAX_IMAGE_BYTES) finish({ error: TOO_LARGE });
            else chunks.push(c);
        });
        // only the end is kept: ffmpeg's last lines say why it stopped
        cp.stderr.on('data', (c: Buffer) => { stderr = Buffer.concat([stderr, c]).subarray(-STDERR_TAIL); });
        cp.on('error', e => finish({ error: e.message }));
        cp.on('close', (code, signal) => {
            const out = Buffer.concat(chunks);
            if (code !== 0) finish({ error: `exit ${code ?? signal}${stderr.length ? `: ${stderr.toString('utf8').trim()}` : ''}` });
            else finish(isJpeg(out) ? { jpeg: out } : { error: 'no JPEG out' });
        });
        cp.stdin.on('error', () => { /* ffmpeg gone: close reports it */ });
        cp.stdin.end(jpeg);
    });
}
