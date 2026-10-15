// ffmpeg frame generator: one ffmpeg per active camera, low-res JPEG frames on stdout.
// Pattern follows plugins/objectdetector/src/ffmpeg-audiosensor.ts (getVideoStream ->
// FFmpegInput -> spawn getFFmpegPath()).

import sdk, { FFmpegInput, ScryptedMimeTypes, VideoCamera } from '@scrypted/sdk';
import child_process from 'child_process';
import fs from 'fs';

import { JpegSplitter, jpegDimensions, qualityToQscale } from './jpeg';
import { FFMPEG_PROTOCOL_WHITELIST, maskSecrets, maskUrls, monoNow, validStreamUrl } from './util';

const { mediaManager } = sdk;

export interface FrameStream {
    stop(): void;
    /** burst (fast confirm): JPEG frames at the burst rate instead of the interval; no-op when the stream was started without burstFps. */
    setBurst?(on: boolean): void;
    /** resolves when ffmpeg has exited (after stop() or on its own). */
    exited: Promise<string | undefined>;
}

export interface FrameOptions {
    camera: VideoCamera;
    /** optional override: an ffmpeg input URL (e.g. rtsp://...). Skips getVideoStream. */
    inputUrl?: string;
    intervalMs: number;
    /**
     * burst rate (fps) the JPEG branch can switch to at runtime (setBurst). The decoder output is sampled at this rate and a
     * `framestep@burst` filter keeps every Nth frame (N = burstFps x interval) BEFORE scaling and JPEG encode, so idle cost stays
     * the interval's; setBurst toggles that filter's timeline (`enable`) through ffmpeg's stdin command interface.
     */
    burstFps?: number;
    /** pre-roll in ms, 0 = none. Passed as RequestMediaStreamOptions.prebuffer. */
    prebufferMs: number;
    /** width cap in px; never upscales. */
    maxWidth: number;
    /** width cap when the chosen stream is wide (aspect > 2); defaults to maxWidth. */
    maxWidthWide?: number;
    /** JPEG quality 1..100. */
    quality: number;
    console: Console;
    onFrame: (jpeg: Buffer, arrivedAt: number) => void;
    /** monotonic time of the raw trigger, for the once-per-start timing diagnostics (default: now). */
    triggerAt?: number;
    /** drop JPEG frames arriving closer than this to the previous one (harmless safety net: the fps filter already samples one per interval). */
    minGapMs?: number;
    /** try VAAPI hardware decode (decoder returns nv12 in system memory, CPU scale). The caller falls back on 'hwaccel failed'. */
    /** LIBVA_DRIVER_NAME for a hardware decode ('' = leave unset). */
    vaDriver?: string;
    hwaccel?: boolean;
    /** a GPU start already failed for this camera: this software start says so in its decode log line. */
    hwFailedBefore?: boolean;
    /** override of STREAM_RESOLVE_MS (tests). */
    resolveTimeoutMs?: number;
    /** opt-in: request the stream chosen by resolution (RTSP ids only) instead of the 'low-resolution' destination. */
    bySize?: boolean;
    /** per-camera 'Stream' setting: a stream id to request (undefined = default). Ignored by the long-probe fallback. */
    streamId?: string;
    /** told which stream is requested; unknown = the stored id is not among the camera's stream options (the default is used). */
    onStream?: (s: { name: string; explicit: boolean; unknown?: boolean }) => void;
    /** 'default' (no destination: the camera's default, already prebuffered stream) or 'low-resolution'. */
    destination?: 'default' | 'low-resolution';
    /** fallback after a codec-probe failure: the default stream with a longer probe. */
    longProbe?: boolean;
    /** second output: tiny grayscale frames (1 fps, area-scaled) for the pixel gate, raw bytes on fd 3. */
    gray?: { width: number; height: number; minGapMs?: number; onGray: (frame: Uint8Array, arrivedAt: number) => void };
}

const versionCache = new Map<string, Promise<{ text: string; passthrough: string[] }>>();

/**
 * ffmpeg version line (logged once per start). `passthrough` is the per-output option that keeps one output frame per
 * decoded frame instead of constant frame rate: unused now that frames are dense and sampled by the fps filter,
 * kept for sparse (keyframe-only) sources. `-fps_mode` exists from 5.1, `-vsync` is removed in 9.
 */
export function ffmpegVersion(ffmpegPath: string): Promise<{ text: string; passthrough: string[] }> {
    let p = versionCache.get(ffmpegPath);
    if (!p) {
        p = new Promise(resolve => {
            child_process.execFile(ffmpegPath, ['-version'], { timeout: 5000 }, (err, out) => {
                const text = String(out || '').split('\n')[0].trim() || 'unknown';
                const m = /version\s+n?(\d+)\.(\d+)/i.exec(text);
                const old = !!m && (Number(m[1]) < 5 || (Number(m[1]) === 5 && Number(m[2]) < 1));
                resolve({ text, passthrough: old ? ['-vsync', '0'] : ['-fps_mode', 'passthrough'] });
            });
        });
        versionCache.set(ffmpegPath, p);
    }
    return p;
}

/** a hung getVideoStream/convert must not leave the placeholder source forever: fail and let the caller retry with backoff. */
export const STREAM_RESOLVE_MS = 15000;

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const t = setTimeout(() => reject(new Error(`${what} timed out after ${ms} ms`)), ms);
        p.then(v => { clearTimeout(t); resolve(v); }, e => { clearTimeout(t); reject(e); });
    });
}

export interface StreamChoice { rtsp?: boolean; id?: string; name?: string; width: number; height: number; codec?: string; bitrate?: number }

const MAX_ANALYSIS_PIXELS = 2.2e6;

/**
 * Pick the analysis stream by resolution instead of by destination. Wide pictures (aspect > 2, e.g. dual lens) need
 * the smallest stream at 1280 px wide or more (the service splits them into two tiles); others the smallest at 896 px
 * or more. If none qualifies, the largest one below 1920 px wide. Streams above ~2 Mpx are never chosen.
 */
export function chooseStream(options: any[] | undefined, rtspOnly = false): { choice?: StreamChoice; all: StreamChoice[]; wide: boolean } {
    const all: StreamChoice[] = (options || [])
        .map((o: any) => ({ id: o?.id, name: o?.name, rtsp: o?.container === 'rtsp' || /^rtsp/i.test(String(o?.name || '')), width: Number(o?.video?.width), height: Number(o?.video?.height), codec: o?.video?.codec, bitrate: o?.video?.bitrate }))
        .filter(s => Number.isFinite(s.width) && Number.isFinite(s.height) && s.width > 0 && s.height > 0);
    if (!all.length)
        return { all, wide: false };
    const widest = all.reduce((a, b) => (b.width * b.height > a.width * a.height ? b : a));
    const wide = widest.width / widest.height > 2;
    const usable = all.filter(s => s.width * s.height <= MAX_ANALYSIS_PIXELS && (!rtspOnly || s.rtsp));
    const need = wide ? 1280 : 896;
    const area = (s: StreamChoice) => s.width * s.height;
    const enough = usable.filter(s => s.width >= need).sort((a, b) => area(a) - area(b));
    const choice = enough[0] ?? usable.filter(s => s.width < 1920).sort((a, b) => area(b) - area(a))[0];
    return { choice, all, wide };
}

export interface StreamLabel { id: string; label: string; open: boolean }

/**
 * Stream setting choices from getVideoStreamOptions: '<name or id> WxH', suffixed ' (open, prebuffered)' when the option
 * reports prebuffering (MediaStreamOptions.prebuffer > 0, set by the Rebroadcast plugin). Open streams first, then by resolution ascending.
 */
export function streamLabels(options: any[] | undefined): StreamLabel[] {
    return (options || []).filter((o: any) => o?.id).map((o: any) => {
        const w = Number(o.video?.width), h = Number(o.video?.height);
        const open = Number(o.prebuffer) > 0;
        return { id: String(o.id), open, area: w > 0 && h > 0 ? w * h : Infinity, label: `${o.name || o.id}${w > 0 && h > 0 ? ` ${w}x${h}` : ''}${open ? ' (open, prebuffered)' : ''}` };
    }).sort((a, b) => (a.open === b.open ? a.area - b.area : a.open ? -1 : 1)).map(({ id, label, open }) => ({ id, label, open }));
}

export const VAAPI_DEVICE = '/dev/dri/renderD128';
/** a slow start (camera key frame wait, VAAPI init) is not a hardware failure: give it this long. */
export const HW_FIRST_FRAME_MS = 20000;
export const HW_RUNTIME_RE = /vaapi|va_|hwaccel|hwdownload|transfer data|sync surface|read image from surface|not implemented|AVHWFramesContext/i;
export const HW_FAILED_PREFIX = 'hwaccel failed';
export const PROBE_FAILED_PREFIX = 'codec probe failed';
export const PROBE_FAILED_RE = /Could not find codec parameters/i;

/** Longer stream probe for the fallback: drop any probe options of the input arguments, put ours right before -i. */
export function withLongProbe(args: string[]): string[] {
    const out: string[] = [];
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '-analyzeduration' || args[i] === '-probesize') { i++; continue; }
        out.push(args[i]);
    }
    const at = out.indexOf('-i');
    const probe = ['-analyzeduration', '1000000', '-probesize', '500000'];
    return at < 0 ? [...probe, ...out] : [...out.slice(0, at), ...probe, ...out.slice(at)];
}

/** True when the VAAPI render node exists and is readable/writable by this process. */
export function vaapiAvailable(device = VAAPI_DEVICE): boolean {
    try { fs.accessSync(device, fs.constants.R_OK | fs.constants.W_OK); return true; } catch { return false; }
}

/** Global input options for the next -i: decoder threads and optional VAAPI decode. */
export function decodeArgs(hw: boolean): string[] {
    return [
        '-threads', '1', // decoder: one thread per stream (many processes should not oversubscribe the cores)
        ...(hw ? ['-hwaccel', 'vaapi', '-hwaccel_device', VAAPI_DEVICE, '-hwaccel_output_format', 'nv12'] : []),
    ];
}

/** Last `n` lines of ffmpeg stderr, credentials and query masked, every URL reduced to scheme://host:port. */
export function stderrTail(text: string, n = 15): string {
    return maskUrls(text).split(/\r?\n/).map(l => l.trimEnd()).filter(Boolean).slice(-n).join(' | ');
}

/** frames the burst filter keeps out of burstFps when not bursting (1 = no burst filter). */
export function burstStep(fps: number, burstFps?: number): number {
    return burstFps && burstFps > fps ? Math.max(1, Math.round(burstFps / fps)) : 1;
}

/** name of the runtime-switchable filter (ffmpeg stdin command target). */
export const BURST_FILTER = 'framestep@burst';

/** ffmpeg stdin command line that turns the burst on (framestep disabled: every burstFps frame passes) or off. */
export function burstCommand(on: boolean): string {
    return `c${BURST_FILTER} -1 enable ${on ? 0 : 1}\n`;
}

function chainHead(hw: boolean, fps: number, burstFps?: number): string {
    // hardware decode returns system-memory frames (nv12): no hwdownload step
    const step = burstStep(fps, burstFps);
    return step > 1 ? `fps=${(fps * step).toFixed(4)},${BURST_FILTER}=step=${step},` : `fps=${fps.toFixed(4)},`;
}

/** Filter graph with two outputs: [v1] the JPEG frames, [v2] the tiny gray frames for the pixel gate (1 fps, area scaling). */
export function grayGraph(hw: boolean, fps: number, maxWidth: number, gw: number, gh: number, burstFps?: number): string {
    // fps first (JPEG branch rate), one full-size scale, then the gray branch is sampled at exactly 1 fps from the reduced picture
    return `${chainHead(hw, fps, burstFps)}scale='min(${maxWidth},iw)':-2,split=2[v1][b];[b]fps=1,scale=${gw}:${gh}:flags=area,format=gray[v2]`;
}

/** Video filter chain: optional hwdownload (never scale_vaapi: not available without VPP), fps, CPU scale. */
export function filterChain(hw: boolean, fps: number, maxWidth: number, burstFps?: number): string {
    // fps FIRST: every frame is decoded, but only one per interval is downloaded, scaled and JPEG-encoded.
    return `${chainHead(hw, fps, burstFps)}scale='min(${maxWidth},iw)':-2`;
}

/** scheme://host:port of a stream URL, credentials and query dropped (safe to log). */
export function describeUrl(url: string): string {
    const m = /^([a-z][a-z0-9+.-]*):\/\/(?:[^\s/@]*@)?([^\s/?#]*)/i.exec(url);
    return m ? `${m[1]}://${m[2]}` : '(unparsable url)';
}

/**
 * Scrypted's FFmpegInput.inputArguments already ends with `-i <url>` (the prebuffer rebroadcast
 * input). Its url is served by a single-client listener, so a second `-i url` is a second
 * connection that is refused. Append `-i url` only when the arguments do not carry one.
 */
export function buildInputArgs(inputArguments: string[] | undefined, url: string): string[] {
    const a = inputArguments || [];
    return a.includes('-i') ? [...a] : [...a, '-i', url];
}

export async function startFrames(o: FrameOptions): Promise<FrameStream> {
    const t0 = o.triggerAt ?? monoNow();
    let source = 'override';
    let args: string[];
    let streamWide = false;
    let streamDesc = '';
    let ffmpegPath: string | undefined;
    let env: NodeJS.ProcessEnv = process.env;
    if (o.inputUrl) {
        if (!validStreamUrl(o.inputUrl))
            throw new Error('stream override must be an rtsp:// or rtsps:// URL');
        // ffmpeg reads files and many protocols by default: restrict what an override may open.
        args = ['-protocol_whitelist', FFMPEG_PROTOCOL_WHITELIST, '-rtsp_transport', 'tcp', ...decodeArgs(!!o.hwaccel), '-i', o.inputUrl];
    }
    else {
        let pick: ReturnType<typeof chooseStream> = { all: [], wide: false };
        let rawOpts: any[] = [];
        try {
            if (typeof o.camera.getVideoStreamOptions === 'function') {
                rawOpts = (await withTimeout(o.camera.getVideoStreamOptions(), o.resolveTimeoutMs ?? STREAM_RESOLVE_MS, 'getVideoStreamOptions')) || [];
                pick = chooseStream(rawOpts, true);
            }
        }
        catch (e) { /* no options: fall back to the low-resolution destination */ }
        streamWide = pick.wide;
        if (pick.all.length)
            o.console.log(`frame source: stream options: ${pick.all.map(s => `${s.name || s.id || '?'} ${s.width}x${s.height}${s.codec ? ' ' + s.codec : ''}${s.bitrate ? ' ' + Math.round(s.bitrate / 1000) + 'k' : ''}`).join(', ')}`);
        let explicit = o.streamId && !o.longProbe ? rawOpts.find((x: any) => x?.id === o.streamId) : undefined;
        const unknown = !!o.streamId && !o.longProbe && !explicit;
        if (!explicit && !o.longProbe && !o.bySize && (o.destination ?? 'default') === 'default') {
            // Default: the first stream that is already open (prebuffered), else the camera default (no id)
            const open = streamLabels(rawOpts).find(l => l.open);
            if (open)
                explicit = rawOpts.find((x: any) => x?.id === open.id);
        }
        const useId = !explicit && !!o.bySize && !o.longProbe && !!pick.choice?.id;
        const dest = o.longProbe ? 'default' : (o.destination ?? 'default');
        if (explicit)
            streamDesc = `${explicit.name || explicit.id}${explicit.video?.width ? ` ${explicit.video.width}x${explicit.video.height}` : ''}`;
        else if (useId && pick.choice)
            streamDesc = `${pick.choice.name || pick.choice.id || '?'} ${pick.choice.width}x${pick.choice.height}`;
        o.onStream?.({ name: streamDesc || 'default', explicit: !!o.streamId && !unknown && !o.longProbe, unknown });
        const mo = await withTimeout(o.camera.getVideoStream({
            ...(explicit ? { id: explicit.id } : useId ? { id: pick.choice!.id } : dest === 'low-resolution' ? { destination: 'low-resolution' } : {}),
            video: {},
            // ask the rebroadcast prebuffer for the last N ms: the stream then starts with a burst
            // of past frames. Only effective if the stream used for 'low-resolution' is prebuffered.
            ...(o.prebufferMs > 0 ? { prebuffer: o.prebufferMs } : {}),
        }), o.resolveTimeoutMs ?? STREAM_RESOLVE_MS, 'getVideoStream');
        const input = await withTimeout(mediaManager.convertMediaObjectToJSON<FFmpegInput>(mo, ScryptedMimeTypes.FFmpegInput), o.resolveTimeoutMs ?? STREAM_RESOLVE_MS, 'stream conversion');
        if (!input.url)
            throw new Error('getVideoStream returned no url');
        ffmpegPath = input.ffmpegPath;
        if (input.env)
            env = { ...process.env, ...input.env };
        const inArgs = buildInputArgs(input.inputArguments, input.url);
        args = [...decodeArgs(!!o.hwaccel), ...(o.longProbe ? withLongProbe(inArgs) : inArgs)];
        const mso: any = input.mediaStreamOptions || {};
        if (!streamDesc)
            streamDesc = `${mso.name || mso.id || dest}${mso.video?.width && mso.video?.height ? ` ${mso.video.width}x${mso.video.height}` : ''}`;
        o.console.log(`frame source: getVideoStream asked ${explicit || useId ? 'stream ' + streamDesc : dest === 'low-resolution' ? 'destination low-resolution' : 'default stream'}${o.longProbe ? ' (long probe fallback)' : ''}, prebuffer ${o.prebufferMs} ms; stream reports prebuffer ${mso.prebuffer ?? 'n/a'} ms / ${mso.prebufferBytes ?? 'n/a'} bytes`);
        source = `${describeUrl(input.url)} (query masked), ${(input.inputArguments || []).includes('-i') ? 'url carried in inputArguments' : 'url appended'}`;
    }
    const tResolved = monoNow();
    ffmpegPath = ffmpegPath || await mediaManager.getFFmpegPath();

    const fps = 1000 / o.intervalMs;
    const ver = await ffmpegVersion(ffmpegPath);
    const width = streamWide ? (o.maxWidthWide ?? o.maxWidth) : o.maxWidth;
    const jpegOut = ['-c:v', 'mjpeg', '-q:v', String(qualityToQscale(o.quality)), '-f', 'image2pipe', 'pipe:1'];
    args = [
        '-hide_banner', '-loglevel', 'warning',
        ...args,
        '-an', '-sn',
        ...(o.gray
            ? ['-filter_complex', grayGraph(!!o.hwaccel, fps, width, o.gray.width, o.gray.height, o.burstFps),
                '-map', '[v1]', ...jpegOut,
                '-map', '[v2]', '-c:v', 'rawvideo', '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:3']
            : ['-vf', filterChain(!!o.hwaccel, fps, width, o.burstFps), ...jpegOut]),
    ];

    if (o.hwaccel && o.vaDriver)
        env = { ...env, LIBVA_DRIVER_NAME: o.vaDriver };
    const canBurst = burstStep(fps, o.burstFps) > 1;
    // stdin is a pipe only for the burst commands (ffmpeg reads its interactive commands from it); otherwise ignored as before
    const cp = child_process.spawn(ffmpegPath, args, { env, stdio: [canBurst ? 'pipe' : 'ignore', 'pipe', 'pipe', ...(o.gray ? ['pipe' as const] : [])] }) as unknown as child_process.ChildProcessWithoutNullStreams;
    cp.stdin?.on?.('error', () => { /* ffmpeg gone: a burst command is best effort */ });
    let burstOn = false;
    if (o.gray) {
        const { width, height, onGray } = o.gray;
        const size = width * height;
        let grayLogged = false;
        let lastGray = 0;
        let acc: Buffer = Buffer.alloc(0);
        (cp.stdio[3] as NodeJS.ReadableStream | null)?.on('data', (d: Buffer) => {
            acc = acc.length ? Buffer.concat([acc, d]) : d;
            while (acc.length >= size) {
                const f = new Uint8Array(acc.subarray(0, size)); // copy: the gate keeps no reference
                acc = acc.subarray(size);
                const gt = monoNow();
                if (o.gray?.minGapMs && gt - lastGray < o.gray.minGapMs)
                    continue;
                lastGray = gt;
                if (!grayLogged) {
                    grayLogged = true;
                    o.console.log(`frame source: first gray +${gt - t0} ms after start`);
                }
                onGray(f, gt);
            }
        });
    }
    const splitter = new JpegSplitter();
    let lastErr = '';
    let gotFrame = false;
    let hwTimedOut = false;
    let sizeLogged = false;
    let stopping = false;
    let lastJpeg = 0;
    const tSpawn = monoNow();
    o.console.log(`frame source: ${ver.text}; ${source}; +${tResolved - t0} ms stream resolved, +${tSpawn - t0} ms ffmpeg spawned`);
    cp.stdout.on('data', (d: Buffer) => {
        const now = monoNow();
        if (!gotFrame) {
            gotFrame = true;
            o.console.log(`frame source: first jpeg +${now - t0} ms after start`);
            o.console.log(`frame source: decode ${o.hwaccel ? `GPU(vaapi${o.vaDriver ? `, driver ${o.vaDriver}` : ''}) verified` : o.hwFailedBefore ? 'CPU(requested GPU, not initialised)' : 'CPU'}`);
        }
        splitter.push(d, jpeg => {
            if (!sizeLogged) {
                sizeLogged = true;
                const dim = jpegDimensions(jpeg);
                o.console.log(`frame source: analysis stream ${streamDesc}, posting ${dim ? `${dim.w}x${dim.h}` : 'unknown size'}`);
            }
            if (o.minGapMs && now - lastJpeg < o.minGapMs)
                return;
            lastJpeg = now;
            o.onFrame(jpeg, now);
        });
    });
    cp.stderr.on('data', (d: Buffer) => {
        // keep only the tail; may contain the stream URL, so it is never logged verbatim except on exit.
        lastErr = (lastErr + d.toString()).slice(-6000);
    });
    let hwTimer: NodeJS.Timeout | undefined;
    if (o.hwaccel) {
        hwTimer = setTimeout(() => { if (!gotFrame) { hwTimedOut = true; stopping = true; try { cp.kill('SIGKILL'); } catch (e) { /* gone */ } } }, HW_FIRST_FRAME_MS);
    }
    const exited = new Promise<string | undefined>(resolve => {
        cp.on('error', e => { clearTimeout(hwTimer); resolve(maskUrls(String(e.message))); });
        cp.on('exit', (code, signal) => {
            clearTimeout(hwTimer);
            if (!stopping)
                o.console.warn(`frame source: ffmpeg ended unexpectedly (code ${code}, signal ${signal}) +${monoNow() - t0} ms after start, first jpeg: ${gotFrame ? 'yes' : 'no'}; stderr tail: ${stderrTail(lastErr) || '(empty)'}`);
            // hardware decode: ANY exit before the first JPEG (code 0 included) is a hardware failure, so the caller falls back to software
            if (o.hwaccel && (hwTimedOut || (!stopping && (!gotFrame || (code && HW_RUNTIME_RE.test(lastErr)))))) {
                resolve(`${HW_FAILED_PREFIX}: ${hwTimedOut ? 'no frame within ' + HW_FIRST_FRAME_MS + ' ms' : 'ffmpeg exited ' + code}; ${maskUrls(lastErr).trim().slice(-300)}`);
                return;
            }
            if (code && !gotFrame && (code === 183 || PROBE_FAILED_RE.test(lastErr))) {
                resolve(`${PROBE_FAILED_PREFIX}: ffmpeg exited ${code}; ${maskUrls(lastErr).trim().slice(-300)}`);
                return;
            }
            resolve(code ? `ffmpeg exited ${code}: ${maskUrls(lastErr).trim().slice(-400)}` : undefined);
        });
    });
    return {
        stop: () => { stopping = true; clearTimeout(hwTimer); try { cp.kill('SIGKILL'); } catch (e) { /* already gone */ } },
        exited,
        setBurst: (on: boolean) => {
            if (!canBurst || on === burstOn || stopping || !cp.stdin || cp.stdin.destroyed)
                return;
            burstOn = on;
            try { cp.stdin.write(burstCommand(on)); } catch (e) { /* best effort */ }
        },
    };
}
