// Small pure helpers: clock, secrets masking, id and URL validation.

const T0 = Date.now() - performance.now();

/** Monotonic clock in unix-ms scale: immune to wall-clock steps (NTP, manual changes). */
export function monoNow(): number {
    return Math.round(T0 + performance.now());
}

/** Remove credentials and query strings (tokens) from text that may contain stream URLs. */
export function maskSecrets(text: string): string {
    return text
        .replace(/([a-z][a-z0-9+.-]*:\/\/)[^\s/@'"]*@/gi, '$1***@')
        .replace(/\?[^\s'"]*/g, '?***');
}

/** maskSecrets, and every URL reduced to scheme://host:port (no path, credentials or query): safe for any log line or status text. */
export function maskUrls(text: string): string {
    return maskSecrets(text).replace(/([a-z][a-z0-9+.-]*:\/\/)(?:[^/\s]*@)?([^/\s?]*)\/[^\s'"]*/gi, '$1$2');
}

/** Accept only rtsp/rtsps stream overrides (ffmpeg can otherwise read files and many protocols). */
export function validStreamUrl(url: string): boolean {
    return /^rtsps?:\/\/[^\s]+$/i.test(url);
}

export const FFMPEG_PROTOCOL_WHITELIST = 'rtsp,rtsps,rtp,udp,tcp,tls,crypto';

/** A camera id usable in a URL path segment of the service: never '.', '..', slashes or spaces. */
export function validCameraId(id: string): boolean {
    return /^[A-Za-z0-9._-]{1,64}$/.test(id) && id !== '.' && id !== '..';
}

type Box4 = readonly number[];
const FRAME_BORDER = 0.01;
/**
 * Two boxes of one object at different places? The verifier's geom.moved (0.3.60), boxes normalised [x0, y0, x1, y1]: compared only
 * where both are really drawn (a side at the frame border is cut to the inner of the two), it moved when two of its sides travelled
 * more than `limit`, or, cut by the border, one inner side did. A single side alone is the detector redrawing what it sees (laundry
 * blown wider on a drying rack, a post in front): nothing moved.
 */
export function boxMoved(a: Box4, b: Box4, limit: number): boolean {
    const p = [...a], q = [...b];
    for (const k of [0, 1]) if (p[k] <= FRAME_BORDER || q[k] <= FRAME_BORDER) p[k] = q[k] = Math.max(p[k], q[k]);
    for (const k of [2, 3]) if (p[k] >= 1 - FRAME_BORDER || q[k] >= 1 - FRAME_BORDER) p[k] = q[k] = Math.min(p[k], q[k]);
    const cut = [0, 1].some(k => Math.min(a[k], b[k]) <= FRAME_BORDER) || [2, 3].some(k => Math.max(a[k], b[k]) >= 1 - FRAME_BORDER);
    const sides = [0, 1, 2, 3].filter(k => Math.abs(p[k] - q[k]) > limit).length;
    return sides >= (cut ? 1 : 2);
}
