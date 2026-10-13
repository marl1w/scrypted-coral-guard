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
