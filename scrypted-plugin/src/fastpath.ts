// Snapshot frames and the camera-AI fast path: small helpers kept apart from the mixin so they are testable alone.

/** a snapshot slower than this (or failing) counts as a failure: the paced path falls back to the stream. */
export const SNAP_TIMEOUT_MS = 1500;
/** consecutive snapshot failures before a camera falls back to its own continuous decode (until the setting changes or the plugin restarts). */
export const SNAP_MAX_FAILS = 3;

export interface SnapshotCamera {
    takePicture?: (options?: any) => Promise<any>;
}
export interface MediaConverter {
    convertMediaObjectToBuffer(mo: any, mime: string): Promise<Buffer>;
}

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
    let timer: NodeJS.Timeout | undefined;
    const t = new Promise<never>((_, rej) => { timer = setTimeout(() => rej(new Error(`${what} timed out after ${ms} ms`)), ms); });
    return Promise.race([p, t]).finally(() => clearTimeout(timer));
}

/** One JPEG from the camera's takePicture(), within timeoutMs in total; rejects on anything that is not a JPEG. */
export async function fetchSnapshot(camera: SnapshotCamera, media: MediaConverter, reason: 'event' | 'periodic', timeoutMs = SNAP_TIMEOUT_MS): Promise<Buffer> {
    if (typeof camera.takePicture !== 'function')
        throw new Error('camera has no snapshots');
    const work = (async () => {
        const mo = await camera.takePicture!({ reason });
        const buf = await media.convertMediaObjectToBuffer(mo, 'image/jpeg');
        if (!Buffer.isBuffer(buf) || buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8)
            throw new Error('snapshot is not a JPEG');
        return buf;
    })();
    return withTimeout(work, timeoutMs, 'snapshot');
}

/**
 * Timing of one camera-AI claim through the episode: claim (camera push) -> scan start -> camera motion edge -> verdict.
 * All times monotonic ms. One line per claim, written at the verdict (or when the episode ends without one).
 */
export class FastPathTimer {
    private claimAt: number | undefined;
    private cls = '';
    private scanAt: number | undefined;
    private edgeAt: number | undefined;
    private firstPostAt: number | undefined;

    get armed() { return this.claimAt !== undefined; }

    claim(className: string, now: number) {
        if (this.claimAt !== undefined)
            return; // the first claim of the episode is the one that started it
        this.claimAt = now; this.cls = className;
        this.scanAt = this.edgeAt = this.firstPostAt = undefined;
    }
    scan(now: number) { if (this.claimAt !== undefined && this.scanAt === undefined) this.scanAt = now; }
    edge(now: number) { if (this.claimAt !== undefined && this.edgeAt === undefined) this.edgeAt = now; }
    posted(now: number) { if (this.claimAt !== undefined && this.firstPostAt === undefined) this.firstPostAt = now; }

    /** the verdict: returns the log line and disarms (undefined when not armed). */
    verdict(outcome: string, now: number): string | undefined {
        if (this.claimAt === undefined)
            return undefined;
        const rel = (t: number | undefined) => t === undefined ? '-' : `+${t - this.claimAt!} ms`;
        const line = `fast path: camera AI ${this.cls} -> ${outcome} ${rel(now)} (scan ${rel(this.scanAt)}, first frame ${rel(this.firstPostAt)}, camera motion ${rel(this.edgeAt)})`;
        this.claimAt = undefined;
        return line;
    }
    /** a claim that never led to a verdict (e.g. not verifiable, mode off) is forgotten after this long. */
    expire(now: number, ms = 30000) { if (this.claimAt !== undefined && now - this.claimAt > ms) this.claimAt = undefined; }
}
