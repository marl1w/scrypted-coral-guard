// Cheap pixel-change gate: decides which 1 fps keyframes are worth sending to the detector.
// Input: tiny 8-bit grayscale frames (default 128x72). Per push: 3x3 gaussian blur, abs diff vs the
// previous frame, threshold, 3x3 dilation, 8-connected components; opens on a big blob or a big changed fraction.
// A per-camera overlay mask (burned-in clock) is learned without labels during a warm-up.

export type GateReason = 'first' | 'blob' | 'fraction' | 'global' | 'closed' | 'learning';

export interface GateOptions {
    width: number;
    height: number;
    /** grey-level threshold on the blurred abs diff */
    threshold: number;
    /** open when the largest changed blob (pixels, after dilation) is larger than this */
    blobPixels: number;
    /** open when changed pixels / total pixels is larger than this */
    fraction: number;
    /** 'global' when more than this fraction of pixels changed by more than globalLevel */
    globalFraction: number;
    globalLevel: number;
    /** frames used to learn the overlay mask (0 disables learning) */
    warmupFrames: number;
    /** overlay candidates only in the outer fraction of rows (top and bottom bands) */
    bandFraction: number;
    /** a pixel must change in at least this share of the warm-up pairs to be masked */
    maskRatio: number;
}

export const DEFAULT_GATE_OPTIONS: GateOptions = {
    width: 128, height: 72, threshold: 12, blobPixels: 24, fraction: 0.016,
    globalFraction: 0.4, globalLevel: 24, warmupFrames: 30, bandFraction: 0.25, maskRatio: 0.8,
};

export interface GateResult { open: boolean; reason: GateReason; blob: number; fraction: number }

export class PixelGate {
    readonly opts: GateOptions;
    private readonly w: number;
    private readonly h: number;
    private readonly n: number;
    private prev: Uint8Array;
    private cur: Uint8Array;
    private readonly tmp: Uint16Array;
    private readonly diff: Uint8Array;
    private readonly dil: Uint8Array;
    private readonly seen: Uint8Array;
    private readonly stack: Int32Array;
    private readonly counts: Uint16Array;
    private readonly mask: Uint8Array;
    private hasPrev = false;
    private pairs = 0;
    private learned: boolean;
    private maskedCount = 0;

    constructor(opts: Partial<GateOptions> = {}) {
        this.opts = { ...DEFAULT_GATE_OPTIONS, ...opts };
        this.w = this.opts.width;
        this.h = this.opts.height;
        this.n = this.w * this.h;
        this.prev = new Uint8Array(this.n);
        this.cur = new Uint8Array(this.n);
        this.tmp = new Uint16Array(this.n);
        this.diff = new Uint8Array(this.n);
        this.dil = new Uint8Array(this.n);
        this.seen = new Uint8Array(this.n);
        this.stack = new Int32Array(this.n);
        this.counts = new Uint16Array(this.n);
        this.mask = new Uint8Array(this.n);
        this.learned = this.opts.warmupFrames <= 1;
    }

    /** number of masked (ignored) pixels; 0 until the overlay mask has been learned */
    get masked(): number { return this.maskedCount; }
    get learning(): boolean { return !this.learned; }

    /** stream restart: forget the previous frame (the next push opens as 'first'); the learned mask is kept */
    reset(): void { this.hasPrev = false; }

    /** discard the learned mask and learn again from the next frames */
    relearn(): void {
        this.hasPrev = false; this.pairs = 0; this.counts.fill(0); this.mask.fill(0);
        this.maskedCount = 0; this.learned = this.opts.warmupFrames <= 1;
    }

    push(gray: Uint8Array, _timeMs: number = 0): GateResult {
        if (gray.length < this.n) throw new Error(`gate: frame has ${gray.length} bytes, expected ${this.n}`);
        this.blur(gray, this.cur);
        if (!this.hasPrev) {
            this.swap(); this.hasPrev = true;
            return { open: true, reason: 'first', blob: 0, fraction: 0 };
        }
        const { w, h, n, prev, cur, diff, mask } = this;
        const T = this.opts.threshold, G = this.opts.globalLevel;
        let big = 0;
        for (let i = 0; i < n; i++) {
            const d = cur[i] > prev[i] ? cur[i] - prev[i] : prev[i] - cur[i];
            if (d > G) big++;
            diff[i] = d > T && !mask[i] ? 1 : 0;
        }
        const isGlobal = big > this.opts.globalFraction * n;
        if (!this.learned && !isGlobal) this.learnPair();
        this.swap();
        if (isGlobal) return { open: true, reason: 'global', blob: 0, fraction: big / n };
        if (!this.learned && this.pairs >= this.opts.warmupFrames - 1) {
            this.finishLearning();
            for (let i = 0; i < n; i++) if (mask[i]) diff[i] = 0;
        }

        // 3x3 dilation (separable would need a second buffer; direct is fine at this size)
        const dil = this.dil;
        dil.fill(0);
        let changed = 0;
        for (let y = 0; y < h; y++) {
            const y0 = y > 0 ? y - 1 : 0, y1 = y < h - 1 ? y + 1 : h - 1;
            for (let x = 0; x < w; x++) {
                const i = y * w + x;
                if (mask[i]) continue;
                const x0 = x > 0 ? x - 1 : 0, x1 = x < w - 1 ? x + 1 : w - 1;
                let on = 0;
                for (let yy = y0; yy <= y1 && !on; yy++) {
                    const r = yy * w;
                    for (let xx = x0; xx <= x1; xx++) if (diff[r + xx]) { on = 1; break; }
                }
                if (on) { dil[i] = 1; changed++; }
            }
        }
        const fraction = changed / n;
        const blob = changed > 0 ? this.largest() : 0;
        if (!this.learned) return { open: true, reason: 'learning', blob, fraction };
        if (blob > this.opts.blobPixels) return { open: true, reason: 'blob', blob, fraction };
        if (fraction > this.opts.fraction) return { open: true, reason: 'fraction', blob, fraction };
        return { open: false, reason: 'closed', blob, fraction };
    }

    // 3x3 gaussian [1 2 1]x[1 2 1]/16 with clamped borders
    private blur(src: Uint8Array, out: Uint8Array): void {
        const { w, h, tmp } = this;
        for (let y = 0; y < h; y++) {
            const r = y * w;
            for (let x = 0; x < w; x++) {
                const a = src[r + (x > 0 ? x - 1 : 0)], b = src[r + x], c = src[r + (x < w - 1 ? x + 1 : w - 1)];
                tmp[r + x] = a + 2 * b + c;
            }
        }
        for (let y = 0; y < h; y++) {
            const up = (y > 0 ? y - 1 : 0) * w, mid = y * w, dn = (y < h - 1 ? y + 1 : h - 1) * w;
            for (let x = 0; x < w; x++) out[mid + x] = (tmp[up + x] + 2 * tmp[mid + x] + tmp[dn + x] + 8) >> 4;
        }
    }

    private swap(): void { const t = this.prev; this.prev = this.cur; this.cur = t; }

    // 8-connected components on `dil`, returns the size of the largest
    private largest(): number {
        const { w, h, n, dil, seen, stack } = this;
        seen.fill(0);
        let best = 0;
        for (let s = 0; s < n; s++) {
            if (!dil[s] || seen[s]) continue;
            let sp = 0, size = 0;
            stack[sp++] = s; seen[s] = 1;
            while (sp > 0) {
                const i = stack[--sp];
                size++;
                const x = i % w, y = (i - x) / w;
                for (let yy = y > 0 ? y - 1 : 0, ye = y < h - 1 ? y + 1 : h - 1; yy <= ye; yy++) {
                    for (let xx = x > 0 ? x - 1 : 0, xe = x < w - 1 ? x + 1 : w - 1; xx <= xe; xx++) {
                        const j = yy * w + xx;
                        if (dil[j] && !seen[j]) { seen[j] = 1; stack[sp++] = j; }
                    }
                }
            }
            if (size > best) best = size;
        }
        return best;
    }

    // count unmasked-by-threshold changes (diff was computed with an empty mask while learning)
    private learnPair(): void {
        const c = this.counts, d = this.diff;
        for (let i = 0; i < this.n; i++) c[i] += d[i];
        this.pairs++;
    }

    private finishLearning(): void {
        const { w, h, mask, counts } = this;
        const band = Math.floor(h * this.opts.bandFraction);
        const need = this.opts.maskRatio * this.pairs;
        const raw = this.dil; // scratch, rebuilt by the caller afterwards
        raw.fill(0);
        for (let y = 0; y < h; y++) {
            if (y >= band && y < h - band) continue;
            for (let x = 0; x < w; x++) if (counts[y * w + x] >= need) raw[y * w + x] = 1;
        }
        let m = 0;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            let on = 0;
            for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1) && !on; yy++)
                for (let xx = Math.max(0, x - 1); xx <= Math.min(w - 1, x + 1); xx++) if (raw[yy * w + xx]) { on = 1; break; }
            mask[y * w + x] = on; m += on;
        }
        this.maskedCount = m;
        this.learned = true;
    }
}
