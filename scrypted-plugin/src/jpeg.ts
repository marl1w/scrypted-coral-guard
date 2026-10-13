// Incremental splitter for a concatenated MJPEG stream (ffmpeg image2pipe output).
// One reusable accumulator; each byte is scanned once. The only per-frame allocation is the
// copy handed to the caller, which it must own (the request is async).

const MAX_ACC = 8 << 20;

export class JpegSplitter {
    private acc = Buffer.allocUnsafe(1 << 18);
    private len = 0;
    private scanned = 0; // bytes already searched for the marker
    private start = -1;  // offset of the current SOI, -1 if none yet

    push(chunk: Buffer, onFrame: (jpeg: Buffer) => void) {
        if (this.len + chunk.length > MAX_ACC) {
            // runaway or garbage stream: drop what is buffered and resynchronise on the next SOI.
            this.len = 0; this.scanned = 0; this.start = -1;
            if (chunk.length > MAX_ACC)
                return;
        }
        if (this.len + chunk.length > this.acc.length) {
            const next = Buffer.allocUnsafe(Math.max(this.acc.length * 2, this.len + chunk.length));
            this.acc.copy(next, 0, 0, this.len);
            this.acc = next;
        }
        chunk.copy(this.acc, this.len);
        this.len += chunk.length;
        const a = this.acc;
        let i = this.scanned;
        for (; i < this.len - 1; i++) {
            if (a[i] !== 0xff)
                continue;
            const m = a[i + 1];
            if (this.start < 0) {
                if (m === 0xd8) { this.start = i; i++; }
            }
            else if (m === 0xd9) {
                onFrame(Buffer.from(a.subarray(this.start, i + 2)));
                this.start = -1;
                i++;
            }
        }
        this.scanned = i;
        // compact: drop consumed bytes, keep the open frame (or the last byte, may be half a marker).
        const keepFrom = this.start >= 0 ? this.start : Math.max(0, this.len - 1);
        if (keepFrom > 0) {
            a.copy(a, 0, keepFrom, this.len);
            this.len -= keepFrom;
            this.scanned -= keepFrom;
            if (this.start >= 0) this.start = 0;
        }
        if (this.scanned < 0) this.scanned = 0;
    }
}

/** Width and height of a JPEG from its SOF marker, or undefined. */
export function jpegDimensions(b: Buffer): { w: number; h: number } | undefined {
    try {
        let i = 2;
        while (i + 9 < b.length) {
            if (b[i] !== 0xff) { i++; continue; }
            const m = b[i + 1];
            if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc)
                return { h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7) };
            i += 2 + b.readUInt16BE(i + 2);
        }
    }
    catch (e) { /* fall through */ }
    return undefined;
}

/** JPEG quality 1..100 to ffmpeg mjpeg -q:v (2 best .. 31 worst). 85 -> 6. */
export function qualityToQscale(quality: number): number {
    const q = Math.min(100, Math.max(1, quality || 85));
    return Math.max(2, Math.min(31, Math.round(31 - q * 0.29)));
}
