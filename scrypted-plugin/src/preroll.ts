// Pure helper for pre-roll frame timing.
//
// With a prebuffer request the stream starts with a burst of past keyframes, then goes live.
// Once ffmpeg has decoded and fps-resampled them, the burst frames carry no wall-clock timestamp,
// so capture times are ESTIMATED without waiting for the burst to end (waiting costs latency):
// the first frame is assumed to be the oldest frame of a FULL pre-roll window and later burst
// frames sit one interval apart. If the real prebuffer was shorter, burst frames are labelled
// older than they really are (never newer, never in the future), which is harmless: the service
// logic is frame-count based for confirmation, and live frames that follow carry arrival time.
// A frame arriving at least half an interval after the previous one is live.

export class FrameClock {
    private count = 0;
    private firstAt = 0;
    private lastAt = 0;
    private live: boolean;

    constructor(private intervalMs: number, private prerollMs: number, enabled: boolean) {
        this.live = !enabled || prerollMs <= 0;
    }

    get inBurst() { return !this.live; }

    /** capture time (unix ms) to send for a frame that arrived at `arrivedAt`. */
    stamp(arrivedAt: number): number {
        if (this.live)
            return arrivedAt;
        if (this.count > 0 && arrivedAt - this.lastAt >= this.intervalMs / 2) {
            this.live = true;
            return arrivedAt;
        }
        if (this.count === 0)
            this.firstAt = arrivedAt;
        const t = Math.min(arrivedAt, this.firstAt - this.prerollMs + this.count * this.intervalMs);
        this.count++;
        this.lastAt = arrivedAt;
        return t;
    }
}
