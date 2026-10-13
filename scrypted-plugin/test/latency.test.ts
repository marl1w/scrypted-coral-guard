// Plugin-side latency with a mock service (HTTP loopback, scripted Coral time). Excludes ffmpeg start
// and camera/prebuffer delivery, which only a live Scrypted can measure. Prints the numbers.
import test from 'node:test';
import assert from 'node:assert/strict';
import { VerifierClient } from '../src/client';
import { Episode } from '../src/episode';
import { FrameClock } from '../src/preroll';
import { MockServer, candidate, confirmed } from './mock-server';

const cfg = { mode: 'gate' as const, confirmTimeoutMs: 6000, maxHoldMs: 600000, killSwitch: false };
const jpeg = Buffer.alloc(40000, 1);
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

/** sequentially post frames like the mixin (one in flight), return ms from trigger to confirmed */
async function run(m: MockServer, c: VerifierClient, frames: { t: number; wait: number }[]): Promise<{ posted: number; verdict: number; confirmed: number }> {
    const e = new Episode(cfg);
    const t0 = Date.now();
    e.raw(true, t0);
    let confirmedAt = 0;
    for (const f of frames) {
        if (f.wait) await sleep(f.wait);
        e.posted(Date.now());
        const r = await c.frame('cam', jpeg, f.t);
        const fx = e.reply(r!, Date.now(), f.t);
        if (fx.motion) { confirmedAt = Date.now(); break; }
    }
    const [s] = [e.tick(t0 + 6000)].flatMap(x => x.logs).filter(l => l.kind === 'summary').map(l => (l as any).summary);
    void s;
    return { posted: 0, verdict: 0, confirmed: confirmedAt - t0 };
}

test('plugin-side trigger-to-decision with a 35 ms Coral, pre-roll vs live', async () => {
    const m = new MockServer();
    await m.start();
    const c = new VerifierClient(() => ({ baseUrl: m.url, timeoutMs: 2000, frameTimeoutMs: 800 }));
    try {
        m.script.delayMs = 35;
        // pre-roll: 4 burst frames available immediately, person seen on the 2nd and 3rd (2-of-3)
        m.script.frames = [candidate, confirmed];
        const clock = new FrameClock(1000, 4000, true);
        const now = Date.now();
        const burst = [0, 1, 2, 3].map(i => ({ t: clock.stamp(now + i), wait: 0 }));
        const withPre: number[] = [];
        for (let i = 0; i < 10; i++) { m.script.frames = [candidate, confirmed]; withPre.push((await run(m, c, burst)).confirmed); }
        // without pre-roll: frames arrive one interval apart (here 1000 ms): confirm = first + 1 interval
        m.script.frames = [candidate, confirmed];
        const live = (await run(m, c, [{ t: Date.now(), wait: 0 }, { t: Date.now() + 1000, wait: 1000 }])).confirmed;
        const sorted = withPre.sort((a, b) => a - b);
        console.log(`LATENCY plugin-side, mock Coral 35 ms: pre-roll p50 ${sorted[5]} ms, max ${sorted[9]} ms; live 1 fps (2 frames) ${live} ms (of which 1000 ms is the frame interval)`);
        assert.ok(sorted[5] < 200);
        assert.ok(live < 1200);
    } finally { await m.stop(); }
});
