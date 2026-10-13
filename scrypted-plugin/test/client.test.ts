import test from 'node:test';
import assert from 'node:assert/strict';
import { slugify, VerifierClient } from '../src/client';
import { MockServer, confirmed, idle, stale, stalled } from './mock-server';

async function withServer(fn: (m: MockServer, c: VerifierClient) => Promise<void>, timeoutMs = 300, token = 'tok') {
    const m = new MockServer();
    await m.start();
    const c = new VerifierClient(() => ({ baseUrl: m.url, token, timeoutMs }));
    try { await fn(m, c); } finally { await m.stop(); }
}

test('slugify', () => {
    assert.equal(slugify('Front Door'), 'front_door');
    assert.equal(slugify('Cortile Nord!'), 'cortile_nord');
    assert.equal(slugify('Caffè'), 'caffe');
});

test('health needs no token; frame sends token, query t, jpeg body and parses the reply', () => withServer(async (m, c) => {
    m.script.frames = [confirmed];
    const h = await c.health();
    assert.equal(h.ok, true);
    const r = await c.frame('cam_a', Buffer.from([0xff, 0xd8, 0xff, 0xd9]), 1791886000000.4);
    assert.equal(r!.person.confirmed, true);
    assert.equal(m.calls[0].token, undefined);
    assert.equal(m.calls[1].token, 'tok');
    assert.equal(m.calls[1].path, '/v1/cameras/cam_a/frame?t=1791886000000');
    assert.equal(m.calls[1].bytes, 4);
}));

test('reset and config hit the contract paths', () => withServer(async (m, c) => {
    await c.reset('cam_a');
    await c.config('cam_a', { start_thr: 0.8 });
    assert.deepEqual(m.calls.map(x => `${x.method} ${x.path}`), ['POST /v1/cameras/cam_a/reset', 'POST /v1/cameras/cam_a/config']);
}));

test('non-200 statuses throw with the status (401, 400, 503, 500)', () => withServer(async (m, c) => {
    for (const status of [401, 400, 503, 500]) {
        m.script.frames = [status];
        await assert.rejects(c.frame('c', Buffer.from([1]), 0), (e: any) => e.status === status);
    }
}));

test('wrong token gives 401', () => withServer(async (m, c) => {
    m.script.token = 'other';
    await assert.rejects(c.reset('c'), (e: any) => e.status === 401);
}));

test('slow server times out via AbortController', () => withServer(async (m, c) => {
    m.script.delayMs = 800;
    m.script.frames = [idle];
    const t0 = Date.now();
    await assert.rejects(c.frame('c', Buffer.from([1]), 0), /timeout/);
    assert.ok(Date.now() - t0 < 700);
}));

test('connection refused is a VerifierError, not a crash', async () => {
    const c = new VerifierClient(() => ({ baseUrl: 'http://127.0.0.1:1', timeoutMs: 300 }));
    await assert.rejects(c.health(), /network error|timeout/);
});

test('503 {reason:"stale"} is a skipped frame (null), other 503s still throw', () => withServer(async (m, c) => {
    m.script.frames = [stale];
    assert.equal(await c.frame('c', Buffer.from([1]), 0), null);
    m.script.frames = [stalled];
    await assert.rejects(c.frame('c', Buffer.from([1]), 0), (e: any) => e.status === 503);
}));

test('frame uses its own short timeout, health/config keep the long one', async () => {
    const m = new MockServer();
    await m.start();
    try {
        const c = new VerifierClient(() => ({ baseUrl: m.url, timeoutMs: 1500, frameTimeoutMs: 150 }));
        m.script.delayMs = 400;
        m.script.frames = [idle];
        await assert.rejects(c.frame('c', Buffer.from([1]), 0), /timeout after 150/);
        assert.equal((await c.health()).ok, true); // 400 ms < 1500 ms
    } finally { await m.stop(); }
});

// ---- review fixes ------------------------------------------------------------------------------
import http from 'http';
import { validFrameReply } from '../src/client';

test('M4: a cross-origin redirect is refused and the token never reaches the target', async () => {
    let seen: any = null;
    const B = http.createServer((q, s) => { seen = q.headers['x-verifier-token']; s.end('{}'); });
    await new Promise<void>(r => B.listen(0, '127.0.0.1', r));
    const A = http.createServer((q, s) => { s.statusCode = 307; s.setHeader('location', `http://127.0.0.1:${(B.address() as any).port}${q.url}`); s.end(); });
    await new Promise<void>(r => A.listen(0, '127.0.0.1', r));
    const c = new VerifierClient(() => ({ baseUrl: `http://127.0.0.1:${(A.address() as any).port}`, token: 'SECRET', timeoutMs: 1000 }));
    await assert.rejects(c.reset('cam'), /network error|redirect/);
    assert.equal(seen, null);
    A.close(); B.close();
});

test('L1: malformed 200 replies are failures, never "person gone"', () => withServer(async (m, c) => {
    const bad: any[] = [{}, { person: null }, { person: { state: 'idle' } }, { person: { state: 'weird', confirmed: false, present: false } },
        { person: { state: 'idle', confirmed: 'no', present: false } }, { person: { state: 'idle', confirmed: false, present: false }, vehicle: { moving: 'x' } },
        { person: { state: 'idle', confirmed: false, present: false }, total_ms: 'fast' }, []];
    for (const b of bad) {
        m.script.frames = [b];
        await assert.rejects(c.frame('c', Buffer.from([1]), 0), /invalid frame reply/, JSON.stringify(b));
    }
    assert.equal(validFrameReply({ person: { state: 'holding', confirmed: false, present: true }, vehicle: { moving: false }, total_ms: 41 }), true);
}));

test('L1: an invalid health reply is an error', () => withServer(async (m, c) => {
    m.script.health = { version: 'x' };
    await assert.rejects(c.health(), /invalid health/);
}));

test('L2: a huge response body is cut off instead of buffered', async () => {
    const H = http.createServer((_q, s) => {
        s.setHeader('content-type', 'application/json');
        s.write('{"person":{"state":"idle","confirmed":false,"present":false},"pad":"');
        const chunk = 'x'.repeat(1 << 20);
        let n = 0;
        const iv = setInterval(() => { if (s.destroyed) { clearInterval(iv); return; } s.write(chunk); if (++n >= 50) { clearInterval(iv); s.end('"}'); } }, 1);
    });
    await new Promise<void>(r => H.listen(0, '127.0.0.1', r));
    const c = new VerifierClient(() => ({ baseUrl: `http://127.0.0.1:${(H.address() as any).port}`, timeoutMs: 3000, frameTimeoutMs: 2000 }));
    await assert.rejects(c.frame('c', Buffer.from([1]), 0), /too large/);
    H.closeAllConnections?.(); H.close();
});

test('H3: check() runs an authenticated call, so a wrong token is not "ready" although /health is open', () => withServer(async (m, c) => {
    m.script.token = 'tok';
    assert.equal((await c.check()).ok, true);
    const bad = new VerifierClient(() => ({ baseUrl: m.url, token: 'WRONG', timeoutMs: 500 }));
    assert.equal((await bad.health()).ok, true);
    await assert.rejects(bad.check(), (e: any) => e.status === 401);
}));

test('H3: 429 is a skip (null), like stale', () => withServer(async (m, c) => {
    m.script.frames = [{ status: 429, body: { retry_after_ms: 100 } }];
    assert.equal(await c.frame('c', Buffer.from([1]), 0), null);
}));

test('L3: dot ids are rejected before any request', () => withServer(async (m, c) => {
    for (const id of ['.', '..', 'a/b', 'a b', '', 'x?y']) {
        await assert.rejects(c.reset(id), /invalid camera id/);
        await assert.rejects(c.frame(id, Buffer.from([1]), 0), /invalid camera id/);
    }
    assert.equal(m.calls.length, 0);
}));

test('reset path url-encodes ids that are valid but unusual', () => withServer(async (m, c) => {
    await c.reset('a_b-1.c');
    assert.equal(m.calls[0].path, '/v1/cameras/a_b-1.c/reset');
}));

test('only http(s) service URLs are accepted; malformed detections are dropped, never thrown on', async () => {
    const { VerifierClient, validFrameReply } = require('../src/client') as typeof import('../src/client');
    const c = new VerifierClient(() => ({ baseUrl: 'file:///etc/passwd', timeoutMs: 500 }));
    await assert.rejects(c.health(), /http/);
    const r: any = { person: { state: 'idle', confirmed: false, present: false }, detections: [{ cls: 'car', score: 0.9, box: [0, 0, 1, 1] }, { cls: 3, score: 'x', box: null }, null, { cls: 'dog', score: NaN, box: [0, 0, 1, 1] }], vehicle: { moving: false, tracks: [null, { id: 1, cls: 'car', parked: true, since_ms: 1, box: [0, 0, 1, 1] }] } };
    assert.equal(validFrameReply(r), true);
    assert.equal(r.detections.length, 1);
    assert.equal(r.vehicle.tracks.length, 1);
});
