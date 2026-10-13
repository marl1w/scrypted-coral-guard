// Mock of the Coral verifier service (docs/SERVICE_API.md) with scripted replies.
import http from 'http';
import type { AddressInfo } from 'net';

export interface MockScript {
    /** replies consumed one per POST /frame; the last one repeats. A number = HTTP status to return. */
    frames: (object | number | { status: number; body: object })[];
    health: object | number;
    delayMs: number;
    token?: string;
}

export class MockServer {
    script: MockScript = { frames: [], health: { ok: true, version: 'mock' }, delayMs: 0 };
    calls: { method: string; path: string; token?: string; bytes: number }[] = [];
    private server!: http.Server;
    url = '';

    async start() {
        this.server = http.createServer((req, res) => {
            const chunks: Buffer[] = [];
            req.on('data', c => chunks.push(c));
            req.on('end', () => {
                const body = Buffer.concat(chunks);
                const path = req.url || '';
                this.calls.push({ method: req.method || '', path, token: req.headers['x-verifier-token'] as string, bytes: body.length });
                const send = (status: number, payload: object) => setTimeout(() => {
                    res.writeHead(status, { 'content-type': 'application/json' });
                    res.end(JSON.stringify(payload));
                }, this.script.delayMs);
                if (path === '/v1/health')
                    return typeof this.script.health === 'number' ? send(this.script.health, {}) : send(200, this.script.health);
                if (this.script.token && req.headers['x-verifier-token'] !== this.script.token)
                    return send(401, {});
                if (path.includes('/frame')) {
                    const next = this.script.frames.length > 1 ? this.script.frames.shift()! : this.script.frames[0];
                    if (typeof next === 'number') return send(next, {});
                    if ('status' in next) return send((next as any).status, (next as any).body);
                    return send(200, next);
                }
                send(200, {});
            });
        });
        await new Promise<void>(r => this.server.listen(0, '127.0.0.1', r));
        this.url = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
    }

    stop() { return new Promise<void>(r => { this.server.closeAllConnections?.(); this.server.close(() => r()); }); }
}

export const stale = { status: 503, body: { reason: 'stale' } };
export const stalled = { status: 503, body: { coral: 'stalled' } };
export const idle = { person: { state: 'idle', confirmed: false, present: false } };
export const candidate = { person: { state: 'candidate', confirmed: false, present: false } };
export const confirmed = { person: { state: 'confirmed', confirmed: true, present: true } };
export const holding = { person: { state: 'holding', confirmed: false, present: true } };
export const vehicle = { person: { state: 'idle', confirmed: false, present: false }, vehicle: { moving: true }, detections: [{ cls: 'car', score: 0.9, box: [0.1, 0.1, 0.5, 0.5], group: 'vehicle' }] };
