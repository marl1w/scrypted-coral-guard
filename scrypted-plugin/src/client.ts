// Typed HTTP client for the Coral verifier contract (../coral-verifier/API.md).
// No Scrypted imports: usable from tests against a mock server.

import { validCameraId } from './util';

const MAX_BODY_BYTES = 1 << 20;

export interface Health {
    ok: boolean;
    version?: string;
    model?: { name?: string; input?: number; coral?: 'ok' | 'stalled' | 'missing' };
    coral_ms?: { median?: number; p95?: number; last?: number };
    resets?: number;
    queue_depth?: number;
    cameras_active?: number;
    uptime_s?: number;
}

export interface PersonState {
    state: 'idle' | 'candidate' | 'confirmed' | 'holding';
    confirmed: boolean;
    present: boolean;
    since_ms?: number;
    last_seen_ms?: number;
    tracks?: number;
    /** service 0.3.0+: live person tracks */
    tracks_detail?: { id: number; box?: number[]; score?: number; last_seen_ms?: number; since_ms?: number; stationary?: boolean }[];
    /** service 0.3.0+: frame time until which the group is held */
    hold_until_ms?: number | null;
    /** service 0.3.12+: how the current confirmation was reached ('fast_move' | 'fast_claim' | 'ticks'), null while not confirmed */
    confirmed_by?: string | null;
    /** fast-confirm diagnostics of this frame: strong person boxes, best track's frames in the window, its shift (box heights). */
    fast?: { hi: number; n: number; moved: number } | null;
    /** rule B (logged only): frame time at which the camera AI person + one weak Coral box would have confirmed. */
    would_b_ms?: number | null;
}

export interface VehicleState {
    /** a non-parked vehicle is present (parked ones never count). */
    moving: boolean;
    hold_until_ms?: number | null;
    tracks?: { id: number; cls: string; parked: boolean; /** newer service: 'moving' | 'still' | 'parked' (new tracks start 'still') */ state?: string; since_ms: number; box?: number[] }[];
}

export interface AnimalState {
    confirmed?: boolean;
    present?: boolean;
    hold_until_ms?: number | null;
    tracks?: { id: number; cls: string; since_ms?: number; last_seen_ms?: number; box?: number[] }[];
}

export interface FrameReply {
    camera?: string;
    t?: number;
    coral_ms?: number;
    queue_ms?: number;
    infer_ms?: number;
    total_ms?: number;
    /** group is absent on an old service: derive it from cls (see groupOf). parked/track_id are optional per-detection extras. */
    detections?: { cls: string; score: number; box: number[]; group?: string; parked?: boolean; track_id?: number; /** display-only candidate flag from the service (vehicles 0.3-0.5): never decides anything */ weak?: boolean }[];
    person: PersonState;
    vehicle?: VehicleState;
    /** absent on an old service. */
    animal?: AnimalState;
    session?: { frames: number; started_ms: number };
}

export interface EpisodeEvent {
    id: string;
    action: 'start' | 'update' | 'end';
    t_ms: number;
    mode: string;
    kind: 'camera' | 'unflagged';
    outcome?: 'confirmed' | 'vetoed' | 'aborted' | 'released' | 'parked';
    groups?: string[];
    timeline_event?: { t_ms: number; text: string };
    note?: string;
    /** service 0.3.12+: a veto resting on too little evidence */
    inconclusive?: boolean;
    /** service 0.3.12+: the camera's enabled detection groups (the review question of an inconclusive veto names them) */
    detect?: string[];
    /** service 0.3.19+ (older services ignore it): end only, confirmed episodes; occupancy vs motion time and still releases */
    motion?: { occupied_s: number; motion_s: number; still_releases: number; reraises: number };
}

export interface TrackerConfig {
    /** newer service: the camera's display name (journal, panels) */
    name?: string;
    /** newer service: vehicle detect, animal confirm / keep thresholds */
    vehicle_thr?: number;
    /** newer service (0.3.23): minimum box size per group, share of the frame area, 0 = off */
    min_area_person?: number;
    min_area_vehicle?: number;
    min_area_animal?: number;
    /** service 0.3.30+: shape hold of still/parked vehicles the detector misses */
    shape_hold?: boolean;
    shape_thr?: number;
    occlusion_hold_s?: number;
    /** service 0.3.31+: learn and use known parking spots */
    parking_spots?: boolean;
    /** service 0.3.31+: ignore person/animal boxes that match a look-alike labelled false alarm */
    lookalikes?: boolean;
    /** newer service: extra shrunk-with-border pass, 0 = off */
    zoom_out?: number;
    /** newer service: re-check uncertain detections with a zoomed-in pass */
    second_look?: boolean;
    animal_start_thr?: number;
    animal_cont_thr?: number;
    start_thr?: number;
    cont_thr?: number;
    hold_s?: number;
    confirm_n?: number;
    confirm_m?: number;
    still_s?: number;
    still_hold_s?: number;
    max_hold_s?: number;
    parked_after_s?: number;
    session_ttl_s?: number;
    /** service 0.3.12+: fast person confirmation (moving track or camera person claim) */
    fast_confirm?: boolean;
    fast_n?: number;
    fast_window_ms?: number;
    fast_move?: number;
}

export interface FrameOptions {
    /** background frame (heartbeat, pixel gate): served only when no normal frame waits */
    low?: boolean;
    /** the longest the frame may wait in the service queue, ms (stale otherwise) */
    deadlineMs?: number;
    /** the camera's own AI claims this class for the current episode */
    claim?: 'person' | 'animal';
}

/** Plain explanation of an HTTP status for logs and the panel, the code only in brackets. */
export function httpText(status: number): string {
    const what = status === 400 ? 'request rejected by the service'
        : status === 401 || status === 403 ? 'token rejected by the service'
            : status === 404 ? 'not found on the service'
                : status === 413 ? 'image too large for the service'
                    : status === 429 ? 'too many frames for the service'
                        : status === 503 ? 'service unavailable'
                            : status >= 500 ? 'service error' : 'unexpected reply from the service';
    return `${what} (HTTP ${status})`;
}

export class VerifierError extends Error {
    constructor(message: string, public status?: number) {
        super(message);
    }
}

export interface ClientOptions {
    baseUrl: string;
    token?: string;
    /** timeout for health / config / reset. */
    timeoutMs: number;
    /** timeout for POST /frame (short: a late verdict is useless). Defaults to timeoutMs. */
    frameTimeoutMs?: number;
}

export class VerifierClient {
    constructor(private opts: () => ClientOptions) { }

    private framePaths = new Map<string, string>();

    /** One request. Returns null only when allowSkip and the service says "skip this frame". */
    private async request<T>(method: string, path: string, body?: Buffer | string, contentType?: string, auth = true, timeoutOverride?: number, allowSkip = false): Promise<T | null> {
        const o = this.opts();
        const timeoutMs = timeoutOverride ?? o.timeoutMs;
        const ac = new AbortController();
        const timer = setTimeout(() => ac.abort(), timeoutMs);
        try {
            const headers: Record<string, string> = {};
            if (contentType)
                headers['Content-Type'] = contentType;
            if (auth && o.token)
                headers['X-Verifier-Token'] = o.token;
            let res: Response;
            try {
                // redirect: 'error' so the token is never forwarded to another origin.
                if (!/^https?:\/\//i.test(o.baseUrl))
                    throw new VerifierError('the service URL must start with http:// or https://');
                res = await fetch(o.baseUrl.replace(/\/+$/, '') + path, { method, headers, body: body as any, signal: ac.signal, redirect: 'error' });
            }
            catch (e: any) {
                throw new VerifierError(ac.signal.aborted ? `timeout after ${timeoutMs} ms` : `network error: ${e?.cause?.code || e?.message || e}`);
            }
            if (res.status !== 200) {
                // Skip (not an outage): 429 (too many frames from this camera) and 503 {reason:"stale"}
                // (the service's latest-wins queue dropped an old frame).
                if (allowSkip && res.status === 429) {
                    res.body?.cancel().catch(() => { });
                    return null;
                }
                if (allowSkip && res.status === 503) {
                    const text = await readCapped(res, 4096).catch(() => '');
                    if (text.includes('"stale"'))
                        return null;
                }
                else
                    res.body?.cancel().catch(() => { });
                throw new VerifierError(httpText(res.status), res.status);
            }
            let text: string;
            try {
                text = await readCapped(res, MAX_BODY_BYTES);
            }
            catch (e: any) {
                throw new VerifierError(ac.signal.aborted ? `timeout after ${timeoutMs} ms` : (e instanceof VerifierError ? e.message : 'body read failed'));
            }
            try {
                return JSON.parse(text) as T;
            }
            catch (e) {
                throw new VerifierError('invalid JSON reply');
            }
        }
        finally {
            clearTimeout(timer);
        }
    }

    // health is unauthenticated per the contract.
    async health(): Promise<Health> {
        const h = await this.request<any>('GET', '/v1/health', undefined, undefined, false);
        if (!h || typeof h !== 'object' || typeof h.ok !== 'boolean')
            throw new VerifierError('invalid health reply');
        return h as Health;
    }

    /**
     * Readiness probe: health AND an authenticated, cheap call (GET /v1/cameras). /health needs no
     * token, so health alone would report a wrong token as ready.
     */
    async check(): Promise<Health> {
        const h = await this.health();
        if (h.ok)
            await this.request('GET', '/v1/cameras');
        return h;
    }

    /** GET /v1/metrics (token, 2 s): the performance feed. Throws (status 404 on an older service). */
    async metrics(): Promise<any> {
        const m = await this.request<any>('GET', '/v1/metrics', undefined, undefined, true, 2000);
        if (!m || typeof m !== 'object')
            throw new VerifierError('invalid metrics reply');
        return m;
    }

    /** Returns null when the service says to skip the frame (stale or 429). */
    async frame(camera: string, jpeg: Buffer, tMs: number, opt: boolean | FrameOptions = false): Promise<FrameReply | null> {
        const o2: FrameOptions = typeof opt === 'boolean' ? { low: opt } : opt;
        let prefix = this.framePaths.get(camera);
        if (!prefix) {
            assertCameraId(camera);
            this.framePaths.set(camera, prefix = `/v1/cameras/${encodeURIComponent(camera)}/frame?t=`);
        }
        const o = this.opts();
        const q = (o2.low ? '&prio=low' : '') + (o2.deadlineMs && o2.deadlineMs > 0 ? `&deadline_ms=${Math.round(o2.deadlineMs)}` : '') + (o2.claim === 'person' || o2.claim === 'animal' ? `&claim=${o2.claim}` : '');
        const r = await this.request<any>('POST', prefix + Math.round(tMs) + q, jpeg, 'image/jpeg', true, o.frameTimeoutMs ?? o.timeoutMs, true);
        if (r === null)
            return null;
        if (!validFrameReply(r))
            throw new VerifierError('invalid frame reply');
        return r;
    }

    async reset(camera: string): Promise<unknown> {
        assertCameraId(camera);
        return this.request('POST', `/v1/cameras/${encodeURIComponent(camera)}/reset`, '{}', 'application/json');
    }

    /** POST /v1/cameras/{camera}/episodes (event journal, service 0.3.0+): fire-and-forget for the caller. */
    async episodeEvent(camera: string, ev: EpisodeEvent): Promise<unknown> {
        assertCameraId(camera);
        return this.request('POST', `/v1/cameras/${encodeURIComponent(camera)}/episodes`, JSON.stringify(ev), 'application/json');
    }

    /** POST /v1/cameras/{camera}/spots (service 0.3.31+): forget the camera's known parking spots. */
    async forgetSpots(camera: string): Promise<{ forgotten?: boolean } | null> {
        assertCameraId(camera);
        return this.request('POST', `/v1/cameras/${encodeURIComponent(camera)}/spots`, '{}', 'application/json');
    }

    /** POST /v1/cameras/{camera}/lookalikes (service 0.3.31+): forget the camera's known look-alikes. */
    async forgetLookalikes(camera: string): Promise<{ forgotten?: boolean } | null> {
        assertCameraId(camera);
        return this.request('POST', `/v1/cameras/${encodeURIComponent(camera)}/lookalikes`, '{}', 'application/json');
    }

    async config(camera: string, cfg: TrackerConfig): Promise<TrackerConfig | null> {
        assertCameraId(camera);
        return this.request('POST', `/v1/cameras/${encodeURIComponent(camera)}/config`, JSON.stringify(cfg), 'application/json');
    }
}

function assertCameraId(id: string) {
    if (!validCameraId(id))
        throw new VerifierError(`invalid camera id "${id}"`);
}

const PERSON_STATES = ['idle', 'candidate', 'confirmed', 'holding'];
const optNum = (v: unknown) => v === undefined || (typeof v === 'number' && Number.isFinite(v));

/** A malformed reply must never be read as "person gone": it is a failure. */
export function validFrameReply(r: any): r is FrameReply {
    if (!r || typeof r !== 'object' || Array.isArray(r))
        return false;
    const p = r.person;
    if (!p || typeof p !== 'object' || !PERSON_STATES.includes(p.state) || typeof p.confirmed !== 'boolean' || typeof p.present !== 'boolean')
        return false;
    if (r.vehicle !== undefined && (!r.vehicle || typeof r.vehicle !== 'object' || typeof r.vehicle.moving !== 'boolean'))
        return false;
    if (r.animal !== undefined && (!r.animal || typeof r.animal !== 'object'))
        return false;
    sanitizeDetections(r);
    return optNum(r.coral_ms) && optNum(r.queue_ms) && optNum(r.infer_ms) && optNum(r.total_ms);
}

/** Read a response body as text, failing once it exceeds maxBytes. */
async function readCapped(res: Response, maxBytes: number): Promise<string> {
    if (!res.body)
        return '';
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    for (; ;) {
        const { done, value } = await reader.read();
        if (done)
            break;
        total += value.length;
        if (total > maxBytes) {
            reader.cancel().catch(() => { });
            throw new VerifierError('response too large');
        }
        chunks.push(value);
    }
    return Buffer.concat(chunks).toString('utf8');
}

export function slugify(name: string): string {
    return name.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 64) || 'camera';
}

const VEHICLE_CLS = ['bicycle', 'car', 'motorcycle', 'bus', 'truck'];
const ANIMAL_CLS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bear'];

/** Detection group: the service's `group` when present, else derived from the class (old services). */
export function groupOf(d: { cls: string; group?: string }): 'person' | 'vehicle' | 'animal' | undefined {
    if (d.group === 'person' || d.group === 'vehicle' || d.group === 'animal')
        return d.group;
    if (d.cls === 'person')
        return 'person';
    if (VEHICLE_CLS.includes(d.cls))
        return 'vehicle';
    if (ANIMAL_CLS.includes(d.cls))
        return 'animal';
    return undefined;
}

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);

/** Keep only well-formed detections (cls string, finite score, 4 finite box numbers) and tracks: a malformed entry is dropped, never thrown on. */
function sanitizeDetections(r: any): void {
    if (r.detections !== undefined) {
        r.detections = Array.isArray(r.detections)
            ? r.detections.filter((d: any) => d && typeof d === 'object' && typeof d.cls === 'string' && finite(d.score) && Array.isArray(d.box) && d.box.length === 4 && d.box.every(finite))
            : undefined;
    }
    for (const g of [r.vehicle, r.animal]) {
        if (g && g.tracks !== undefined)
            g.tracks = Array.isArray(g.tracks) ? g.tracks.filter((t: any) => t && typeof t === 'object' && (t.box === undefined || (Array.isArray(t.box) && t.box.length === 4 && t.box.every(finite)))) : undefined;
    }
}
