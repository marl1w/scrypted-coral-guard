// Test harness: loads the real compiled mixin/provider/frames with a stubbed Scrypted SDK, a
// controllable frame source (startFrames) and a controllable service client.
/* eslint-disable @typescript-eslint/no-explicit-any */
import Module = require('module');

export interface Ctl {
    /** ObjectDetector events the mixin published */
    detections: any[];
    /** the camera's getObjectTypes classes (undefined: a default list) */
    cameraClasses: string[] | undefined;
    starts: { o: any; stopped: boolean; exit: (e?: string) => void }[];
    listeners: number;
    removed: number;
    logs: [string, string][];
    writes: [number, boolean][];
    occWrites?: [number, boolean][];
    /** the "shared persisted state" the mixin reads back through mixinDevice.motionDetected */
    state: { motionDetected: boolean; occupied?: boolean };
    /** when false, writes by the mixin do not change `state` (simulates not being outermost) */
    persistWrites: boolean;
    hangStart: boolean;
    pre: Record<string, string | number>;
}

export const ctl: Ctl = { detections: [], cameraClasses: undefined, starts: [], listeners: 0, removed: 0, logs: [], writes: [], state: { motionDetected: false }, persistWrites: true, hangStart: false, pre: {} };
let listenCb: (s: any, d: any, v: boolean) => void = () => { };
let claimCb: (s: any, d: any, v: any) => void = () => { };

export const sdkStub: any = {
    systemManager: { getDeviceById: () => ({ listen: (o: any, cb: any) => { ctl.listeners++; if (o?.event === 'ObjectDetector') claimCb = cb; else listenCb = cb; return { removeListener: () => { ctl.removed++; } }; } }) },
    deviceManager: { onMixinEvent: (_id: any, _m: any, iface: string, data: any) => { if (iface === 'ObjectDetector') { (ctl as any).detections.push(data); return Promise.resolve(); } (ctl as any).notified = ((ctl as any).notified || 0) + 1; return Promise.resolve(); } },
    mediaManager: {},
};

class Base {
    id = 'cam1';
    name = 'Cam 1';
    mixinDevice = { get motionDetected() { return ctl.state.motionDetected; }, getObjectTypes: async () => ({ classes: ctl.cameraClasses ?? ['people', 'vehicle', 'dog_cat', 'package'] }), getDetectionInput: async () => undefined };
    console = {
        log: (...a: any[]) => ctl.logs.push(['log', a.join(' ')]),
        warn: (...a: any[]) => ctl.logs.push(['warn', a.join(' ')]),
        error: (...a: any[]) => ctl.logs.push(['error', a.join(' ')]),
    };
    store: Record<string, any> = { ...ctl.pre };
    storage = { getItem: (k: string) => this.store[k] ?? null, setItem: (k: string, v: any) => { this.store[k] = String(v); } };
    constructor(_o: any) { }
    get motionDetected() { return ctl.state.motionDetected; }
    set motionDetected(v: boolean) {
        ctl.writes.push([Date.now(), v]);
        if (ctl.persistWrites)
            ctl.state.motionDetected = v;
    }
    get occupied() { return !!ctl.state.occupied; }
    set occupied(v: boolean) {
        (ctl.occWrites ??= []).push([Date.now(), v]);
        ctl.state.occupied = v;
    }
    async release() { }
}

class StorageSettingsStub {
    values: Record<string, any> = {};
    constructor(private dev: any, private defs: Record<string, any>) {
        for (const k of Object.keys(defs))
            Object.defineProperty(this.values, k, { get: () => this.get(k), enumerable: true });
    }
    private get(k: string) {
        const d = this.defs[k];
        const raw = this.dev.storage.getItem(k);
        if (raw == null) return d.defaultValue;
        if (d.type === 'number') { const n = parseFloat(raw); return isNaN(n) ? (d.defaultValue || 0) : n; }
        if (d.type === 'boolean') return raw === 'true';
        return raw || d.defaultValue;
    }
    async putSetting(k: string, v: any) { this.dev.storage.setItem(k, v); }
    async getSettings() { return Object.entries(this.defs).map(([key, d]: [string, any]) => ({ key, group: d.group, title: d.title, description: d.description, value: this.values[key] })); }
}

const framesStub = {
    HW_FAILED_PREFIX: 'hwaccel failed',
    PROBE_FAILED_PREFIX: 'codec probe failed',
    vaapiAvailable: () => !!(ctl as any).vaapi,
    streamLabels: (o: any[]) => (o || []).filter((x: any) => x?.id).map((x: any) => ({ id: x.id, label: `${x.name || x.id}`, open: false })),
    startFrames: (o: any) => {
        const r: Ctl['starts'][number] = { o, stopped: false, exit: () => { } };
        ctl.starts.push(r);
        if (ctl.hangStart)
            return new Promise(() => { });
        if ((ctl as any).failStart)
            return Promise.reject(new Error('getVideoStream failed'));
        let exitRes: (e?: string) => void = () => { };
        (r as any).burst = [] as boolean[];
        const fs = { setBurst: (on: boolean) => { (r as any).burst.push(on); }, stop: () => { r.stopped = true; }, exited: new Promise<string | undefined>(res => { exitRes = res; }) };
        r.exit = exitRes;
        return Promise.resolve(fs);
    },
};

const origLoad = (Module as any)._load;
let hooked = false;
function hook(stubFrames: boolean) {
    if (hooked) return;
    hooked = true;
    (Module as any)._load = function (req: string, ...rest: any[]) {
        if (req === '@scrypted/sdk')
            return {
                __esModule: true, default: sdkStub,
                ScryptedInterface: { MotionSensor: 'MotionSensor', OccupancySensor: 'OccupancySensor', Settings: 'Settings', VideoCamera: 'VideoCamera', MixinProvider: 'MixinProvider', ObjectDetector: 'ObjectDetector' },
                ScryptedDeviceType: { Camera: 'Camera', Doorbell: 'Doorbell', Light: 'Light' },
                ScryptedDeviceBase: class { id = 'prov1'; nativeId = 'n'; storage = { getItem: () => null, setItem: () => { } }; log = { i() { } }; console = console; onDeviceEvent() { } constructor(_n?: string) { } },
                ScryptedMimeTypes: { FFmpegInput: 'x-scrypted/x-ffmpeg-input' },
            };
        if (req === '@scrypted/sdk/settings-mixin') return { SettingsMixinDeviceBase: Base };
        if (req === '@scrypted/sdk/storage-settings') return { StorageSettings: StorageSettingsStub };
        if (stubFrames && req === './frames') return framesStub;
        return origLoad.call(this, req, ...rest);
    };
}

export interface FakeClient {
    frame: (...a: any[]) => Promise<any>;
    reset: (...a: any[]) => Promise<any>;
    health: () => Promise<any>;
    check: () => Promise<any>;
    config: (...a: any[]) => Promise<any>;
}

export function deferred<T = any>() {
    let res!: (v: T) => void, rej!: (e: any) => void;
    const p = new Promise<T>((a, b) => { res = a; rej = b; });
    return { p, res, rej };
}

const made: any[] = [];
export async function releaseAll() {
    while (made.length)
        await made.pop().release().catch(() => { });
}

import { DEFAULTS } from '../src/defaults';
const PRISTINE = JSON.parse(JSON.stringify(DEFAULTS));

export function make(opts: { settings?: Record<string, any>; initial?: boolean; hangStart?: boolean; client?: any; persistWrites?: boolean; holdReset?: boolean; dup?: string; sensorless?: boolean; interfaces?: string[]; fast?: boolean; graceMs?: number } = {}) {
    hook(true);
    (ctl as any).notified = 0;
    ctl.starts = []; ctl.listeners = 0; ctl.removed = 0; ctl.logs = []; ctl.writes = []; ctl.occWrites = []; (ctl.state as any).occupied = undefined; (ctl as any).detections = [];
    ctl.state.motionDetected = !!opts.initial; ctl.hangStart = !!opts.hangStart; ctl.persistWrites = opts.persistWrites !== false;
    const calls = { frame: [] as { a: any[]; d: ReturnType<typeof deferred> }[], reset: [] as { a: any[]; d: ReturnType<typeof deferred> }[], check: 0, episodes: [] as any[] };
    const client = opts.client || {
        frame: (...a: any[]) => { const d = deferred(); const o = a[3]; if (o && typeof o === 'object') a = [a[0], a[1], a[2], !!o.low, o]; calls.frame.push({ a, d }); return d.p; }, // a[3] = low priority, a[4] = the frame options
        reset: (...a: any[]) => { const d = deferred(); calls.reset.push({ a, d }); if (!opts.holdReset) d.res({}); return d.p; },
        health: async () => ({ ok: true }),
        check: async () => { calls.check++; return { ok: true }; },
        config: async () => ({}),
        episodeEvent: async (_cam: string, ev: any) => { if ((ctl as any).journalDelay && ev.action === 'start') await new Promise(r => setTimeout(r, (ctl as any).journalDelay)); calls.episodes.push(ev); if ((ctl as any).journal404) { const e: any = new Error('not found'); e.status = 404; throw e; } return {}; },
    };
    const dups: string[] = opts.dup ? [opts.dup] : [];
    const plugin: any = { config: { killSwitch: false, baseUrl: '', token: '', timeoutMs: 2000, frameTimeoutMs: 800 }, client, register() { }, unregister() { }, hw: new Map<string, any>(), acquireHw(m: any, max: number) { if (this.hw.has(m.id)) { this.hw.set(m.id, m); return true; } if (this.hw.size >= max) return false; this.hw.set(m.id, m); return true; }, releaseHw(m: any) { if (this.hw.get(m.id) === m) this.hw.delete(m.id); }, get hwSlotsUsed() { return this.hw.size; }, findDuplicate: () => dups[0], fastConfirm: !!opts.fast, graceMs: opts.graceMs ?? 0 }; // fast confirm and grace: off unless a test turns them on (today's behaviour)
    // Mode is a real stored setting; every other former setting is an internal constant (src/defaults.ts)
    // that the tests override here. Reset to the pristine values first, then apply the test's overrides.
    const { mode, detect, ...tuning } = (opts.settings || {}) as Record<string, any>;
    Object.assign(DEFAULTS, PRISTINE, { frameProcess: 'Per trigger', hwDecode: 'Off', heartbeatMs: 1e9, holdIntervalMs: 0 }, tuning);
    ctl.pre = { ...(mode === undefined ? {} : { mode }), ...(detect === undefined ? {} : { detect }) };
    const { GuardMixin } = require('../src/mixin') as typeof import('../src/mixin');
    const mixin = new GuardMixin({ mixinDevice: {}, ...(opts.sensorless ? { mixinDeviceInterfaces: ['VideoCamera', 'Camera'] } : opts.interfaces ? { mixinDeviceInterfaces: opts.interfaces } : {}) } as any, plugin);
    made.push(mixin);
    return { mixin, plugin, calls, dups, raw: (v: boolean) => listenCb(null, null, v), claimRaw: (data: any) => claimCb(null, null, data), claim: (...classes: string[]) => claimCb(null, null, { detections: classes.map(className => ({ className, score: 1 })) }), ctl };
}

export function loadFrames() {
    hook(false);
    return require('../src/frames') as typeof import('../src/frames');
}

export function loadProvider() {
    hook(true);
    const { GuardProvider } = require('../src/main') as typeof import('../src/main');
    return GuardProvider;
}

export const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));
export const J = Buffer.from([0xff, 0xd8, 0xff, 0xd9]);
export const idleReply = { person: { state: 'idle', confirmed: false, present: false } };
export const confirmedReply = { person: { state: 'confirmed', confirmed: true, present: true } };
export const holdingReply = { person: { state: 'holding', confirmed: true, present: true } };
