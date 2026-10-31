import sdk, { DeviceProvider, MixinProvider, ScryptedDeviceBase, ScryptedDeviceType, ScryptedInterface, Setting, Settings, SettingValue, WritableDeviceState } from '@scrypted/sdk';
import { StorageSettings } from '@scrypted/sdk/storage-settings';
import { VerifierClient } from './client';
import { DEFAULTS } from './defaults';
import { GuardMixin } from './mixin';
import { parsePresenceNativeId, PRESENCE_LABEL, presenceName, presenceNativeId, PresenceGroup } from './presence';
import { PresenceSensor } from './presence-sensor';
import { maskUrls } from './util';
import { AddonEntity, MqttAddon, slug } from './mqtt-addon';
import { renderOverview } from './overview';
import { MetricsRing, renderPerformance } from './performance';
import { DEFAULT_THRESHOLDS, resolveThresholds, THRESHOLD_TITLES, Thresholds } from './thresholds';

export const TRIGGER_CHOICES = ['Camera motion', 'Camera AI', 'Paced'];
export const PACED_SOURCE_CHOICES = ['Stream', 'Camera snapshots'];

const thresholdSetting = (k: keyof Thresholds, description: string, onPut: () => void) => ({
    group: 'Detection', title: THRESHOLD_TITLES[k], type: 'number' as const, defaultValue: DEFAULT_THRESHOLDS[k], description, onPut,
});

/** One occupancy entity per enabled group (several are on together when several are present); ids as Home Assistant names them. */
export function mqttPresenceEntities(cameraName: string, groups: string[]): AddonEntity[] {
    const base = slug(cameraName);
    return (Object.keys(PRESENCE_LABEL) as PresenceGroup[]).filter(g => groups.includes(g)).map((g): AddonEntity => ({
        key: g, name: `${PRESENCE_LABEL[g]} presence`, platform: 'binary_sensor', device_class: 'occupancy',
        ...(base ? { object_id: `${base}_${g}_occupancy` } : {}),
    }));
}

// Marker interface: lets exactly one Coral Guard mixin exist per camera (objectdetector pattern).
const MARKER = 'scrypted-coral-guard:mixin';
const HEALTH_INTERVAL_MS = 30000;

export interface GlobalConfig {
    baseUrl: string;
    token: string;
    timeoutMs: number;
    frameTimeoutMs: number;
    killSwitch: boolean;
}

export class GuardProvider extends ScryptedDeviceBase implements MixinProvider, DeviceProvider, Settings {
    // Enabling is opt-in: nothing here ever calls plugins.setMixins, so the user turns the
    // mixin on per camera in the Scrypted UI (Extensions).
    storageSettings = new StorageSettings(this, {
        baseUrl: {
            title: 'Coral service URL',
            defaultValue: 'http://127.0.0.1:8750',
            onPut: () => this.settingsChanged(),
        },
        token: {
            title: 'Token',
            type: 'password',
            description: 'Service token (X-Verifier-Token).',
            onPut: () => this.settingsChanged(),
        },
        killSwitch: {
            title: 'Disable everywhere',
            description: 'Pass raw motion through everywhere.',
            type: 'boolean',
            defaultValue: false,
            onPut: () => this.settingsChanged(),
        },
        personConfirm: thresholdSetting('personConfirm', 'Score that confirms a person.', () => this.thresholdsChanged()),
        personKeep: thresholdSetting('personKeep', 'Score that keeps a person tracked.', () => this.thresholdsChanged()),
        vehicleDetect: thresholdSetting('vehicleDetect', 'Score that detects a vehicle.', () => this.thresholdsChanged()),
        vehicleConfirm: thresholdSetting('vehicleConfirm', 'Score that confirms a moving vehicle.', () => this.thresholdsChanged()),
        animalDetect: thresholdSetting('animalDetect', 'Score that detects an animal.', () => this.thresholdsChanged()),
        triggers: {
            group: 'Detection',
            title: 'Triggers',
            description: 'What starts an analysis.',
            choices: TRIGGER_CHOICES,
            multiple: true,
            defaultValue: TRIGGER_CHOICES,
            onPut: () => { for (const m of this.active) m.applyConfig(); },
        },
        pacedSource: {
            group: 'Detection',
            title: 'Paced frames',
            description: 'Stream: own continuous decode (pixel gate, pre-roll ring). Camera snapshots: takePicture() every ~2 s, no idle decode; only for cameras with cheap snapshots. Falls back to Stream per camera after repeated snapshot failures.',
            choices: PACED_SOURCE_CHOICES,
            defaultValue: 'Stream',
            onPut: () => { for (const m of this.active) m.applyConfig(); },
        },
        aiFastPath: {
            group: 'Detection',
            title: 'Camera AI fast path',
            description: 'A person/vehicle/animal class from the camera starts verification at once at normal priority, before the camera motion edge. Same verification rules.',
            type: 'boolean',
            defaultValue: true,
        },
        fastConfirm: {
            group: 'Detection',
            title: 'Fast confirm',
            description: 'Burst frames while a trigger is unconfirmed and let the service confirm a moving person (or one the camera AI claims) after 3 frames in 0.7 s instead of 2 of 3 seconds. Off: the 2-of-3-seconds rule only.',
            type: 'boolean',
            defaultValue: true,
            onPut: () => this.thresholdsChanged(),
        },
        graceMs: {
            group: 'Detection',
            title: 'Grace after motion ends (ms)',
            description: 'An unconfirmed trigger keeps being verified this long after the camera motion ends before it is vetoed (0 = veto at once).',
            type: 'number',
            defaultValue: 3000,
            onPut: () => { for (const m of this.active) m.applyConfig(); },
        },
        stillReleaseS: {
            group: 'Detection',
            title: 'Release motion after still (s)',
            description: 'Gate mode: once every held subject has been still this long, motion goes off while occupancy stays on; a new or moving subject raises motion again. Default 170 (just under the 180 s HomeKit recording cap, so a still subject is covered by one full clip); 0 = motion held while anyone is present; at most 180.',
            type: 'number',
            defaultValue: 170,
            onPut: () => { for (const m of this.active) m.applyConfig(); },
        },
        secondLook: {
            group: 'Detection',
            title: 'Second look',
            description: 'Zoomed re-check of uncertain detections.',
            type: 'boolean',
            defaultValue: true,
            onPut: () => this.thresholdsChanged(),
        },
        animalConfirm: thresholdSetting('animalConfirm', 'Score that confirms an animal.', () => this.thresholdsChanged()),
        publishVerified: {
            group: 'Detection',
            title: 'Publish verified detections',
            description: "Gate mode: publish Coral-verified detections on the camera's ObjectDetector as verified_person / verified_vehicle / verified_animal, next to the camera's own AI classes (Scrypted does not let a mixin hide those). Off: nothing is published.",
            type: 'boolean',
            defaultValue: true,
        },
    });

    /** 'Triggers' (Detection group): empty or unparsable = all three, never none. */
    get triggers(): string[] {
        let v: any = (this.storageSettings.values as any).triggers;
        if (typeof v === 'string') { try { v = JSON.parse(v); } catch { v = v.split(','); } }
        const out = (Array.isArray(v) ? v : []).map((x: any) => String(x).trim()).filter((x: string) => TRIGGER_CHOICES.includes(x));
        return out.length ? out : TRIGGER_CHOICES.slice();
    }

    /** 'Paced frames' (Detection group, default Stream): true = paced analysis from camera snapshots. */
    get pacedSnapshots(): boolean {
        return (this.storageSettings.values as any).pacedSource === 'Camera snapshots';
    }

    /** 'Camera AI fast path' (Detection group, default on). */
    get aiFastPath(): boolean {
        const v = (this.storageSettings.values as any).aiFastPath;
        return v === undefined || v === null || v === '' ? true : v === true || v === 'true';
    }

    /** 'Publish verified detections' (Detection group, default on). */
    get publishVerified(): boolean {
        const v = (this.storageSettings.values as any).publishVerified;
        return v === undefined || v === null || v === '' ? true : v === true || v === 'true';
    }

    /** 'Fast confirm' (Detection group, default on): burst frames + the service's fast_confirm. */
    get fastConfirm(): boolean {
        const v = (this.storageSettings.values as any).fastConfirm;
        return v === undefined || v === null || v === '' ? true : v === true || v === 'true';
    }

    /** 'Grace after motion ends' in ms (default 3000, clamped 0..10000). */
    get graceMs(): number {
        const n = Number((this.storageSettings.values as any).graceMs);
        return Number.isFinite(n) && (this.storageSettings.values as any).graceMs !== '' ? Math.min(10000, Math.max(0, n)) : 3000;
    }

    /** 'Release motion after still' in s (default 170, clamped 0..180; 0 = off). */
    get stillReleaseS(): number {
        const raw = (this.storageSettings.values as any).stillReleaseS;
        const n = Number(raw);
        return Number.isFinite(n) && raw !== '' && raw !== null && raw !== undefined ? Math.min(180, Math.max(0, n)) : 170;
    }

    /** 'Second look' (Detection group, default on): sent to the service as second_look. */
    get secondLook(): boolean {
        const v = (this.storageSettings.values as any).secondLook;
        return v === undefined || v === null || v === '' ? true : v === true || v === 'true';
    }

    private thresholdWarned = '';
    /** plugin-wide detection thresholds, clamped and checked (invalid pairs fall back to the defaults, warned once). */
    get thresholds(): Thresholds {
        const v = this.storageSettings.values as any;
        const { t, invalid } = resolveThresholds({ personConfirm: v.personConfirm, personKeep: v.personKeep, vehicleDetect: v.vehicleDetect, vehicleConfirm: v.vehicleConfirm, animalDetect: v.animalDetect, animalConfirm: v.animalConfirm });
        const w = invalid.join('; ');
        if (w && w !== this.thresholdWarned) {
            this.thresholdWarned = w;
            this.console.warn(`detection thresholds: ${w}; using the defaults for that pair`);
        }
        return t;
    }

    private thresholdsChanged() {
        for (const m of this.active) {
            m.applyConfig();
            m.postConfig();
        }
    }

    private lastUptime: number | undefined;

    private active = new Set<GuardMixin>();
    private healthText = 'not checked yet';
    private healthTimer: NodeJS.Timeout;
    readonly client: VerifierClient;

    /** Presence and its summary as entities of each camera in the Scrypted MQTT bridge, when it is installed. */
    readonly mqtt: MqttAddon;

    constructor(nativeId?: string) {
        super(nativeId);
        this.client = new VerifierClient(() => this.config);
        this.mqtt = new MqttAddon('coral-guard', this.console);
        this.mqtt.start();
        this.healthTimer = setInterval(() => this.checkHealth(), HEALTH_INTERVAL_MS);
        this.healthTimer.unref?.();
        process.nextTick(() => this.checkHealth());
    }

    get config(): GlobalConfig {
        const v = this.storageSettings.values;
        return {
            baseUrl: v.baseUrl || 'http://127.0.0.1:8750',
            token: v.token || '',
            timeoutMs: Math.max(200, DEFAULTS.timeoutMs),
            frameTimeoutMs: Math.max(100, DEFAULTS.frameTimeoutMs),
            killSwitch: !!v.killSwitch,
        };
    }

    /** VAAPI slots by camera device id: one slot per camera however many mixin instances or starts it has. */
    private hwUsers = new Map<string, GuardMixin>();
    /** Reserve (idempotently) one of `max` VAAPI slots for this camera; false = decode in software. */
    acquireHw(m: GuardMixin, max: number): boolean {
        if (this.hwUsers.has(m.id)) {
            this.hwUsers.set(m.id, m); // a newer instance of the same camera takes the slot over
            return true;
        }
        if (this.hwUsers.size >= max)
            return false;
        this.hwUsers.set(m.id, m);
        return true;
    }
    /** Free the slot, only if this instance still owns it (a stale instance cannot free its successor's). */
    releaseHw(m: GuardMixin) { if (this.hwUsers.get(m.id) === m) this.hwUsers.delete(m.id); }
    get hwSlotsUsed(): number { return this.hwUsers.size; }

    register(m: GuardMixin) { this.active.add(m); }

    /** name of another enabled camera that uses the same service camera id, if any. */
    /** another camera (different device id) that uses the same service camera id, if any */
    findDuplicate(m: GuardMixin): string | undefined {
        for (const o of this.active)
            if (o !== m && String(o.id) !== String(m.id) && o.verifierId === m.verifierId)
                return `${o.name} (device ${o.id})`;
        return undefined;
    }
    unregister(m: GuardMixin) {
        this.active.delete(m);
        this.releaseHw(m);
        if (this.live.get(String(m.id)) === m) this.live.delete(String(m.id));
    }

    /** the one live mixin per device id (a re-created chain replaces the previous instance) */
    private live = new Map<string, GuardMixin>();
    isLive(m: GuardMixin): boolean { const l = this.live.get(String(m.id)); return !l || l === m; }

    private settingsChanged() {
        for (const m of this.active)
            m.applyConfig();
        this.checkHealth();
    }

    private warnedPlainHttp = false;

    /** One-time warning (never a block) when the token would travel over plain http to a non-loopback host. */
    private warnPlainHttp() {
        const { baseUrl, token } = this.config;
        if (this.warnedPlainHttp || !token || !/^http:\/\//i.test(baseUrl))
            return;
        const host = /^http:\/\/\[?([^/:\]]+)/i.exec(baseUrl)?.[1]?.toLowerCase();
        if (host && !['127.0.0.1', 'localhost', '::1'].includes(host)) {
            this.warnedPlainHttp = true;
            this.console.warn('the token is sent over plain http to a non-loopback host: use it only on a trusted network');
        }
    }

    async checkHealth() {
        this.warnPlainHttp();
        if (this.config.killSwitch) {
            this.healthText = 'kill switch on, not checking';
            return;
        }
        let ok = false;
        let reason = '';
        try {
            const h = await this.client.check(); // health plus an authenticated call (wrong token fails)
            ok = !!h.ok;
            const up = typeof h.uptime_s === 'number' ? h.uptime_s : undefined;
            if (up !== undefined && this.lastUptime !== undefined && up < this.lastUptime) {
                this.console.log('the service restarted: pushing the camera configuration again');
                for (const m of this.active) m.postConfig();
            }
            if (up !== undefined) this.lastUptime = up;
            reason = ok ? '' : `health not ok (coral ${h.model?.coral ?? 'unknown'})`;
            this.healthText = ok ? `OK \u00b7 v${h.version ?? '?'}` : reason;
        }
        catch (e: any) {
            const st = e?.status;
            reason = (st === 401 || st === 403)
                ? `the service rejected the token (HTTP ${st}): check the Token setting`
                : `health failed: ${maskUrls(String(e?.message || e))}`;
            this.healthText = reason;
        }
        for (const m of this.active)
            m.onHealth(ok, reason);
        if (typeof (this.client as any).metrics === 'function')
            await this.pollMetrics(ok);
        // no Settings event here: a push would re-render the page and wipe unsaved edits; the health line is read on page load
    }

    async getSettings(): Promise<Setting[]> {
        const stored = await this.storageSettings.getSettings();
        const isThreshold = (x: Setting) => x.group === 'Detection';
        // groups appear in the order of their first setting: Overview, Performance, Service, Detection
        return [
            { key: 'overview', group: 'Overview', title: 'Overview', type: 'html', readonly: true, value: this.overview() },
            { key: 'refreshPerformance', group: 'Performance', title: 'Refresh', type: 'button', description: 'Fetch now.' },
            { key: 'performance', group: 'Performance', title: 'Service and Coral', type: 'html', readonly: true, value: renderPerformance(Date.now(), this.perfLatest, this.perfRing, this.perfDownSince) },
            ...stored.filter(x => !isThreshold(x)).map(x => ({ ...x, group: x.group || 'Service' })), // never ungrouped: no implicit 'General'
            {
                key: 'health',
                group: 'Service',
                title: 'Service health',
                value: this.healthText,
                readonly: true,
                description: 'Checked every 30 s.',
            },
            ...stored.filter(isThreshold),
        ];
    }

    readonly perfRing = new MetricsRing();
    perfLatest: { at: number; metrics: any } | undefined;
    perfDownSince: number | undefined;

    /** Piggybacked on the health poll: the latest /v1/metrics snapshot and one ring sample per ~30 s. */
    async pollMetrics(ok: boolean) {
        const now = Date.now();
        if (ok) {
            try {
                const m = await this.client.metrics();
                this.perfLatest = { at: now, metrics: m };
                this.perfRing.push(now, m);
                this.perfDownSince = undefined;
                return;
            }
            catch (e: any) {
                if (e?.status === 404) { this.perfDownSince = undefined; return; } // older service: no metrics, not down
            }
        }
        this.perfDownSince ??= now;
    }

    /** Overview html (camera table), computed fresh (never pushed as a Settings event). */
    overview(): string {
        const cams = [...this.active].map(m => { try { return m.overviewRow(); } catch { return undefined; } }).filter(Boolean) as any[];
        cams.sort((a, b) => a.name.localeCompare(b.name));
        return renderOverview(cams);
    }

    async putSetting(key: string, value: SettingValue): Promise<void> {
        if (key === 'refreshPerformance') {
            await this.checkHealth(); // health, then /v1/metrics: the snapshot always updates, the ring keeps its spacing
            this.onDeviceEvent(ScryptedInterface.Settings, undefined); // user-initiated: one refresh
            return;
        }
        await this.storageSettings.putSetting(key, value);
        await this.checkHealth();
        this.onDeviceEvent(ScryptedInterface.Settings, undefined); // only after the user's own save
    }

    async canMixin(type: ScryptedDeviceType, interfaces: string[]): Promise<string[] | null | undefined | void> {
        if (type !== ScryptedDeviceType.Camera && type !== ScryptedDeviceType.Doorbell)
            return;
        // a camera without its own MotionSensor (e.g. a derived stream) is accepted too: it gains one, fed only by Coral Guard
        if (!interfaces.includes(ScryptedInterface.VideoCamera))
            return;
        // One mixin per camera: a marker of another provider hides us, our own marker must not
        // (otherwise an enabled extension would be hidden and unticked), as objectdetector does.
        const thisMarker = `${MARKER}:${this.id}`;
        if (interfaces.some(i => i.startsWith(MARKER) && i !== thisMarker))
            return;
        // ObjectDetector only where the camera has one: owning it hides the camera's raw events from other listeners,
        // the mixin re-publishes them (or the verified classes) itself
        return [ScryptedInterface.MotionSensor, ScryptedInterface.OccupancySensor, ScryptedInterface.Settings, ...(interfaces.includes(ScryptedInterface.ObjectDetector) ? [ScryptedInterface.ObjectDetector] : []), thisMarker];
    }

    async getMixin(mixinDevice: any, mixinDeviceInterfaces: ScryptedInterface[], mixinDeviceState: WritableDeviceState): Promise<any> {
        // Scrypted re-creates a camera's mixin chain (e.g. when another mixin or the camera plugin changes) and may call
        // getMixin before (or without) releaseMixin for the old chain: keep exactly one live instance per device.
        const id = String((mixinDeviceState as any)?.id ?? mixinDevice?.id ?? '');
        const old = id ? this.live.get(id) : undefined;
        if (old) {
            this.console.log(`${old.name || id}: replacing previous instance`);
            this.live.delete(id);
            try { await old.release(); } catch (e) { /* best effort */ }
        }
        const m = new GuardMixin({
            group: 'Coral Guard',
            groupKey: 'coral-guard',
            mixinDevice,
            mixinDeviceInterfaces,
            mixinDeviceState,
            mixinProviderNativeId: this.nativeId,
        }, this);
        this.live.set(String(m.id || id), m);
        return m;
    }

    // ---- per-class presence sensors (child devices) -------------------------------------------

    private presence = new Map<string, PresenceSensor>();
    private announced = new Map<string, string>(); // nativeId -> name last announced
    private pruned = new Map<string, string>(); // cameraId -> enabled groups last checked for stale sensors

    /** announces a camera's presence sensors (one per enabled group, repeated only when the name changes) and removes those of disabled groups. */
    ensurePresence(cameraId: string, cameraName: string, groups: string[]) {
        const mqttKey = `${cameraName}|${[...groups].sort().join(',')}`;
        if (this.mqttConfigured.get(cameraId) !== mqttKey) {
            this.mqttConfigured.set(cameraId, mqttKey);
            this.mqtt.config(cameraId, mqttPresenceEntities(cameraName, groups));
        }
        for (const g of groups) {
            if (!(g in PRESENCE_LABEL)) continue;
            const nativeId = presenceNativeId(cameraId, g as PresenceGroup), name = presenceName(cameraName, g as PresenceGroup);
            if (this.announced.get(nativeId) === name) continue;
            this.announced.set(nativeId, name);
            Promise.resolve().then(() => sdk.deviceManager.onDeviceDiscovered({
                nativeId, name, type: ScryptedDeviceType.Sensor, interfaces: [ScryptedInterface.OccupancySensor],
                info: { manufacturer: 'Coral Guard', model: `${PRESENCE_LABEL[g as PresenceGroup]} presence` },
            })).catch(e => this.console.warn(`${name}: could not announce the presence sensor`, e));
        }
        // a disabled group has no sensor: remove one left from an earlier setting (it would linger in HomeKit and Home Assistant); checked when the groups change
        const key = [...groups].sort().join(',');
        if (this.pruned.get(cameraId) === key) return;
        this.pruned.set(cameraId, key);
        const known = new Set<string>((sdk.deviceManager as any).getNativeIds?.() ?? []);
        for (const g of Object.keys(PRESENCE_LABEL) as PresenceGroup[]) {
            if (groups.includes(g)) continue;
            const nativeId = presenceNativeId(cameraId, g);
            if (!known.has(nativeId) && !this.announced.has(nativeId)) continue;
            this.announced.delete(nativeId);
            this.presence.delete(nativeId);
            Promise.resolve().then(() => sdk.deviceManager.onDeviceRemoved(nativeId))
                .catch(e => this.console.warn(`${presenceName(cameraName, g)}: could not remove the presence sensor`, e));
        }
    }

    setPresence(cameraId: string, group: PresenceGroup, on: boolean) {
        const nativeId = presenceNativeId(cameraId, group);
        if (!this.announced.has(nativeId)) return; // group disabled for this camera
        this.presenceDevice(nativeId)?.set(on);
        this.mqtt.state(cameraId, group, on);
    }

    /** name and groups each camera's MQTT entities were last built from (ensurePresence runs on every frame) */
    private mqttConfigured = new Map<string, string>();

    private presenceDevice(nativeId: string): PresenceSensor | undefined {
        if (!parsePresenceNativeId(nativeId)) return undefined;
        let d = this.presence.get(nativeId);
        if (!d) { d = new PresenceSensor(nativeId); this.presence.set(nativeId, d); }
        return d;
    }

    async getDevice(nativeId: string): Promise<any> {
        return this.presenceDevice(nativeId);
    }

    async releaseDevice(_id: string, nativeId: string): Promise<void> {
        this.presence.delete(nativeId);
        this.announced.delete(nativeId);
    }

    async releaseMixin(id: string, mixinDevice: GuardMixin): Promise<void> {
        await mixinDevice?.release(); // idempotent; unregister drops it from the live map if it is still the live one
        if (!this.live.has(id)) {
            this.mqtt.withdraw(id);
            this.mqttConfigured.delete(id);
        }
    }
}

export default GuardProvider;
