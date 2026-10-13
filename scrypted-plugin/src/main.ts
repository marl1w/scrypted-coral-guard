import { MixinProvider, ScryptedDeviceBase, ScryptedDeviceType, ScryptedInterface, Setting, Settings, SettingValue, WritableDeviceState } from '@scrypted/sdk';
import { StorageSettings } from '@scrypted/sdk/storage-settings';
import { VerifierClient } from './client';
import { DEFAULTS } from './defaults';
import { GuardMixin } from './mixin';
import { maskUrls } from './util';
import { renderOverview } from './overview';
import { MetricsRing, renderPerformance } from './performance';
import { DEFAULT_THRESHOLDS, resolveThresholds, THRESHOLD_TITLES, Thresholds } from './thresholds';

const thresholdSetting = (k: keyof Thresholds, description: string, onPut: () => void) => ({
    group: 'Detection', title: THRESHOLD_TITLES[k], type: 'number' as const, defaultValue: DEFAULT_THRESHOLDS[k], description, onPut,
});

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

export class GuardProvider extends ScryptedDeviceBase implements MixinProvider, Settings {
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
            description: 'Sent as X-Verifier-Token. Empty if the service has no token.',
            onPut: () => this.settingsChanged(),
        },
        killSwitch: {
            title: 'Disable everywhere',
            description: 'Kill switch: every camera passes raw motion through and no request is sent.',
            type: 'boolean',
            defaultValue: false,
            onPut: () => this.settingsChanged(),
        },
        personConfirm: thresholdSetting('personConfirm', 'A person is confirmed from this score (0.05-0.95).', () => this.thresholdsChanged()),
        personKeep: thresholdSetting('personKeep', 'A tracked person keeps being followed down to this score; dashed in the panel. Not above Person confirm.', () => this.thresholdsChanged()),
        vehicleDetect: thresholdSetting('vehicleDetect', 'A vehicle is detected (drawn, held) from this score. Not above Vehicle confirm.', () => this.thresholdsChanged()),
        vehicleConfirm: thresholdSetting('vehicleConfirm', 'A moving vehicle confirms from this score (2 of 3 frames).', () => this.thresholdsChanged()),
        animalDetect: thresholdSetting('animalDetect', 'An animal is detected (drawn, kept) from this score. Not above Animal confirm.', () => this.thresholdsChanged()),
        secondLook: {
            group: 'Detection',
            title: 'Second look',
            description: 'Re-checks uncertain detections with a zoomed-in pass on the Coral. Fewer false alerts, slightly more Coral load.',
            type: 'boolean',
            defaultValue: true,
            onPut: () => this.thresholdsChanged(),
        },
        animalConfirm: thresholdSetting('animalConfirm', 'An animal is confirmed from this score.', () => this.thresholdsChanged()),
    });

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

    constructor(nativeId?: string) {
        super(nativeId);
        this.client = new VerifierClient(() => this.config);
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
    findDuplicate(m: GuardMixin): string | undefined {
        for (const o of this.active)
            if (o !== m && o.verifierId === m.verifierId)
                return o.name;
        return undefined;
    }
    unregister(m: GuardMixin) { this.active.delete(m); this.releaseHw(m); }

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
            this.healthText = ok
                ? `ok, v${h.version ?? '?'}, coral ${h.model?.coral ?? '?'}, median ${h.coral_ms?.median ?? '?'} ms, ${h.cameras_active ?? '?'} cameras active`
                : reason;
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
            { key: 'refreshPerformance', group: 'Performance', title: 'Refresh', type: 'button', description: 'Fetch the service metrics now.' },
            { key: 'performance', group: 'Performance', title: 'Service and Coral', type: 'html', readonly: true, value: renderPerformance(Date.now(), this.perfLatest, this.perfRing, this.perfDownSince) },
            ...stored.filter(x => !isThreshold(x)).map(x => ({ ...x, group: x.group || 'Service' })), // never ungrouped: no implicit 'General'
            {
                key: 'health',
                group: 'Service',
                title: 'Service health',
                value: this.healthText,
                readonly: true,
                description: 'Checked every 30 s and whenever a setting here is saved.',
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
        return [ScryptedInterface.MotionSensor, ScryptedInterface.Settings, thisMarker];
    }

    async getMixin(mixinDevice: any, mixinDeviceInterfaces: ScryptedInterface[], mixinDeviceState: WritableDeviceState): Promise<any> {
        return new GuardMixin({
            group: 'Coral Guard',
            groupKey: 'coral-guard',
            mixinDevice,
            mixinDeviceInterfaces,
            mixinDeviceState,
            mixinProviderNativeId: this.nativeId,
        }, this);
    }

    async releaseMixin(id: string, mixinDevice: GuardMixin): Promise<void> {
        await mixinDevice?.release();
    }
}

export default GuardProvider;
