import sdk, { EventListenerRegister } from '@scrypted/sdk';

/**
 * Add-on for the Scrypted MQTT bridge (github.com/marl1w/scrypted-mqtt-ha, docs/contract.md section 10): entities
 * this plugin adds to cameras the bridge publishes, sent over Scrypted's event bus. Nothing happens when the bridge is
 * not installed. Keeps what it sent, so a bridge that (re)starts gets everything again.
 */

const ADDON_EVENT = 'ScryptedMqttAddon';
const BRIDGE_EVENT = 'ScryptedMqttBridge';
const BRIDGE_PLUGIN_ID = '@marl1w/scrypted-mqtt';
const ALIVE_MS = 30_000;
const FIND_BRIDGE_MS = 60_000;
/** the contract's limit for an image */
const MAX_IMAGE_BYTES = 1024 * 1024;

export interface AddonEntity {
    key: string;
    name: string;
    platform: 'binary_sensor' | 'sensor' | 'switch' | 'button' | 'event' | 'image';
    device_class?: string;
    unit?: string;
    state_class?: string;
    event_types?: string[];
    category?: 'diagnostic' | 'config';
    enabled?: boolean;
    icon?: string;
    object_id?: string;
}

/** Lower-case words joined by `_`, as Home Assistant derives entity ids from names. */
export function slug(s: string): string {
    return String(s).normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
        .replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

export class MqttAddon {
    private configs = new Map<string, string>();
    private states = new Map<string, Map<string, string | null>>();
    /** last image per device and key, base64 (images cross the event bus as text) */
    private images = new Map<string, Map<string, string>>();
    private bridge?: { id: string; listener: EventListenerRegister };
    /** entity platforms the bridge's last hello advertised; none before it (or from a bridge too old to say) */
    private bridgePlatforms: readonly string[] = [];
    private timers: NodeJS.Timeout[] = [];

    /** onPlatforms: the bridge's platforms changed (hello, bridge found or gone), called before everything is sent again: a plugin re-sends configs that depend on them. */
    constructor(readonly source: string, private console: Console, private onCommand?: (device: string, key: string, value: string) => void,
        private onPlatforms?: () => void) { }

    /** the bridge is installed and listened to */
    get connected(): boolean {
        return !!this.bridge;
    }

    /** entity platforms the bridge accepts, as its last hello said */
    get platforms(): readonly string[] {
        return this.bridgePlatforms;
    }

    /** whether the bridge accepts entities of this platform; false until its hello says so */
    supports(platform: AddonEntity['platform']): boolean {
        return this.bridgePlatforms.includes(platform);
    }

    start() {
        this.emit({ type: 'alive' });
        const alive = setInterval(() => this.emit({ type: 'alive' }), ALIVE_MS);
        const find = setInterval(() => this.findBridge(), FIND_BRIDGE_MS);
        alive.unref?.();
        find.unref?.();
        this.timers.push(alive, find);
        this.findBridge();
    }

    stop() {
        for (const t of this.timers) clearInterval(t);
        this.timers = [];
        this.bridge?.listener.removeListener();
        this.bridge = undefined;
        this.bridgePlatforms = [];
    }

    /** Listens to the bridge once it exists (again, if it was reinstalled), and sends it everything when first found. */
    private findBridge() {
        try {
            const id = sdk.systemManager.getDeviceById(BRIDGE_PLUGIN_ID)?.id;
            if (id === this.bridge?.id) return;
            this.bridge?.listener.removeListener();
            this.bridge = undefined;
            this.setPlatforms([]); // a new bridge says what it supports in its hello
            if (!id) return; // not installed (yet)
            const listener = sdk.systemManager.listenDevice(id, BRIDGE_EVENT, (_s, _d, msg: any) => this.onBridge(msg));
            this.bridge = { id, listener };
            this.resend();
        }
        catch (e) {
            this.console.warn(`mqtt add-on: ${(e as Error).message}`);
        }
    }

    private onBridge(msg: any) {
        if (msg?.v !== 1) return;
        if (msg.type === 'hello') {
            // a bridge too old to list its platforms supports none of the optional ones
            this.setPlatforms(Array.isArray(msg.platforms) ? msg.platforms.filter((p: unknown) => typeof p === 'string') : []);
            this.resend();
        }
        else if (msg.type === 'command' && msg.source === this.source && this.declares(msg.device, msg.key))
            this.onCommand?.(msg.device, msg.key, String(msg.value));
    }

    private setPlatforms(platforms: string[]) {
        if (platforms.join(',') === this.bridgePlatforms.join(',')) return;
        this.bridgePlatforms = platforms;
        try { this.onPlatforms?.(); }
        catch (e) { this.console.warn(`mqtt add-on: ${(e as Error).message}`); }
    }

    /** Best effort: the add-on never disturbs the plugin it lives in. */
    private emit(msg: object) {
        try {
            (sdk.deviceManager as any).onDeviceEvent?.(undefined, ADDON_EVENT, { v: 1, source: this.source, ...msg })
                ?.catch?.((e: Error) => this.console.warn(`mqtt add-on: ${e.message}`));
        }
        catch (e) {
            this.console.warn(`mqtt add-on: ${(e as Error).message}`);
        }
    }

    private declares(device: unknown, key: unknown): boolean {
        const json = typeof device === 'string' ? this.configs.get(device) : undefined;
        return !!json && typeof key === 'string' && (JSON.parse(json) as AddonEntity[]).some(e => e.key === key);
    }

    /** The entities this plugin adds to a camera; sent only when they changed. */
    config(device: string, entities: AddonEntity[]) {
        const json = JSON.stringify(entities);
        if (this.configs.get(device) === json) return;
        this.configs.set(device, json);
        const keys = new Set(entities.map(e => e.key));
        const states = this.states.get(device);
        for (const k of [...states?.keys() ?? []]) if (!keys.has(k)) states!.delete(k);
        const images = this.images.get(device);
        for (const k of [...images?.keys() ?? []]) if (!keys.has(k)) images!.delete(k);
        this.emit({ type: 'config', device, entities });
        this.sendValues(device);
    }

    withdraw(device: string) {
        if (!this.configs.delete(device)) return;
        this.states.delete(device);
        this.images.delete(device);
        this.emit({ type: 'withdraw', device });
    }

    /** ON/OFF for booleans, text, or null for unknown; sent only when it changed. */
    state(device: string, key: string, value: boolean | string | null) {
        const v = typeof value === 'boolean' ? (value ? 'ON' : 'OFF') : value;
        let states = this.states.get(device);
        if (!states) this.states.set(device, states = new Map());
        if (states.has(key) && states.get(key) === v) return;
        states.set(key, v);
        if (this.configs.has(device)) this.emit({ type: 'state', device, key, value: v });
    }

    event(device: string, key: string, eventType: string, attributes: Record<string, unknown> = {}) {
        if (this.configs.has(device)) this.emit({ type: 'event', device, key, event_type: eventType, attributes });
    }

    /** A JPEG of at most 1 MB for an image entity, or null to clear it; kept (the last per key) and sent again with everything else. */
    image(device: string, key: string, jpeg: Buffer | null) {
        if (jpeg && (jpeg.length > MAX_IMAGE_BYTES || jpeg[0] !== 0xff || jpeg[1] !== 0xd8 || jpeg[2] !== 0xff)) {
            this.console.warn(`mqtt add-on: ${key} image dropped (not a JPEG of at most 1 MB)`);
            return;
        }
        let images = this.images.get(device);
        if (!images) this.images.set(device, images = new Map());
        if (jpeg) images.set(key, jpeg.toString('base64'));
        else if (!images.delete(key)) return;
        if (this.configs.has(device)) this.emit({ type: 'image', device, key, jpeg: jpeg ? images.get(key) : null });
    }

    private sendValues(device: string) {
        for (const [key, value] of this.states.get(device) ?? []) this.emit({ type: 'state', device, key, value });
        for (const [key, jpeg] of this.images.get(device) ?? []) this.emit({ type: 'image', device, key, jpeg });
    }

    private resend() {
        this.emit({ type: 'alive' });
        for (const [device, json] of this.configs) {
            this.emit({ type: 'config', device, entities: JSON.parse(json) });
            this.sendValues(device);
        }
    }
}
