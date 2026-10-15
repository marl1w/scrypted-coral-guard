import { OccupancySensor, ScryptedDeviceBase } from '@scrypted/sdk';

/** One per-class presence child device (see presence.ts). */
export class PresenceSensor extends ScryptedDeviceBase implements OccupancySensor {
    constructor(nativeId: string) {
        super(nativeId);
    }

    /** writes only on change, so HomeKit and Home Assistant see one event per transition; the first write always lands (a never-set value reads as unknown). */
    set(on: boolean) {
        if (this.occupied === undefined || this.occupied === null || !!this.occupied !== on)
            this.occupied = on;
    }
}
