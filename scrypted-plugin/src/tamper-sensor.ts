import { BinarySensor, ScryptedDeviceBase } from '@scrypted/sdk';

/**
 * One Tamper child device per camera (see tamper.ts): a BinarySensor, on while the camera is covered, moved or blurred.
 * The Scrypted MQTT bridge and HomeKit publish it like any other binary sensor.
 */
export const tamperNativeId = (cameraId: string) => `${cameraId}-tamper`;
export const tamperName = (cameraName: string) => `${cameraName} Tamper`;

export function parseTamperNativeId(nativeId: string): string | undefined {
    const m = /^(.+)-tamper$/.exec(nativeId || '');
    return m && !m[1].startsWith('presence:') ? m[1] : undefined;
}

export class TamperSensor extends ScryptedDeviceBase implements BinarySensor {
    constructor(nativeId: string) {
        super(nativeId);
    }

    /** writes only on change; the first write always lands (a never-set value reads as unknown). */
    set(on: boolean) {
        if (this.binaryState === undefined || this.binaryState === null || !!this.binaryState !== on)
            this.binaryState = on;
    }
}
