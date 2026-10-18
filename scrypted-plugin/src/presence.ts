/**
 * Per-class presence: one child Sensor device per camera and detection group (person, vehicle, animal), each an
 * OccupancySensor. Unlike the camera's own occupancy (people and moving subjects, which drives HomeKit recording),
 * these report everything the tracker holds, still and parked vehicles included, so a home dashboard or an
 * automation can tell what is there. Being Sensor devices, the HomeKit plugin publishes them as occupancy sensors.
 */
export const PRESENCE_GROUPS = ['person', 'vehicle', 'animal'] as const;
export type PresenceGroup = typeof PRESENCE_GROUPS[number];
export const PRESENCE_LABEL: Record<PresenceGroup, string> = { person: 'Person', vehicle: 'Vehicle', animal: 'Animal' };
/** device name: `{Camera} {Category} Presence`. */
export const presenceName = (cameraName: string, group: PresenceGroup) => `${cameraName} ${PRESENCE_LABEL[group]} Presence`;

export const presenceNativeId = (cameraId: string, group: PresenceGroup) => `presence:${cameraId}:${group}`;

export function parsePresenceNativeId(nativeId: string): { cameraId: string; group: PresenceGroup } | undefined {
    const m = /^presence:(.+):(person|vehicle|animal)$/.exec(nativeId || '');
    return m ? { cameraId: m[1], group: m[2] as PresenceGroup } : undefined;
}

/**
 * Which groups are present in one analysed frame, from the tracked items built for the status page.
 * A tracked item counts unless its box lies in an exclusion zone (a neighbour's parked car stays out).
 */
export function presenceOf(items: { group: string; box?: number[] }[], enabled: string[], masked: (box: number[]) => boolean): Record<PresenceGroup, boolean> {
    const out = { person: false, vehicle: false, animal: false };
    for (const i of items) {
        if (!(PRESENCE_GROUPS as readonly string[]).includes(i.group) || !enabled.includes(i.group))
            continue;
        if (Array.isArray(i.box) && i.box.length === 4 && masked(i.box))
            continue;
        out[i.group as PresenceGroup] = true;
    }
    return out;
}

/**
 * Presence after a session reset (a camera trigger starts a fresh episode on the service): the fresh trackers report nothing
 * until they confirm again (about 1-3 s), which is not the model saying the subject left. Within `graceMs` of the reset a group
 * that is shown on stays on; it can turn on at once, and goes off only on a reply after the grace that still has nothing.
 */
export function presenceAfterReset(present: Record<PresenceGroup, boolean>, shown: Record<PresenceGroup, boolean>, sinceResetMs: number, graceMs: number): Record<PresenceGroup, boolean> {
    const hold = sinceResetMs >= 0 && sinceResetMs < graceMs;
    const out = { ...present };
    for (const g of PRESENCE_GROUPS)
        out[g] = !!present[g] || (hold && !!shown[g]);
    return out;
}
