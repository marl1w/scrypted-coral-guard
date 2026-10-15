// Plugin-wide detection thresholds (provider settings, group 'Detection'): pushed to the service per camera and
// used by the plugin's own decisions (vehicle confirm guard) and the panel drawing.

export interface Thresholds {
    personConfirm: number;
    personKeep: number;
    vehicleDetect: number;
    vehicleConfirm: number;
    animalDetect: number;
    animalConfirm: number;
}

export const DEFAULT_THRESHOLDS: Thresholds = {
    personConfirm: 0.65, personKeep: 0.20, vehicleDetect: 0.40, vehicleConfirm: 0.60, animalDetect: 0.30, animalConfirm: 0.60,
};

export const THRESHOLD_TITLES: Record<keyof Thresholds, string> = {
    personConfirm: 'Person confirm', personKeep: 'Person keep tracking', vehicleDetect: 'Vehicle detect',
    vehicleConfirm: 'Vehicle confirm (moving)', animalDetect: 'Animal detect', animalConfirm: 'Animal confirm',
};

const clamp = (v: unknown, d: number) => { const n = Number(v); return Number.isFinite(n) ? Math.min(0.95, Math.max(0.05, n)) : d; };

/**
 * Clamp each value to 0.05..0.95; a confirm below its keep/detect pair makes that pair fall back to the defaults
 * (`invalid` names it, for a one-time warning).
 */
export function resolveThresholds(raw: Partial<Record<keyof Thresholds, unknown>>): { t: Thresholds; invalid: string[] } {
    const t = { ...DEFAULT_THRESHOLDS };
    for (const k of Object.keys(t) as (keyof Thresholds)[])
        if (raw[k] !== undefined && raw[k] !== null && raw[k] !== '')
            t[k] = clamp(raw[k], DEFAULT_THRESHOLDS[k]);
    const invalid: string[] = [];
    const pair = (c: keyof Thresholds, l: keyof Thresholds) => {
        if (t[c] < t[l]) {
            invalid.push(`${THRESHOLD_TITLES[c]} ${t[c]} < ${THRESHOLD_TITLES[l]} ${t[l]}`);
            t[c] = DEFAULT_THRESHOLDS[c]; t[l] = DEFAULT_THRESHOLDS[l];
        }
    };
    pair('personConfirm', 'personKeep');
    pair('vehicleConfirm', 'vehicleDetect');
    pair('animalConfirm', 'animalDetect');
    return { t, invalid };
}

/** Service config body for these thresholds (person keys first; the rest needs a newer service). */
export function serviceThresholds(t: Thresholds) {
    return { start_thr: t.personConfirm, cont_thr: t.personKeep, vehicle_thr: t.vehicleDetect, animal_start_thr: t.animalConfirm, animal_cont_thr: t.animalDetect };
}

/**
 * Per-camera animal overrides (camera settings 'Animal start threshold' / 'Animal keep threshold'; empty = the
 * plugin-wide value). Each must be a number with 0.05 <= keep <= start <= 0.95 (an empty one takes the plugin-wide
 * value before the check); otherwise both overrides are ignored and `invalid` says why.
 */
export function applyAnimalOverride(t: Thresholds, start: unknown, keep: unknown): { t: Thresholds; invalid?: string } {
    const blank = (v: unknown) => v === undefined || v === null || String(v).trim() === '';
    if (blank(start) && blank(keep))
        return { t };
    const s = blank(start) ? t.animalConfirm : Number(start);
    const k = blank(keep) ? t.animalDetect : Number(keep);
    if (!Number.isFinite(s) || !Number.isFinite(k) || k < 0.05 || s > 0.95 || k > s)
        return { t, invalid: `animal start ${start ?? ''} / keep ${keep ?? ''}: need 0.05 <= keep <= start <= 0.95` };
    return { t: { ...t, animalConfirm: s, animalDetect: k } };
}
