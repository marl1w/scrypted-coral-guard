import { isMasked, Zone } from './zones';
// Display model of one service reply: which boxes to draw (person, vehicle, animal) and the counts for the
// caption. Display only: nothing here takes part in a decision (that is the Episode state machine).

import { FrameReply, groupOf } from './client';
import type { Box, FrameEvent } from './status';

/** person: solid from here; personKeep: dashed from here (below: not drawn); vehicle / animal: drawn from here. */
export interface ViewThresholds { person: number; personKeep?: number; vehicle: number; animal: number; /** groups enabled in Detect: others are not drawn or counted (undefined = all) */ enabled?: string[]; /** exclusion zones: masked detections are drawn dimmed '(masked)' and never counted */ zones?: Zone[]; minVisible?: number }

const iou = (a: number[], b: number[]): number => {
    const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
    const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    if (w <= 0 || h <= 0) return 0;
    const inter = w * h;
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
};

/**
 * Track state of a vehicle detection: the state of the best overlapping track (IoU > 0.3), 'still' when none overlaps (only a
 * matching 'moving' track makes it eligible). undefined when the reply has no vehicle tracks.
 */
export function vehicleState(d: { box: number[] }, tracks: NonNullable<FrameReply['vehicle']>['tracks']): string | undefined {
    if (!tracks || !tracks.length)
        return undefined;
    let best: { v: number; state: string } | undefined;
    for (const t of tracks) {
        if (!Array.isArray(t.box) || t.box.length !== 4 || !Array.isArray(d.box) || d.box.length !== 4) continue;
        const v = iou(d.box, t.box);
        if (v > 0.3 && (!best || v > best.v)) best = { v, state: String(t.state) };
    }
    return best ? best.state : 'still';
}

/** Is this vehicle detection ineligible (not on a 'moving' track)? With no vehicle tracks in the reply it is eligible. */
export function isParked(d: { box: number[] }, tracks: NonNullable<FrameReply['vehicle']>['tracks']): boolean {
    const st = vehicleState(d, tracks);
    return st !== undefined && st !== 'moving';
}

export function viewOfReply(reply: FrameReply, th: ViewThresholds): { frame: Omit<FrameEvent, 'atMs'>; boxes: Box[]; present: boolean } {
    const boxes: Box[] = [];
    let persons = 0, best: number | undefined, weakBest: number | undefined, vehicles = 0, parked = 0, animals = 0, weakVehicles = 0;
    for (const d of reply.detections || []) {
        const g = groupOf(d);
        if (!g || !Array.isArray(d.box) || d.box.length !== 4) continue;
        if (th.enabled && !th.enabled.includes(g)) continue;
        if (th.zones?.length && isMasked(d.box, th.zones, th.minVisible ?? 0.6)) {
            if (g === 'person' ? d.score >= (th.personKeep ?? 0) : g === 'vehicle' ? (d.score >= th.vehicle || d.weak === true) : d.score >= th.animal)
                boxes.push({ x0: d.box[0], y0: d.box[1], x1: d.box[2], y1: d.box[3], score: d.score, group: g, cls: d.cls, masked: true });
            continue;
        }
        const [x0, y0, x1, y1] = d.box;
        if (g === 'person') {
            if (th.personKeep !== undefined && d.score < th.personKeep) continue;
            const weak = d.score < th.person;
            if (weak) weakBest = Math.max(weakBest ?? 0, d.score);
            else { persons++; best = Math.max(best ?? 0, d.score); }
            boxes.push({ x0, y0, x1, y1, score: d.score, weak, group: 'person', cls: d.cls });
        }
        else if (g === 'vehicle' && d.score < th.vehicle && d.weak === true) {
            // display-only candidate flagged by the service (old services never send it): drawn, counted apart, never decides
            weakVehicles++;
            boxes.push({ x0, y0, x1, y1, score: d.score, weak: true, group: 'vehicle', cls: d.cls });
        }
        else if (g === 'vehicle' && d.score >= th.vehicle) {
            const p = isParked(d, reply.vehicle?.tracks);
            const still = p && vehicleState(d, reply.vehicle?.tracks) === 'still';
            vehicles++;
            if (p) parked++;
            boxes.push({ x0, y0, x1, y1, score: d.score, group: 'vehicle', cls: d.cls, parked: p && !still, ...(still ? { still: true } : {}) });
        }
        else if (g === 'animal' && d.score >= th.animal) {
            animals++;
            boxes.push({ x0, y0, x1, y1, score: d.score, group: 'animal', cls: d.cls });
        }
    }
    return {
        frame: {
            personCount: persons, bestScore: best, candidateScore: weakBest, vehicles, parkedVehicles: parked, weakVehicles, animals,
            totalMs: reply.total_ms, inferMs: reply.infer_ms, queueMs: reply.queue_ms,
        },
        boxes,
        present: !!reply.person?.present || !!reply.vehicle?.moving || !!reply.animal?.present,
    };
}
