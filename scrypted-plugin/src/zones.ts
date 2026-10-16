// Per-camera exclusion zones (plugin-side only: the service tracker still sees every detection).
// Storage: key `zones`, JSON array of {name, points:[[x,y],...]} in normalised 0..1 frame coordinates (x right, y down),
// as stored by Scrypted's clippath editor (legacy 0..100 paths are divided by 100).

import { FrameReply } from './client';

export interface Zone { name: string; points: [number, number][] }

/** 0..100 legacy paths (any |coordinate| >= 2) become 0..1, as @scrypted/objectdetector does. */
export function fixLegacyPath(points: [number, number][]): [number, number][] {
    return points.some(p => Math.abs(p[0]) >= 2 || Math.abs(p[1]) >= 2) ? points.map(([x, y]) => [x / 100, y / 100]) : points;
}

function validPoints(v: unknown): [number, number][] | undefined {
    if (!Array.isArray(v)) return undefined;
    const pts = v.filter(p => Array.isArray(p) && p.length >= 2 && Number.isFinite(Number(p[0])) && Number.isFinite(Number(p[1]))).map(p => [Number(p[0]), Number(p[1])] as [number, number]);
    return pts.length === v.length ? fixLegacyPath(pts) : undefined;
}

/** Parse the stored JSON. Malformed entries are dropped and named in `warnings`; a zone with fewer than 3 points is kept (unconfigured) but never masks. */
export function parseZones(raw: unknown): { zones: Zone[]; warnings: string[] } {
    const warnings: string[] = [];
    if (raw === undefined || raw === null || raw === '') return { zones: [], warnings };
    let v: any = raw;
    if (typeof raw === 'string') {
        try { v = JSON.parse(raw); } catch { return { zones: [], warnings: ['zones: not valid JSON, ignored'] }; }
    }
    if (!Array.isArray(v)) return { zones: [], warnings: ['zones: not an array, ignored'] };
    const zones: Zone[] = [];
    const seen = new Set<string>();
    v.forEach((z: any, i: number) => {
        const name = typeof z?.name === 'string' && z.name.trim() ? z.name.trim().slice(0, 40) : undefined;
        const points = validPoints(z?.points ?? []);
        if (!name || !points || seen.has(name)) { warnings.push(`zones: entry ${i + 1} is malformed or duplicated, ignored`); return; }
        seen.add(name);
        zones.push({ name, points });
    });
    return { zones, warnings };
}

export function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const [xi, yi] = poly[i], [xj, yj] = poly[j];
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
            inside = !inside;
    }
    return inside;
}

/** Shoelace area of a polygon (absolute). */
export function polygonArea(poly: [number, number][]): number {
    let a = 0;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++)
        a += (poly[j][0] + poly[i][0]) * (poly[j][1] - poly[i][1]);
    return Math.abs(a) / 2;
}

/** Share (0..1) of the box [x0,y0,x1,y1] that lies OUTSIDE the union of the zones, sampled on a grid x grid lattice. */
export function visibleFraction(box: number[], zones: Zone[], grid = 64): number {
    const polys = zones.map(z => z.points).filter(p => p.length >= 3);
    if (!polys.length) return 1;
    const x0 = Math.min(box[0], box[2]), x1 = Math.max(box[0], box[2]), y0 = Math.min(box[1], box[3]), y1 = Math.max(box[1], box[3]);
    if (!(x1 > x0) || !(y1 > y0)) return 1;
    let outside = 0;
    for (let i = 0; i < grid; i++) {
        const y = y0 + ((i + 0.5) / grid) * (y1 - y0);
        for (let j = 0; j < grid; j++) {
            const x = x0 + ((j + 0.5) / grid) * (x1 - x0);
            if (!polys.some(p => pointInPolygon(x, y, p))) outside++;
        }
    }
    return outside / (grid * grid);
}

/**
 * A detection is excluded when the point it STANDS on, the bottom centre of its box (feet, wheels), lies in a zone, or when the box
 * lies almost entirely inside the zones (visible share below 1 - minVisible). Judging by the standing point keeps a person walking on
 * a path whose head and shoulders overlap a background zone (fence, street behind) valid; the old area-share rule masked them.
 */
export function isMasked(box: number[], zones: Zone[], minVisible: number): boolean {
    if (!Array.isArray(box) || box.length !== 4) return false;
    const polys = zones.map(z => z.points).filter(p => p.length >= 3);
    if (!polys.length) return false;
    const x0 = Math.min(box[0], box[2]), x1 = Math.max(box[0], box[2]), y0 = Math.min(box[1], box[3]), y1 = Math.max(box[1], box[3]);
    const fx = (x0 + x1) / 2, fy = y1 - 0.02 * (y1 - y0);
    if (polys.some(p => pointInPolygon(fx, fy, p))) return true;
    return visibleFraction(box, zones) < 1 - minVisible;
}

/**
 * The reply as the plugin's decisions see it with exclusion zones: masked detections removed, and the service's verdict per group kept
 * only for subjects outside the zones. When the service reports its tracks with boxes (person `tracks_detail`, animal and vehicle
 * `tracks`), a group is confirmed / present / moving when the service says so AND one of its held tracks stands outside the zones:
 * the service's own rules (fast confirm, rule B, holds through missed frames) apply unchanged. Without track boxes (old service) the
 * old frame rule applies: person.confirmed needs a valid person box at or above `personConfirm` in 2 of the last 3 replies (`ticks`,
 * updated here) and a valid one at or above `personKeep` now; present, animal and vehicle need a valid box of their group in this reply.
 * Without zones the reply is returned unchanged.
 */
export function maskReply(r: FrameReply, zones: Zone[], minVisible: number, th: { personConfirm: number; personKeep: number; animalDetect: number }, ticks: boolean[], groupOf: (d: any) => string | undefined): FrameReply {
    if (!zones.some(z => z.points.length >= 3)) return r;
    const valid = (r.detections || []).filter(d => !isMasked(d.box, zones, minVisible));
    const g = (d: any) => groupOf(d);
    const tick = valid.some(d => g(d) === 'person' && d.score >= th.personConfirm);
    ticks.push(tick);
    while (ticks.length > 3) ticks.shift();
    // undefined = the service sent no track boxes for this group: fall back to this reply's detections
    const outside = (tracks: { box?: number[] }[] | undefined, moving = false): boolean | undefined => {
        const boxed = (tracks || []).filter(t => Array.isArray(t.box) && t.box.length === 4 && (!moving || (t as any).state === 'moving'));
        if (!tracks?.length || boxed.length < tracks.filter(t => !moving || (t as any).state === 'moving').length) return undefined;
        return boxed.some(t => !isMasked(t.box!, zones, minVisible));
    };
    const keep = valid.some(d => g(d) === 'person' && d.score >= th.personKeep);
    const animal = valid.some(d => g(d) === 'animal' && d.score >= th.animalDetect);
    const vehicle = valid.some(d => g(d) === 'vehicle');
    const out: any = { ...r, detections: valid };
    if (r.person) {
        const p = outside(r.person.tracks_detail);
        out.person = p === undefined
            ? { ...r.person, confirmed: !!r.person.confirmed && keep && ticks.filter(Boolean).length >= 2, present: !!r.person.present && keep }
            : { ...r.person, confirmed: !!r.person.confirmed && p, present: !!r.person.present && p };
    }
    if (r.animal) {
        const a = outside(r.animal.tracks) ?? animal;
        out.animal = { ...r.animal, confirmed: !!r.animal.confirmed && a, present: !!r.animal.present && a };
    }
    if (r.vehicle) out.vehicle = { ...r.vehicle, moving: !!r.vehicle.moving && (outside(r.vehicle.tracks, true) ?? vehicle) };
    return out as FrameReply;
}
