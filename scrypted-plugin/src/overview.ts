// Plugin page 'Overview' (read-only html setting): service / Coral / host figures from GET /v1/metrics (or /v1/health
// on an older service) and one row per camera. Computed on getSettings only; every value is escaped.

import { esc } from './status';

export interface OverviewCamera { name: string; mode: string; state: string; today: string; p50: string; decode: string; lastDetection: string }

const dash = '—';

/**
 * metrics: the /v1/metrics reply; health: the /v1/health reply used when metrics is missing (older service);
 * neither = the service is unavailable.
 */
/** Camera table only (service metrics live in the Performance group). */
export function renderOverview(cameras: OverviewCamera[]): string {
    const th = (t: string) => `<th style="text-align:left;font-weight:normal;opacity:.7;padding:2px 10px 2px 0">${esc(t)}</th>`;
    const td = (v: string) => `<td style="padding:2px 10px 2px 0;white-space:nowrap">${v}</td>`;
    const rows = cameras.map(c => `<tr>${td(esc(c.name))}${td(c.mode)}${td(c.state)}${td(c.today)}${td(c.p50)}${td(c.decode)}${td(c.lastDetection)}</tr>`).join('');
    const table = cameras.length
        ? `<div style="overflow-x:auto"><table style="border-collapse:collapse;color:currentColor;font-size:95%"><tr>${th('Camera')}${th('Mode')}${th('State')}${th('Today ep / conf / veto / unfl / still / re-raise')}${th('Confirm p50')}${th('Decode')}${th('Last detection')}</tr>${rows}</table></div>`
        : `<div style="opacity:.7">No camera has Coral Guard enabled ${dash} enable it per camera under Extensions.</div>`;
    return `<div style="color:currentColor;line-height:1.5">${table}</div>`;
}
