// Plugin page 'Overview' (read-only html setting): service / Coral / host figures from GET /v1/metrics (or /v1/health
// on an older service) and one row per camera. Computed on getSettings only; every value is escaped.

import { esc } from './status';

export interface OverviewCamera {
    name: string; mode: string; state: string; today: string; p50: string; decode: string; lastDetection: string;
    /** the camera's own name (without the sensorless mark): what mirrors refer to */
    base?: string;
    /** the stream being analysed, e.g. 'View 928x576' */
    stream?: string;
    /** the chosen stream failed: the default stream is analysed until it is back */
    fallback?: boolean;
    view?: ViewInfo;
}

/** What the camera's own plugin says about its picture: a crop of a wider camera, and mirror cameras that show the same crop in another Home. */
export interface ViewInfo { crop?: string; mirrors?: { name: string; source: string }[] }

/**
 * Reads the camera plugin's own settings (keys of a plugin that crops cameras: `viewCrop` = W:H:X:Y of the camera's own view, `crop` = a crop
 * child's description "<parent> WxH at X,Y ...", `mirrors` = ["<mirror name>=<crop entry>"]). Anything else, or another plugin, gives {}.
 */
export function viewInfoOf(settings: { key?: string; value?: unknown }[]): ViewInfo {
    const val = (k: string) => settings.find(x => x?.key === k)?.value;
    const out: ViewInfo = {};
    const vc = /^(\d+):(\d+):(\d+):(\d+)$/.exec(String(val('viewCrop') ?? '').trim());
    if (vc)
        out.crop = `crop ${vc[1]}\u00d7${vc[2]} at ${vc[3]},${vc[4]}`;
    else {
        const c = /^(.+?) (\d+)x(\d+) at (\d+),(\d+)/.exec(String(val('crop') ?? '').trim());
        if (c)
            out.crop = `crop of ${c[1]} ${c[2]}\u00d7${c[3]} at ${c[4]},${c[5]}`;
    }
    const m = val('mirrors');
    const mirrors = (Array.isArray(m) ? m : []).map(e => /^([^=]+)=(.+)$/.exec(String(e))).filter(Boolean).map(r => ({ name: r![1].trim(), source: r![2].trim() }));
    if (mirrors.length)
        out.mirrors = mirrors;
    return out;
}

const dash = '—';

/**
 * metrics: the /v1/metrics reply; health: the /v1/health reply used when metrics is missing (older service);
 * neither = the service is unavailable.
 */
/** Camera table only (service metrics live in the Performance group). */
export function renderOverview(cameras: OverviewCamera[]): string {
    const th = (t: string) => `<th style="text-align:left;font-weight:normal;opacity:.7;padding:2px 10px 2px 0">${esc(t)}</th>`;
    const td = (v: string) => `<td style="padding:2px 10px 2px 0;white-space:nowrap">${v}</td>`;
    // a mirror entry names its source crop as '<crop name>' or '<crop name> (<plugin>)'
    const shownAs = (c: OverviewCamera) => {
        const base = c.base ?? c.name;
        return cameras.flatMap(o => o.view?.mirrors ?? []).filter(m => m.source === base || m.source.startsWith(base + ' (')).map(m => m.name);
    };
    const analyses = (c: OverviewCamera) => {
        const main = `${c.decode}${c.stream && c.decode !== dash ? ` \u00b7 ${esc(c.stream)}` : ''}${c.fallback ? ` <span style="color:#e5484d">\u00b7 fallback</span>` : ''}`;
        const notes = [c.view?.crop, ...shownAs(c).map(n => `also shown as ${n}`)].filter(Boolean).map(n => esc(n!));
        return notes.length ? `${main}<div style="opacity:.7;font-size:85%">${notes.join(' \u00b7 ')}</div>` : main;
    };
    const rows = cameras.map(c => `<tr>${td(esc(c.name))}${td(c.mode)}${td(c.state)}${td(c.today)}${td(c.p50)}${td(analyses(c))}${td(c.lastDetection)}</tr>`).join('');
    const table = cameras.length
        ? `<div style="overflow-x:auto"><table style="border-collapse:collapse;color:currentColor;font-size:95%"><tr>${th('Camera')}${th('Mode')}${th('State')}${th('Today ep / conf / veto / unfl / still / re-raise')}${th('Confirm p50')}${th('Analyses')}${th('Last detection')}</tr>${rows}</table></div>`
        : `<div style="opacity:.7">No camera has Coral Guard enabled ${dash} enable it per camera under Extensions.</div>`;
    return `<div style="color:currentColor;line-height:1.5">${table}</div>`;
}
