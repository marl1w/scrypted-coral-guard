// Executes the review page's script in a minimal fake DOM (node vm, stubbed fetch) and reports what happened: Load more / empty messages for the three lists
// and the Filter menu (open/close, keyboard, choices, summary words, reset, URL hash). usage: node ui_harness.js <path to ui.html>  -> JSON on stdout
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const script = /<script>([\s\S]*)<\/script>/.exec(html)[1];
const head = html.slice(0, html.indexOf("<script>"));

function boot(hash) {
  const doc = { _l: {}, activeElement: null };
  class El {
    constructor(tag) { this.tagName = String(tag).toUpperCase(); this.children = []; this.hidden = false; this.className = ""; this._text = ""; this.style = {}; this.dataset = {}; this.value = ""; this.attrs = {}; this._l = {}; this.parent = null;
      const self = this; this.classList = { add: c => { if (!self.className.split(" ").includes(c)) self.className = (self.className + " " + c).trim() }, remove: c => { self.className = self.className.split(" ").filter(x => x !== c).join(" ") }, contains: c => self.className.split(" ").includes(c) }; }
    get textContent() { return this._text + this.children.map(c => c.textContent).join("") }
    set textContent(v) { this._text = String(v); this.children = [] }
    appendChild(c) { c.parent = this; this.children.push(c); return c }
    replaceChildren(...c) { this.children = c; this._text = "" }
    setAttribute(k, v) { this.attrs[k] = String(v) }
    getAttribute(k) { return this.attrs[k] === undefined ? null : this.attrs[k] }
    insertRow() { const r = new El("tr"); this.children.push(r); r.insertCell = () => { const c = new El("td"); r.children.push(c); return c }; return r }
    querySelectorAll() { return [] }
    addEventListener(t, f) { (this._l[t] = this._l[t] || []).push(f) }
    focus() { doc.activeElement = this }
    contains(o) { for (let n = o; n; n = n.parent) if (n === this) return true; return false }
    set innerHTML(v) { throw new Error("innerHTML is not allowed") }
  }
  const ids = {};
  for (const m of head.matchAll(/<(\w+)([^>]*)>/g)) {
    const id = /\sid="([^"]+)"/.exec(m[2]); if (!id) continue;
    const e = new El(m[1]); e.id = id[1]; e.hidden = /\shidden(\s|$|=)/.test(m[2]); ids[id[1]] = e;
  }
  const initial = { more: ids.more.hidden, rToMore: ids.rToMore.hidden, rDoMore: ids.rDoMore.hidden, empty: ids.empty.hidden, rToEmpty: ids.rToEmpty.hidden, rDoEmpty: ids.rDoEmpty.hidden };
  const state = { responder: () => ({ total: 0, items: [], cameras: [], rows: [], totals: {}, days: [], groups: [] }), calls: [], hash: [] };
  Object.assign(doc, { querySelector: s => (s[0] === "#" ? ids[s.slice(1)] || new El("x") : new El("x")), querySelectorAll: () => [], createElement: t => new El(t),
    createTextNode: t => { const e = new El("#text"); e._text = String(t); return e }, createElementNS: (ns, t) => new El(t),
    addEventListener(t, f) { (this._l[t] = this._l[t] || []).push(f) }, removeEventListener(t, f) { this._l[t] = (this._l[t] || []).filter(x => x !== f) } });
  const ctx = vm.createContext({ document: doc, console, URLSearchParams, Promise, Date, Math, JSON, Number, String, Array, Set, Object, RegExp, Image: function () {}, setTimeout, encodeURIComponent,
    location: { hash: hash || "" }, history: { replaceState: (a, b, h) => state.hash.push(String(h)) },
    fetch: url => { state.calls.push(String(url)); const body = state.responder(String(url)); return Promise.resolve({ ok: true, json: () => Promise.resolve(body) }) } });
  return { El, ids, doc, ctx, state, initial, script: () => vm.runInContext(script, ctx) };
}
const tick = () => new Promise(r => setTimeout(r, 5));
const meta = i => ({ id: "ep-" + String(i).padStart(6, "0"), camera: "cam", display_name: "Cam", start_ms: 1000 + i, end_ms: 2000 + i, mode: "shadow", kind: "camera", outcome: "confirmed", groups: ["person"], label: null, frames: [], status: "closed" });
const page = (n, total) => ({ total, items: Array.from({ length: n }, (_, i) => meta(i)) });
const SUMMARY = { cameras: [{ camera: "garden_1", display_name: "Garden" }, { camera: "door_2", display_name: "Door" }], totals: { events: 9, closed: 8, ongoing: 1, reviewable: 2, reviewed: 6, correct: 4, false_alarm: 1, missed: 1, wrong_class: 0, precision: 0.6667 }, rows: [], days: [], groups: [] };

(async () => {
  const out = { scenarios: {} };
  const B = boot(""); B.state.responder = u => (u.includes("summary") ? SUMMARY : page(0, 0));
  B.script(); await tick(); await tick();
  const { ids, ctx, state } = B;
  const run = async code => { await vm.runInContext(code, ctx); await tick(); await tick() };
  const ev = code => vm.runInContext(code, ctx);
  out.initial = B.initial;

  // ---- Load more / empty message per list ----
  const lists = [
    ["events", "#more", "empty", "load(true)", "load(false)", u => u.startsWith("v1/journal?") && !u.includes("labelled") && !u.includes("summary") && !u.includes("next"), 24],
    ["to_review", "#rToMore", "rToEmpty", 'reLoad("to",true)', 'reLoad("to",false)', u => u.includes("labelled=false"), 12],
    ["reviewed", "#rDoMore", "rDoEmpty", 'reLoad("do",true)', 'reLoad("do",false)', u => u.includes("labelled=true"), 12],
  ];
  for (const [name, more, emptyId, first, next, match, limit] of lists) {
    const btn = ids[more.slice(1)], empty = ids[emptyId], res = {};
    btn.hidden = false;
    const set = (n, t) => { state.responder = u => (match(u) ? page(n, t) : u.includes("summary") ? SUMMARY : {}) };
    set(0, 0); await run(first); res.empty = { hidden: btn.hidden, emptyShown: !empty.hidden };
    set(limit, limit * 2 + 5); await run(first); res.firstPage = { hidden: btn.hidden, emptyShown: !empty.hidden };
    set(limit, limit * 2 + 5); await run(next); res.middlePage = { hidden: btn.hidden };
    set(5, limit * 2 + 5); await run(next); res.lastPage = { hidden: btn.hidden };
    set(limit, limit); await run(first); res.exactlyOnePage = { hidden: btn.hidden };
    set(0, 0); await run(first); await run(next); await run(next); res.emptyThenMoreClicks = { hidden: btn.hidden, emptyShown: !empty.hidden };
    out.scenarios[name] = res;
  }
  state.responder = u => (u.includes("labelled=false") ? page(0, 0) : u.includes("summary") ? SUMMARY : {});
  await run('reLoad("to",true)'); await run('reLoad("to",false)');
  out.toReviewGridChildren = ids.rToGrid.children.length;

  // ---- Filter menu ----
  state.responder = u => (u.includes("summary") ? SUMMARY : page(0, 0));
  await run("cams()");
  const bar = ids.evBar.children[0], wrap = bar.children[0], btn = wrap.children[0], menu = wrap.children[1], stateEl = bar.children[1];
  const labelsOf = () => menu.children.filter(c => c.tagName === "BUTTON").map(b => b.textContent);
  const fire = (el, type, e) => { e.preventDefault = () => {}; for (const f of el._l[type] || []) f(e) };
  const m = out.menu = {};
  m.initial = { menuHidden: menu.hidden, expanded: btn.getAttribute("aria-expanded"), stateHidden: stateEl.hidden, text: btn.textContent };
  btn.onclick(); m.open = { menuHidden: menu.hidden, expanded: btn.getAttribute("aria-expanded"), focusOnChecked: B.doc.activeElement && B.doc.activeElement.getAttribute("aria-checked") === "true", labels: labelsOf() };
  m.groupsEvents = JSON.parse(ev('JSON.stringify(groupsFor("ev").map(g=>g.id))')); m.groupsReviewed = JSON.parse(ev('JSON.stringify(groupsFor("re").map(g=>g.id))'));
  m.groupsSummary = JSON.parse(ev('JSON.stringify(groupsFor("su").map(g=>g.id))')); m.groupsValidate = JSON.parse(ev('JSON.stringify(groupsFor("va").map(g=>g.id))'));
  m.cameraOptions = JSON.parse(ev('JSON.stringify(groupsFor("ev")[0].options)'));
  // keyboard
  const items = []; for (const c of menu.children) { if (c.tagName === "BUTTON") items.push(c); else if (c.tagName === "LABEL") items.push(c.children[1]) }      // buttons and date inputs, in order
  const first = items[0], last = items[items.length - 1];
  fire(wrap, "keydown", { key: "ArrowDown" }); const afterDown = B.doc.activeElement;
  fire(wrap, "keydown", { key: "End" }); const atEnd = B.doc.activeElement;
  fire(wrap, "keydown", { key: "ArrowDown" }); const wrapDown = B.doc.activeElement;                 // past the last item: back to the first
  fire(wrap, "keydown", { key: "Home" }); const atHome = B.doc.activeElement;
  fire(wrap, "keydown", { key: "ArrowUp" }); const wrapUp = B.doc.activeElement;                      // above the first item: the last
  m.keyboard = { down: afterDown === items[1], end: atEnd === last, wrapDown: wrapDown === first, home: atHome === first, wrapUp: wrapUp === last, items: items.length };
  fire(wrap, "keydown", { key: "Escape" }); m.escape = { menuHidden: menu.hidden, focusOnButton: B.doc.activeElement === btn };
  btn.onclick(); fire(wrap, "keydown", { key: "Tab" }); m.tab = { menuHidden: menu.hidden };
  btn.onclick(); for (const f of B.doc._l.pointerdown || []) f({ target: ids.grid }); m.outside = { menuHidden: menu.hidden };
  // choices
  const pick = (text, nth = 0) => { btn.onclick(); const b = menu.children.filter(c => c.tagName === "BUTTON" && c.textContent === text)[nth]; b.onclick(); };
  state.calls.length = 0; state.hash.length = 0;
  pick("Garden"); await tick(); await tick();
  m.camera = { filter: ev("F.ev.camera"), menuHidden: menu.hidden, request: state.calls.some(u => u.startsWith("v1/journal?") && u.includes("camera=garden_1")), hash: state.hash[state.hash.length - 1] || "", summary: stateEl.textContent };
  pick("person"); pick("7 days");
  m.words = { summary: stateEl.textContent, stateHidden: stateEl.hidden, dotShown: !btn.children[2].hidden, filter: ev("JSON.stringify(F.ev)"), hash: state.hash[state.hash.length - 1],
              requestHasAll: state.calls.some(u => u.includes("camera=garden_1") && u.includes("group=person") && u.includes("from=")) };
  pick("confirmed");
  m.outcome = { filter: ev("F.ev.outcome"), request: state.calls.some(u => u.includes("outcome=confirmed")) };
  btn.onclick(); const labelsBefore = labelsOf(); m.hasReset = labelsBefore.includes("Reset"); fire(wrap, "keydown", { key: "Escape" });
  pick("Reset"); await tick(); await tick();
  m.reset = { filter: ev("JSON.stringify(F.ev)"), stateHidden: stateEl.hidden, dotHidden: btn.children[2].hidden, summary: stateEl.textContent, hash: state.hash[state.hash.length - 1] };
  // date fields override the period
  btn.onclick(); const dates = menu.children.filter(c => c.tagName === "LABEL"); dates[0].children[1].value = "2026-10-01"; dates[0].children[1].onchange();
  m.dates = { filter: ev("F.ev.from"), summary: stateEl.textContent }; fire(wrap, "keydown", { key: "Escape" }); pick("Reset");
  // summary tab: its default period is 7 days (not shown), another choice is
  m.summaryDefault = ev('summaryOf("su")'); ev('F.su.period="14d"'); m.summaryChoice = ev('summaryOf("su")'); ev('F.su.period="7d"');
  // the Reviewed counts line is ONE compact line
  await run("reAll()"); m.countsLine = ids.rTot.textContent;
  // validate and summary requests use their own filters
  ev('F.va.camera="door_2";F.va.category="vehicle"'); state.calls.length = 0; await run("vLoad(null,false)");
  m.validateRequest = state.calls.find(u => u.startsWith("v1/journal/next")) || "";
  ev('F.va.camera="";F.va.category=""');

  // ---- hash persistence: a reload keeps the filters and the tab ----
  const B2 = boot("#t=re&re.camera=door_2&re.label=missed&re.period=14d&su.category=animal&ev.from=2026-10-01&ev.period=bogus&va.from=garbage");
  B2.state.responder = u => (u.includes("summary") ? SUMMARY : page(0, 0));
  B2.script(); await tick(); await tick();
  const c2 = code => vm.runInContext(code, B2.ctx);
  out.hash = { tab: c2("TAB"), re: JSON.parse(c2("JSON.stringify(F.re)")), su: JSON.parse(c2("JSON.stringify(F.su)")), ev: JSON.parse(c2("JSON.stringify(F.ev)")), va: JSON.parse(c2("JSON.stringify(F.va)")),
    reviewedShown: !B2.ids.reView.hidden, eventsShown: !B2.ids.evView.hidden, summaryText: c2('summaryOf("re")'),
    request: B2.state.calls.some(u => u.includes("labelled=true") && u.includes("camera=door_2") && u.includes("label=missed")) };
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1) });
