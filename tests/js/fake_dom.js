// Tiny fake browser for driving the real page scripts under Node (no npm deps).
// Elements keep their innerHTML as a string; querySelector(sel) on an element
// returns one stable stub per selector so state set on it can be inspected.
// fetch is routed by URL path to handler functions supplied by the harness.
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function createEnv({ root, routes, brokenImages = new Set(), mood = 'happy', mode = 'match' }) {
  const registry = [];
  const roots = new Set();
  const doc = { activeElement: null, listeners: {} };
  const state = { mood, mode, requests: [], toasts: [] };
  const session = {};

  function el(tag, id) {
    const set = new Set();
    const node = {
      tag, id, children: [], parent: null, dataset: {}, attrs: {}, listeners: {}, subs: {},
      style: { setProperty() {} }, inert: false, disabled: false, _html: '', _text: '', type: '', tabIndex: -1, onclick: null,
      classList: {
        add: (...c) => c.forEach((x) => set.add(x)), remove: (...c) => c.forEach((x) => set.delete(x)),
        toggle: (c, on) => { const v = on === undefined ? !set.has(c) : !!on; if (v) set.add(c); else set.delete(c); return v; },
        contains: (c) => set.has(c),
      },
      set className(v) { set.clear(); String(v).split(/\s+/).filter(Boolean).forEach((c) => set.add(c)); },
      get className() { return [...set].join(' '); },
      set innerHTML(v) { this._html = String(v); this._text = ''; this.children.forEach((c) => { c.parent = null; }); this.children = []; this.subs = {}; },
      get innerHTML() { return this._html; },
      set textContent(v) { this._text = String(v); this._html = this._text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); },
      get textContent() { return this._text || this._html.replace(/<[^>]*>/g, ''); },
      appendChild(c) { c.parent = this; this.children.push(c); return c; },
      remove() { if (this.parent) { this.parent.children = this.parent.children.filter((c) => c !== this); this.parent = null; } },
      setAttribute(k, v) { this.attrs[k] = String(v); },
      getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
      addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
      dispatch(t, ev = {}) { (this.listeners[t] || []).forEach((f) => f(ev)); },
      focus() { doc.activeElement = this; },
      querySelector(sel) { return (this.subs[sel] = this.subs[sel] || Object.assign(el('sub'), { selector: sel, owner: this })); },
      querySelectorAll(sel) {
        if (sel === 'button') return this.children.filter((c) => c.tag === 'button');
        if (sel.startsWith('.')) return this.children.filter((c) => c.classList.contains(sel.slice(1)));
        return [];
      },
      get isConnected() { let n = this; while (n.parent) n = n.parent; return roots.has(n); },
    };
    registry.push(node);
    return node;
  }

  const byId = {};
  function getElementById(id) {
    if (!byId[id]) { byId[id] = el('div', id); roots.add(byId[id]); }
    return byId[id];
  }
  const modeToggle = getElementById('modeToggle');
  for (const m of ['match', 'uplift']) { const b = el('button'); b.dataset.mode = m; modeToggle.appendChild(b); }

  // '.cls' and '.cls[data-x="v"]' selectors over connected elements
  function matches(node, sel) {
    const m = sel.match(/^\.([\w-]+)(?:\[data-([\w-]+)="([^"]*)"\])?$/);
    if (!m || !node.isConnected || !node.classList.contains(m[1])) return false;
    if (!m[2]) return true;
    const key = m[2].replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    return String(node.dataset[key]) === m[3];
  }

  class FakeImage {
    set src(url) {
      this._src = url;
      if (!url) return;
      setTimeout(() => {
        if (brokenImages.has(url)) { this.naturalWidth = 0; if (this.onerror) this.onerror(); }
        else { this.naturalWidth = 300; this.naturalHeight = 450; if (this.onload) this.onload(); }
      }, 0);
    }
    get src() { return this._src; }
  }

  const themeHandlers = [];
  const MOODS = ['happy', 'sad', 'angry', 'chill', 'energetic', 'stressed'];
  const ctx = {
    console, Math, JSON, Number, String, Set, Map, Array, Object, Promise, Error, TypeError, isNaN, parseFloat,
    encodeURIComponent, decodeURIComponent, encodeURI, Date,
    setTimeout, clearTimeout, requestAnimationFrame: (f) => setTimeout(f, 0), Image: FakeImage,
    localStorage: { getItem: () => null, setItem() {} },
    sessionStorage: { getItem: (k) => (k in session ? session[k] : null), setItem: (k, v) => { session[k] = String(v); } },
    navigator: {},
    getSelection: () => '',
    document: {
      getElementById,
      createElement: (tag) => el(tag),
      querySelector: (sel) => registry.find((n) => matches(n, sel)) || null,
      querySelectorAll: (sel) => registry.filter((n) => matches(n, sel)),
      addEventListener: (t, f) => { (doc.listeners[t] = doc.listeners[t] || []).push(f); },
      get activeElement() { return doc.activeElement; },
    },
    fetch: async (url, opts = {}) => {
      const u = new URL(url, 'http://moodcast.test');
      const body = opts.body ? JSON.parse(opts.body) : null;
      state.requests.push({ path: u.pathname, query: Object.fromEntries(u.searchParams), body });
      const handler = routes[u.pathname];
      if (!handler) return { ok: false, status: 404, json: async () => ({ error: 'not found' }) };
      const out = await handler({ query: Object.fromEntries(u.searchParams), body });
      if (out instanceof Error) throw out;
      return { ok: (out.status || 200) < 400, status: out.status || 200, json: async () => out.json };
    },
    // mood-theme.js / landing-effects.js stand-ins
    MOOD_ORDER: MOODS,
    MOOD_THEME: Object.fromEntries(MOODS.map((m) => [m, { emoji: '*', label: m[0].toUpperCase() + m.slice(1), colors: { glow: '#fff' } }])),
    getCurrentMood: () => state.mood, getCurrentMode: () => state.mode,
    getMoodTheme: (m) => ({ emoji: '*', label: (m || state.mood)[0].toUpperCase() + (m || state.mood).slice(1) }),
    setMoodTheme: (m) => { state.mood = m; themeHandlers.forEach((h) => h({ detail: { mood: m } })); },
    setMode: (m) => { state.mode = m; },
    onMoodThemeChange: (h) => { themeHandlers.push(h); },
    initParticles() {}, observeReveal() {},
  };
  ctx.window = ctx;
  vm.createContext(ctx);

  return {
    ctx, state, getElementById, el, doc, session,
    load(files) { for (const f of files) vm.runInContext(fs.readFileSync(path.join(root, f), 'utf8'), ctx, { filename: f }); },
    run(code) { return vm.runInContext(code, ctx); },
    start() { (doc.listeners.DOMContentLoaded || []).forEach((f) => f()); },
    keydown(ev) { (doc.listeners.keydown || []).forEach((f) => f(ev)); },
    settle: (ms = 40) => new Promise((r) => setTimeout(r, ms)),
  };
}

module.exports = { createEnv };
