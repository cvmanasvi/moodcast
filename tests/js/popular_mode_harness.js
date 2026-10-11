// Drives the real static/js/ui-utils.js + track-identity.js + music.js through the
// Popular Artists ON/OFF switch and its Refresh, using the tiny fake DOM in fake_dom.js.
// Usage: node tests/js/popular_mode_harness.js <project root> <fixtures.json>
//   fixtures.json: { normal, normalRefresh, popular, popularRefreshes: [...] } -- /api/recommend
//   and /api/popular responses. Each popular request carrying `exclude` is answered by the next
//   entry of popularRefreshes, which may be a response body or {"status": 500, "error": "..."}.
// Every response is scripted, so nothing depends on random draws.
// Prints a JSON summary; tests/test_popular_frontend.py makes the assertions.
const fs = require('fs');
const { createEnv } = require('./fake_dom');

const [root, fixturesPath] = process.argv.slice(2);
const fx = JSON.parse(fs.readFileSync(fixturesPath, 'utf8'));
const refreshQueue = [...fx.popularRefreshes];

let popularMode = 'ok';
const env = createEnv({
  root,
  routes: {
    '/api/recommend': ({ body }) => ({ json: body.exclude ? fx.normalRefresh : fx.normal }),
    '/api/popular': ({ body }) => {
      if (body.exclude) {
        const next = refreshQueue.shift();
        return next && next.status ? { status: next.status, json: { error: next.error } } : { json: next };
      }
      if (popularMode === 'unavailable') {
        return { json: { mood: body.mood, mode: body.mode, available: false, results: [], message: 'Popular Artists needs its local dataset.' } };
      }
      if (popularMode === 'empty') {
        return { json: { mood: body.mood, mode: body.mode, available: true, results: [], eligible: 0, limited: true, limited_by: 'pool' } };
      }
      return { json: fx.popular };
    },
    '/api/track-details': () => ({ json: { enabled: true, source: 'Spotify oEmbed', tracks: {} } }),
  },
});
env.load(['static/js/ui-utils.js', 'static/js/track-identity.js', 'static/js/music.js']);
const list = () => env.getElementById('trackList');
const rows = () => list().children.filter((c) => c.classList.contains('track-card'));
const ids = () => rows().map((r) => r.dataset.trackId);
const reqs = () => env.state.requests.filter((q) => q.path !== '/api/track-details');
const since = (n) => reqs().slice(n).map((q) => ({ path: q.path, body: q.body }));
const sw = () => env.getElementById('popularSwitch');
const switchState = () => ({
  checked: sw().getAttribute('aria-checked'), data: sw().dataset.state,
  text: sw().querySelector('.switch-state').textContent,
});
const status = () => env.getElementById('statusLine').textContent;
const html = () => rows().map((r) => r._html).join('\n');
const artistLines = () => rows().map((r) => {
  const h = (r.subs['.track-info'] ? r.subs['.track-info']._html : '') || r._html;
  return (h.match(/class="track-artist[^"]*">([^<]*)</) || [])[1] || null;
});
const cachedIds = (view) => {
  const v = env.ctx.sessionStorage.getItem('mc:view:' + view);
  return v ? JSON.parse(v).value.results.map((t) => t.track_id) : null;
};
const seenIds = (view) => JSON.parse(env.ctx.sessionStorage.getItem('mc:seen:' + view) || '[]');
const step = async (action) => {
  const n = reqs().length;
  action();
  await env.settle(60);
  return { requests: since(n), rows: ids(), switch: switchState(), status: status() };
};
const flip = () => sw().dispatch('click');
const refresh = () => env.getElementById('refreshBtn').dispatch('click');

(async () => {
  const out = {};
  out.initial = await step(() => env.start());
  out.on = await step(flip);
  Object.assign(out.on, {
    kicker: env.getElementById('pageKicker').textContent, title: env.getElementById('pageTitle').textContent,
    sourceNote: env.getElementById('sourceNote').textContent, artistLines: artistLines(), html: html(),
    spotifyLinks: rows().map((r) => (r._html.match(/<a class="spotify-link" href="([^"]+)" target="_blank" rel="noopener noreferrer"/) || [])[1] || null),
  });
  out.off = await step(flip);
  out.off.kicker = env.getElementById('pageKicker').textContent;
  out.onAgain = await step(flip);

  // ---- Popular Artists Refresh ----
  out.refresh1 = await step(refresh);                          // a full fresh list
  out.refresh1.normalCache = cachedIds('music:happy:match');
  out.refresh1.popularCache = cachedIds('popular:happy:match');
  out.refresh2 = await step(refresh);                          // short: per-artist limit
  out.refresh2.refreshDisabled = env.getElementById('refreshBtn').disabled;
  out.refresh3 = await step(refresh);                          // nothing unseen left
  out.refresh4 = await step(refresh);                          // server error
  out.refresh5 = await step(refresh);                          // dataset became unavailable
  out.afterRefreshes = { popularCache: cachedIds('popular:happy:match'), popularSeen: seenIds('popular:happy:match'),
    normalCache: cachedIds('music:happy:match'), normalSeen: seenIds('music:happy:match') };

  // ---- normal mode is independent ----
  out.offAfterRefresh = await step(flip);
  out.normalRefresh = await step(refresh);
  out.normalRefresh.popularCache = cachedIds('popular:happy:match');
  out.normalRefresh.normalCache = cachedIds('music:happy:match');
  out.onAfterNormalRefresh = await step(flip);

  // ---- selection changes while ON ----
  out.moodChange = await step(() => env.run('document.querySelectorAll(".mood-pill").find((c) => c.dataset.mood === "sad").dispatch("click")'));
  out.modeChange = await step(() => env.getElementById('modeToggle').children.find((b) => b.dataset.mode === 'uplift').dispatch('click'));
  out.modeChange.title = env.getElementById('pageTitle').textContent;

  // ---- empty and unavailable (not a refresh) ----
  popularMode = 'empty';
  out.empty = await step(() => { env.state.mood = 'stressed'; env.run('loadMusic()'); });
  out.empty.list = list().innerHTML;
  popularMode = 'unavailable';
  out.unavailable = await step(() => { env.state.mood = 'angry'; env.run('loadMusic()'); });
  out.unavailable.list = list().innerHTML;
  out.unavailable.popularCache = cachedIds('popular:angry:uplift');
  out.backToNormal = await step(flip);

  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
