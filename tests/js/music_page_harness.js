// Drives the real static/js/ui-utils.js + track-identity.js + music.js with a
// real /api/recommend response, using the tiny fake DOM in fake_dom.js.
// Usage: node tests/js/music_page_harness.js <project root> <response.json> <details.json> <refresh.json>
//   details.json: the /api/track-details "tracks" map to serve (Spotify oEmbed stand-in)
//   refresh.json: the response to serve when a request carries `exclude`
// Prints a JSON summary; tests/test_music_api.py and tests/test_frontend_pages.py make the assertions.
const fs = require('fs');
const { createEnv } = require('./fake_dom');

const [root, responsePath, detailsPath, refreshPath] = process.argv.slice(2);
const response = JSON.parse(fs.readFileSync(responsePath, 'utf8'));
const details = detailsPath ? JSON.parse(fs.readFileSync(detailsPath, 'utf8')) : {};
const refreshResponse = refreshPath ? JSON.parse(fs.readFileSync(refreshPath, 'utf8')) : { results: [] };

let recommendMode = 'ok';
let detailsMode = 'ok';
const env = createEnv({
  root,
  routes: {
    '/api/recommend': ({ body }) => {
      if (recommendMode === 'error') return { status: 500, json: { error: 'Server exploded' } };
      if (body.exclude) return { json: refreshResponse };
      return { json: response };
    },
    '/api/track-details': ({ query }) => {
      if (detailsMode === 'error') return new TypeError('Failed to fetch');
      const tracks = {};
      for (const id of (query.ids || '').split(',')) if (details[id]) tracks[id] = details[id];
      return { json: { enabled: true, source: 'Spotify oEmbed', tracks } };
    },
  },
});
const storage = {};
env.ctx.localStorage = { getItem: (k) => (k in storage ? storage[k] : null), setItem: (k, v) => { storage[k] = String(v); } };
env.load(['static/js/ui-utils.js', 'static/js/track-identity.js', 'static/js/music.js']);
const list = () => env.getElementById('trackList');
const rows = () => list().children.filter((c) => c.classList.contains('track-card'));
const headingOf = (r) => {
  const html = (r.subs['.track-info'] ? r.subs['.track-info']._html : '') || r._html;
  return (html.match(/class="track-title[^"]*"[^>]*>([^<]*)</) || [])[1];
};

(async () => {
  const out = {};
  // ---- render the real response before any oEmbed details arrive ----
  env.run('currentTracks = __r; trackDetails = {}; renderTracks(currentTracks);'.replace('__r', JSON.stringify(response.results)));
  const r0 = rows();
  out.rows = r0.length;
  out.rowTrackIds = r0.map((r) => r.dataset.trackId || null);
  out.badText = r0.filter((r) => /\bundefined\b|\bnull\b|\[object Object\]/.test(r._html)).length;
  out.durationsShown = r0.map((r) => (r._html.match(/track-duration">([^<]*)</) || [])[1] || '');
  out.spotifyLinks = r0.map((r) => (r._html.match(/<a class="spotify-link" href="([^"]+)" target="_blank" rel="noopener noreferrer"/) || [])[1] || null);
  out.linkUnavailable = r0.map((r) => r._html.includes('Spotify link unavailable'));
  out.headingsBeforeDetails = r0.map(headingOf);
  out.placeholderText = r0.filter((r) => /Unknown (title|artist)/.test(r._html)).length;
  out.rowsAreButtons = r0.filter((r) => r.attrs.role === 'button').length;
  out.playbackApi = ['playTrack', 'nextTrack', 'prevTrack', 'togglePlay', 'startPlayback', 'onQueueClick', 'waveformHeights']
    .filter((name) => env.run(`typeof ${name}`) !== 'undefined');
  env.run('toggleFavorite(currentTracks[currentTracks.length - 1]);');
  out.favorites = JSON.parse(storage.moodcast_favorite_tracks || '[]');

  // ---- full page flow: load, oEmbed fill-in, cache, refresh, mood/mode, errors ----
  env.start();
  await env.settle(60);
  const reqs = () => env.state.requests;
  out.firstRequests = reqs().map((q) => ({ path: q.path, body: q.body, ids: q.query.ids ? q.query.ids.split(',').length : 0 }));
  const after = rows();
  out.headingsAfterDetails = after.map(headingOf);
  out.artistLines = after.map((r) => {
    const html = (r.subs['.track-info'] ? r.subs['.track-info']._html : '') || r._html;
    return (html.match(/class="track-artist[^"]*">([^<]*)</) || [])[1] || null;
  });
  out.covers = after.map((r) => /class="track-cover"/.test((r.subs['.track-art'] || {})._html || r._html));
  out.status = env.getElementById('statusLine').textContent;

  const n = reqs().length;
  env.run('loadMusic()');                                   // same mood/mode again: stable, no request
  await env.settle();
  out.requestsOnRevisit = reqs().length - n;

  env.getElementById('refreshBtn').dispatch('click');       // refresh asks for unseen tracks
  await env.settle(60);
  const refreshReq = reqs().slice(n).find((q) => q.path === '/api/recommend');
  out.refreshBody = refreshReq ? refreshReq.body : null;
  out.rowsAfterRefresh = rows().map((r) => r.dataset.trackId);
  out.statusAfterRefresh = env.getElementById('statusLine').textContent;

  const m = reqs().length;
  env.run('document.querySelectorAll(".mood-pill").find((c) => c.dataset.mood === "sad").dispatch("click")');
  await env.settle(60);
  env.getElementById('modeToggle').children.find((b) => b.dataset.mode === 'uplift').dispatch('click');
  await env.settle(60);
  out.changeRequests = reqs().slice(m).filter((q) => q.path === '/api/recommend').map((q) => q.body);
  out.titleAfterChanges = env.getElementById('pageTitle').textContent;
  out.activePill = env.run('document.querySelectorAll(".mood-pill").filter((c) => c.classList.contains("active")).map((c) => c.dataset.mood)');

  detailsMode = 'error';                                     // oEmbed down: titles settle to the honest fallback
  env.state.mood = 'angry';
  env.run('loadMusic()');
  await env.settle(60);
  out.headingsWhenDetailsFail = rows().map(headingOf);

  recommendMode = 'error';
  env.state.mood = 'chill';
  env.run('loadMusic()');
  await env.settle(60);
  out.errorRows = rows().length;
  out.errorHtml = list().innerHTML;
  out.refreshDisabledAfterError = env.getElementById('refreshBtn').disabled;

  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
