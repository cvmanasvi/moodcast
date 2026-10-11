// Drives the real static/js/ui-utils.js + movies.js with a real /api/recommend
// response, using the tiny fake DOM in fake_dom.js.
// Usage: node tests/js/movies_page_harness.js <project root> <fixture.json>
//   fixture.json: { response, refresh, details: {movie_id: {...}}, broken: [image urls that fail to load] }
// Prints a JSON summary; tests/test_frontend_pages.py makes the assertions.
const fs = require('fs');
const { createEnv } = require('./fake_dom');

const [root, fixturePath] = process.argv.slice(2);
const fx = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));

let recommendMode = 'ok';
let detailsMode = 'ok';
const env = createEnv({
  root,
  brokenImages: new Set(fx.broken),
  routes: {
    '/api/recommend': ({ body }) => {
      if (recommendMode === 'error') return new TypeError('Failed to fetch');
      if (recommendMode === 'empty') return { json: { results: [] } };
      if (recommendMode === 'noposter') return { json: { results: [{ ...fx.response.results[0], image: '', movie_id: 999999 }] } };
      return { json: body.exclude ? fx.refresh : fx.response };
    },
    '/api/movie-details': ({ body }) => {
      if (detailsMode === 'error') return { status: 500, json: { error: 'TMDB down' } };
      const movies = {};
      for (const id of body.movie_ids) movies[String(id)] = fx.details[String(id)] || { status: 'unmatched', tmdb_id: null, poster: null, providers: { region: 'IN', link: null, groups: [] } };
      return { json: { region: 'IN', enabled: true, attribution: 'Streaming data from JustWatch via TMDB', movies } };
    },
  },
});
env.load(['static/js/ui-utils.js', 'static/js/movies.js']);

const grid = () => env.getElementById('movieGrid');
const cards = () => grid().children.filter((c) => c.classList.contains('movie-card'));
const titleOf = (c) => (c._html.match(/class="movie-title" id="[^"]+">([^<]*)</) || [])[1];
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const target = (matches) => ({ closest: (sel) => matches[sel] || null });   // fake event target

(async () => {
  const out = {};
  env.start();
  out.loadingBeforePosters = grid().innerHTML.includes('poster-skeleton');
  await env.settle(80);
  const reqs = () => env.state.requests;
  out.requests = reqs().map((q) => ({ path: q.path, body: q.body }));

  const shown = cards();
  out.cards = shown.length;
  out.cardSrcs = shown.map((c) => (c._html.match(/class="movie-poster" src="([^"]+)"/) || [])[1]);
  out.cardApiIndex = shown.map((c) => fx.response.results.findIndex((m) => esc(m.title) === titleOf(c)));
  out.status = env.getElementById('resultsSubtitle').textContent;
  out.attribution = env.getElementById('watchAttribution').textContent;
  out.cardHtml = shown.map((c) => c._html);
  out.backHasInert = shown.every((c) => /class="movie-face movie-back"[^>]*\binert\b/.test(c._html));

  // ---- flip behaviour on the first card ----
  const card = shown[0];
  const click = (matches) => card.dispatch('click', { target: target(matches) });
  click({});                                                  // poster area: flips (mouse/tap)
  out.flipByCardClick = card.classList.contains('is-flipped');
  out.frontInertWhenFlipped = card.querySelector('.movie-front').inert;
  out.backInertWhenFlipped = card.querySelector('.movie-back').inert;
  out.backAriaHiddenWhenFlipped = card.querySelector('.movie-back').getAttribute('aria-hidden');
  const fav = env.el('button'); fav.dataset.action = 'favorite';
  click({ '[data-action]': fav });
  out.flippedAfterFavorite = card.classList.contains('is-flipped');
  out.favoritePressed = fav.getAttribute('aria-pressed');
  const providerLink = env.el('a');
  click({ 'a, button, input': providerLink });                // a "Where to watch" link: opens, no flip
  out.flippedAfterProviderLink = card.classList.contains('is-flipped');
  const details = env.el('button'); details.dataset.action = 'details';
  click({ '[data-action]': details });
  out.flippedAfterDetails = card.classList.contains('is-flipped');
  out.modalShown = env.getElementById('modalOverlay').classList.contains('show');
  out.modalInert = env.getElementById('modalOverlay').inert;
  out.modalTitle = env.getElementById('modalTitle').textContent;
  out.modalDesc = env.getElementById('modalDesc').textContent;
  out.modalWatch = env.getElementById('modalWatch').innerHTML;
  env.keydown({ key: 'Escape' });
  out.modalClosedByEscape = !env.getElementById('modalOverlay').classList.contains('show') && env.getElementById('modalOverlay').inert === true;
  const close = env.el('button'); close.dataset.flip = 'close';
  click({ '[data-flip]': close });
  out.unflippedByBackButton = !card.classList.contains('is-flipped');
  out.focusAfterUnflip = env.doc.activeElement && env.doc.activeElement.selector;
  const open = env.el('button'); open.dataset.flip = 'open';
  click({ '[data-flip]': open });                             // Enter/Space on the flip button = click
  out.flippedByFlipButton = card.classList.contains('is-flipped');
  out.focusAfterFlip = env.doc.activeElement && env.doc.activeElement.selector;
  out.ariaExpanded = card.querySelector('[data-flip="open"]').getAttribute('aria-expanded');
  card.dispatch('keydown', { key: 'Escape', stopPropagation() {} });
  out.unflippedByEscape = !card.classList.contains('is-flipped');

  // ---- a poster that fails after rendering is removed, not replaced ----
  const before = cards().length;
  cards()[1].querySelector('.movie-poster').dispatch('error', {});
  out.cardsAfterLateError = [before, cards().length];

  // ---- same mood/mode again: same picks, no new request ----
  let n = reqs().length;
  env.run('loadMovies()');
  await env.settle();
  out.requestsOnRevisit = reqs().length - n;
  out.cardsOnRevisit = cards().length;

  // ---- refresh asks for movies not shown or checked before ----
  n = reqs().length;
  env.getElementById('refreshBtn').dispatch('click');
  await env.settle(80);
  const refreshReq = reqs().slice(n).find((q) => q.path === '/api/recommend');
  out.refreshBody = refreshReq ? refreshReq.body : null;
  out.cardsAfterRefresh = cards().map((c) => fx.refresh.results.findIndex((m) => esc(m.title) === titleOf(c)));
  out.statusAfterRefresh = env.getElementById('resultsSubtitle').textContent;

  // ---- mood and mode changes call the API with the new values ----
  n = reqs().length;
  env.run('document.querySelectorAll(".mood-pill").find((c) => c.dataset.mood === "sad").dispatch("click")');
  await env.settle(80);
  env.getElementById('modeToggle').children.find((b) => b.dataset.mode === 'uplift').dispatch('click');
  await env.settle(80);
  out.changeRequests = reqs().slice(n).filter((q) => q.path === '/api/recommend').map((q) => q.body);
  out.activePills = env.run('document.querySelectorAll(".mood-pill").filter((c) => c.classList.contains("active")).map((c) => c.dataset.mood)');
  out.titleAfterChanges = env.getElementById('resultsTitle').textContent;

  // ---- TMDB details unavailable: posters still checked, "where to watch" says so ----
  detailsMode = 'error';
  env.state.mood = 'angry';
  env.run('loadMovies()');
  await env.settle(80);
  out.cardsWhenDetailsFail = cards().length;
  out.watchWhenDetailsFail = cards().length ? (cards()[0]._html.match(/<section class="watch"[\s\S]*?<\/section>/) || [''])[0] : '';
  out.attributionWhenDetailsFail = env.getElementById('watchAttribution').textContent;

  // ---- API error, empty results, no loadable poster: messages, never stale cards ----
  recommendMode = 'error'; env.state.mood = 'chill';
  env.run('loadMovies()'); await env.settle(60);
  out.errorCards = cards().length; out.errorHtml = grid().innerHTML;
  out.errorStatus = env.getElementById('resultsSubtitle').textContent;
  recommendMode = 'empty'; env.state.mood = 'energetic';
  env.run('loadMovies()'); await env.settle(60);
  out.emptyCards = cards().length; out.emptyHtml = grid().innerHTML;
  recommendMode = 'noposter'; detailsMode = 'ok'; env.state.mood = 'stressed';
  env.run('loadMovies()'); await env.settle(60);
  out.noPosterCards = cards().length; out.noPosterHtml = grid().innerHTML;
  out.noPosterStatus = env.getElementById('resultsSubtitle').textContent;

  // ---- pure helpers ----
  out.missingFields = env.run(`(() => {
    const f = MovieCards.movieFields({ title: 'T', year: 'N/A', genre: '', rating: 'N/A', duration: 'N/A',
      description: 'No description available.', overview: '', image: 'https://m.media-amazon.com/images/M/x.jpg', mood_confidence: null });
    return { fields: f, html: cardHtml(f, 99, false, { title: 'T', details: null }) };
  })()`);
  out.posterUrls = env.run(`[
    posterDisplayUrl('https://m.media-amazon.com/images/M/MV5BNDVk@.jpg'),
    posterDisplayUrl('https://m.media-amazon.com/images/M/MV5Bx@@._V1_SX300.jpg'),
    posterDisplayUrl('https://example.org/poster.jpg'),
    posterDisplayUrl('javascript:alert(1)'), posterDisplayUrl(''), posterDisplayUrl(null)]`);
  out.watchKinds = env.run(`[
    MovieCards.watchInfo(null).kind,
    MovieCards.watchInfo({ status: 'unmatched', providers: { groups: [] } }).kind,
    MovieCards.watchInfo({ status: 'matched', providers: { groups: [] } }).kind,
    MovieCards.watchInfo({ status: 'matched', providers: { link: 'https://evil.example/x', groups: [{ label: 'Stream', providers: [{ name: 'X' }] }] } }),
    MovieCards.watchInfo({ status: 'error' }).kind,
    MovieCards.watchInfo({ status: 'matched', providers: { verified: true, groups: [] } }).kind]`);
  // "Where to watch" for given TMDB details (see test_frontend_pages.py for the fixtures)
  out.watchHtmlFor = env.run(`(${JSON.stringify(fx.watchCases || {})})`);
  for (const [name, details] of Object.entries(out.watchHtmlFor)) {
    out.watchHtmlFor[name] = env.run(`watchHtml({ title: 'T', details: ${JSON.stringify(details)} })`);
  }
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
