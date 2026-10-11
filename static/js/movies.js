/**
 * MoodCast — Movies page driver.
 *
 * Talks to POST /api/recommend ({ mood, mode, type: 'movies', limit }) and
 * renders the movies it returns, in the order returned (the backend's ranking).
 *
 * Posters: only movies with a poster that actually loads are shown -- the
 * dataset's poster, or else the verified TMDB poster of the same movie
 * (POST /api/movie-details, matched server-side by exact title + year). No
 * poster is ever substituted; the page says how many were skipped.
 *
 * Each movie is a two-sided card: poster on the front; the full overview,
 * rating, year, genre and "Where to watch" (India, from TMDB/JustWatch) on
 * the back. Missing fields are labelled as missing, never filled in.
 *
 * A mood/mode keeps showing the same picks (per browser tab) until the user
 * presses Refresh, which asks the backend for movies not shown recently.
 */

const MovieCards = (() => {
  const MISSING = new Set(['', 'N/A', 'No description available.']);

  function realField(value) {
    return typeof value === 'string' && !MISSING.has(value.trim()) ? value.trim() : null;
  }

  /** Real display fields of one API result; null for anything missing. */
  function movieFields(m) {
    const rating = realField(m.rating);
    return {
      title: realField(m.title) || 'Untitled',
      year: realField(m.year),
      genre: realField(m.genre),
      duration: realField(m.duration),
      rating: rating && !isNaN(parseFloat(rating)) ? rating : null,
      overview: realField(m.overview) || realField(m.description),
      poster: posterUrl(m),
      match: m.mood_confidence != null ? Math.round(m.mood_confidence * 100) : null,
    };
  }

  /** The poster URL that was checked to load (poster_src), else the API's image; absolute http(s) only. */
  function posterUrl(m) {
    const ok = (u) => typeof u === 'string' && /^https?:\/\/[^\s"'<>]+$/i.test(u);
    if (!m) return null;
    if (ok(m.poster_src)) return m.poster_src;
    return ok(m.image) ? m.image : null;
  }

  /**
   * "Where to watch" state for a title's TMDB details:
   * { kind: 'providers'|'none'|'unverified'|'unmatched'|'error'|'disabled'|'unknown', groups, link }.
   * 'none' (no listed options) only when TMDB answered with provider data
   * (providers.verified) that lists nothing for the region; a missing or
   * unverified answer is 'unverified', never 'none'.
   */
  function watchInfo(details) {
    if (!details) return { kind: 'unknown', groups: [], link: null };
    const p = details.providers || {};
    const link = typeof p.link === 'string' && /^https:\/\/www\.themoviedb\.org\//.test(p.link) ? p.link : null;
    const groups = Array.isArray(p.groups) ? p.groups.filter((g) => g && Array.isArray(g.providers) && g.providers.length) : [];
    if (details.status === 'matched' && groups.length) return { kind: 'providers', groups, link };
    if (details.status === 'matched') return { kind: p.verified === true ? 'none' : 'unverified', groups: [], link };
    if (details.status === 'error') return { kind: 'error', groups: [], link: null };
    if (details.status === 'disabled') return { kind: 'disabled', groups: [], link: null };
    return { kind: 'unmatched', groups: [], link: null };
  }

  return { movieFields, posterUrl, watchInfo };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = MovieCards;

const POOL_SIZE = 60;      // recommendations requested per load (the API allows up to 100)
const SHOW_MAX = 24;       // poster-backed cards shown
const POSTER_TIMEOUT_MS = 10000;
const REGION_NAME = { IN: 'India' };
const WATCH_ATTRIBUTION = 'Streaming data from JustWatch via TMDB';   // JustWatch credit TMDB requires

function movieKey(m) { return (m.title || '') + '::' + (m.year || ''); }

/* ---------------------------------------------------------------------- */
/* Favorites (localStorage — frontend-only, no backend/user account)       */
/* ---------------------------------------------------------------------- */

const FAV_KEY = 'moodcast_favorite_movies';
function loadFavorites() {
  try { return new Set(JSON.parse(localStorage.getItem(FAV_KEY)) || []); }
  catch { return new Set(); }
}
function saveFavorites(set) {
  try { localStorage.setItem(FAV_KEY, JSON.stringify([...set])); } catch { /* storage unavailable */ }
}
let favorites = loadFavorites();
function toggleFavorite(m) {
  const key = movieKey(m);
  if (favorites.has(key)) favorites.delete(key);
  else favorites.add(key);
  saveFavorites(favorites);
  return favorites.has(key);
}

/* ---------------------------------------------------------------------- */
/* Mood chips + mode toggle                                                */
/* ---------------------------------------------------------------------- */

function renderMoodChips() {
  const strip = document.getElementById('moodStrip');
  strip.innerHTML = '';
  MOOD_ORDER.forEach((id) => {
    const theme = MOOD_THEME[id];
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'mood-pill glass' + (id === getCurrentMood() ? ' active' : '');
    chip.dataset.mood = id;
    chip.setAttribute('aria-pressed', String(id === getCurrentMood()));
    chip.style.setProperty('--chip-glow', theme.colors.glow);
    chip.innerHTML = `<span aria-hidden="true">${theme.emoji}</span><span>${theme.label}</span>`;
    // setMoodTheme fires moodthemechange; that listener (in init) reloads.
    chip.addEventListener('click', () => setMoodTheme(id));
    strip.appendChild(chip);
  });
}

function markActiveMood(id) {
  document.querySelectorAll('.mood-pill').forEach((c) => {
    const on = c.dataset.mood === id;
    c.classList.toggle('active', on);
    c.setAttribute('aria-pressed', String(on));
  });
}

function initModeToggle() {
  const buttons = [...document.getElementById('modeToggle').querySelectorAll('button')];
  const mark = (mode) => buttons.forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', String(on));
  });
  mark(getCurrentMode());
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.dataset.mode === getCurrentMode()) return;
      setMode(btn.dataset.mode);
      mark(btn.dataset.mode);
      loadMovies();
    });
  });
}

/* ---------------------------------------------------------------------- */
/* Loading movies                                                          */
/* ---------------------------------------------------------------------- */

let currentMovies = [];
let requestSeq = 0;
let requestedMood = null;  // mood of the latest request, so a theme change elsewhere (assistant) reloads
let watchAttribution = '';

const viewKey = () => `movies:${getCurrentMood()}:${getCurrentMode()}`;
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

function setStatus(text, busy = false) {
  document.getElementById('resultsSubtitle').innerHTML =
    (busy ? '<span class="spinner" aria-hidden="true"></span>' : '') + escapeHtml(text);
}

function setRefreshBusy(busy) {
  const btn = document.getElementById('refreshBtn');
  btn.disabled = busy;
  btn.setAttribute('aria-busy', String(busy));
}

function setHeader(mood, mode) {
  const theme = getMoodTheme(mood);
  document.getElementById('pageKicker').textContent =
    `${theme.emoji} Movies · ${mode === 'uplift' ? 'Cheer me up' : 'Match my mood'}`;
  document.getElementById('resultsTitle').textContent =
    mode === 'uplift' ? `Films to lift a ${theme.label.toLowerCase()} mood` : `${theme.label} movies`;
}

/**
 * Show this mood/mode's picks. Uses the picks already shown in this tab
 * unless `refresh` is set, in which case the backend is asked for movies not
 * shown recently.
 */
async function loadMovies({ refresh = false } = {}) {
  const mood = getCurrentMood();
  const mode = getCurrentMode();
  const view = viewKey();
  const seq = ++requestSeq;
  requestedMood = mood;
  setHeader(mood, mode);
  const grid = document.getElementById('movieGrid');

  const cached = !refresh && ResultMemory.load(view);
  if (cached && Array.isArray(cached.movies) && cached.movies.length) {
    currentMovies = cached.movies;
    watchAttribution = cached.attribution || '';
    renderMovies(currentMovies, mood);
    setStatus(`${plural(currentMovies.length, 'movie')} · same picks as before for this mood · press Refresh for new ones`);
    return;
  }

  const previous = currentMovies;
  grid.setAttribute('aria-busy', 'true');
  if (!refresh) { grid.innerHTML = skeletonPosters(12); currentMovies = []; }
  setRefreshBusy(true);
  setStatus(refresh ? 'Finding movies you haven\'t seen…' : 'Loading recommendations…', true);

  try {
    const exclude = refresh ? ResultMemory.recent(view) : [];
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mood, mode, type: 'movies', limit: POOL_SIZE, ...(exclude.length ? { exclude } : {}) }),
    });
    const data = await readJsonResponse(res);
    if (seq !== requestSeq) return;
    const pool = Array.isArray(data.results) ? data.results : [];
    if (!pool.length) {
      if (refresh && previous.length) {
        currentMovies = previous;
        renderMovies(currentMovies, mood);
        setStatus('No other movies are available for this mood right now — these are still your picks');
      } else {
        grid.innerHTML = '<div class="state-box"><strong>No movies found</strong>Try another mood or mode.</div>';
        setStatus('0 results');
      }
      return;
    }

    setStatus(`Checking posters and where to watch for ${plural(pool.length, 'recommendation')}…`, true);
    const withPoster = await resolveMoviePosters(pool, POSTER_TIMEOUT_MS);   // ui-utils.js
    if (seq !== requestSeq) return;

    const shown = withPoster.slice(0, SHOW_MAX);
    if (!shown.length && refresh && previous.length) {
      currentMovies = previous;
      renderMovies(currentMovies, mood);
      setStatus('None of the other recommendations has a working poster — these are still your picks');
      return;
    }
    currentMovies = shown;
    watchAttribution = withPoster.meta ? withPoster.meta.attribution : '';
    renderMovies(currentMovies, mood);
    // every movie checked counts as seen, so refreshes move on to new ones
    ResultMemory.remember(view, pool.map((m) => m.movie_id).filter(Number.isInteger));
    if (currentMovies.length) ResultMemory.save(view, { movies: currentMovies, attribution: watchAttribution });

    const skipped = pool.length - withPoster.length;
    if (!currentMovies.length) {
      setStatus(`No poster loads for any of the top ${plural(pool.length, 'recommendation')}`);
    } else {
      const fromTmdb = currentMovies.filter((m) => m.poster_source === 'tmdb').length;
      setStatus(`Showing ${currentMovies.length} poster-backed picks from ${plural(pool.length, 'recommendation')}`
        + (skipped ? ` · ${skipped} skipped (no working poster)` : '')
        + (fromTmdb ? ` · ${fromTmdb} posters from TMDB` : '')
        + (data.refresh && data.refresh.exhausted ? ' · fewer new movies were available, none repeated' : ''));
    }
  } catch (err) {
    if (seq !== requestSeq) return;
    if (refresh && previous.length) {
      currentMovies = previous;
      renderMovies(currentMovies, mood);
      setStatus(`Couldn't refresh (${err.message || 'request failed'}) — showing your previous picks`);
    } else {
      currentMovies = [];
      setStatus('Recommendations could not be loaded');
      grid.innerHTML = `<div class="state-box is-error" role="alert"><strong>Couldn't load movies</strong>${escapeHtml(err.message || 'Request failed')}.
        <br><button type="button" class="btn btn-glass retry-btn">Try again</button></div>`;
      grid.querySelector('.retry-btn').addEventListener('click', () => loadMovies());
    }
    console.error(err);
  } finally {
    if (seq === requestSeq) {
      grid.setAttribute('aria-busy', 'false');
      setRefreshBusy(false);
    }
  }
}

function skeletonPosters(n) {
  return Array.from({ length: n }, () => '<div class="skeleton poster-skeleton" aria-hidden="true"></div>').join('');
}

/* ---------------------------------------------------------------------- */
/* Where to watch                                                          */
/* ---------------------------------------------------------------------- */

function watchHtml(m, { compact = true } = {}) {
  const w = MovieCards.watchInfo(m.details);
  const region = REGION_NAME[(m.details && m.details.providers && m.details.providers.region) || 'IN'] || 'India';
  const tmdbLink = (text) => (w.link ? `<a class="watch-link" href="${escapeHtml(w.link)}" target="_blank" rel="noopener noreferrer"
        aria-label="See where to watch ${escapeHtml(m.title)} in ${region} on TMDB (new tab)">${text} ↗</a>` : '');
  let body;
  if (w.kind === 'providers') {
    body = w.groups.map((g) => `
      <div class="watch-group">
        <span class="watch-type">${escapeHtml(g.label)}</span>
        <ul class="watch-list">
          ${g.providers.map((p) => `<li class="watch-provider" title="${escapeHtml(p.name)}">
              ${p.logo ? `<img src="${escapeHtml(p.logo)}" alt="" loading="lazy" decoding="async">` : ''}
              <span class="${p.logo && compact ? 'visually-hidden' : ''}">${escapeHtml(p.name)}</span></li>`).join('')}
        </ul>
      </div>`).join('');
    body += tmdbLink('All options on TMDB');
  } else if (w.kind === 'none') {
    body = `<p class="watch-empty">No listed options for ${region}.</p>` + tmdbLink('Check on TMDB');
  } else if (w.kind === 'disabled') {
    body = `<p class="watch-empty">Availability could not be verified (not configured on this server).</p>`;
  } else {
    body = `<p class="watch-empty">Availability could not be verified.</p>` + tmdbLink('Check on TMDB');
  }
  return `<section class="watch" aria-label="Where to watch in ${region}">
      <h4 class="watch-head">Where to watch <span>· ${region}</span></h4>${body}</section>`;
}

/* ---------------------------------------------------------------------- */
/* Flippable poster cards                                                  */
/* ---------------------------------------------------------------------- */

function cardHtml(f, i, isFav, m = {}) {
  const title = escapeHtml(f.title);
  const metaParts = [f.year, f.genre && f.genre.split(',')[0].trim()].filter(Boolean).map(escapeHtml);
  const backMeta = [f.year, f.duration].filter(Boolean).map(escapeHtml).join(' · ');
  return `
    <div class="movie-card-inner">
      <div class="movie-face movie-front">
        <img class="movie-poster" src="${escapeHtml(f.poster)}" alt="Poster for ${title}" decoding="async">
        ${f.match != null ? `<span class="poster-match" title="Mood classifier confidence">${f.match}%</span>` : ''}
        <div class="movie-front-info">
          <h3 class="movie-title" id="movie-title-${i}">${title}</h3>
          ${metaParts.length ? `<p class="movie-meta">${metaParts.join(' · ')}</p>` : ''}
          <button type="button" class="flip-btn" data-flip="open" aria-label="Show storyline, rating and where to watch for ${title}">
            <span aria-hidden="true">⟲</span> Details
          </button>
        </div>
      </div>
      <div class="movie-face movie-back" role="group" aria-labelledby="movie-back-title-${i}" inert aria-hidden="true">
        <div class="back-scroll">
          <h3 class="movie-back-title" id="movie-back-title-${i}">${title}</h3>
          ${backMeta ? `<p class="movie-meta">${backMeta}</p>` : ''}
          <p class="movie-meta">${f.genre ? escapeHtml(f.genre) : '<span class="is-missing">Genre unavailable</span>'}</p>
          <p class="movie-rating">${f.rating
            ? `<span class="star" aria-hidden="true">★</span> ${escapeHtml(f.rating)}<span class="out-of"> / 10</span>`
            : '<span class="is-missing">Rating unavailable</span>'}</p>
          <p class="movie-overview">${f.overview ? escapeHtml(f.overview) : '<span class="is-missing">No storyline available for this title.</span>'}</p>
          ${watchHtml(m)}
        </div>
        <div class="movie-back-actions">
          <button type="button" class="back-btn" data-action="details">Full details</button>
          <button type="button" class="back-btn icon${isFav ? ' is-fav' : ''}" data-action="favorite"
            aria-pressed="${isFav}" aria-label="Save ${title}">${isFav ? '❤️' : '🤍'}</button>
          <button type="button" class="back-btn icon" data-flip="close" aria-label="Back to poster for ${title}">
            <span aria-hidden="true">↺</span>
          </button>
        </div>
      </div>
    </div>`;
}

function setFlipped(card, flipped, { moveFocus = true } = {}) {
  const front = card.querySelector('.movie-front');
  const back = card.querySelector('.movie-back');
  card.classList.toggle('is-flipped', flipped);
  front.inert = flipped;
  back.inert = !flipped;
  front.setAttribute('aria-hidden', String(flipped));
  back.setAttribute('aria-hidden', String(!flipped));
  card.querySelector('[data-flip="open"]').setAttribute('aria-expanded', String(flipped));
  if (moveFocus) {
    // the face that just went inert can't keep focus; hand it to the other face's toggle
    const target = card.querySelector(flipped ? '[data-flip="close"]' : '[data-flip="open"]');
    if (target) target.focus({ preventScroll: true });
  }
}

function renderMovies(movies, mood) {
  const grid = document.getElementById('movieGrid');
  grid.innerHTML = '';
  document.getElementById('watchAttribution').textContent = watchAttribution || WATCH_ATTRIBUTION;
  if (!movies.length) {
    grid.innerHTML = '<div class="state-box"><strong>No poster-backed picks</strong>None of these recommendations has a working poster. Try another mood or mode, or refresh.</div>';
    return;
  }
  movies.forEach((m, i) => {
    const f = MovieCards.movieFields(m);
    const card = document.createElement('article');
    card.className = 'movie-card';
    card.setAttribute('aria-labelledby', `movie-title-${i}`);
    card.innerHTML = cardHtml(f, i, favorites.has(movieKey(m)), m);
    card.querySelector('[data-flip="open"]').setAttribute('aria-expanded', 'false');

    const poster = card.querySelector('.movie-poster');
    poster.addEventListener('error', () => {   // a poster that fails later is dropped, not replaced
      card.remove();
      setStatus(`Showing ${grid.querySelectorAll('.movie-card').length} poster-backed picks`);
    });

    card.addEventListener('click', (e) => {
      const flip = e.target.closest('[data-flip]');
      if (flip) { setFlipped(card, flip.dataset.flip === 'open'); return; }
      const action = e.target.closest('[data-action]');
      if (action) {
        if (action.dataset.action === 'details') openModal(m, mood);
        else if (action.dataset.action === 'favorite') onFavorite(action, m);
        return;                                // action buttons never flip the card
      }
      if (e.target.closest('a, button, input')) return;   // e.g. provider links open, card stays put
      if (String(window.getSelection ? window.getSelection() : '').trim()) return;   // selecting text doesn't flip
      // a click/tap anywhere else on the card flips it
      setFlipped(card, !card.classList.contains('is-flipped'), { moveFocus: false });
    });
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && card.classList.contains('is-flipped')) {
        e.stopPropagation();
        setFlipped(card, false);
      }
    });
    grid.appendChild(card);
  });
}

function onFavorite(btn, m) {
  const fav = toggleFavorite(m);
  btn.textContent = fav ? '❤️' : '🤍';
  btn.classList.toggle('is-fav', fav);
  btn.setAttribute('aria-pressed', String(fav));
  showToast(fav ? 'Saved to your list' : 'Removed from your list');
}

/* ---------------------------------------------------------------------- */
/* Modal — full details for one movie                                      */
/* ---------------------------------------------------------------------- */

let modalOpenerEl = null;

function openModal(m, mood) {
  const f = MovieCards.movieFields(m);
  modalOpenerEl = document.activeElement;
  const heroSrc = (m.poster_source !== 'tmdb' && posterDisplayUrl(m.image, 720)) || f.poster;   // same poster, larger
  document.getElementById('modalHero').style.backgroundImage = heroSrc ? `url("${heroSrc}")` : 'none';
  document.getElementById('modalTitle').textContent = f.title;

  document.getElementById('modalMeta').innerHTML = [
    f.rating ? `<span class="rating">★ ${escapeHtml(f.rating)} / 10</span>` : '<span>Rating unavailable</span>',
    f.year ? `<span>${escapeHtml(f.year)}</span>` : '',
    f.duration ? `<span>${escapeHtml(f.duration)}</span>` : '',
    f.genre ? `<span>${escapeHtml(f.genre)}</span>` : '<span>Genre unavailable</span>',
    f.match != null ? `<span class="badge badge-mood">${f.match}% mood match</span>` : '',
  ].join('');
  document.getElementById('modalDesc').textContent = f.overview || 'No storyline available for this title.';
  document.getElementById('modalWatch').innerHTML = watchHtml(m, { compact: false });

  const isFav = favorites.has(movieKey(m));
  const saveBtn = document.getElementById('modalSaveBtn');
  saveBtn.textContent = isFav ? '❤️ Saved' : '🤍 Save';
  saveBtn.classList.toggle('is-fav', isFav);
  saveBtn.onclick = () => {
    const fav = toggleFavorite(m);
    saveBtn.textContent = fav ? '❤️ Saved' : '🤍 Save';
    saveBtn.classList.toggle('is-fav', fav);
    const cardBtn = [...document.querySelectorAll('.movie-card')]
      .map((c) => c.querySelector('[data-action="favorite"]'))
      .find((b, i) => b && currentMovies[i] && movieKey(currentMovies[i]) === movieKey(m));
    if (cardBtn) { cardBtn.textContent = fav ? '❤️' : '🤍'; cardBtn.classList.toggle('is-fav', fav); cardBtn.setAttribute('aria-pressed', String(fav)); }
    showToast(fav ? 'Saved to your list' : 'Removed from your list');
  };

  renderExplain(f, m, mood);

  const overlay = document.getElementById('modalOverlay');
  overlay.inert = false;
  overlay.classList.add('show');
  overlay.setAttribute('aria-hidden', 'false');
  document.getElementById('modalClose').focus();
}

function renderExplain(f, m, mood) {
  const theme = getMoodTheme(m.mood || mood);
  const wrap = document.getElementById('modalExplain');
  wrap.innerHTML = `
    <div class="explain-head">Why this was recommended</div>
    ${f.match != null ? `
      <div class="explain-bar-row">
        <span class="explain-bar-label"><span aria-hidden="true">🎭</span> Mood match</span>
        <div class="progress-glow explain-bar-track"><div class="progress-glow-fill explain-bar-fill" style="width:${f.match}%"></div></div>
        <span class="explain-bar-value">${f.match}%</span>
      </div>` : ''}
    <div class="explain-note">The movie mood classifier reads this film's plot and genre text and gives it a
      ${f.match != null ? `${f.match}% ` : ''}probability of fitting ${escapeHtml(theme.label)}. Rankings blend that
      with the film's rating and popularity.</div>
  `;
}

function closeModal() {
  const overlay = document.getElementById('modalOverlay');
  if (!overlay.classList.contains('show')) return;
  overlay.classList.remove('show');
  overlay.setAttribute('aria-hidden', 'true');
  overlay.inert = true;   // hidden dialog controls must not stay in the tab order
  if (modalOpenerEl && typeof modalOpenerEl.focus === 'function') modalOpenerEl.focus();
}

/* ---------------------------------------------------------------------- */
/* Init                                                                     */
/* ---------------------------------------------------------------------- */

if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', () => {
  initParticles(document.getElementById('particles'));
  renderMoodChips();
  initModeToggle();
  document.getElementById('refreshBtn').addEventListener('click', () => loadMovies({ refresh: true }));

  document.getElementById('modalClose').addEventListener('click', closeModal);
  document.getElementById('modalOverlay').addEventListener('click', (e) => {
    if (e.target.id === 'modalOverlay') closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });

  onMoodThemeChange((e) => {
    const mood = e.detail && e.detail.mood;
    markActiveMood(mood);
    if (mood && mood !== requestedMood) loadMovies();
  });

  loadMovies();
});
