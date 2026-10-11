/**
 * MoodCast — AI Dashboard.
 * Frontend-only orchestration: reads the current mood/mode from the same
 * sessionStorage keys ('mood', 'mode') that mood-theme.js manages, so
 * selections made on the landing page (or on /music, /movies) carry over
 * here automatically. Talks to the existing
 * POST /api/recommend endpoint for the Music/Movies preview tabs — no new
 * backend behavior, just consuming what's already there.
 */

const AI_SUMMARY = {
  happy: {
    match: "You're feeling happy today — we've lined up upbeat music and feel-good movies to match your energy.",
    uplift: "Already feeling good — we've added even more upbeat music and joyful movies to keep the momentum going.",
  },
  sad: {
    match: "You seem a little down today. We've gathered reflective music and heartfelt movies that match how you're feeling.",
    uplift: "You seem a little down today. We've prepared warm, comforting music and uplifting movies to help lift your spirits.",
  },
  angry: {
    match: "You're feeling fired up today. We've pulled high-intensity tracks and movies with real edge.",
    uplift: "Feeling fired up? We've queued cool-down tracks and movies to help you ease into a better headspace.",
  },
  chill: {
    match: "You're in a chill headspace. We've curated laid-back music and easy-watching movies to match.",
    uplift: "You're relaxed — we've added a touch of energetic music and movies to keep things interesting.",
  },
  energetic: {
    match: "You're buzzing with energy. We've queued up high-tempo tracks and adrenaline-charged movies.",
    uplift: "You're already energized — we've balanced it with some feel-good music and movies to match.",
  },
  stressed: {
    match: "You seem stressed today. We've matched calming music and grounding movies to your mood.",
    uplift: "You seem stressed today. We've prepared relaxing music and uplifting movies to help you unwind.",
  },
};

// Opposite-mood map used as a deliberate, explainable fallback for
// Surprise Me when there isn't enough history yet to personalize the
// pick. Roughly the opposite corner of the valence/energy circumplex the
// real ML pipeline uses (src/mood_utils.py / K-Means mood mapping).
const OPPOSITE_MOOD = {
  happy: 'sad', sad: 'happy',
  angry: 'chill', chill: 'angry',
  energetic: 'stressed', stressed: 'energetic',
};

/* ---------------------------------------------------------------------- */
/* Mood summary card                                                       */
/* ---------------------------------------------------------------------- */

function renderSummary() {
  const mood = getCurrentMood();
  const mode = getCurrentMode();
  const theme = getMoodTheme(mood);

  document.getElementById('summaryOrb').textContent = theme.emoji;
  document.getElementById('summaryMoodName').textContent = theme.label;
  document.getElementById('summaryAiText').textContent = AI_SUMMARY[mood][mode === 'uplift' ? 'uplift' : 'match'];
  document.getElementById('summaryModeBadge').innerHTML =
    mode === 'uplift' ? '🚀 Cheer me up' : '🎯 Match my mood';

  // Confidence is only known if the current mood came from text detection
  // (mood-theme.js's setMoodTheme(mood, { confidence }) sets this, and
  // clears it again whenever a mood is picked directly instead — so a
  // stale confidence from an earlier detection never gets misattributed
  // to a manually-picked mood).
  const stored = sessionStorage.getItem('moodConfidence');
  const confEl = document.getElementById('summaryConfValue');
  const fillEl = document.getElementById('summaryConfFill');
  if (stored !== null && !isNaN(parseFloat(stored))) {
    const pct = Math.round(parseFloat(stored) * 100);
    confEl.textContent = pct + '%';
    requestAnimationFrame(() => { fillEl.style.width = pct + '%'; });
  } else {
    confEl.textContent = 'Manually selected';
    fillEl.style.width = '100%';
  }
}

/* ---------------------------------------------------------------------- */
/* Tabs                                                                     */
/* ---------------------------------------------------------------------- */

function initTabs() {
  const buttons = document.querySelectorAll('.tab-btn');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.classList.contains('active')) return;
      buttons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
      const panel = document.getElementById('panel-' + btn.dataset.tab);
      panel.classList.add('active');

      if (btn.dataset.tab === 'music' && !panel.dataset.loaded) loadMusicPreview();
      if (btn.dataset.tab === 'movies' && !panel.dataset.loaded) loadMoviePreview();
    });
  });
}

/* ---------------------------------------------------------------------- */
/* Music / Movies preview (reuses the existing /api/recommend endpoint)    */
/* ---------------------------------------------------------------------- */

/**
 * One music card: real title/artist only (local metadata, else the Spotify
 * oEmbed title in `details`), and an exact Spotify link when the track has one.
 */
function musicCardHtml(t, details) {
  const Track = window.MoodcastTrack;
  const info = Track.trackDisplay(t, details);
  const art = Track.trackArtwork(details);
  const url = Track.spotifyTrackUrl(t);
  const sub = info.title ? (info.artist || '') : (info.artist || 'Not in the available metadata');
  return `
      <div class="card music-card glass">
        <div class="art-wrap">${art
          ? `<img src="${escapeHtml(art)}" alt="" loading="lazy" decoding="async">`
          : '<div class="art-fallback" aria-hidden="true">🎵</div>'}</div>
        <div class="title${info.available ? '' : ' is-missing'}">${escapeHtml(info.heading)}</div>
        ${sub ? `<div class="artist">${escapeHtml(sub)}</div>` : ''}
        ${url
          ? `<a class="card-link" href="${url}" target="_blank" rel="noopener noreferrer"
               aria-label="Open ${escapeHtml(info.title || 'this track')} on Spotify (new tab)">Open in Spotify ↗</a>`
          : '<span class="card-link-missing">Spotify link unavailable</span>'}
      </div>`;
}

/** One poster-backed movie card (only called for movies whose poster loaded). */
function movieCardHtml(m) {
  const rating = m.rating && m.rating !== 'N/A' ? m.rating : null;
  const year = m.year && m.year !== 'N/A' ? m.year : '';
  return `
      <div class="card movie-card glass">
        <div class="poster-wrap">
          <img src="${escapeHtml(m.poster_src)}" alt="Poster for ${escapeHtml(m.title)}" decoding="async">
          ${rating ? `<span class="poster-badge">★ ${escapeHtml(rating)}</span>` : ''}
        </div>
        <div class="movie-meta">
          <div class="title">${escapeHtml(m.title)}</div>
          <div class="year">${escapeHtml(year)}</div>
        </div>
      </div>`;
}

const PREVIEW_COUNT = 8;
let previewSeq = { music: 0, movies: 0 };

async function loadMusicPreview() {
  const grid = document.getElementById('musicGrid');
  const panel = document.getElementById('panel-music');
  const seq = ++previewSeq.music;
  grid.innerHTML = '<div class="loading-state">Loading tracks…</div>';
  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mood: getCurrentMood(), mode: getCurrentMode(), type: 'music', limit: PREVIEW_COUNT }),
    });
    const data = await readJsonResponse(res);
    if (seq !== previewSeq.music) return;
    const tracks = data.results || [];
    panel.dataset.loaded = '1';
    if (!tracks.length) {
      grid.innerHTML = '<div class="empty-state">No tracks found for this mood yet.</div>';
      return;
    }
    grid.innerHTML = tracks.map((t) => musicCardHtml(t)).join('');
    // fill in real titles + covers from Spotify oEmbed (exact track ids); keep the honest fallback on failure
    try {
      const ids = tracks.map((t) => t.track_id).filter(Boolean);
      const details = await readJsonResponse(await fetch('/api/track-details?ids=' + encodeURIComponent(ids.join(','))));
      if (seq !== previewSeq.music) return;
      grid.innerHTML = tracks.map((t) => musicCardHtml(t, (details.tracks || {})[t.track_id])).join('');
    } catch (e) { console.error(e); }
  } catch (err) {
    if (seq !== previewSeq.music) return;
    grid.innerHTML = `<div class="empty-state" role="alert">Couldn't load tracks: ${escapeHtml(err.message || 'request failed')}.</div>`;
    console.error(err);
  }
}

async function loadMoviePreview() {
  const grid = document.getElementById('movieGrid');
  const panel = document.getElementById('panel-movies');
  const seq = ++previewSeq.movies;
  grid.innerHTML = '<div class="loading-state">Loading movies…</div>';
  try {
    // ask for more than we show: only movies whose poster actually loads are displayed
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mood: getCurrentMood(), mode: getCurrentMode(), type: 'movies', limit: PREVIEW_COUNT * 3 }),
    });
    const data = await readJsonResponse(res);
    if (seq !== previewSeq.movies) return;
    const pool = data.results || [];
    panel.dataset.loaded = '1';
    const movies = (await resolveMoviePosters(pool)).slice(0, PREVIEW_COUNT);   // dataset or verified TMDB poster
    if (seq !== previewSeq.movies) return;
    if (!movies.length) {
      grid.innerHTML = pool.length
        ? '<div class="empty-state">None of these recommendations has a poster that loads right now.</div>'
        : '<div class="empty-state">No movies found for this mood yet.</div>';
      return;
    }
    grid.innerHTML = movies.map(movieCardHtml).join('');
  } catch (err) {
    if (seq !== previewSeq.movies) return;
    grid.innerHTML = `<div class="empty-state" role="alert">Couldn't load movies: ${escapeHtml(err.message || 'request failed')}.</div>`;
    console.error(err);
  }
}

/* ---------------------------------------------------------------------- */
/* Mood insights                                                           */
/* ---------------------------------------------------------------------- */

function renderInsightSwatches() {
  const wrap = document.getElementById('insightSwatches');
  const current = getCurrentMood();
  wrap.innerHTML = MOOD_ORDER.map((id) => {
    const theme = MOOD_THEME[id];
    const active = id === current;
    return `<span class="insight-swatch glass" style="${active ? `border-color:${theme.colors.glow};box-shadow:0 0 20px -8px ${theme.colors.glow};` : ''}">
      <span class="dot" style="background:${theme.colors.glow}"></span>${theme.emoji} ${theme.label}
    </span>`;
  }).join('');
}

/* ---------------------------------------------------------------------- */
/* History & Statistics — storage lives in mood-history.js (shared across  */
/* every page); this just renders it here as a timeline + stat cards.      */
/* ---------------------------------------------------------------------- */

function relativeTime(ts) {
  const diff = Math.max(0, Date.now() - ts);
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return mins + 'm ago';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ago';
  const days = Math.floor(hrs / 24);
  return days + 'd ago';
}

function dayLabel(ts) {
  const d = new Date(ts);
  const now = new Date();
  const startOfDay = (date) => new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const diffDays = Math.round((startOfDay(now) - startOfDay(d)) / 86400000);
  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  return d.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' });
}

function renderStats() {
  const wrap = document.getElementById('historyStats');
  const stats = window.MoodHistory ? window.MoodHistory.getStats() : { recommendationCount: 0, avgConfidencePct: null, mostSelectedMood: null };
  const mostTheme = stats.mostSelectedMood ? getMoodTheme(stats.mostSelectedMood) : null;
  wrap.innerHTML = `
    <div class="card-stat glass shine-sweep">
      <div class="stat-num">${stats.recommendationCount}</div>
      <div class="stat-label">Recommendations viewed</div>
    </div>
    <div class="card-stat glass shine-sweep">
      <div class="stat-num">${stats.avgConfidencePct != null ? stats.avgConfidencePct + '%' : '—'}</div>
      <div class="stat-label">Avg. mood confidence</div>
    </div>
    <div class="card-stat glass shine-sweep">
      <div class="stat-num">${mostTheme ? mostTheme.emoji + ' ' + mostTheme.label : '—'}</div>
      <div class="stat-label">Most selected mood</div>
    </div>
  `;
  // Tilt is attached by polish.js's global static-tilt pass (it also
  // re-runs on moodthemechange, after this function regenerates the
  // nodes), so nothing to wire up manually here.
}

function renderHistory() {
  const list = document.getElementById('historyList');
  const history = window.MoodHistory ? window.MoodHistory.getHistory() : [];
  if (!history.length) {
    list.innerHTML = '<div class="empty-state">No mood history yet — it\'ll build up as you use MoodCast.</div>';
    return;
  }

  let html = '';
  let lastDay = null;
  history.forEach((h) => {
    const theme = getMoodTheme(h.mood);
    const label = dayLabel(h.ts);
    if (label !== lastDay) {
      html += `<div class="timeline-day-label">${label}</div>`;
      lastDay = label;
    }
    html += `
      <div class="timeline-row">
        <div class="timeline-dot" style="background:${theme.colors.glow}"></div>
        <div class="timeline-card glass card-float">
          <span class="h-emoji">${theme.emoji}</span>
          <div class="h-body">
            <div class="h-mood">${theme.label}</div>
            <div class="h-mode">${h.mode === 'uplift' ? '🚀 Cheer me up' : '🎯 Match my mood'}</div>
          </div>
          <span class="h-time">${relativeTime(h.ts)}</span>
        </div>
      </div>
    `;
  });
  list.innerHTML = `<div class="timeline">${html}</div>`;
}

/* ---------------------------------------------------------------------- */
/* Surprise Me — intentionally recommends something unexpected, and       */
/* explains why, using the same real /api/recommend endpoint.              */
/* ---------------------------------------------------------------------- */

function pickSurpriseMood() {
  const current = getCurrentMood();
  const stats = window.MoodHistory ? window.MoodHistory.getStats() : null;
  const history = window.MoodHistory ? window.MoodHistory.getHistory() : [];

  if (history.length >= 3) {
    // Personalized pick: the mood you've explored the least.
    const tally = {};
    MOOD_ORDER.forEach((id) => { tally[id] = 0; });
    history.forEach((h) => { if (tally[h.mood] != null) tally[h.mood]++; });
    const least = MOOD_ORDER.filter((id) => id !== current).sort((a, b) => tally[a] - tally[b])[0];
    return { mood: least, reason: 'least-explored' };
  }
  // Fallback: the deliberate opposite of your current mood.
  return { mood: OPPOSITE_MOOD[current] || 'happy', reason: 'opposite' };
}

async function runSurpriseMe() {
  const btn = document.getElementById('surpriseBtn');
  const resultWrap = document.getElementById('surpriseResult');
  btn.disabled = true;
  resultWrap.innerHTML = '<div class="loading-state">Thinking of something unexpected…</div>';
  resultWrap.classList.add('show');

  const { mood, reason } = pickSurpriseMood();
  const theme = getMoodTheme(mood);
  const type = Math.random() < 0.5 ? 'music' : 'movies';
  const mode = Math.random() < 0.5 ? 'match' : 'uplift';

  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mood, mode, type, limit: 6 }),
    });
    const data = await readJsonResponse(res);
    const pool = data.results || [];
    const items = type === 'movies' ? (await resolveMoviePosters(pool)).slice(0, 6) : pool;

    const currentTheme = getMoodTheme(getCurrentMood());
    const why = reason === 'least-explored'
      ? `You've barely explored <strong>${theme.emoji} ${theme.label}</strong> so far — here's something from that side of the spectrum instead of your usual picks.`
      : `You're currently in a <strong>${currentTheme.emoji} ${currentTheme.label}</strong> headspace, so instead of matching it, we deliberately picked <strong>${theme.emoji} ${theme.label}</strong> — its opposite — for a bit of contrast.`;

    resultWrap.innerHTML = `
      <div class="surprise-why glass-panel">
        <div class="surprise-why-label">Why this pick?</div>
        <div class="surprise-why-text">${why}</div>
      </div>
      <div class="surprise-grid">
        ${items.length
          ? items.map((it) => (type === 'movies' ? movieCardHtml(it) : musicCardHtml(it))).join('')
          : '<div class="empty-state">No pick with a working poster this time — try again.</div>'}
      </div>
    `;
    if (type === 'music' && items.length) {
      try {
        const ids = items.map((t) => t.track_id).filter(Boolean);
        const details = await readJsonResponse(await fetch('/api/track-details?ids=' + encodeURIComponent(ids.join(','))));
        resultWrap.querySelector('.surprise-grid').innerHTML =
          items.map((t) => musicCardHtml(t, (details.tracks || {})[t.track_id])).join('');
      } catch (e) { console.error(e); }   // cards keep the honest fallback
    }
  } catch (err) {
    resultWrap.innerHTML = `<div class="empty-state" role="alert">Couldn't fetch a surprise pick: ${escapeHtml(err.message || 'request failed')}.</div>`;
    console.error(err);
  } finally {
    btn.disabled = false;
  }
}

/* ---------------------------------------------------------------------- */
/* Init                                                                     */
/* ---------------------------------------------------------------------- */

document.addEventListener('DOMContentLoaded', () => {
  initParticles(document.getElementById('particles'));
  renderSummary();
  initTabs();
  renderInsightSwatches();
  // Log this dashboard visit itself as a history entry too.
  if (window.MoodHistory) window.MoodHistory.logMoodEvent(getCurrentMood(), getCurrentMode());
  renderStats();
  renderHistory();
  loadMusicPreview();
  document.getElementById('panel-music').dataset.loaded = '1';

  document.getElementById('clearHistoryBtn').addEventListener('click', () => {
    if (window.MoodHistory) window.MoodHistory.clearAll();
    renderStats();
    renderHistory();
    showToast('History cleared');
  });

  document.getElementById('surpriseBtn').addEventListener('click', runSurpriseMe);

  onMoodThemeChange(() => {
    renderSummary(); renderInsightSwatches(); renderStats(); renderHistory();
    // previews show the current mood's recommendations, so refresh any that were loaded
    loadMusicPreview();
    if (document.getElementById('panel-movies').dataset.loaded === '1') loadMoviePreview();
  });
});
