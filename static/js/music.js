/**
 * MoodCast — Music page driver.
 *
 * Talks to POST /api/recommend ({ mood, mode, type: 'music', limit }) and
 * renders the tracks it returns in the order returned (the backend's ranking).
 *
 * - No audio plays in MoodCast: each track's primary action opens that exact
 *   track on Spotify (open.spotify.com/track/<track_id>) in a new tab.
 * - Titles/artists come from local metadata. For tracks without a local
 *   title, GET /api/track-details fetches the real title and cover art for
 *   the exact track_id from Spotify's public oEmbed endpoint (no artist).
 *   Nothing is guessed; tracks with no title anywhere say so.
 * - A mood/mode keeps showing the same picks (per browser tab) until the user
 *   presses Refresh, which asks the backend for picks not shown recently.
 * - The "Popular Artists" ON/OFF switch changes the list to POST /api/popular (a
 *   separate dataset and pipeline) for the same mood/mode; switching it OFF
 *   restores the normal list. Each mode keeps its own remembered picks.
 */

const Track = window.MoodcastTrack;
const { trackIdentity: trackKey, trackDisplay, trackArtwork, spotifyTrackUrl } = Track;

const LIMIT = 30;
const TEMPO_MIN_BPM = 60;
const TEMPO_MAX_BPM = 200;

/** Real audio features as 0-100 bars, or null if the track lacks any of them. */
function audioProfile(t) {
  const af = t.audio_features;
  const hasReal = af && af.energy != null && af.valence != null && af.danceability != null
    && af.tempo != null && af.acousticness != null;
  if (!hasReal) return null;
  const pct = (v) => Math.max(2, Math.min(98, Math.round(v * 100)));
  return {
    energy: pct(af.energy),
    dance: pct(af.danceability),
    valence: pct(af.valence),
    tempo: Math.max(2, Math.min(98, Math.round(((af.tempo - TEMPO_MIN_BPM) / (TEMPO_MAX_BPM - TEMPO_MIN_BPM)) * 100))),
    acoustic: pct(af.acousticness),
  };
}

/* ---------------------------------------------------------------------- */
/* Favorites (localStorage, frontend-only — no backend/user account)       */
/* ---------------------------------------------------------------------- */

const FAV_KEY = 'moodcast_favorite_tracks';

function loadFavorites() {
  try { return new Set(JSON.parse(localStorage.getItem(FAV_KEY)) || []); }
  catch { return new Set(); }
}
function saveFavorites(set) {
  try { localStorage.setItem(FAV_KEY, JSON.stringify([...set])); } catch { /* storage unavailable */ }
}
let favorites = loadFavorites();

function toggleFavorite(t) {
  const isFav = Track.toggleFavorite(favorites, t);
  saveFavorites(favorites);
  return isFav;
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
      loadMusic();
    });
  });
}

/* ---------------------------------------------------------------------- */
/* Loading tracks                                                          */
/* ---------------------------------------------------------------------- */

let currentTracks = [];
let trackDetails = {};      // track_id -> { title, artwork } from /api/track-details
let requestSeq = 0;
let requestedMood = null;   // mood of the latest request, so a theme change elsewhere (assistant) reloads
let popularMode = false;    // Popular Artists toggle; normal recommendations when off

const viewKey = () => `${popularMode ? 'popular' : 'music'}:${getCurrentMood()}:${getCurrentMode()}`;

const SOURCE_NOTE = {
  music: 'Titles and artists come from a local dataset matched by Spotify track ID; for tracks it doesn\'t cover, the title and cover come from Spotify\'s public oEmbed service, which doesn\'t provide the artist.',
  popular: 'Popular Artists uses a separate local dataset: titles and artist names are shown as listed there, and covers come from Spotify\'s public oEmbed service. Moods are predicted by MoodCast\'s mood model from audio features, not chosen by people.',
};

function setHeader(mood, mode) {
  const theme = getMoodTheme(mood);
  document.getElementById('pageKicker').textContent =
    `${theme.emoji} Music · ${popularMode ? 'Popular Artists · ' : ''}${mode === 'uplift' ? 'Cheer me up' : 'Match my mood'}`;
  document.getElementById('pageTitle').textContent = popularMode
    ? (mode === 'uplift' ? `Popular songs to lift a ${theme.label.toLowerCase()} mood` : `Popular ${theme.label.toLowerCase()} songs`)
    : (mode === 'uplift' ? `Tracks to lift a ${theme.label.toLowerCase()} mood` : `${theme.label} tracks`);
  document.getElementById('sourceNote').textContent = SOURCE_NOTE[popularMode ? 'popular' : 'music'];
}

function setStatus(html, busy = false) {
  document.getElementById('statusLine').innerHTML = (busy ? '<span class="spinner" aria-hidden="true"></span>' : '') + html;
}

function setRefreshBusy(busy) {
  const btn = document.getElementById('refreshBtn');
  btn.disabled = busy;
  btn.setAttribute('aria-busy', String(busy));
}

/**
 * Show this mood/mode's picks. Uses the picks already shown in this tab
 * unless `refresh` is set, in which case the backend is asked for tracks not
 * shown recently.
 */
async function loadMusic({ refresh = false } = {}) {
  const mood = getCurrentMood();
  const mode = getCurrentMode();
  const view = viewKey();
  const seq = ++requestSeq;
  requestedMood = mood;
  setHeader(mood, mode);

  const popular = popularMode;
  const cached = !refresh && ResultMemory.load(view);
  if (cached && Array.isArray(cached.results) && cached.results.length) {
    currentTracks = cached.results;
    trackDetails = cached.details || {};
    renderTracks(currentTracks);
    summarize('Same picks as before for this mood · press Refresh for new ones' + (cached.limitedNote || ''));
    fetchDetails(seq, view);
    return;
  }

  const list = document.getElementById('trackList');
  const previous = currentTracks;
  list.setAttribute('aria-busy', 'true');
  if (!refresh) list.innerHTML = skeletonRows(8);
  setRefreshBusy(true);
  setStatus(refresh ? 'Finding tracks you haven\'t seen…' : 'Loading recommendations…', true);

  try {
    const exclude = refresh ? ResultMemory.recent(view) : [];
    const res = await fetch(popular ? '/api/popular' : '/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(popular
        ? { mood, mode, limit: LIMIT, ...(exclude.length ? { exclude } : {}) }
        : { mood, mode, type: 'music', limit: LIMIT, ...(exclude.length ? { exclude } : {}) }),
    });
    const data = await readJsonResponse(res);
    if (seq !== requestSeq) return;   // a newer mood/mode request superseded this one
    const results = Array.isArray(data.results) ? data.results : [];

    if (popular && data.available === false) {
      if (refresh && previous.length) {
        // keep the valid list on screen rather than replacing it with an explanation
        currentTracks = previous;
        renderTracks(currentTracks);
        setStatus('Couldn\'t refresh: Popular Artists is unavailable on this server right now — showing your previous picks.');
        return;
      }
      // the separate dataset isn't prepared here: explain, keep the normal list's cache untouched
      currentTracks = [];
      list.innerHTML = `<div class="state-box" role="status"><strong>Popular Artists isn't available yet</strong>${escapeHtml(data.message || 'Its dataset has not been prepared on this server.')}
        <br>Press “Popular Artists” again to return to your recommendations.</div>`;
      setStatus('Popular Artists is unavailable on this server.');
      return;
    }

    if (refresh && !results.length) {
      // nothing new is eligible: keep the current picks rather than repeating ones already seen
      currentTracks = previous;
      renderTracks(currentTracks);
      summarize('No other tracks are available for this mood right now — these are still your picks');
      return;
    }
    currentTracks = results;
    trackDetails = {};
    if (popular) {
      const n = currentTracks.length;
      let limitedNote = '';
      if (data.limited && n && data.limited_by === 'artist_cap') {
        // more songs fit the mood, but each would repeat an artist more than twice
        limitedNote = ` · only ${n}${refresh ? ' new' : ''} songs could be picked without repeating an artist more than twice`;
      } else if (data.limited && n) {
        limitedNote = refresh ? ` · only ${n} songs you haven't seen fit this mood, so none are repeated`
                              : ` · only ${n} songs by these artists fit this mood`;
      }
      const note = (refresh ? 'Fresh picks from popular artists' : 'Popular artists · mood predicted from audio features') + limitedNote;
      ResultMemory.save(view, { results: currentTracks, details: {}, limitedNote });
      ResultMemory.remember(view, currentTracks.map((t) => t.track_id).filter(Boolean));
      renderTracks(currentTracks);
      if (!currentTracks.length) {
        list.innerHTML = '<div class="state-box"><strong>No popular songs fit this mood yet</strong>None of the popular artists\' songs in the dataset are a confident match for this mood and mode. Try another mood or mode.</div>';
        setStatus('No popular-artist songs match this mood and mode.');
        return;
      }
      summarize(note);
      fetchDetails(seq, view);
      return;
    }
    ResultMemory.save(view, { results: currentTracks, details: {} });
    ResultMemory.remember(view, currentTracks.map((t) => t.track_id).filter(Boolean));
    renderTracks(currentTracks);
    if (!currentTracks.length) { setStatus('No tracks found for this mood and mode.'); return; }
    const short = data.refresh && data.refresh.exhausted
      ? ` · only ${currentTracks.length} new tracks were available, so none you've already seen are repeated` : '';
    summarize((refresh ? 'Fresh picks' : 'Ranked by the mood model from each track\'s audio features') + short);
    fetchDetails(seq, view);
  } catch (err) {
    if (seq !== requestSeq) return;
    if (refresh && previous.length) {
      currentTracks = previous;
      renderTracks(currentTracks);
      setStatus(`Couldn't refresh (${escapeHtml(err.message || 'request failed')}) — showing your previous picks.`);
    } else {
      currentTracks = [];
      list.innerHTML = `<div class="state-box is-error" role="alert"><strong>Couldn't load tracks</strong>${escapeHtml(err.message || 'Request failed')}.
        <br><button type="button" class="btn btn-glass retry-btn">Try again</button></div>`;
      list.querySelector('.retry-btn').addEventListener('click', () => loadMusic());
      setStatus('Recommendations could not be loaded.');
    }
    console.error(err);
  } finally {
    if (seq === requestSeq) {
      list.setAttribute('aria-busy', 'false');
      setRefreshBusy(false);
    }
  }
}

function summarize(note) {
  const named = currentTracks.filter((t) => trackDisplay(t, trackDetails[t.track_id]).available).length;
  setStatus(`${currentTracks.length} tracks · ${named} with a title · ${escapeHtml(note)}`);
}

/** Real titles + cover art for the shown tracks (Spotify oEmbed, exact track_id), applied in place. */
async function fetchDetails(seq, view) {
  const ids = currentTracks.map((t) => t.track_id)
    .filter((id) => typeof id === 'string' && /^[0-9A-Za-z]{22}$/.test(id) && !(id in trackDetails));
  if (!ids.length) return;
  document.querySelectorAll('.track-title.is-pending').forEach((el) => el.setAttribute('aria-busy', 'true'));
  try {
    const res = await fetch('/api/track-details?ids=' + encodeURIComponent(ids.join(',')));
    const data = await readJsonResponse(res);
    if (seq !== requestSeq) return;
    Object.assign(trackDetails, data.tracks || {});
  } catch (err) {
    console.error(err);   // titles fall back to the honest "unavailable" text below
  }
  if (seq !== requestSeq) return;
  ids.forEach((id) => { if (!trackDetails[id]) trackDetails[id] = { title: null, artwork: null }; });  // settled
  ResultMemory.save(view, { ...(ResultMemory.load(view) || {}), results: currentTracks, details: trackDetails });
  currentTracks.forEach((t, i) => updateRowIdentity(i, t));
  const status = document.getElementById('statusLine').textContent;
  const named = currentTracks.filter((t) => trackDisplay(t, trackDetails[t.track_id]).available).length;
  document.getElementById('statusLine').textContent = status.replace(/· \d+ with a title/, `· ${named} with a title`);
}

function skeletonRows(n) {
  return Array.from({ length: n }, () => '<div class="skeleton track-skeleton" aria-hidden="true"></div>').join('');
}

/* ---------------------------------------------------------------------- */
/* Track list rendering                                                    */
/* ---------------------------------------------------------------------- */

// a track without a local title whose oEmbed lookup hasn't finished yet
const pendingDetails = (t) => !trackDisplay(t).available && Track.hasTrackId(t) && !(t.track_id in trackDetails);

function identityHtml(t, i) {
  const d = trackDetails[t.track_id];
  const info = trackDisplay(t, d);
  const art = trackArtwork(d);
  const pending = pendingDetails(t);
  const title = pending ? 'Loading title…' : info.heading;
  const sub = [];
  if (info.artist) sub.push(escapeHtml(info.artist));
  else if (!info.title && !pending) sub.push('Not in the available metadata');
  if (t.release_year) sub.push(escapeHtml(String(t.release_year)));
  const label = info.title ? info.title + (info.artist ? ` by ${info.artist}` : '') : `track ${i + 1}`;
  return {
    label,
    art: art
      ? `<img class="track-cover" src="${escapeHtml(art)}" alt="" loading="lazy" decoding="async" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'track-cover-placeholder',textContent:'♪'}))">`
      : '<span class="track-cover-placeholder" aria-hidden="true">♪</span>',
    info: `<div class="track-title${info.available ? '' : ' is-missing'}${pending ? ' is-pending' : ''}"
             title="${escapeHtml(info.title || '')}">${escapeHtml(title)}</div>
           ${sub.length ? `<div class="track-artist${info.artist ? '' : ' is-missing'}">${sub.join(' · ')}</div>` : ''}`,
  };
}

function updateRowIdentity(i, t) {
  const row = document.querySelector(`.track-card[data-idx="${i}"]`);
  if (!row) return;
  const id = identityHtml(t, i);
  row.querySelector('.track-art').innerHTML = id.art;
  row.querySelector('.track-info').innerHTML = id.info;
  const link = row.querySelector('.spotify-link');
  if (link) link.setAttribute('aria-label', `Open ${id.label} on Spotify (new tab)`);
}

function renderTracks(tracks) {
  const list = document.getElementById('trackList');
  list.innerHTML = '';
  if (!tracks.length) {
    list.innerHTML = '<div class="state-box"><strong>No tracks found</strong>Try another mood or mode.</div>';
    return;
  }
  tracks.forEach((t, i) => {
    const row = document.createElement('div');
    row.className = 'track-card glass';
    row.dataset.idx = i;
    row.setAttribute('role', 'listitem');
    if (Track.hasTrackId(t)) row.dataset.trackId = t.track_id;

    const id = identityHtml(t, i);
    const url = spotifyTrackUrl(t);
    const conf = t.mood_confidence != null ? Math.round(t.mood_confidence * 100) : null;
    const isFav = favorites.has(trackKey(t));

    row.innerHTML = `
      <span class="track-idx">${i + 1}</span>
      <div class="track-art">${id.art}</div>
      <div class="track-info">${id.info}</div>
      <div class="track-match">
        ${conf != null ? `
          <div class="match-bar-wrap" title="Mood model confidence">
            <div class="progress-glow match-bar"><div class="progress-glow-fill match-bar-fill" style="width:${conf}%"></div></div>
            <span class="match-pct">${conf}%</span>
          </div>` : '<span class="match-none">—</span>'}
      </div>
      <span class="track-duration">${escapeHtml(t.duration || '')}</span>
      <div class="track-actions">
        ${url
          ? `<a class="spotify-link" href="${url}" target="_blank" rel="noopener noreferrer"
               aria-label="Open ${escapeHtml(id.label)} on Spotify (new tab)">Open in Spotify <span aria-hidden="true">↗</span></a>`
          : '<span class="spotify-missing">Spotify link unavailable</span>'}
        <button type="button" class="track-action-btn${isFav ? ' is-fav' : ''}" data-action="favorite"
          aria-pressed="${isFav}" aria-label="Favorite ${escapeHtml(id.label)}" title="Favorite">${isFav ? '❤️' : '🤍'}</button>
        <button type="button" class="track-action-btn" data-action="share" aria-label="Share ${escapeHtml(id.label)}" title="Share">🔗</button>
        <button type="button" class="track-action-btn" data-action="explain" aria-expanded="false"
          aria-controls="explain-${i}" aria-label="Why ${escapeHtml(id.label)}?" title="Why this track?">✨</button>
      </div>
      <div class="track-explain" id="explain-${i}"></div>
    `;

    row.addEventListener('click', (e) => {
      const actionBtn = e.target.closest('[data-action]');
      if (!actionBtn) return;
      const action = actionBtn.dataset.action;
      if (action === 'favorite') onFavoriteClick(actionBtn, t);
      else if (action === 'share') onShareClick(t);
      else if (action === 'explain') toggleExplain(row, i, t, actionBtn);
    });

    list.appendChild(row);
  });
}

function onFavoriteClick(btn, t) {
  const isFav = toggleFavorite(t);
  btn.textContent = isFav ? '❤️' : '🤍';
  btn.classList.toggle('is-fav', isFav);
  btn.setAttribute('aria-pressed', String(isFav));
  showToast(isFav ? 'Added to favorites' : 'Removed from favorites');
}

function onShareClick(t) {
  const info = trackDisplay(t, trackDetails[t.track_id]);
  const url = spotifyTrackUrl(t);
  const name = info.title ? info.title + (info.artist ? ` — ${info.artist}` : '') : 'A track';
  const text = `${name} (matched to my ${t.mood || getCurrentMood()} mood on MoodCast)${url ? ' ' + url : ''}`;
  if (navigator.share) {
    navigator.share({ title: info.title || 'MoodCast track', text, ...(url ? { url } : {}) }).catch(() => {});
  } else if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(() => showToast('Copied to clipboard'), () => showToast('Could not copy'));
  } else {
    showToast('Sharing not supported in this browser');
  }
}

function toggleExplain(row, i, t, btn) {
  const panel = row.querySelector(`#explain-${i}`);
  const isOpen = panel.classList.contains('open');
  document.querySelectorAll('.track-explain.open').forEach((p) => {
    if (p !== panel) {
      p.classList.remove('open'); p.innerHTML = '';
      const other = p.closest('.track-card').querySelector('[data-action="explain"]');
      if (other) other.setAttribute('aria-expanded', 'false');
    }
  });
  if (isOpen) {
    panel.classList.remove('open');
    panel.innerHTML = '';
    btn.setAttribute('aria-expanded', 'false');
    return;
  }
  const profile = audioProfile(t);
  const mood = escapeHtml(t.mood || getCurrentMood());
  if (!profile) {
    panel.innerHTML = `<div class="explain-inner glass-panel"><div class="explain-note">Audio features are unavailable for this track, so no breakdown is shown.</div></div>`;
  } else {
    const bars = [
      { label: 'Energy', value: profile.energy, icon: '⚡' },
      { label: 'Danceability', value: profile.dance, icon: '💃' },
      { label: 'Valence', value: profile.valence, icon: '☀️' },
      { label: 'Tempo', value: profile.tempo, icon: '🥁' },
      { label: 'Acousticness', value: profile.acoustic, icon: '🎻' },
    ];
    panel.innerHTML = `
      <div class="explain-inner glass-panel">
        <div class="explain-head">Why this track matches your ${mood} mood</div>
        <div class="explain-bars">
          ${bars.map((b) => `
            <div class="explain-bar-row">
              <span class="explain-bar-label"><span aria-hidden="true">${b.icon}</span> ${b.label}</span>
              <div class="progress-glow explain-bar-track"><div class="progress-glow-fill explain-bar-fill" style="width:${b.value}%"></div></div>
              <span class="explain-bar-value">${b.value}</span>
            </div>
          `).join('')}
        </div>
        <div class="explain-note">${popularMode
          ? 'Audio features from the Popular Artists dataset; MoodCast\'s mood model predicted the mood from them (not human-validated).'
          : 'Audio features from MoodCast\'s music dataset, the same values its mood model uses.'}</div>
      </div>
    `;
  }
  panel.classList.add('open');
  btn.setAttribute('aria-expanded', 'true');
}

/* ---------------------------------------------------------------------- */
/* Init                                                                     */
/* ---------------------------------------------------------------------- */

/** Render the Popular Artists switch from `popularMode`, the only source of truth. */
function renderPopularSwitch() {
  const sw = document.getElementById('popularSwitch');
  sw.setAttribute('aria-checked', String(popularMode));
  sw.dataset.state = popularMode ? 'on' : 'off';
  sw.querySelector('.switch-state').textContent = popularMode ? 'On' : 'Off';
}

function initPopularSwitch() {
  renderPopularSwitch();
  // a native <button role="switch">: click, Enter and Space all arrive as 'click'
  document.getElementById('popularSwitch').addEventListener('click', () => {
    popularMode = !popularMode;
    renderPopularSwitch();
    loadMusic();      // each mode shows its own remembered picks, or loads them once
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initParticles(document.getElementById('particles'));
  renderMoodChips();
  initModeToggle();
  initPopularSwitch();
  document.getElementById('refreshBtn').addEventListener('click', () => loadMusic({ refresh: true }));
  onMoodThemeChange((e) => {
    const mood = e.detail && e.detail.mood;
    markActiveMood(mood);
    if (mood && mood !== requestedMood) loadMusic();
  });
  loadMusic();
});
