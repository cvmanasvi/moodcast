/**
 * MoodCast — Universal Search.
 * Injects a search trigger into the navbar and a command-palette-style
 * dropdown, on every page that includes this script. Talks only to the
 * existing GET /api/search endpoint (movies/songs/artists/genres,
 * read-only substring search) — no other backend involvement.
 */

(function () {
  const DEBOUNCE_MS = 220;
  let debounceTimer = null;
  let lastQuery = '';

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s == null ? '' : String(s);
    return div.innerHTML;
  }

  function injectDOM() {
    const navInner = document.querySelector('.navbar nav');
    if (!navInner) return; // page doesn't have the shared navbar shape

    const trigger = document.createElement('button');
    trigger.className = 'search-trigger';
    trigger.id = 'searchTrigger';
    trigger.setAttribute('aria-label', 'Search MoodCast');
    trigger.innerHTML = '🔍';
    navInner.parentElement.insertBefore(trigger, navInner);

    const backdrop = document.createElement('div');
    backdrop.className = 'search-backdrop';
    backdrop.id = 'searchBackdrop';
    document.body.appendChild(backdrop);

    const dropdown = document.createElement('div');
    dropdown.className = 'search-dropdown glass-strong';
    dropdown.id = 'searchDropdown';
    dropdown.setAttribute('role', 'dialog');
    dropdown.setAttribute('aria-modal', 'true');
    dropdown.setAttribute('aria-label', 'Search MoodCast');
    dropdown.innerHTML = `
      <div class="search-input-row">
        <span class="search-icon" aria-hidden="true">🔍</span>
        <label for="searchInput" class="visually-hidden">Search movies, songs, artists, genres</label>
        <input type="text" id="searchInput" placeholder="Search movies, songs, artists, genres…" autocomplete="off">
        <span class="search-esc-hint" aria-hidden="true">ESC</span>
      </div>
      <div class="search-results" id="searchResults" aria-live="polite">
        <div class="search-empty">Start typing to search across everything in MoodCast.</div>
      </div>
    `;
    document.body.appendChild(dropdown);

    trigger.addEventListener('click', openSearch);
    backdrop.addEventListener('click', closeSearch);
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        openSearch();
      } else if (e.key === 'Escape') {
        closeSearch();
      }
    });

    document.getElementById('searchInput').addEventListener('input', onInput);
  }

  let searchOpenerEl = null;

  function openSearch() {
    searchOpenerEl = document.activeElement;
    document.getElementById('searchBackdrop').classList.add('show');
    document.getElementById('searchDropdown').classList.add('show');
    const input = document.getElementById('searchInput');
    setTimeout(() => input.focus(), 80);
  }
  function closeSearch() {
    document.getElementById('searchBackdrop').classList.remove('show');
    document.getElementById('searchDropdown').classList.remove('show');
    if (searchOpenerEl && typeof searchOpenerEl.focus === 'function') searchOpenerEl.focus();
  }

  function onInput(e) {
    const q = e.target.value.trim();
    clearTimeout(debounceTimer);
    if (q.length < 2) {
      renderEmpty(q.length === 0 ? 'Start typing to search across everything in MoodCast.' : 'Keep typing…');
      return;
    }
    debounceTimer = setTimeout(() => runSearch(q), DEBOUNCE_MS);
  }

  function renderEmpty(msg) {
    document.getElementById('searchResults').innerHTML = `<div class="search-empty">${msg}</div>`;
  }

  async function runSearch(q) {
    lastQuery = q;
    document.getElementById('searchResults').innerHTML = '<div class="search-loading">Searching…</div>';
    try {
      const res = await fetch('/api/search?q=' + encodeURIComponent(q));
      const data = await readJsonResponse(res);
      if (q !== lastQuery) return; // stale response from a faster later keystroke
      renderResults(data, q);
    } catch (err) {
      if (q !== lastQuery) return;
      document.getElementById('searchResults').innerHTML = '<div class="search-empty">Search is unavailable right now.</div>';
      console.error(err);
    }
  }

  function renderResults(data, q) {
    const wrap = document.getElementById('searchResults');
    const total = (data.movies || []).length + (data.songs || []).length + (data.artists || []).length + (data.genres || []).length;
    if (!total) {
      wrap.innerHTML = `<div class="search-empty">No results for "${escapeHtml(q)}"</div>`;
      return;
    }

    let html = '';

    if (data.movies && data.movies.length) {
      html += `<div class="search-group"><div class="search-group-label">Movies</div>`;
      html += data.movies.map((m) => `
        <div class="search-row">
          ${m.image ? `<img class="search-row-art" src="${m.image}" alt="${escapeHtml(m.title)}" loading="lazy" decoding="async" onerror="this.outerHTML='<div class=&quot;search-row-art-fallback&quot;>🎬</div>'">` : '<div class="search-row-art-fallback">🎬</div>'}
          <div class="search-row-body">
            <div class="search-row-title">${escapeHtml(m.title)}</div>
            <div class="search-row-sub">${escapeHtml(m.year)}</div>
          </div>
          <span class="search-row-badge">★ ${escapeHtml(m.rating)}</span>
        </div>
      `).join('');
      html += `<div class="search-group-footer"><a href="/movies">Open full Movies page →</a></div></div>`;
    }

    if (data.songs && data.songs.length) {
      html += `<div class="search-group"><div class="search-group-label">Songs</div>`;
      // each song row keeps its track_id (songs sharing a title/artist stay distinct)
      const Track = window.MoodcastTrack;
      html += data.songs.map((s) => {
        const info = Track.trackDisplay(s);
        const url = Track.spotifyTrackUrl(s);
        const inner = `
          <div class="search-row-art-fallback">🎵</div>
          <div class="search-row-body">
            <div class="search-row-title">${escapeHtml(info.heading)}</div>
            <div class="search-row-sub">${escapeHtml(info.artist || '')}</div>
          </div>
          ${url ? '<span class="search-row-badge">Spotify ↗</span>' : ''}`;
        // the song's exact Spotify track page, in a new tab (MoodCast doesn't play audio)
        return url
          ? `<a class="search-row" href="${url}" target="_blank" rel="noopener noreferrer"${Track.trackIdAttr(s)}
               aria-label="Open ${escapeHtml(info.title || 'this track')} on Spotify (new tab)">${inner}</a>`
          : `<div class="search-row"${Track.trackIdAttr(s)}>${inner}</div>`;
      }).join('');
      html += `<div class="search-group-footer"><a href="/music">Open full Music page →</a></div></div>`;
    }

    if (data.artists && data.artists.length) {
      html += `<div class="search-group"><div class="search-group-label">Artists</div>`;
      html += data.artists.map((a) => `
        <div class="search-row">
          <div class="search-row-icon">🎤</div>
          <div class="search-row-body"><div class="search-row-title">${escapeHtml(a.name)}</div></div>
        </div>
      `).join('');
      html += `</div>`;
    }

    if (data.genres && data.genres.length) {
      html += `<div class="search-group"><div class="search-group-label">Genres</div>`;
      html += data.genres.map((g) => `
        <div class="search-row">
          <div class="search-row-icon">🏷️</div>
          <div class="search-row-body"><div class="search-row-title">${escapeHtml(g.name)}</div></div>
        </div>
      `).join('');
      html += `</div>`;
    }

    wrap.innerHTML = html;
  }

  document.addEventListener('DOMContentLoaded', injectDOM);
})();
