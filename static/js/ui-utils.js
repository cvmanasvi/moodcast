/**
 * MoodCast — shared UI utilities.
 * Small, dependency-free helpers used by every page script (previously
 * copy-pasted verbatim into music.js, movies.js, dashboard.js, assistant.js
 * and index.html's inline script). Loaded first so every other script can
 * rely on these being defined. Frontend-only, no backend involvement.
 */

/** Toast notification — expects a `#toastStack` element on the page (all
 * five templates include one). Announces politely to screen readers via
 * the container's aria-live attribute (set in the HTML). */
function showToast(message) {
  const stack = document.getElementById('toastStack');
  if (!stack) return;
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.setAttribute('role', 'status');
  toast.innerHTML = `<span class="toast-dot" aria-hidden="true"></span>${message}`;
  stack.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 2200);
}

/** Escapes a string for safe innerHTML interpolation. */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str == null ? '' : String(str);
  // innerHTML escapes & < > only; also escape quotes so the result is safe in attributes
  return div.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/**
 * Reveal-on-scroll: adds `.in-view` to matched elements the first time
 * they enter the viewport, instead of animating everything immediately on
 * mount (which is wasted work for rows far below the fold in long lists
 * like the poster grid or track list). Falls back to revealing
 * everything immediately if IntersectionObserver isn't available.
 */
function observeReveal(selectorOrList, options = {}) {
  const els = typeof selectorOrList === 'string'
    ? document.querySelectorAll(selectorOrList)
    : selectorOrList;
  if (!els || !els.length) return;

  if (typeof IntersectionObserver === 'undefined') {
    els.forEach((el) => el.classList.add('in-view'));
    return;
  }

  const io = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        observer.unobserve(entry.target);
      }
    });
  }, { rootMargin: '80px 0px', threshold: 0.05, ...options });

  els.forEach((el) => io.observe(el));
  return io;
}

/**
 * Parses a fetch Response as JSON, throwing if the server returned an error
 * status. Without this, a 4xx/5xx body (which has no `results`) would render
 * as an empty "no results" state instead of reaching the caller's catch.
 */
async function readJsonResponse(res) {
  let data = null;
  try {
    data = await res.json();
  } catch {
    // non-JSON body (e.g. a proxy error page) -- fall through to status check
  }
  if (!res.ok) {
    throw new Error((data && data.error) || `Request failed (${res.status})`);
  }
  return data;
}

/**
 * Resolves true if the image at `url` actually loads (a real image, not an
 * error page) within `timeoutMs`, else false. Used to show only movies whose
 * dataset poster still loads, instead of empty poster frames.
 */
function imageLoads(url, timeoutMs = 10000) {
  return new Promise((resolve) => {
    if (!url) { resolve(false); return; }
    const img = new Image();
    const timer = setTimeout(() => { img.onload = img.onerror = null; resolve(false); }, timeoutMs);
    img.onload = () => { clearTimeout(timer); resolve(img.naturalWidth > 1 && img.naturalHeight > 1); };
    img.onerror = () => { clearTimeout(timer); resolve(false); };
    img.src = url;
  });
}

/**
 * Display-size URL for a dataset poster. The movie data's posters are IMDb
 * CDN originals (m.media-amazon.com/images/M/<id>.jpg), often 1-2 MB; that
 * CDN serves the same image resized when "._V1_SX<width>" precedes the
 * extension. Other URLs (and ones already sized) are returned unchanged.
 * Returns null for anything that is not an absolute http(s) URL.
 */
function posterDisplayUrl(url, width = 400) {
  if (typeof url !== 'string' || !/^https?:\/\/[^\s"'<>]+$/i.test(url)) return null;
  const m = url.match(/^(https:\/\/m\.media-amazon\.com\/images\/M\/[^\s"'<>?#]+?)\.(jpe?g|png)$/i);
  if (!m || /\._V1_/.test(m[1])) return url;
  return `${m[1]}._V1_SX${width}.${m[2]}`;
}

/**
 * Movie results whose poster actually loads, in their original (ranked)
 * order, as shallow copies with `poster_src` set to the checked URL.
 */
async function withLoadablePosters(movies, timeoutMs) {
  const srcs = movies.map((m) => posterDisplayUrl(m && m.image));
  const ok = await Promise.all(srcs.map((src) => imageLoads(src, timeoutMs)));
  return movies.flatMap((m, i) => (ok[i] ? [{ ...m, poster_src: srcs[i] }] : []));
}

/** POST /api/movie-details: TMDB poster + India watch providers per movie_id (server-side, cached). */
async function fetchMovieDetails(movies) {
  const ids = movies.map((m) => m.movie_id).filter((id) => Number.isInteger(id));
  if (!ids.length) return { movies: {}, region: 'IN', attribution: '', enabled: false };
  const res = await fetch('/api/movie-details', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ movie_ids: ids.slice(0, 60) }),
  });
  return readJsonResponse(res);
}

/**
 * Movies (in their original order) that have a poster which actually loads:
 * the dataset poster first, else the verified TMDB poster of the same movie.
 * Each result is a copy with `poster_src` and `details` (TMDB lookup, or
 * null if that lookup failed). Movies with neither loadable poster are dropped.
 */
async function resolveMoviePosters(movies, timeoutMs) {
  const detailsReq = fetchMovieDetails(movies).catch(() => null);   // posters still work without it
  const dataset = movies.map((m) => posterDisplayUrl(m && m.image));
  const [datasetOk, details] = await Promise.all([Promise.all(dataset.map((u) => imageLoads(u, timeoutMs))), detailsReq]);
  const info = (m) => (details && details.movies ? details.movies[String(m.movie_id)] || null : null);
  const tmdb = movies.map((m, i) => {
    const d = info(m);
    return !datasetOk[i] && d && typeof d.poster === 'string' && /^https:\/\/image\.tmdb\.org\/t\/p\/\w+\/[\w-]+\.(jpe?g|png)$/i.test(d.poster) ? d.poster : null;
  });
  const tmdbOk = await Promise.all(tmdb.map((u) => imageLoads(u, timeoutMs)));
  const out = movies.flatMap((m, i) => {
    const src = datasetOk[i] ? dataset[i] : tmdbOk[i] ? tmdb[i] : null;
    return src ? [{ ...m, poster_src: src, poster_source: datasetOk[i] ? 'dataset' : 'tmdb', details: info(m) }] : [];
  });
  out.meta = { region: (details && details.region) || 'IN', attribution: (details && details.attribution) || '', detailsOk: !!details };
  return out;
}

/* ---------------------------------------------------------------------- */
/* Per-tab memory of results, so a mood/mode shows the same picks until the */
/* user asks to refresh, and refreshes can avoid recently shown items.      */
/* ---------------------------------------------------------------------- */

const ResultMemory = {
  TTL_MS: 30 * 60 * 1000,
  _get(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
  _set(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch { /* storage full or blocked */ } },
  /** Cached value for this view, or null if missing/expired. */
  load(view) {
    const v = this._get('mc:view:' + view);
    return v && Date.now() - v.at < this.TTL_MS ? v.value : null;
  },
  save(view, value) { this._set('mc:view:' + view, { at: Date.now(), value }); },
  /** Ids shown recently for this view (newest last), capped. */
  recent(view) { return this._get('mc:seen:' + view) || []; },
  remember(view, ids, cap = 300) {
    const seen = this.recent(view).filter((id) => !ids.includes(id)).concat(ids);
    this._set('mc:seen:' + view, seen.slice(-cap));
  },
};

/** Generic debounce helper (used by search.js and available to any other
 * script that wants it, replacing ad-hoc setTimeout/clearTimeout pairs). */
function debounce(fn, wait) {
  let t = null;
  return function (...args) {
    clearTimeout(t);
    t = setTimeout(() => fn.apply(this, args), wait);
  };
}
