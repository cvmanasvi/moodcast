/**
 * MoodCast — Mood History & Stats tracker.
 * Frontend-only, single source of truth for everything stored in
 * localStorage about mood usage. Included on every page (before the
 * page-specific script) so history/stats build up no matter which page
 * the user is on, not just the Dashboard.
 *
 * How it observes activity without editing every page's code:
 *  - Listens for the 'moodthemechange' event that mood-theme.js already
 *    dispatches on every setMoodTheme() call, and logs a history entry.
 *  - Wraps window.fetch once to observe responses from the two existing
 *    endpoints (/api/recommend, /api/detect-mood) and update counters —
 *    the real network call and its return value are completely
 *    untouched, this only reads the response for bookkeeping.
 *
 * No new backend endpoints are used here at all — this is 100% derived
 * from data already flowing through the two existing APIs.
 */

(function () {
  const HISTORY_KEY = 'moodcast_history';
  const HISTORY_MAX = 30;
  const REC_COUNT_KEY = 'moodcast_rec_count';
  const CONF_LOG_KEY = 'moodcast_confidence_log';
  const CONF_LOG_MAX = 40;

  function safeParse(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
    catch { return fallback; }
  }

  function getHistory() {
    return safeParse(HISTORY_KEY, []);
  }
  function saveHistory(list) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, HISTORY_MAX)));
  }

  function logMoodEvent(mood, mode) {
    const history = getHistory();
    const last = history[0];
    const now = Date.now();
    // Collapse rapid duplicate events (e.g. re-render firing the same
    // change twice) instead of spamming the timeline.
    if (!last || last.mood !== mood || last.mode !== mode || now - last.ts > 15000) {
      history.unshift({ mood, mode, ts: now });
      saveHistory(history);
    }
  }

  function incrementRecCount() {
    const n = parseInt(localStorage.getItem(REC_COUNT_KEY) || '0', 10) + 1;
    localStorage.setItem(REC_COUNT_KEY, String(n));
  }

  function logConfidence(confidence) {
    if (confidence == null || isNaN(confidence)) return;
    const log = safeParse(CONF_LOG_KEY, []);
    log.unshift({ v: confidence, ts: Date.now() });
    localStorage.setItem(CONF_LOG_KEY, JSON.stringify(log.slice(0, CONF_LOG_MAX)));
  }

  function getStats() {
    const history = getHistory();
    const recCount = parseInt(localStorage.getItem(REC_COUNT_KEY) || '0', 10);
    const confLog = safeParse(CONF_LOG_KEY, []);

    let avgConfidencePct = null;
    if (confLog.length) {
      const avg = confLog.reduce((s, e) => s + e.v, 0) / confLog.length;
      avgConfidencePct = Math.round(avg * 100);
    }

    let mostSelectedMood = null;
    if (history.length) {
      const tally = {};
      history.forEach((h) => { tally[h.mood] = (tally[h.mood] || 0) + 1; });
      mostSelectedMood = Object.keys(tally).sort((a, b) => tally[b] - tally[a])[0];
    }

    return { recommendationCount: recCount, avgConfidencePct, mostSelectedMood, historyCount: history.length };
  }

  function clearAll() {
    localStorage.removeItem(HISTORY_KEY);
    localStorage.removeItem(REC_COUNT_KEY);
    localStorage.removeItem(CONF_LOG_KEY);
  }

  /* ---- passive observation, no behavior change to existing calls ---- */

  document.addEventListener('moodthemechange', (e) => {
    const mood = e.detail && e.detail.mood;
    if (mood) logMoodEvent(mood, typeof getCurrentMode === 'function' ? getCurrentMode() : 'match');
  });

  const _realFetch = window.fetch.bind(window);
  window.fetch = function (url, opts) {
    const p = _realFetch(url, opts);
    if (url === '/api/recommend') {
      p.then((res) => {
        res.clone().json().then((data) => {
          if (data && Array.isArray(data.results) && data.results.length) incrementRecCount();
        }).catch(() => {});
      }).catch(() => {});
    } else if (url === '/api/detect-mood') {
      p.then((res) => {
        res.clone().json().then((data) => {
          if (data && typeof data.confidence === 'number') logConfidence(data.confidence);
        }).catch(() => {});
      }).catch(() => {});
    }
    return p;
  };

  window.MoodHistory = { getHistory, logMoodEvent, getStats, clearAll };
})();
