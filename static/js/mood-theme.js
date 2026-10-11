/**
 * MoodCast — Mood Theme Engine
 * Frontend-only. Reads/writes sessionStorage (same key the existing pages
 * already use) and toggles a `data-mood` attribute on <html>. All color
 * theming lives in CSS (design-system.css `[data-mood="..."]` blocks) —
 * this module's only job is to set the attribute and notify listeners.
 * No network calls, no dependency on any backend endpoint or API shape.
 */

const MOOD_THEME = Object.freeze({
  happy: {
    id: 'happy',
    label: 'Happy',
    emoji: '\u{1F60A}',
    colors: { a: '#ffb020', b: '#ff7a00', glow: '#ffb020' },
    description: 'Yellow → Orange',
  },
  sad: {
    id: 'sad',
    label: 'Sad',
    emoji: '\u{1F622}',
    colors: { a: '#5b6ef5', b: '#8b5cf6', glow: '#7c8cff' },
    description: 'Blue → Purple',
  },
  angry: {
    id: 'angry',
    label: 'Angry',
    emoji: '\u{1F620}',
    colors: { a: '#8b0000', b: '#e53935', glow: '#ff3b30' },
    description: 'Dark red',
  },
  chill: {
    id: 'chill',
    label: 'Chill',
    emoji: '\u{1F60C}',
    colors: { a: '#34d9c4', b: '#22c1dc', glow: '#2de8c8' },
    description: 'Mint → Cyan',
  },
  energetic: {
    id: 'energetic',
    label: 'Energetic',
    emoji: '⚡',
    colors: { a: '#c026d3', b: '#ec4899', glow: '#e040fb' },
    description: 'Purple → Pink',
  },
  stressed: {
    id: 'stressed',
    label: 'Stressed',
    emoji: '\u{1F630}',
    colors: { a: '#0f766e', b: '#14b8a6', glow: '#2dd4bf' },
    description: 'Teal → Green',
  },
});

const MOOD_ORDER = ['happy', 'sad', 'angry', 'chill', 'energetic', 'stressed'];
const DEFAULT_MOOD = 'happy';
const STORAGE_KEY_MOOD = 'mood';
const STORAGE_KEY_MODE = 'mode';
const EVENT_NAME = 'moodthemechange';

function getMoodTheme(id) {
  return MOOD_THEME[id] || MOOD_THEME[DEFAULT_MOOD];
}

function getCurrentMood() {
  const stored = sessionStorage.getItem(STORAGE_KEY_MOOD);
  return MOOD_THEME[stored] ? stored : DEFAULT_MOOD;
}

function getCurrentMode() {
  return sessionStorage.getItem(STORAGE_KEY_MODE) || 'match';
}

/**
 * Apply a mood theme to the document. Safe to call before or after
 * DOMContentLoaded. Persists the choice and dispatches a `moodthemechange`
 * CustomEvent (detail: { mood, theme }) so page-specific scripts can react
 * (e.g. re-render a hero image) without this module knowing about them.
 *
 * `confidence` (0-1) should be passed only when the mood came from the ML
 * text detector (/api/detect-mood) — the Dashboard's "Mood confidence"
 * stat and summary card read it back out of sessionStorage. When it's
 * omitted (the normal case for a direct mood-card/chip click), any
 * confidence value left over from a *previous* text detection is cleared
 * automatically, so it doesn't get misattributed to a mood the user just
 * picked manually.
 */
function setMoodTheme(moodId, { persist = true, silent = false, confidence } = {}) {
  const theme = getMoodTheme(moodId);
  document.documentElement.setAttribute('data-mood', theme.id);

  if (persist) {
    sessionStorage.setItem(STORAGE_KEY_MOOD, theme.id);
    if (confidence !== undefined && confidence !== null) {
      sessionStorage.setItem('moodConfidence', String(confidence));
    } else {
      sessionStorage.removeItem('moodConfidence');
    }
  }

  if (!silent) {
    document.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: { mood: theme.id, theme } }));
  }
  return theme;
}

function setMode(mode) {
  sessionStorage.setItem(STORAGE_KEY_MODE, mode);
  document.dispatchEvent(new CustomEvent('moodmodechange', { detail: { mode } }));
}

function onMoodThemeChange(handler) {
  document.addEventListener(EVENT_NAME, handler);
  return () => document.removeEventListener(EVENT_NAME, handler);
}

/**
 * Wires up any element with [data-mood-select="<moodId>"] to call
 * setMoodTheme on click, and toggles an "active" class based on current
 * selection. Call once per page after the mood picker markup exists.
 */
function bindMoodSelectors(root = document) {
  const els = root.querySelectorAll('[data-mood-select]');
  els.forEach((el) => {
    const moodId = el.getAttribute('data-mood-select');
    el.classList.toggle('active', moodId === getCurrentMood());
    el.addEventListener('click', () => {
      els.forEach((e) => e.classList.remove('active'));
      el.classList.add('active');
      setMoodTheme(moodId);
    });
  });
}

// Auto-initialize on load so theme is correct before first paint of
// mood-dependent components; pages can still call setMoodTheme manually.
(function initMoodThemeEarly() {
  const theme = getMoodTheme(getCurrentMood());
  document.documentElement.setAttribute('data-mood', theme.id);
})();
