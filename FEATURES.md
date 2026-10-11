# MoodCast — Features & Functionality Reference

> **Data note.** Music features need the locally built music catalog, and movie
> features need `data/movies_final.csv`. Neither is distributed with this
> repository (see [README.md](README.md) and [DATA.md](DATA.md)). Without them,
> those features answer with an explanation instead of results.

A reference of everything the app currently does and every piece of data available to display, meant as a planning doc for a UI redesign.

## Core concept

The user picks (or types) a mood, picks a mode, and gets a ranked list of music tracks and/or movies that fit. There are two independent recommendation engines (music, movies) sharing one mood vocabulary and one mode concept.

## The 6 moods

`happy`, `sad`, `angry`, `chill`, `energetic`, `stressed` — this is the full set, defined in `src/config.py`, and both backend and frontend now use these exact 6 names everywhere (no more alias translation).

## The 2 modes

- **Match my mood** — recommends content that reflects the mood you're in.
- **Cheer me up** — recommends content aimed at shifting you toward a better mood (each source mood has its own deliberately different target: sad → warmth, angry → cool off then lift, stressed → pure calm-down, etc. — see `MOOD_TO_TAGS_CHEER` in `src/config.py`).

## Mood input methods

1. **Click a mood card/chip** — direct selection, persisted in `sessionStorage` so it carries across pages.
2. **Free-text mood detection** (`src/mood_utils.py`, exposed as `POST /api/detect-mood` and used by the Flask assistant's free-text input, `static/js/assistant.js`) — type a sentence like "I've been so stressed about work" and a trained classifier detects the mood. Returns the mood, a confidence score and `uncertain: true` when the model's probability is below 0.30, which the pages show as a tentative guess. On small hand-written checks it gets about 83% of sentences it hasn't seen right (15/18 held-out, 30/36 blind probe); it is not validated on real user text (see `evaluate_text_mood.py`).

## Music feature set

**Data available per track** (default: the 278k catalog, `data/music_catalog_278k.csv`, 277,938 tracks):
- `track_id` — the canonical identity; `spotify_url` is built from it (no API call)
- `title`, `artist` — from the optional local metadata when available, otherwise "Unknown title" / "Unknown artist" (see METADATA.md)
- `image` — empty (no artwork source); the UI shows a placeholder. `album` is null
- `duration` — formatted `M:SS` from the catalog's real duration, or null if missing
- `mood` — the dataset's label for that track; `mood_confidence` — the audio-feature classifier's probability for that label
- `audio_features` — valence, energy, danceability, tempo, acousticness from the catalog (inherited from the Moodify dataset, see DATA.md), used by the "why this track" panel

**Ranking logic**: target-coverage ranking (see README "Music Recommendations"); popularity plays no part.

The 278k catalog is the only music catalog. The legacy `data/music_final.csv` (1,023 tracks, Spotify-hosted album art, **synthetic** audio features, cosine-similarity + popularity ranking) and its rollback switch were retired in Phase 2H.4.

**Current UI behavior** (`templates/music.html`):
- Mood chip strip + match/cheer-up toggle at the top
- A "hero panel" showing the selected mood with a colored gradient and playlist-style framing
- Scrollable track list, each row: index number (becomes a play icon on hover), album art, title, artist, match-confidence badge, duration, play button
- Clicking a row or its play button starts a **simulated** now-playing bar fixed to the bottom: art, title, artist, prev/play-pause/next, a seekable progress bar. This is fake — there's no real audio, just a timer counting up to the track's real duration.
- "Play all" button starts from the top of the list.

**Not implemented**: real audio playback/preview, search within results, filtering/sorting (by popularity, duration, etc.), volume control, queue management, saving/liking tracks, pagination (currently caps at ~30 results per request).

## Movie feature set

**Data available per movie** (from `data/movies_final.csv`, ~12,500 movies):
- `title` (cleaned of a baked-in year suffix)
- `description` — plot overview (missing for ~36 movies, shown as "No description available." rather than breaking)
- `image` — poster URL (some are placeholder images when the source had none)
- `rating` — /10
- `year`
- `duration` — runtime in minutes (recovered from a raw source file the original pipeline had dropped; ~98.5% coverage, "N/A" for the rest)
- `genre` — comma-separated, cleaned from a raw Python-list string
- `mood_confidence` — 0–1, the classifier's confidence that this movie fits the selected mood, based on reading the plot + genre text

**Ranking logic**: 65% predicted mood-match probability + 20% rating + 15% popularity.

**Current UI behavior** (`templates/movies.html`):
- Mood chip strip + match/cheer-up toggle at the top
- A large Netflix-style hero banner featuring the #1 ranked movie: blurred backdrop image, title, rating/year/runtime/genre line, truncated description, match-confidence badge, "View details" button
- A poster grid below (responsive, auto-fills columns) — each poster shows rating badge (top-right), match-confidence badge (top-left), and on-hover overlay with title + primary genre + year
- Clicking any poster (or "View details") opens a modal with the full backdrop, title, full meta line, and complete (untruncated) description

**Not implemented**: search, genre filtering, "more like this," watchlist/favorites, trailers, cast/crew info, pagination beyond the first ~30 results, sorting controls.

## Navigation & state

- Three pages: `/` (landing/mood picker), `/music`, `/movies` — classic multi-page navigation, no SPA framework.
- Selected mood and mode are stored in `sessionStorage` and read on every page load, so switching pages preserves your selection.
- Landing page also has two direct CTA buttons ("Show me music" / "Show me movies") that jump straight to results with whatever mood/mode is currently selected.

## Landing page content

- Animated gradient headline + tagline
- Mood/mode selectors (same as above)
- A static 3-card "How it works" section explaining the ML in plain terms (text detection, audio matching, plot-aware movie matching)
- A static 3-stat strip (277k+ training tracks, text accuracy, 6 moods). The earlier "94% text accuracy" figure was inflated: 6 of its 18 "unseen" test sentences were in the training data (corrected 2026-10-10; see the text classifier entry above).

## Visual/animation system (shared across all 3 pages)

- Dark Netflix/Spotify-inspired palette: red `#e50914`, green `#1db954`, gold `#f5c518`, dark base
- Animated particle canvas (drifting colored dots)
- Drifting blurred "poster" and "album art" shaped tiles rising through the background
- Rotating conic-gradient spotlight sweep
- Cursor-following radial glow
- Subtle animated grid mesh + noise texture overlay
- Equalizer-style bar animation
- Fade-up entrance animations on cards/rows, hover scale/glow on interactive elements
- Skeleton loading states while fetching results
- A small toast notification system (used today for mood-selection feedback)

## Backend API

Single endpoint: `POST /api/recommend`
- Request: `{ mood, mode, type: 'music'|'movies', limit }`
- Response: `{ results: [...], mood, mode }` — `results` is the array of track/movie objects described above
- Invalid mood values silently fall back to `happy` rather than erroring

## What's explicitly NOT there yet (for planning)

- No user accounts, login, or personalization/history
- No real audio or video playback
- No search or text filtering of results
- No sort/filter controls (genre, year, rating range, duration, etc.)
- No pagination UI (backend supports `limit`, but no "load more" or page numbers)
- No mobile-specific layout testing beyond basic responsive breakpoints
- No accessibility audit done (contrast, keyboard nav, screen reader labels)
