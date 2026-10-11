/**
 * Canonical music-track identity for the frontend.
 *
 * 278k-catalog tracks from /api/recommend and /api/search carry a Spotify
 * `track_id`; that ID is the track's identity (favorites, queue lookups,
 * current track, per-track visual seeds). Different releases can share the
 * same title and artist, so display fields are never used when an ID exists.
 * The API no longer serves tracks without an ID (the legacy music_final.csv
 * catalog was retired in Phase 2H.4); the "title::artist" fallback remains
 * only so an ID-less track object can never crash identity lookups.
 *
 * Loaded before search.js / music.js; also importable from Node for tests.
 */
(function (root) {
  function hasTrackId(t) {
    return !!t && typeof t.track_id === 'string' && t.track_id !== '';
  }

  /** track_id when present, else the legacy "title::artist" key. Never invents an ID. */
  function trackIdentity(t) {
    if (hasTrackId(t)) return t.track_id;
    return `${(t && t.title) || ''}::${(t && t.artist) || ''}`;
  }

  function sameTrack(a, b) {
    return trackIdentity(a) === trackIdentity(b);
  }

  /** Position of `t` in `tracks` by identity, or -1. */
  function indexOfTrack(tracks, t) {
    const key = trackIdentity(t);
    return tracks.findIndex((candidate) => trackIdentity(candidate) === key);
  }

  /** Add/remove `t` in a Set of identities; returns whether it's now a favorite. */
  function toggleFavorite(favorites, t) {
    const key = trackIdentity(t);
    if (favorites.has(key)) favorites.delete(key);
    else favorites.add(key);
    return favorites.has(key);
  }

  function isFavorite(favorites, t) {
    return favorites.has(trackIdentity(t));
  }

  /** Structured artist list from the backend; never derived by splitting `artist` on commas. */
  function trackArtists(t) {
    return t && Array.isArray(t.artists) ? t.artists.filter((a) => typeof a === 'string') : [];
  }

  /** A string that is safe to show: the value if it's a non-empty string, else the fallback. */
  function displayText(value, fallback) {
    return typeof value === 'string' && value.trim() !== '' ? value : fallback;
  }

  function displayTitle(t) {
    return displayText(t && t.title, 'Unknown title');
  }

  function displayArtist(t) {
    return displayText(t && t.artist, 'Unknown artist');
  }

  /** ` data-track-id="…"` for tracks with an ID (IDs are [0-9A-Za-z]), else ''. */
  function trackIdAttr(t) {
    return hasTrackId(t) && /^[0-9A-Za-z]+$/.test(t.track_id) ? ` data-track-id="${t.track_id}"` : '';
  }

  // The backend's placeholders for a track with no local metadata (src/music_metadata.py).
  const PLACEHOLDERS = new Set(['Unknown title', 'Unknown artist']);
  const SPOTIFY_ID_RE = /^[0-9A-Za-z]{22}$/;
  const SPOTIFY_TRACK_URL_RE = /^https:\/\/open\.spotify\.com\/track\/([0-9A-Za-z]{22})$/;

  function realText(value) {
    return typeof value === 'string' && value.trim() !== '' && !PLACEHOLDERS.has(value.trim()) ? value : null;
  }

  /**
   * What a track card may honestly show: the real title and artist, or null
   * for each one the metadata doesn't have. Placeholder strings and tracks
   * flagged has_metadata: false count as missing; nothing is inferred.
   */
  function trackDisplay(t, details) {
    const known = !!t && t.has_metadata !== false;
    let title = known ? realText(t.title) : null;
    const artist = known ? realText(t.artist) : null;
    let titleSource = title ? 'metadata' : null;
    // Spotify oEmbed (fetched for this exact track_id) can supply a title -- never an artist.
    const oembedTitle = details && typeof details === 'object' ? realText(details.title) : null;
    if (!title && oembedTitle) { title = oembedTitle; titleSource = 'spotify'; }
    let heading = title;
    if (!title) heading = artist ? 'Title unavailable' : 'Track details unavailable';
    return { title, artist, heading, available: !!title, titleSource };
  }

  /** Cover-art URL from oEmbed details if it is a Spotify image-CDN URL, else null. */
  function trackArtwork(details) {
    const art = details && details.artwork;
    return typeof art === 'string'
      && /^https:\/\/(i\.scdn\.co|image-cdn-[a-z0-9-]+\.spotifycdn\.com|mosaic\.scdn\.co)\/image\/[0-9a-f]+$/.test(art) ? art : null;
  }

  /**
   * Exact open.spotify.com/track/<id> link for this track, or null. Uses the
   * backend's spotify_url when it is a well-formed track link for this same
   * track_id, else builds it from a well-formed track_id. Never a search URL.
   */
  function spotifyTrackUrl(t) {
    if (!t) return null;
    const id = hasTrackId(t) && SPOTIFY_ID_RE.test(t.track_id) ? t.track_id : null;
    const m = typeof t.spotify_url === 'string' ? t.spotify_url.match(SPOTIFY_TRACK_URL_RE) : null;
    if (m && (!hasTrackId(t) || m[1] === t.track_id)) return t.spotify_url;
    if (m) return null;  // a link that names a different track than track_id: trust neither
    return id ? `https://open.spotify.com/track/${id}` : null;
  }

  const api = {
    hasTrackId, trackIdentity, sameTrack, indexOfTrack, toggleFavorite, isFavorite,
    trackArtists, displayText, displayTitle, displayArtist, trackIdAttr, trackDisplay, trackArtwork, spotifyTrackUrl,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MoodcastTrack = api;
})(typeof window !== 'undefined' ? window : this);
