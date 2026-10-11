// Unit tests for static/js/track-identity.js, using Node's built-in test
// runner (no npm dependencies). Run: node --test tests/js/track_identity.test.js
// Also run from pytest via tests/test_frontend_js.py.
const test = require('node:test');
const assert = require('node:assert/strict');
const T = require('../../static/js/track-identity.js');

// Two different releases with identical display fields (as in the real data).
const twinA = { track_id: '1xAAAAAAAAAAAAAAAAAAAA', title: 'Bohemian Like You', artist: 'The Dandy Warhols' };
const twinB = { track_id: '2xBBBBBBBBBBBBBBBBBBBB', title: 'Bohemian Like You', artist: 'The Dandy Warhols' };
const legacy = { track_id: null, title: 'Golden', artist: 'HUNTR/X' };
const noMeta = {
  track_id: '0PBbGtt09Jefz2FqHx99fp', title: 'Unknown title', artist: 'Unknown artist', artists: [],
  has_metadata: false, release_year: null, spotify_url: 'https://open.spotify.com/track/0PBbGtt09Jefz2FqHx99fp',
};
const tyler = {
  track_id: '078C2jXg7XsMgW78Gfx1JA', title: 'SMUCKERS (feat. Lil Wayne & Kanye West)',
  artist: 'Tyler, The Creator, Lil Wayne, Kanye West', artists: ['Tyler, The Creator', 'Lil Wayne', 'Kanye West'],
  has_metadata: true, release_year: 2015,
};

// ---- identity ----

test('same title + artist but different track_ids are different tracks', () => {
  assert.notEqual(T.trackIdentity(twinA), T.trackIdentity(twinB));
  assert.equal(T.sameTrack(twinA, twinB), false);
});

test('same track_id is the same track, whatever the display fields say', () => {
  assert.equal(T.sameTrack(twinA, { ...twinA, title: 'Renamed', artist: 'Someone else' }), true);
  assert.equal(T.trackIdentity(twinA), twinA.track_id);
});

test('legacy records without track_id keep the title::artist key', () => {
  assert.equal(T.trackIdentity(legacy), 'Golden::HUNTR/X');
  assert.equal(T.trackIdentity({ title: 'Golden', artist: 'HUNTR/X' }), 'Golden::HUNTR/X');  // field absent
  assert.equal(T.trackIdentity({ track_id: '', title: 'Golden', artist: 'HUNTR/X' }), 'Golden::HUNTR/X');
  assert.equal(T.hasTrackId(legacy), false);
});

test('no identity is invented from nothing', () => {
  assert.equal(T.trackIdentity({}), '::');
  assert.equal(T.trackIdentity(null), '::');
  assert.equal(T.hasTrackId({ track_id: 12345 }), false);  // only real string IDs count
});

// ---- metadata display ----

test('real metadata displays as-is', () => {
  assert.equal(T.displayTitle(tyler), 'SMUCKERS (feat. Lil Wayne & Kanye West)');
  assert.equal(T.displayArtist(tyler), 'Tyler, The Creator, Lil Wayne, Kanye West');
});

test('missing metadata displays safe placeholders, never null/undefined/[object Object]', () => {
  assert.equal(T.displayTitle(noMeta), 'Unknown title');
  for (const bad of [{}, { title: null, artist: undefined }, { title: { x: 1 }, artist: ['a'] }, { title: '  ' }, null]) {
    for (const text of [T.displayTitle(bad), T.displayArtist(bad)]) {
      assert.ok(['Unknown title', 'Unknown artist'].includes(text), text);
      assert.doesNotMatch(text, /null|undefined|\[object Object\]/);
    }
  }
});

test('artists array is kept as structured data, never comma-split', () => {
  assert.deepEqual(T.trackArtists(tyler), ['Tyler, The Creator', 'Lil Wayne', 'Kanye West']);
  assert.equal(T.trackArtists(tyler).length, 3);
  assert.deepEqual(T.trackArtists({ artist: 'Earth, Wind & Fire' }), []);  // no array -> nothing derived
  assert.deepEqual(T.trackArtists(noMeta), []);
  assert.deepEqual(T.trackArtists(legacy), []);  // legacy sends artists: null
});

// ---- state ----

test('favorites distinguish different track_ids and survive a reload', () => {
  const favs = new Set();
  assert.equal(T.toggleFavorite(favs, twinA), true);
  assert.equal(T.isFavorite(favs, twinA), true);
  assert.equal(T.isFavorite(favs, twinB), false);  // same title/artist, different track
  const reloaded = new Set(JSON.parse(JSON.stringify([...favs])));  // localStorage round trip
  assert.equal(T.isFavorite(reloaded, { ...twinA }), true);
  assert.equal(T.toggleFavorite(reloaded, twinA), false);
});

test('legacy favorites keep working', () => {
  const favs = new Set(['Golden::HUNTR/X']);  // key stored before this change
  assert.equal(T.isFavorite(favs, legacy), true);
});

test('current-track lookup uses track_id when available', () => {
  const current = [twinA, twinB, noMeta];
  assert.equal(T.indexOfTrack(current, { ...twinB }), 1);  // not 0, despite identical title/artist
  assert.equal(T.indexOfTrack(current, noMeta), 2);
  assert.equal(T.indexOfTrack(current, { track_id: 'zzzzzzzzzzzzzzzzzzzzzz' }), -1);
  assert.equal(T.indexOfTrack([legacy], { title: 'Golden', artist: 'HUNTR/X' }), 0);
});

test('queue entries keep their track_id and resolve to the right track', () => {
  const queue = [];
  queue.push(twinB);  // music.js queues the whole track object
  const queued = JSON.parse(JSON.stringify(queue.shift()));
  assert.equal(queued.track_id, twinB.track_id);
  assert.equal(T.indexOfTrack([twinA, twinB], queued), 1);
});

test('search result rows keep their track_id as a data attribute', () => {
  assert.equal(T.trackIdAttr(twinA), ' data-track-id="1xAAAAAAAAAAAAAAAAAAAA"');
  assert.notEqual(T.trackIdAttr(twinA), T.trackIdAttr(twinB));  // duplicate display names stay distinct
  assert.equal(T.trackIdAttr(legacy), '');
  assert.equal(T.trackIdAttr({ track_id: '"><script>' }), '');  // never emits unsafe markup
});

// ---- honest card display (music page, dashboard, search, assistant) ----

test('a track with real metadata shows its real title and artist', () => {
  assert.deepEqual(T.trackDisplay(tyler), {
    title: 'SMUCKERS (feat. Lil Wayne & Kanye West)', artist: 'Tyler, The Creator, Lil Wayne, Kanye West',
    heading: 'SMUCKERS (feat. Lil Wayne & Kanye West)', available: true, titleSource: 'metadata',
  });
});

test('backend placeholders are treated as missing, not shown as a title', () => {
  const d = T.trackDisplay(noMeta);
  assert.equal(d.title, null);
  assert.equal(d.artist, null);
  assert.equal(d.heading, 'Track details unavailable');
  assert.equal(d.available, false);
  // the same placeholder strings without the has_metadata flag are still not real metadata
  assert.equal(T.trackDisplay({ title: 'Unknown title', artist: 'Unknown artist' }).heading, 'Track details unavailable');
});

test('a title without an artist shows the title and omits the artist', () => {
  const d = T.trackDisplay({ track_id: twinA.track_id, title: 'Only Title', artist: 'Unknown artist', has_metadata: true });
  assert.equal(d.heading, 'Only Title');
  assert.equal(d.artist, null);
});

test('an artist without a title never gets an invented title', () => {
  const d = T.trackDisplay({ title: '  ', artist: 'Real Artist', has_metadata: true });
  assert.equal(d.title, null);
  assert.equal(d.heading, 'Title unavailable');
  assert.equal(d.artist, 'Real Artist');
});

test('nothing is derived from other fields', () => {
  for (const t of [{}, null, { mood: 'happy', track_id: twinA.track_id, audio_features: { energy: 0.9 } }, { title: { x: 1 } }]) {
    assert.equal(T.trackDisplay(t).heading, 'Track details unavailable');
  }
});

// ---- exact Spotify track links ----

test('the Spotify link is the exact track page for the track_id', () => {
  assert.equal(T.spotifyTrackUrl(noMeta), 'https://open.spotify.com/track/0PBbGtt09Jefz2FqHx99fp');
  assert.equal(T.spotifyTrackUrl({ track_id: twinA.track_id }), 'https://open.spotify.com/track/1xAAAAAAAAAAAAAAAAAAAA');
  assert.notEqual(T.spotifyTrackUrl(twinA), T.spotifyTrackUrl(twinB));  // twins open different tracks
});

test('no link without a valid track id or track URL, and never a search URL', () => {
  assert.equal(T.spotifyTrackUrl(legacy), null);
  assert.equal(T.spotifyTrackUrl({}), null);
  assert.equal(T.spotifyTrackUrl(null), null);
  assert.equal(T.spotifyTrackUrl({ track_id: 'too-short' }), null);
  assert.equal(T.spotifyTrackUrl({ track_id: '"><script>alert(1)</script>xx' }), null);
  assert.equal(T.spotifyTrackUrl({ spotify_url: 'https://open.spotify.com/search/golden' }), null);
  assert.equal(T.spotifyTrackUrl({ spotify_url: 'javascript:alert(1)' }), null);
});

test('a spotify_url naming a different track than track_id is not trusted', () => {
  assert.equal(T.spotifyTrackUrl({ track_id: twinA.track_id, spotify_url: `https://open.spotify.com/track/${twinB.track_id}` }), null);
  assert.equal(T.spotifyTrackUrl({ spotify_url: `https://open.spotify.com/track/${twinB.track_id}` }),
    `https://open.spotify.com/track/${twinB.track_id}`);
});
