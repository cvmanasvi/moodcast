# Music metadata (local-only)

MoodCast's music recommender runs on the 278k Moodify catalog
(`data/music_catalog_278k.csv`, built locally by `python train_music_278k.py`;
not distributed, see DATA.md), keyed by Spotify `track_id`. That catalog has
no titles or artists. Display metadata is an optional, separate enrichment
layer:

```text
278k Moodify catalog
        |
        +--> ML / mood classification / recommendation
        |
        +--> track_id
                 |
                 v
       Optional local metadata  (data/metadata/music_metadata.csv)
                 |
                 v
       title / artist / year
```

Metadata never flows back into model training, mood labels, derived moods
(angry/stressed), classifier probabilities or recommendation scoring, and the
recommender works when the metadata file is absent. Tracks without metadata
stay in the catalog.

Data provenance and licensing for the metadata source, the 278k catalog and
the project's other music data are documented in [DATA.md](DATA.md).

## Building it

```bash
python build_music_metadata.py
```

- Output: `data/metadata/music_metadata.csv` and
  `data/metadata/metadata_sources.json` (provenance manifest with source and
  output SHA-256 hashes, row counts and coverage).
- Both are **gitignored and must not be committed**.
- Matching is **exact `track_id` equality only**: no title/artist or
  audio-feature matching. Unmatched tracks get no row; nothing is guessed.
- The build is deterministic: the same inputs produce byte-identical files.
- The catalog is never modified (its SHA-256 is checked before and after).

To check existing outputs without rebuilding (table integrity plus manifest
hashes and counts): `python build_music_metadata.py --verify`.

## Columns

| Column | Meaning |
|---|---|
| `track_id` | Spotify track ID. The canonical identity, shared with the catalog |
| `track_name` | Track title, as in the source |
| `artist_name` | **Display only.** The artists joined with ", " |
| `artist_names_json` | **Structured artist list**: a JSON array of strings, e.g. `["Tyler, The Creator", "Frank Ocean"]` |
| `release_year` | Year from the source's release date (`YYYY`, `YYYY-MM` or `YYYY-MM-DD`) |
| `metadata_source` | Which source the row came from (`A_yamaerenay_600k`) |

There is no `album_name`: the source has no album column.

**Artists.** Commas legitimately occur inside artist names ("Tyler, The
Creator", "Earth, Wind & Fire", "Harry Connick, Jr."), so `artist_name` is
ambiguous and must never be parsed by splitting on commas. Anything that needs
individual artists must use `json.loads(artist_names_json)`. Both columns are
built from the same parsed source list, in source order, with no reordering,
deduplication or rewriting. The only change is that surrounding whitespace is
trimmed from each name (9 names in the current build, e.g. a trailing
non-breaking space); names that would be blank after trimming are skipped.

## Source

Only one source is approved (Phase 2B licence review):

| | |
|---|---|
| Dataset | Spotify Dataset 1921-2020, 600k+ Tracks |
| Creator | Yamac Eren Ay |
| URL | https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks |
| Licence | CDLA-Sharing-1.0 (https://cdla.dev/sharing-1-0/) |
| Version | Kaggle version 1, last updated 2022-03-13 |
| Local file | `data/external/tracks.csv`, extracted from `data/external/A_600k.zip` |
| Fields used | `id`, `name`, `artists`, `release_date` |

To rebuild, download the dataset from the URL above and extract `tracks.csv`
into `data/external/` (also gitignored). Build the music catalog first: the
metadata build reads its track IDs.

The creator describes the data as collected from the Spotify Web API. The
licence covers the creator's dataset; it does not establish rights in the
underlying Spotify data. That question is unresolved, which is one reason the
generated metadata is kept local-only. If it were ever published,
CDLA-Sharing-1.0 would require keeping this attribution, noting changes and
publishing under the same licence.

Not used from this source: popularity, audio features, genres, artist IDs
and artwork. Nothing here is fetched from the Spotify Web API.

## On-demand titles and cover art (Spotify oEmbed)

For displayed tracks that this local file doesn't cover, the app asks
Spotify's public oEmbed endpoint
(`https://open.spotify.com/oembed?url=https://open.spotify.com/track/<id>`)
for the title and cover image of that exact `track_id` (approved by the
project owner on 2026-10-09). Details:

- No login, token or Premium subscription is involved; oEmbed returns no
  artist, so the artist is shown only when the local file has it.
- Server-side and on demand (`GET /api/track-details`, `src/spotify_oembed.py`),
  validated (non-empty title; cover must be a Spotify image-CDN URL), cached
  in memory only. Nothing is written to `data/`, and the 278k catalog,
  mood labels, training and ranking are untouched.
- Local metadata always wins when it has a title. If oEmbed has nothing,
  the track is shown as "Track details unavailable", never guessed.
- Disable with `MOODCAST_SPOTIFY_OEMBED=0`.
