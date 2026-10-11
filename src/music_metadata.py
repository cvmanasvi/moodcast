"""
Local-only display metadata for the 278k music catalog.

Builds a table of title / artist / release year keyed by the canonical
Spotify `track_id`, from a third-party dataset downloaded into
data/external/. This is an enrichment layer only: nothing here feeds model
training, mood labels, derived moods or recommendation scoring, and the
recommender works without it.

Matching is by exact `track_id` equality only -- never by title/artist
similarity or audio features. Tracks without a match simply have no
metadata row; nothing is guessed. Every structural surprise fails loudly.
Driven by build_music_metadata.py; see METADATA.md for provenance.
"""
from __future__ import annotations

import ast
import datetime
import json
import os
import re

import pandas as pd

# artist_name is for display only (names can contain commas, so it must never
# be split); artist_names_json is the structured artist list (a JSON array).
METADATA_COLUMNS = ["track_id", "track_name", "artist_name", "artist_names_json", "release_year",
                    "metadata_source"]

# Source A: the only metadata source approved for Phase 2C (Phase 2B review).
# Licence and version facts come from Kaggle's dataset metadata.
SOURCE_A = {
    "source_id": "A_yamaerenay_600k",
    "source_name": "Spotify Dataset 1921-2020, 600k+ Tracks",
    "creator": "Yamac Eren Ay",
    "source_url": "https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks",
    "license": "CDLA-Sharing-1.0",
    "license_url": "https://cdla.dev/sharing-1-0/",
    "source_version": "Kaggle version 1, last updated 2022-03-13",
    "archive_filename": "A_600k.zip",
    "source_filename": "tracks.csv",
    "columns": {"id": "id", "name": "name", "artists": "artists", "release_date": "release_date"},
}

_TRACK_ID_RE = re.compile(r"^[0-9A-Za-z]{22}$")
_RELEASE_DATE_RE = re.compile(r"^(\d{4})(-\d{2}(-\d{2})?)?$")


class MetadataBuildError(ValueError):
    """The inputs violate an assumption the metadata build relies on."""


def validate_catalog_ids(track_ids: pd.Series) -> None:
    """The canonical catalog's identity column must be unique 22-char Spotify IDs."""
    ids = track_ids.astype(str)
    if track_ids.isna().any():
        raise MetadataBuildError(f"Catalog has {int(track_ids.isna().sum())} missing track_ids")
    bad = ~ids.str.match(_TRACK_ID_RE)
    if bad.any():
        raise MetadataBuildError(f"Catalog has {int(bad.sum())} malformed track_ids, e.g. {ids[bad].iloc[0]!r}")
    dup = ids.duplicated()
    if dup.any():
        raise MetadataBuildError(f"Catalog has {int(dup.sum())} duplicate track_ids, e.g. {ids[dup].iloc[0]!r}")


def load_source(path: str, columns: dict) -> pd.DataFrame:
    """
    Read the needed columns of a metadata source as raw strings (no NA
    guessing), renamed to id/name/artists/release_date.
    """
    if not os.path.exists(path):
        raise MetadataBuildError(f"Metadata source file not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    missing = [src for src in columns.values() if src not in header]
    if missing:
        raise MetadataBuildError(f"Metadata source {path} is missing required columns: {missing}")
    df = pd.read_csv(path, usecols=list(columns.values()), dtype=str, keep_default_na=False)
    return df.rename(columns={src: ours for ours, src in columns.items()})


def dedupe_source(source: pd.DataFrame) -> pd.DataFrame:
    """
    One row per id. Exact duplicate rows collapse to one; the same id with
    different title/artist/date is ambiguous and fails instead of picking one.
    """
    fields = ["id", "name", "artists", "release_date"]
    collapsed = source[fields].drop_duplicates()
    conflict = collapsed["id"].duplicated(keep=False)
    if conflict.any():
        example = collapsed.loc[conflict, "id"].iloc[0]
        raise MetadataBuildError(f"{int(collapsed.loc[conflict, 'id'].nunique())} source ids have conflicting "
                                 f"metadata, e.g. {example!r}")
    return collapsed


def parse_artist_list(raw: str, track_id: str) -> list:
    """
    Source A stores artists as a Python-list literal, e.g. "['A', 'Earth, Wind & Fire']".
    Parsed with ast.literal_eval (literals only, never executes code). Returns
    the names in source order; the only change is trimming outer whitespace,
    and names that are blank after trimming are skipped.
    """
    try:
        artists = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        raise MetadataBuildError(f"Unparseable artists value for {track_id}: {raw!r}") from None
    if not isinstance(artists, list) or not all(isinstance(a, str) for a in artists):
        raise MetadataBuildError(f"Unexpected artists value for {track_id}: {raw!r}")
    return [a.strip() for a in artists if a.strip()]


def artist_display(artists: list):
    """Display string: names joined with ", ". Not reversible -- names may contain commas."""
    return ", ".join(artists) if artists else pd.NA


def artist_json(artists: list):
    """Structured, unambiguous artist list: a JSON array of strings, source order kept."""
    return json.dumps(artists, ensure_ascii=False) if artists else pd.NA


def parse_release_year(raw: str):
    """Year from 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'; anything else is left missing, not guessed."""
    match = _RELEASE_DATE_RE.match(raw.strip())
    return int(match.group(1)) if match else pd.NA


def build_metadata(catalog_ids: pd.Series, source: pd.DataFrame, source_id: str) -> pd.DataFrame:
    """
    Metadata rows for catalog tracks whose track_id exactly equals a source
    id. Sorted by track_id, columns METADATA_COLUMNS. Empty source values
    stay missing.
    """
    validate_catalog_ids(catalog_ids)
    source = dedupe_source(source)
    matched = source[source["id"].isin(set(catalog_ids.astype(str)))].sort_values("id", kind="stable")

    names = matched["name"].where(matched["name"].str.strip() != "")
    artists = [parse_artist_list(raw, tid) for raw, tid in zip(matched["artists"], matched["id"])]
    metadata = pd.DataFrame({
        "track_id": matched["id"].to_numpy(),
        "track_name": names.to_numpy(),
        "artist_name": pd.array([artist_display(a) for a in artists], dtype=object),
        "artist_names_json": pd.array([artist_json(a) for a in artists], dtype=object),
        "release_year": pd.array([parse_release_year(r) for r in matched["release_date"]], dtype="Int64"),
        "metadata_source": source_id,
    })
    return metadata.reset_index(drop=True)[METADATA_COLUMNS]


def write_metadata(metadata: pd.DataFrame, path: str) -> None:
    """Byte-stable CSV (fixed column order, row order and line endings)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metadata.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def read_metadata_table(path: str) -> pd.DataFrame:
    """
    Read a built metadata CSV without pandas' NA guessing (a song titled
    "NA" or "None" stays a string); only truly empty fields become missing.
    """
    if not os.path.exists(path):
        raise MetadataBuildError(f"Metadata file not found: {path}")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ["track_name", "artist_name", "artist_names_json", "release_year"]:
        if col in raw.columns:
            raw[col] = raw[col].replace("", pd.NA)
    return raw


def decode_artist_json(value, track_id: str = "?") -> list:
    """Decode one artist_names_json value; it must be a non-empty JSON array of non-blank strings."""
    try:
        artists = json.loads(value)
    except (TypeError, ValueError):
        raise MetadataBuildError(f"artist_names_json for {track_id} is not valid JSON: {value!r}") from None
    if (not isinstance(artists, list) or not artists
            or not all(isinstance(a, str) and a.strip() for a in artists)):
        raise MetadataBuildError(f"artist_names_json for {track_id} is not a non-empty list of names: {value!r}")
    return artists


def validate_metadata_table(metadata: pd.DataFrame, catalog_ids: pd.Series) -> None:
    """
    Integrity checks on a built metadata table: exactly the approved
    columns, unique well-formed track_ids that all exist in the catalog,
    non-blank text, artist_names_json a valid JSON list consistent with
    artist_name, and plausible integer years. Raises on the first problem.
    """
    if list(metadata.columns) != METADATA_COLUMNS:
        raise MetadataBuildError(f"Metadata columns {list(metadata.columns)} != approved {METADATA_COLUMNS}")
    ids = metadata["track_id"]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise MetadataBuildError("Metadata has missing track_ids")
    ids = ids.astype(str)
    bad = ~ids.str.match(_TRACK_ID_RE)
    if bad.any():
        raise MetadataBuildError(f"Metadata has malformed track_ids, e.g. {ids[bad].iloc[0]!r}")
    if ids.duplicated().any():
        raise MetadataBuildError(f"Metadata has duplicate track_ids, e.g. {ids[ids.duplicated()].iloc[0]!r}")
    unknown = ~ids.isin(set(catalog_ids.astype(str)))
    if unknown.any():
        raise MetadataBuildError(f"{int(unknown.sum())} metadata track_ids are not in the catalog, "
                                 f"e.g. {ids[unknown].iloc[0]!r}")
    for col in ["track_name", "artist_name"]:
        text = metadata[col].dropna().astype(str)
        if (text.str.strip() == "").any():
            raise MetadataBuildError(f"Metadata {col} has blank values (should be missing instead)")
    # the structured list and the display string must describe the same artists
    for tid, display, encoded in zip(ids, metadata["artist_name"], metadata["artist_names_json"]):
        if pd.isna(display) != pd.isna(encoded):
            raise MetadataBuildError(f"{tid}: artist_name and artist_names_json must both be present or both missing")
        if not pd.isna(encoded) and artist_display(decode_artist_json(encoded, tid)) != display:
            raise MetadataBuildError(f"{tid}: artist_name does not match artist_names_json")
    years = metadata["release_year"].dropna().astype(str)
    bad_year = ~years.str.fullmatch(r"\d{4}")
    if bad_year.any():
        raise MetadataBuildError(f"Metadata has non-integer release_year values, e.g. {years[bad_year].iloc[0]!r}")
    future = years.astype(int) > datetime.date.today().year
    if future.any():
        raise MetadataBuildError(f"Metadata has future release_year values, e.g. {years[future].iloc[0]!r}")


def verify_manifest(manifest: dict, source_path: str, catalog_path: str, metadata_path: str,
                    sha256=None) -> None:
    """Recompute the hashes and counts a manifest claims; raise listing every mismatch."""
    from .music_pipeline import file_sha256
    sha256 = sha256 or file_sha256
    problems = []
    for key, path in [("source_sha256", source_path), ("catalog_sha256", catalog_path),
                      ("generated_sha256", metadata_path)]:
        if not os.path.exists(path):
            problems.append(f"{key}: {path} not found")
        elif manifest.get(key) != sha256(path):
            problems.append(f"{key} does not match {path}")
    if os.path.exists(metadata_path):
        rows = len(read_metadata_table(metadata_path))
        if manifest.get("generated_row_count") != rows:
            problems.append(f"generated_row_count {manifest.get('generated_row_count')} != {rows}")
    if manifest.get("matching_method") != "exact track_id":
        problems.append(f"matching_method is {manifest.get('matching_method')!r}")
    if problems:
        raise MetadataBuildError("Manifest verification failed: " + "; ".join(problems))


def coverage_report(catalog: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Coverage of the catalog (track_id, mood) by the metadata table."""
    has_meta = catalog["track_id"].isin(set(metadata["track_id"]))
    by_label = catalog.assign(has_meta=has_meta).groupby("mood")["has_meta"].agg(["sum", "count", "mean"])
    return {
        "catalog_tracks": int(len(catalog)),
        "metadata_rows": int(len(metadata)),
        "exact_id_matches": int(has_meta.sum()),
        "unmatched_tracks": int((~has_meta).sum()),
        "coverage_pct": round(100 * float(has_meta.mean()), 2),
        "coverage_by_label": {mood: {"matched": int(r["sum"]), "tracks": int(r["count"]),
                                     "coverage_pct": round(100 * float(r["mean"]), 2)}
                              for mood, r in by_label.iterrows()},
        "missing_in_metadata_rows": {col: int(metadata[col].isna().sum())
                                     for col in ["track_name", "artist_name", "artist_names_json", "release_year"]},
        "duplicate_track_ids_in_metadata": int(metadata["track_id"].duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# Runtime (backend) access: read-only lookup and search over the built table.
# ---------------------------------------------------------------------------

FALLBACK_TITLE = "Unknown title"
FALLBACK_ARTIST = "Unknown artist"
SPOTIFY_TRACK_URL = "https://open.spotify.com/track/{}"


def spotify_track_url(track_id):
    """Link to a track on Spotify, built from its ID alone (no API call). None if the ID is malformed."""
    if not isinstance(track_id, str) or not _TRACK_ID_RE.match(track_id):
        return None
    return SPOTIFY_TRACK_URL.format(track_id)


def missing_display() -> dict:
    """Display fields for a track with no metadata: placeholders, never guesses."""
    return {"title": FALLBACK_TITLE, "artist": FALLBACK_ARTIST, "artists": [],
            "release_year": None, "has_metadata": False}


class MetadataStore:
    """
    In-memory, read-only view of the metadata table for the backend. Lookups
    are by exact track_id only; a track without a row gets fallback display
    values, never another track's metadata.
    """

    def __init__(self, table: pd.DataFrame):
        if list(table.columns) != METADATA_COLUMNS:
            raise MetadataBuildError(f"Metadata columns {list(table.columns)} != approved {METADATA_COLUMNS}")
        if table["track_id"].isna().any() or table["track_id"].duplicated().any():
            raise MetadataBuildError("Metadata track_ids must be present and unique")
        self._rows = {}
        artist_names = set()
        for tid, title, artist, encoded, year in zip(table["track_id"], table["track_name"], table["artist_name"],
                                                     table["artist_names_json"], table["release_year"]):
            artists = decode_artist_json(encoded, tid) if not pd.isna(encoded) else []
            artist_names.update(artists)
            self._rows[tid] = {
                "title": None if pd.isna(title) else str(title),
                "artist": None if pd.isna(artist) else str(artist),
                "artists": artists,
                "release_year": None if pd.isna(year) else int(year),
            }
        self._ids = list(self._rows)
        self._titles_lower = [(self._rows[t]["title"] or "").lower() for t in self._ids]
        self._artists_lower = [(self._rows[t]["artist"] or "").lower() for t in self._ids]
        self._artist_names = sorted(artist_names)

    def __len__(self):
        return len(self._rows)

    def __contains__(self, track_id):
        return track_id in self._rows

    def track_ids(self) -> list:
        """Every track_id that has metadata."""
        return list(self._ids)

    def artists_of(self, track_id) -> list:
        """The track's individual artist names (artist_names_json); [] when it has no metadata."""
        row = self._rows.get(track_id)
        return list(row["artists"]) if row else []

    def display(self, track_id) -> dict:
        """Display fields for one track_id, with fallbacks when it has no metadata."""
        row = self._rows.get(track_id)
        if row is None:
            return missing_display()
        return {"title": row["title"] or FALLBACK_TITLE, "artist": row["artist"] or FALLBACK_ARTIST,
                "artists": list(row["artists"]), "release_year": row["release_year"], "has_metadata": True}

    def search_tracks(self, query: str, limit: int = 6) -> list:
        """
        Case-insensitive substring search over title and display artist.
        Any match on an individual artist (artist_names_json) is always a match
        on the display string too, so both are covered. Order: title prefix,
        title substring, artist match; ties by title then track_id.
        """
        q = query.strip().lower()
        if not q:
            return []
        hits = []
        for i, tid in enumerate(self._ids):
            title, artist = self._titles_lower[i], self._artists_lower[i]
            if title.startswith(q):
                rank = 0
            elif q in title:
                rank = 1
            elif q in artist:
                rank = 2
            else:
                continue
            hits.append((rank, title, tid))
        hits.sort()
        return [dict(track_id=tid, **self.display(tid)) for _, _, tid in hits[:limit]]

    def search_artists(self, query: str, limit: int = 6) -> list:
        """
        Individual artist names (from artist_names_json, never from splitting
        artist_name) containing the query, case-insensitive. Prefix matches
        first, then alphabetical.
        """
        q = query.strip().lower()
        if not q:
            return []
        hits = [(0 if name.lower().startswith(q) else 1, name.lower(), name)
                for name in self._artist_names if q in name.lower()]
        hits.sort()
        return [name for _, _, name in hits[:limit]]


def load_metadata_store(path: str):
    """The MetadataStore for a built metadata file, or None if it hasn't been built."""
    if not os.path.exists(path):
        return None
    return MetadataStore(read_metadata_table(path))
