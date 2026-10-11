"""Tests for the local-only music metadata build (src/music_metadata.py,
build_music_metadata.py): exact track_id matching, fail-loud validation,
determinism, the approved schema, git-ignore protection, and separation from
the ML/recommendation code.

The tests at the bottom need `python build_music_metadata.py` to have been
run; they are skipped (not failed) if its outputs are missing."""
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from src import music_metadata as mm
from src import recommend
from src.config import (BASE_DIR, METADATA_MANIFEST_PATH, MUSIC_CATALOG_278K_PATH, MUSIC_METADATA_PATH)
from tests.local_data import requires_git_checkout

APPROVED_COLUMNS = ["track_id", "track_name", "artist_name", "artist_names_json", "release_year",
                    "metadata_source"]
FORBIDDEN_COLUMNS = {"popularity", "danceability", "energy", "loudness", "speechiness", "acousticness",
                     "instrumentalness", "liveness", "valence", "tempo", "key", "mode", "duration_ms",
                     "explicit", "genre", "genres", "id_artists", "album_cover", "image", "label", "mood"}


def _tid(i):
    return f"T{i:021d}"


def _catalog_ids(n=5):
    return pd.Series([_tid(i) for i in range(n)])


def _source(rows):
    """rows: list of (id, name, artists_literal, release_date)."""
    return pd.DataFrame(rows, columns=["id", "name", "artists", "release_date"])


def _basic_source():
    return _source([
        (_tid(0), "Song Zero", "['Artist A']", "1999-05-01"),
        (_tid(2), "Song Two", "['Artist B', 'Artist C']", "2005"),
        (_tid(3), "Song Three", "['Artist D']", "2010-07"),
        (_tid(99), "Not In Catalog", "['Nobody']", "2001"),
    ])


# ---- matching ----

def test_exact_track_id_matching_only():
    meta = mm.build_metadata(_catalog_ids(), _basic_source(), "SRC")
    assert meta["track_id"].tolist() == [_tid(0), _tid(2), _tid(3)]  # 1 and 4 unmatched; 99 not in catalog
    assert meta["track_name"].tolist() == ["Song Zero", "Song Two", "Song Three"]
    assert meta["artist_name"].tolist() == ["Artist A", "Artist B, Artist C", "Artist D"]
    assert meta["release_year"].tolist() == [1999, 2005, 2010]
    assert (meta["metadata_source"] == "SRC").all()


@pytest.mark.parametrize("near_miss", [_tid(1).lower(), _tid(1) + " ", " " + _tid(1), _tid(1)[:-1]])
def test_near_identical_ids_do_not_match(near_miss):
    source = _source([(near_miss, "Song One", "['Artist']", "2000")])
    assert mm.build_metadata(_catalog_ids(), source, "SRC").empty


def test_no_fuzzy_matching_on_title_or_artist():
    # the source has a different id but the same title/artist as a catalog
    # track that already matched -- it must not attach to any other track
    source = _basic_source()
    source.loc[len(source)] = (_tid(50), "Song Zero", "['Artist A']", "1999-05-01")
    meta = mm.build_metadata(_catalog_ids(), source, "SRC")
    assert _tid(1) not in set(meta["track_id"]) and _tid(4) not in set(meta["track_id"])
    assert (meta["track_name"] == "Song Zero").sum() == 1


# ---- no fabrication ----

def test_missing_values_stay_missing():
    source = _source([
        (_tid(0), "", "['Artist A']", "1999"),          # empty title
        (_tid(1), "Song", "[]", "2000"),                  # no artists
        (_tid(2), "Song", "['Artist']", "unknown"),       # unparseable date
        (_tid(3), "Song", "['Artist']", ""),              # empty date
    ])
    meta = mm.build_metadata(_catalog_ids(), source, "SRC").set_index("track_id")
    assert pd.isna(meta.loc[_tid(0), "track_name"])
    assert pd.isna(meta.loc[_tid(1), "artist_name"])
    assert pd.isna(meta.loc[_tid(2), "release_year"]) and pd.isna(meta.loc[_tid(3), "release_year"])


def test_unmatched_tracks_get_no_row():
    meta = mm.build_metadata(_catalog_ids(10), _basic_source(), "SRC")
    assert len(meta) == 3
    assert not meta.isna().any().any()


# ---- duplicates ----

def test_identical_duplicate_source_rows_collapse():
    source = pd.concat([_basic_source(), _basic_source().iloc[[0]]], ignore_index=True)
    meta = mm.build_metadata(_catalog_ids(), source, "SRC")
    assert meta["track_id"].is_unique and len(meta) == 3


def test_conflicting_duplicate_source_rows_fail():
    source = _basic_source()
    source.loc[len(source)] = (_tid(0), "A Different Title", "['Artist A']", "1999-05-01")
    with pytest.raises(mm.MetadataBuildError, match="conflicting"):
        mm.build_metadata(_catalog_ids(), source, "SRC")


# ---- fail loudly ----

def test_missing_source_file_fails(tmp_path):
    with pytest.raises(mm.MetadataBuildError, match="not found"):
        mm.load_source(str(tmp_path / "nope.csv"), mm.SOURCE_A["columns"])


@pytest.mark.parametrize("dropped", ["id", "name", "artists", "release_date"])
def test_missing_required_source_column_fails(tmp_path, dropped):
    path = tmp_path / "tracks.csv"
    _basic_source().drop(columns=[dropped]).to_csv(path, index=False)
    with pytest.raises(mm.MetadataBuildError, match="missing required columns"):
        mm.load_source(str(path), mm.SOURCE_A["columns"])


def test_load_source_keeps_raw_strings(tmp_path):
    path = tmp_path / "tracks.csv"
    _source([(_tid(0), "NA", "['None']", "2000")]).to_csv(path, index=False)
    loaded = mm.load_source(str(path), mm.SOURCE_A["columns"])
    assert loaded.loc[0, "name"] == "NA"  # a song literally titled "NA" is not treated as missing


@pytest.mark.parametrize("ids", [
    [_tid(0), _tid(0), _tid(1)],     # duplicate
    [_tid(0), "short"],              # malformed
    [_tid(0), None],                 # missing
])
def test_catalog_identity_problems_fail(ids):
    with pytest.raises(mm.MetadataBuildError):
        mm.build_metadata(pd.Series(ids, dtype=object), _basic_source(), "SRC")


@pytest.mark.parametrize("artists", ["not a list", "['unterminated", "[1, 2]", "'A'"])
def test_unparseable_artists_fail(artists):
    source = _source([(_tid(0), "Song", artists, "2000")])
    with pytest.raises(mm.MetadataBuildError):
        mm.build_metadata(_catalog_ids(), source, "SRC")


# ---- determinism + schema ----

def test_output_is_deterministic_regardless_of_source_order(tmp_path):
    source = _basic_source()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    mm.write_metadata(mm.build_metadata(_catalog_ids(), source, "SRC"), str(a))
    shuffled = source.sample(frac=1, random_state=3).reset_index(drop=True)
    mm.write_metadata(mm.build_metadata(_catalog_ids()[::-1], shuffled, "SRC"), str(b))
    assert a.read_bytes() == b.read_bytes()


def test_schema_is_exactly_the_approved_columns():
    meta = mm.build_metadata(_catalog_ids(), _basic_source(), "SRC")
    assert list(meta.columns) == APPROVED_COLUMNS == mm.METADATA_COLUMNS
    assert not FORBIDDEN_COLUMNS & set(meta.columns)


def test_build_does_not_touch_the_catalog_frame():
    ids = _catalog_ids()
    before = ids.copy()
    mm.build_metadata(ids, _basic_source(), "SRC")
    pd.testing.assert_series_equal(ids, before)


# ---- git protection + separation ----

def _git_check_ignore(path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=BASE_DIR).returncode


@requires_git_checkout
def test_metadata_directory_is_gitignored():
    assert _git_check_ignore("data/metadata/music_metadata.csv") == 0
    assert _git_check_ignore("data/metadata/metadata_sources.json") == 0


@requires_git_checkout
def test_rest_of_data_directory_is_not_ignored():
    assert _git_check_ignore("data/text_mood_dataset.csv") == 1
    assert _git_check_ignore("build_music_metadata.py") == 1


@requires_git_checkout
@pytest.mark.parametrize("path", ["data/music_dataset2/278k_labelled_uri.csv", "data/music_catalog_278k.csv",
                                  "models/music_mood_classifier_278k.pkl", "data/movies_final.csv",
                                  "data/movie_dataset2/Movies (1970-2023).csv", "models/movie_mood_classifier.pkl"])
def test_undistributed_datasets_are_ignored(path):
    assert _git_check_ignore(path) == 0


@pytest.mark.parametrize("path", ["src/recommend.py", "src/music_pipeline.py", "train_music_278k.py",
                                  "train_all_models.py", "src/ml_inference.py"])
def test_ml_and_ranking_code_never_reads_metadata(path):
    with open(os.path.join(BASE_DIR, path), encoding="utf-8") as f:
        source = f.read()
    for needle in ["music_metadata", "MUSIC_METADATA_PATH", "METADATA_DIR", "data/metadata"]:
        assert needle not in source, f"{path} references {needle}"


# ---- built artifacts (need build_music_metadata.py to have run) ----

requires_metadata = pytest.mark.skipif(
    not (os.path.exists(MUSIC_METADATA_PATH) and os.path.exists(METADATA_MANIFEST_PATH)),
    reason="run `python build_music_metadata.py` to build the local metadata")


@pytest.fixture(scope="module")
def metadata():
    return pd.read_csv(MUSIC_METADATA_PATH, dtype={"track_id": str})


@pytest.fixture(scope="module")
def manifest():
    with open(METADATA_MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@requires_metadata
def test_built_metadata_schema_and_identity(metadata):
    assert list(metadata.columns) == APPROVED_COLUMNS
    assert metadata["track_id"].is_unique
    assert metadata["track_id"].str.fullmatch(r"[0-9A-Za-z]{22}").all()
    catalog_ids = set(pd.read_csv(MUSIC_CATALOG_278K_PATH, usecols=["track_id"], dtype=str)["track_id"])
    assert set(metadata["track_id"]) <= catalog_ids
    assert (metadata["metadata_source"] == mm.SOURCE_A["source_id"]).all()


@requires_metadata
def test_manifest_matches_the_files(metadata, manifest):
    assert manifest["license"] == "CDLA-Sharing-1.0"
    assert manifest["matching_method"] == "exact track_id"
    assert manifest["generated_row_count"] == len(metadata)
    assert manifest["generated_columns"] == APPROVED_COLUMNS
    assert manifest["generated_sha256"] == _sha256(MUSIC_METADATA_PATH)
    # the canonical catalog is the one the metadata was built against, unchanged
    assert manifest["catalog_sha256"] == _sha256(MUSIC_CATALOG_278K_PATH)


def test_recommendation_works_without_metadata(monkeypatch, tmp_path):
    """The recommender never needs the metadata file: point its path at nothing."""
    import src.config as config
    monkeypatch.setattr(config, "MUSIC_METADATA_PATH", str(tmp_path / "absent.csv"))
    rng = np.random.RandomState(0)
    n = 200
    catalog = pd.DataFrame({
        "track_id": [_tid(i) for i in range(n)],
        "label": [i % 4 for i in range(n)],
        "mood": [["sad", "happy", "energetic", "calm"][i % 4] for i in range(n)],
        **{f"proba_{m}": p for m, p in zip(["sad", "happy", "energetic", "calm"],
                                             rng.dirichlet(np.ones(4), size=n).T)},
    })
    out = recommend.recommend_music(catalog, "sad", "Cheer me up", top_n=10)
    assert len(out) == 10 and out["track_id"].is_unique


# ---- Phase 2D: validation of a built metadata table + manifest ----

def _valid_table():
    return mm.build_metadata(_catalog_ids(), _basic_source(), "SRC")


def test_valid_table_passes_validation():
    mm.validate_metadata_table(_valid_table(), _catalog_ids())


def _corrupted(kind):
    meta = _valid_table().astype(object)
    if kind == "nonexistent_id":
        meta.loc[0, "track_id"] = _tid(77)
    elif kind == "duplicate_id":
        meta.loc[1, "track_id"] = meta.loc[0, "track_id"]
    elif kind == "null_id":
        meta.loc[0, "track_id"] = None
    elif kind == "malformed_id":
        meta.loc[0, "track_id"] = "abc"
    elif kind == "non_integer_year":
        meta.loc[0, "release_year"] = "19x9"
    elif kind == "future_year":
        meta.loc[0, "release_year"] = "3000"
    elif kind == "blank_artist":
        meta.loc[0, "artist_name"] = "   "
    elif kind == "blank_title":
        meta.loc[0, "track_name"] = ""
    elif kind == "unauthorized_column":
        meta = meta.assign(album_cover="https://example.invalid/x.jpg")
    elif kind == "popularity":
        meta = meta.assign(popularity=50)
    elif kind == "audio_feature":
        meta = meta.assign(energy=0.5)
    elif kind == "reordered_columns":
        meta = meta[list(reversed(meta.columns))]
    return meta


@pytest.mark.parametrize("kind", ["nonexistent_id", "duplicate_id", "null_id", "malformed_id", "non_integer_year",
                                  "future_year", "blank_artist", "blank_title", "unauthorized_column",
                                  "popularity", "audio_feature", "reordered_columns"])
def test_validation_rejects_corrupted_tables(kind):
    with pytest.raises(mm.MetadataBuildError):
        mm.validate_metadata_table(_corrupted(kind), _catalog_ids())


def test_missing_metadata_file_fails_validation_read(tmp_path):
    with pytest.raises(mm.MetadataBuildError, match="not found"):
        mm.read_metadata_table(str(tmp_path / "music_metadata.csv"))


def test_read_keeps_na_like_titles_as_text(tmp_path):
    path = tmp_path / "music_metadata.csv"
    source = _source([(_tid(0), "NA", "['None']", "2000"), (_tid(1), "null", "['N/A']", "2001")])
    mm.write_metadata(mm.build_metadata(_catalog_ids(), source, "SRC"), str(path))
    meta = mm.read_metadata_table(str(path))
    assert meta["track_name"].tolist() == ["NA", "null"]
    assert meta["artist_name"].tolist() == ["None", "N/A"]
    mm.validate_metadata_table(meta, _catalog_ids())


@pytest.fixture
def built_files(tmp_path):
    """A small, real build on disk plus its matching manifest."""
    source_path, catalog_path, meta_path = tmp_path / "tracks.csv", tmp_path / "catalog.csv", tmp_path / "meta.csv"
    _basic_source().to_csv(source_path, index=False)
    pd.DataFrame({"track_id": _catalog_ids()}).to_csv(catalog_path, index=False)
    mm.write_metadata(_valid_table(), str(meta_path))
    manifest = {"source_sha256": _sha256(source_path), "catalog_sha256": _sha256(catalog_path),
                "generated_sha256": _sha256(meta_path), "generated_row_count": 3,
                "matching_method": "exact track_id"}
    return manifest, str(source_path), str(catalog_path), str(meta_path)


def test_matching_manifest_verifies(built_files):
    mm.verify_manifest(*built_files)


def test_changed_source_fails_manifest_verification(built_files):
    manifest, source_path, catalog_path, meta_path = built_files
    with open(source_path, "a") as f:
        f.write(f"{_tid(4)},Late Addition,['X'],2020\n")
    with pytest.raises(mm.MetadataBuildError, match="source_sha256"):
        mm.verify_manifest(manifest, source_path, catalog_path, meta_path)


def test_changed_catalog_fails_manifest_verification(built_files):
    manifest, source_path, catalog_path, meta_path = built_files
    with open(catalog_path, "a") as f:
        f.write(f"{_tid(5)}\n")
    with pytest.raises(mm.MetadataBuildError, match="catalog_sha256"):
        mm.verify_manifest(manifest, source_path, catalog_path, meta_path)


def test_edited_metadata_fails_manifest_verification(built_files):
    manifest, source_path, catalog_path, meta_path = built_files
    with open(meta_path, "a") as f:
        f.write(f'{_tid(4)},Injected,Someone,"[""Someone""]",2020,SRC\n')
    with pytest.raises(mm.MetadataBuildError, match="generated_sha256.*generated_row_count"):
        mm.verify_manifest(manifest, source_path, catalog_path, meta_path)


def test_missing_metadata_file_fails_manifest_verification(built_files):
    manifest, source_path, catalog_path, meta_path = built_files
    os.remove(meta_path)
    with pytest.raises(mm.MetadataBuildError, match="not found"):
        mm.verify_manifest(manifest, source_path, catalog_path, meta_path)


@requires_metadata
def test_real_build_passes_validation_and_manifest_verification(manifest):
    catalog_ids = pd.read_csv(MUSIC_CATALOG_278K_PATH, usecols=["track_id"], dtype=str)["track_id"]
    mm.validate_metadata_table(mm.read_metadata_table(MUSIC_METADATA_PATH), catalog_ids)
    source_path = os.path.join(BASE_DIR, manifest["source_filename"])
    if not os.path.exists(source_path):
        pytest.skip("Source A not present locally")
    mm.verify_manifest(manifest, source_path, MUSIC_CATALOG_278K_PATH, MUSIC_METADATA_PATH)


# ---- Phase 2D.1: structured artist representation ----

KNOWN_COMMA_ARTISTS = ["Tyler, The Creator", "Earth, Wind & Fire", "Harry Connick, Jr.", "Emerson, Lake & Palmer"]


def _build_one(artists):
    """Build metadata for one track whose Source A artists field is the Python repr of `artists`
    (the exact format Source A uses, e.g. "['A', 'B']")."""
    source = _source([(_tid(0), "Song", repr(artists), "2000")])
    return mm.build_metadata(_catalog_ids(1), source, "SRC").iloc[0]


@pytest.mark.parametrize("artists", [
    ["Artist"],
    ["Artist One", "Artist Two"],
    ["Tyler, The Creator"],
    ["Artist One", "Earth, Wind & Fire", "Artist Three"],
    ["Harry Connick, Jr.", "Emerson, Lake & Palmer"],
    ["AC/DC", "P!nk", "Guns N' Roses", "Destiny's Child", "Florence + The Machine", '"Weird Al" Yankovic',
     "Sigur Rós", "Beyoncé", "The Notorious B.I.G.", "(Hed) P.E.", "ÀÇÈ; semi;colon", "back\\slash"],
])
def test_artist_json_preserves_exact_boundaries(artists):
    row = _build_one(artists)
    assert json.loads(row["artist_names_json"]) == artists  # exact names, exact order, exact boundaries
    assert row["artist_name"] == ", ".join(artists)         # natural display string


def test_display_string_is_not_split_or_json():
    row = _build_one(["Tyler, The Creator"])
    assert row["artist_name"] == "Tyler, The Creator"
    assert not row["artist_name"].startswith("[")
    # the display string is ambiguous by design; only the JSON gives the true count
    assert len(json.loads(row["artist_names_json"])) == 1
    assert len(row["artist_name"].split(", ")) == 2


def test_artist_order_and_repeats_are_kept():
    assert json.loads(_build_one(["Zed", "Alpha", "Zed"])["artist_names_json"]) == ["Zed", "Alpha", "Zed"]


def test_only_outer_whitespace_is_trimmed():
    row = _build_one(["Calcutta ", "\xa0Babes In Toyland", "  Inner  Spaces  "])
    assert json.loads(row["artist_names_json"]) == ["Calcutta", "Babes In Toyland", "Inner  Spaces"]


def test_artist_json_is_real_json_not_a_python_repr():
    encoded = _build_one(["Guns N' Roses", "Destiny's Child"])["artist_names_json"]
    assert encoded == '["Guns N\' Roses", "Destiny\'s Child"]'
    assert json.loads(encoded) == ["Guns N' Roses", "Destiny's Child"]


def test_no_artists_means_both_artist_fields_missing():
    row = _build_one([])
    assert pd.isna(row["artist_name"]) and pd.isna(row["artist_names_json"])


def test_artist_json_survives_a_csv_round_trip(tmp_path):
    path = tmp_path / "meta.csv"
    artists = ["Artist One", "Earth, Wind & Fire", '"Weird Al" Yankovic']
    source = _source([(_tid(0), "Song", repr(artists), "2000")])
    mm.write_metadata(mm.build_metadata(_catalog_ids(1), source, "SRC"), str(path))
    meta = mm.read_metadata_table(str(path))
    assert json.loads(meta.loc[0, "artist_names_json"]) == artists
    assert meta.loc[0, "artist_name"] == 'Artist One, Earth, Wind & Fire, "Weird Al" Yankovic'
    mm.validate_metadata_table(meta, _catalog_ids(1))


@pytest.mark.parametrize("encoded,display", [
    ("not json", "A"),                       # invalid JSON
    ('"A"', "A"),                            # JSON, but not a list
    ("[]", "A"),                             # empty list
    ('["A", ""]', "A, "),                    # blank element
    ('["A", 1]', "A, 1"),                    # non-string element
    ("['A']", "A"),                          # Python repr, not JSON
    ('["Tyler", "The Creator"]', "Tyler, The Creator"),  # display matches, but boundaries differ...
])
def test_validation_rejects_bad_artist_json(encoded, display):
    meta = _valid_table().astype(object)
    meta.loc[0, "artist_name"], meta.loc[0, "artist_names_json"] = display, encoded
    if encoded == '["Tyler", "The Creator"]':
        # ...which validation cannot detect from the table alone (the display string is
        # ambiguous by design); the real-data test below checks boundaries against Source A.
        mm.validate_metadata_table(meta, _catalog_ids())
        return
    with pytest.raises(mm.MetadataBuildError):
        mm.validate_metadata_table(meta, _catalog_ids())


@pytest.mark.parametrize("display,encoded", [
    ("Someone Else", '["Artist A"]'),        # display disagrees with the list
    (None, '["Artist A"]'),                  # list without display
    ("Artist A", None),                      # display without list
])
def test_validation_rejects_inconsistent_artist_fields(display, encoded):
    meta = _valid_table().astype(object)
    meta.loc[0, "artist_name"], meta.loc[0, "artist_names_json"] = display, encoded
    with pytest.raises(mm.MetadataBuildError):
        mm.validate_metadata_table(meta, _catalog_ids())


@pytest.fixture(scope="module")
def real_artists():
    """(metadata, {track_id: artist list parsed independently from Source A})."""
    source_path = os.path.join(BASE_DIR, "data", "external", mm.SOURCE_A["source_filename"])
    if not os.path.exists(source_path):
        pytest.skip("Source A not present locally")
    meta = mm.read_metadata_table(MUSIC_METADATA_PATH)
    src = pd.read_csv(source_path, usecols=["id", "artists"], dtype=str, keep_default_na=False)
    src = src[src["id"].isin(set(meta["track_id"]))]
    import ast
    expected = {tid: [a.strip() for a in ast.literal_eval(raw) if a.strip()]
                for tid, raw in zip(src["id"], src["artists"])}
    return meta, expected


@requires_metadata
def test_every_real_row_round_trips_to_the_source_artist_list(real_artists):
    meta, expected = real_artists
    for tid, display, encoded in zip(meta["track_id"], meta["artist_name"], meta["artist_names_json"]):
        artists = json.loads(encoded)
        assert isinstance(artists, list) and artists, tid
        assert all(isinstance(a, str) and a.strip() for a in artists), tid
        assert artists == expected[tid], tid              # order + boundaries match Source A
        assert display == ", ".join(artists), tid


@requires_metadata
@pytest.mark.parametrize("name", KNOWN_COMMA_ARTISTS)
def test_known_comma_artists_keep_their_boundaries_in_real_data(real_artists, name):
    meta, expected = real_artists
    hits = [(tid, json.loads(enc)) for tid, enc in zip(meta["track_id"], meta["artist_names_json"])
            if name in json.loads(enc)]
    if not hits:
        pytest.skip(f"{name!r} is not in the matched subset (covered by the synthetic tests)")
    for tid, artists in hits:
        assert artists == expected[tid]
        assert name in meta.loc[meta["track_id"] == tid, "artist_name"].iloc[0]
