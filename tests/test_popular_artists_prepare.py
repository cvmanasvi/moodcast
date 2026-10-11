"""Tests for prepare_popular_artists.py on SYNTHETIC raw files (made-up ids and names).

A hash mismatch or an unexpected schema stops preparation before anything is
written; rows with malformed ids, mismatched track URLs, placeholder or
out-of-range values, spoken word or failed audio analysis are dropped;
duplicate ids are kept once when identical and dropped when they disagree;
versions of one recording collapse to the most popular; only whole curated
artist entries match; mood scoring uses the recorded 278k cutoffs; and the
manifest carries aggregates only.
"""
import json
import os

import pandas as pd
import pytest

import prepare_popular_artists as prep
from src import music_pipeline as mp
from src import popular_artists as pa
from src.data_loader import load_music_catalog
from tests.local_data import requires_music_catalog, requires_music_classifier, skip_if_missing


@pytest.fixture(scope="module")
def catalog():
    # every test using this runs prep.build, which also needs the 278k classifier
    skip_if_missing(requires_music_catalog, requires_music_classifier)
    return load_music_catalog()


def _row(i, **over):
    tid = f"{i:022d}"
    row = {"track_id": tid, "track_name": f"Synthetic Song {i}", "artist_name": "Synthetic Star", "year": "2021",
           "popularity": "50", "artwork_url": "https://example.invalid/a.jpg", "album_name": "Synthetic Album",
           "acousticness": "0.2", "danceability": "0.6", "duration_ms": "200000.0", "energy": "0.7",
           "instrumentalness": "0.0", "key": "5.0", "liveness": "0.1", "loudness": "-5.0", "mode": "1.0",
           "speechiness": "0.05", "tempo": "120.0", "time_signature": "4.0", "valence": "0.8",
           "track_url": f"https://open.spotify.com/track/{tid}", "language": "English"}
    row.update({k: str(v) for k, v in over.items()})
    return row


def _write(tmp_path, rows, name="raw.csv", columns=None):
    path = tmp_path / name
    pd.DataFrame(rows, columns=columns or prep.EXPECTED_COLUMNS).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "artists.json"
    path.write_text(json.dumps({"categories": {
        "international": ["Synthetic Star", "Comma, The Band"],
        "kpop": ["SYNTH"],
        "indian": ["Solo"]}}), encoding="utf-8")
    return str(path)


@pytest.fixture
def pinned(monkeypatch):
    """Pin EXPECTED_SHA256 to whatever file the test writes."""
    def pin(path):
        monkeypatch.setattr(prep, "EXPECTED_SHA256", mp.file_sha256(path))
        return path
    return pin


@pytest.fixture
def outdir(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(pa, "DATA_DIR", str(out))
    monkeypatch.setattr(pa, "PREPARED_PATH", str(out / "popular_tracks.csv"))
    monkeypatch.setattr(pa, "MANIFEST_PATH", str(out / "manifest.json"))
    monkeypatch.setattr(prep, "RAW_PATH", str(out / "spotify_tracks.csv"))
    return out


def test_hash_mismatch_stops_before_anything_is_written(tmp_path, outdir, config):
    raw = _write(tmp_path, [_row(1)])
    with pytest.raises(prep.PreparationError, match="SHA-256"):
        prep.build(raw, config)
    with pytest.raises(prep.PreparationError):
        prep.import_source(raw)                                   # a modified file is never installed
    assert not os.path.exists(prep.RAW_PATH) and not os.path.exists(pa.PREPARED_PATH)
    assert prep.main(["--source", raw]) == 1 and not os.path.exists(pa.PREPARED_PATH)


def test_main_without_a_raw_file_explains_how_to_get_it(outdir, capsys):
    assert prep.main([]) == 1
    assert "--source" in capsys.readouterr().err and not os.path.exists(pa.PREPARED_PATH)


def test_unexpected_schema_or_non_numeric_values_stop(tmp_path, config, pinned):
    extra = [{**_row(1), "surprise": "x"}]
    with pytest.raises(prep.PreparationError, match="Unexpected columns"):
        prep.build(pinned(_write(tmp_path, extra, "extra.csv", prep.EXPECTED_COLUMNS + ["surprise"])), config)
    reordered = prep.EXPECTED_COLUMNS[1:] + prep.EXPECTED_COLUMNS[:1]
    with pytest.raises(prep.PreparationError, match="Unexpected columns"):
        prep.build(pinned(_write(tmp_path, [_row(1)], "order.csv", reordered)), config)
    with pytest.raises(prep.PreparationError, match="non-numeric"):
        prep.build(pinned(_write(tmp_path, [_row(1, energy="high")], "text.csv")), config)


def test_bad_rows_are_dropped_and_counted(tmp_path, config, pinned, catalog):
    rows = [
        _row(1),                                                    # kept
        _row(2, track_id="short", track_url="https://open.spotify.com/track/short"),
        _row(3, track_url="https://open.spotify.com/track/0000000000000000000099"),
        _row(4, valence="-1.0"),                                    # the source's placeholder
        _row(5, loudness="-100000.0"),                              # another placeholder form
        _row(6, duration_ms="42000.0"),                             # under a minute: interlude
        _row(7, speechiness="0.9"),                                 # spoken word
        _row(8, tempo="0.0"),                                       # failed audio analysis
        _row(9, track_name="  "),
        _row(10, artist_name="Someone Else"),                       # not curated
    ]
    prepared, manifest = prep.build(pinned(_write(tmp_path, rows)), config, catalog)
    c = manifest["counts"]
    assert prepared.track_id.tolist() == [f"{1:022d}"] and c["prepared_rows"] == 1 and c["raw_rows"] == 10
    assert c["dropped_malformed_track_id"] == 1 and c["dropped_track_url_mismatch"] == 1
    assert c["dropped_placeholder_or_out_of_range"] == 3 and c["dropped_spoken_word"] == 1
    assert c["dropped_failed_audio_analysis"] == 1 and c["dropped_empty_text"] == 1
    assert c["dropped_not_curated_artist"] == 1


def test_duplicate_ids_and_versions(tmp_path, config, pinned, catalog):
    rows = [
        _row(1), _row(1),                                           # identical copy: kept once
        _row(2), _row(2, popularity="70"),                          # disagreeing copies: both dropped
        _row(3, track_name="Hit", album_name="Original", popularity="80"),
        _row(4, track_name="hit ", album_name="Pop Hits 2023", popularity="5"),     # same recording, compilation
        _row(5, track_name="Hit", popularity="60", duration_ms="230000.0"),         # different length: kept
        _row(6, track_name="Hit (Remix)", popularity="40"),                         # different title: kept
    ]
    prepared, manifest = prep.build(pinned(_write(tmp_path, rows)), config, catalog)
    c = manifest["counts"]
    assert c["dropped_exact_duplicate_rows"] == 1 and c["dropped_conflicting_duplicate_ids"] == 2
    assert c["dropped_duplicate_versions"] == 1
    assert sorted(prepared.track_id) == [f"{i:022d}" for i in (1, 3, 5, 6)]


def test_only_whole_curated_entries_match_and_strings_are_kept(tmp_path, config, pinned, catalog):
    rows = [
        _row(1, artist_name="Comma, The Band, Guest"),              # comma-containing curated name
        _row(2, artist_name="Solo Chakraborty"),                    # partial name: no match
        _row(3, artist_name="Guest, SYNTH, Solo"),
        _row(4, artist_name="Comma"),                               # half of a comma name: no match
    ]
    prepared, manifest = prep.build(pinned(_write(tmp_path, rows)), config, catalog)
    by_id = prepared.set_index("track_id")
    assert sorted(by_id.index) == [f"{1:022d}", f"{3:022d}"]
    assert by_id.loc[f"{1:022d}", "artist_name"] == "Comma, The Band, Guest"
    assert json.loads(by_id.loc[f"{1:022d}", "matched_artists"]) == ["Comma, The Band"]
    assert json.loads(by_id.loc[f"{3:022d}", "matched_artists"]) == ["SYNTH", "Solo"]
    assert by_id.loc[f"{3:022d}", "category"] == "kpop"             # first matched artist's category
    assert manifest["curated_artists_missing"] == ["Synthetic Star"]


def test_moods_use_the_existing_classifier_and_recorded_cutoffs(tmp_path, config, pinned, catalog):
    angry = dict(valence="0.1", energy="0.95", loudness="-3.0")      # inside the recorded angry cutoffs
    rows = [_row(1, **angry), _row(2, valence="0.9", energy="0.3", loudness="-12.0")]
    prepared, manifest = prep.build(pinned(_write(tmp_path, rows)), config, catalog)
    prepared = prepared.set_index("track_id")
    assert bool(prepared.loc[f"{1:022d}", "derived_angry"]) and not bool(prepared.loc[f"{2:022d}", "derived_angry"])
    probs = prepared[["proba_sad", "proba_happy", "proba_energetic", "proba_calm"]].sum(axis=1)
    assert ((probs - 1).abs() < 1e-4).all()
    recorded = json.load(open(prep.EVAL_PATH, encoding="utf-8"))["derived_moods"]["angry"]["rule"]
    used = manifest["mood_method"]["derived_moods"]["angry"]["rule"]
    assert [r["cutoff"] for r in used] == [r["cutoff"] for r in recorded]   # never recomputed on the new data
    assert manifest["mood_method"]["validation"] == "model-derived, not human-validated"


def test_stale_recorded_cutoffs_stop(tmp_path, config, pinned, catalog, monkeypatch):
    stale = json.load(open(prep.EVAL_PATH, encoding="utf-8"))
    stale["derived_moods"]["angry"]["rule"][0]["cutoff"] = 0.5
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(stale), encoding="utf-8")
    monkeypatch.setattr(prep, "EVAL_PATH", str(path))
    with pytest.raises(prep.PreparationError, match="cutoff"):
        prep.build(pinned(_write(tmp_path, [_row(1)])), config, catalog)


def test_outputs_are_written_together_and_the_manifest_is_aggregate_only(tmp_path, config, pinned, catalog, outdir):
    raw = pinned(_write(tmp_path, [_row(i, artist_name="Synthetic Star" if i % 2 else "SYNTH") for i in range(1, 7)]))
    prepared, manifest = prep.build(raw, config, catalog)
    written = prep.write_outputs(prepared, manifest)
    text = open(pa.MANIFEST_PATH, encoding="utf-8").read()
    assert not any(f"{i:022d}" in text for i in range(1, 7))
    assert written["prepared_sha256"] == mp.file_sha256(pa.PREPARED_PATH)
    assert written["contains_track_ids"] is False and written["source"]["collection_method"] == "not documented by the author"
    loaded = pa.load_prepared(pa.PREPARED_PATH)
    assert loaded is not None and len(loaded) == 6 and "artwork_url" not in loaded.columns
    assert not [f for f in os.listdir(outdir) if f.endswith(".tmp")]
