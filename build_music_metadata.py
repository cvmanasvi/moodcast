"""
Build the local-only music display metadata (title / artist / release year).

Run from the project root:  python build_music_metadata.py
Check existing outputs:     python build_music_metadata.py --verify

Reads   data/music_catalog_278k.csv        (canonical catalog; never modified,
                                            SHA-256 checked before and after)
        data/external/tracks.csv           (Source A, extracted from A_600k.zip)
Writes  data/metadata/music_metadata.csv    (gitignored)
        data/metadata/metadata_sources.json (gitignored provenance manifest)

Matching is exact track_id equality only. The output is deterministic: the
manifest records hashes and counts but no run timestamp, so rebuilding from
the same inputs produces byte-identical files. See METADATA.md.
"""
import datetime
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (BASE_DIR, EXTERNAL_DATA_DIR, METADATA_MANIFEST_PATH, MUSIC_CATALOG_278K_PATH,
                        MUSIC_METADATA_PATH)
from src import music_metadata as mm
from src.music_pipeline import file_sha256


def _rel(path):
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")


def verify():
    """Validate the existing outputs against the catalog and source without rebuilding."""
    source_path = os.path.join(EXTERNAL_DATA_DIR, mm.SOURCE_A["source_filename"])
    catalog_ids = pd.read_csv(MUSIC_CATALOG_278K_PATH, usecols=["track_id"], dtype=str)["track_id"]
    mm.validate_metadata_table(mm.read_metadata_table(MUSIC_METADATA_PATH), catalog_ids)
    if not os.path.exists(METADATA_MANIFEST_PATH):
        raise mm.MetadataBuildError(f"Manifest not found: {METADATA_MANIFEST_PATH}")
    with open(METADATA_MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    mm.verify_manifest(manifest, source_path, MUSIC_CATALOG_278K_PATH, MUSIC_METADATA_PATH)
    print(f"OK: {_rel(MUSIC_METADATA_PATH)} and {_rel(METADATA_MANIFEST_PATH)} verified "
          f"({manifest['generated_row_count']} rows).")


def main():
    if "--verify" in sys.argv[1:]:
        verify()
        return

    source = mm.SOURCE_A
    source_path = os.path.join(EXTERNAL_DATA_DIR, source["source_filename"])
    archive_path = os.path.join(EXTERNAL_DATA_DIR, source["archive_filename"])

    catalog_hash_before = file_sha256(MUSIC_CATALOG_278K_PATH)
    catalog = pd.read_csv(MUSIC_CATALOG_278K_PATH, usecols=["track_id", "mood"], dtype={"track_id": str})
    raw = mm.load_source(source_path, source["columns"])
    source_duplicate_ids = int(raw["id"].duplicated().sum())

    metadata = mm.build_metadata(catalog["track_id"], raw, source["source_id"])
    mm.write_metadata(metadata, MUSIC_METADATA_PATH)
    # validate what was actually written, read back the way consumers will read it
    mm.validate_metadata_table(mm.read_metadata_table(MUSIC_METADATA_PATH), catalog["track_id"])
    coverage = mm.coverage_report(catalog, metadata)

    if file_sha256(MUSIC_CATALOG_278K_PATH) != catalog_hash_before:
        raise SystemExit("Canonical catalog changed during the metadata build -- this must never happen.")

    acquisition = None
    if os.path.exists(archive_path):
        acquisition = datetime.date.fromtimestamp(os.path.getmtime(archive_path)).isoformat()

    manifest = {
        "source_id": source["source_id"],
        "source_name": source["source_name"],
        "creator": source["creator"],
        "source_url": source["source_url"],
        "source_version": source["source_version"],
        "license": source["license"],
        "license_url": source["license_url"],
        "attribution": f"{source['source_name']} by {source['creator']} ({source['source_url']}), "
                       f"licensed under {source['license']}. Only title, artists (display string + JSON list) and release year were "
                       f"extracted, for tracks whose Spotify ID matches the MoodCast catalog.",
        "acquisition_date": acquisition,
        "acquisition_date_basis": f"filesystem modification time of {_rel(archive_path)} (local download)"
                                  if acquisition else "archive not present; not recorded",
        "source_archive": _rel(archive_path) if os.path.exists(archive_path) else None,
        "source_archive_sha256": file_sha256(archive_path) if os.path.exists(archive_path) else None,
        "source_filename": _rel(source_path),
        "source_sha256": file_sha256(source_path),
        "source_row_count": int(len(raw)),
        "source_duplicate_ids": source_duplicate_ids,
        "source_columns_used": source["columns"],
        "catalog_file": _rel(MUSIC_CATALOG_278K_PATH),
        "catalog_sha256": catalog_hash_before,
        "generated_file": _rel(MUSIC_METADATA_PATH),
        "generated_sha256": file_sha256(MUSIC_METADATA_PATH),
        "generated_row_count": int(len(metadata)),
        "generated_columns": mm.METADATA_COLUMNS,
        "matching_method": "exact track_id",
        "coverage": coverage,
        "notes": "Local-only metadata enrichment. Not used for ML training or recommendation scoring. "
                 "Not committed to git (data/metadata/ is gitignored). Source A has no album column, "
                 "so album_name is not provided.",
    }
    with open(METADATA_MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(coverage, indent=2))
    print(f"\nWrote {len(metadata)} rows to {_rel(MUSIC_METADATA_PATH)} "
          f"(sha256 {manifest['generated_sha256'][:12]}...) and {_rel(METADATA_MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
