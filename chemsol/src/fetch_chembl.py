"""M1 - Download of aqueous solubility data from the public ChEMBL API.

Downloads every activity with standard_type='Solubility' in the units that hold
the bulk of the records (ug.mL-1 and nM) and stores them raw.

Usage:
    python src/fetch_chembl.py --out data/raw/solubility_chembl.csv
    python src/fetch_chembl.py --limit 200 --out data/raw/sample.csv   # quick test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

# Units that hold the bulk of the solubility records in ChEMBL (verified).
TARGET_UNITS = ["ug.mL-1", "nM"]

FIELDS = [
    "activity_id",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "target_pref_name",
    "data_validity_comment",
    "potential_duplicate",
    "document_chembl_id",
    "document_year",
    "document_journal",
    "src_id",
]


def _fetch_unit(units: str, page_size: int, max_records: int | None, verbose: bool) -> list[dict]:
    """Paged download of the activities for a single unit."""
    records: list[dict] = []
    offset = 0
    total: int | None = None

    while True:
        params = {
            "standard_type": "Solubility",
            "standard_units": units,
            "limit": page_size,
            "offset": offset,
        }
        for attempt in range(4):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=60)
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 3:
                    raise
                wait = 2**attempt
                print(f"  [warning] failure at offset={offset} ({exc}); retrying in {wait}s")
                time.sleep(wait)

        if total is None:
            total = payload["page_meta"]["total_count"]

        activities = payload.get("activities", [])
        if not activities:
            break

        records.extend({f: act.get(f) for f in FIELDS} for act in activities)
        offset += page_size

        if verbose:
            print(f"  {units}: {len(records)}/{total}")

        if offset >= total:
            break
        if max_records is not None and len(records) >= max_records:
            records = records[:max_records]
            break

    return records


def fetch(units: list[str], page_size: int, max_records: int | None, verbose: bool) -> pd.DataFrame:
    frames = []
    for unit in units:
        print(f"Downloading standard_units='{unit}' ...")
        rows = _fetch_unit(unit, page_size, max_records, verbose)
        print(f"  -> {len(rows)} records")
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aqueous solubility download from ChEMBL")
    parser.add_argument("--out", default="data/raw/solubility_chembl.csv", help="Output CSV")
    parser.add_argument("--units", nargs="+", default=TARGET_UNITS, help="Units to download")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Max records per unit (testing)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    df = fetch(args.units, args.page_size, args.limit, verbose=not args.quiet)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} records to {out_path}")
    print(f"Unique compounds (SMILES): {df['canonical_smiles'].nunique()}")
    print(f"Years: {df['document_year'].min()} - {df['document_year'].max()}")


if __name__ == "__main__":
    main()
