"""M5 - Construction of the external validation set.

Resolves a list of well-known drugs to ChEMBL identifiers, retrieves their
aqueous solubility measurements from the downloaded raw data, converts them to
logS with the same methodology and writes a CSV with the experimental truth.

These molecules are EXCLUDED from training (see --holdout-file in train.py), so
they constitute a genuine external validation.

Usage:
    python src/make_external_set.py --raw data/raw/solubility_chembl.csv \
        --out data/external/validation_set.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from curate import EXCLUDE_PATTERNS, LOGS_MAX, LOGS_MIN, parent_smiles, to_logs

RDLogger.DisableLog("rdApp.*")

# Well-known drugs with a structure and experimental aqueous solubility in ChEMBL.
DRUG_NAMES = [
    "Warfarin", "Paracetamol", "Caffeine", "Aspirin", "Ibuprofen",
    "Diclofenac", "Propranolol", "Verapamil", "Tamoxifen", "Lidocaine",
    "Diazepam", "Ketoconazole", "Griseofulvin", "Rifampicin", "Furosemide",
]

MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"


def resolve_chembl_id(name: str) -> str | None:
    """Resolves a drug name to its ChEMBL ID (exact preferred name)."""
    try:
        resp = requests.get(MOLECULE_URL, params={"pref_name__iexact": name, "limit": 1},
                            timeout=30)
        resp.raise_for_status()
        molecules = resp.json().get("molecules", [])
        return molecules[0]["molecule_chembl_id"] if molecules else None
    except (requests.RequestException, ValueError) as exc:
        print(f"  [warning] {name}: {exc}")
        return None


def compute_experimental_logs(raw: pd.DataFrame, chembl_id: str) -> dict | None:
    """Applies the same curation as the training dataset to a single molecule."""
    rec = raw[(raw["molecule_chembl_id"] == chembl_id)
              & (raw["standard_relation"] == "=")
              & (raw["standard_value"] > 0)
              & (raw["potential_duplicate"].fillna(0) == 0)
              & (raw["standard_units"].isin(["ug.mL-1", "nM"]))].copy()
    if rec.empty:
        return None

    suspicious = rec["assay_description"].fillna("").str.contains(
        "|".join(EXCLUDE_PATTERNS), case=False, regex=True
    )
    rec = rec[~suspicious]
    if rec.empty:
        return None

    mol = Chem.MolFromSmiles(rec.iloc[0]["canonical_smiles"])
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)

    logS_values = [to_logs(float(v), u, mw) for v, u in
                   zip(rec["standard_value"], rec["standard_units"])]
    logS_values = [x for x in logS_values if LOGS_MIN <= x <= LOGS_MAX]
    if not logS_values:
        return None

    return {
        "smiles": parent_smiles(Chem.MolToSmiles(mol)),
        "logS_experimental": round(float(np.median(logS_values)), 3),
        "n_records": len(logS_values),
        "logS_spread": round(float(np.std(logS_values)), 3) if len(logS_values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation set")
    parser.add_argument("--raw", default="data/raw/solubility_chembl.csv")
    parser.add_argument("--out", default="data/external/validation_set.csv")
    parser.add_argument("--processed", default="data/processed/solubility_logS.csv",
                        help="Curated dataset, to confirm the molecule enters it")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    processed = pd.read_csv(args.processed)
    in_dataset = set(processed["chembl_id"])

    rows = []
    for name in DRUG_NAMES:
        chembl_id = resolve_chembl_id(name)
        if chembl_id is None:
            print(f"  {name:<15} no ChEMBL ID")
            continue
        data = compute_experimental_logs(raw, chembl_id)
        if data is None:
            print(f"  {name:<15} {chembl_id:<13} no usable measurements after curation")
            continue
        rows.append({
            "name": name,
            "chembl_id": chembl_id,
            **data,
            "in_processed_dataset": chembl_id in in_dataset,
        })
        print(f"  {name:<15} {chembl_id:<13} logS_exp={data['logS_experimental']:6.2f} "
              f"(n={data['n_records']}, spread={data['logS_spread']:.2f})")
        time.sleep(0.2)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n{len(out)} external-validation molecules -> {out_path}")
    print(f"Present in the curated dataset: {int(out['in_processed_dataset'].sum())} "
          f"(they will be excluded from training)")


if __name__ == "__main__":
    main()
