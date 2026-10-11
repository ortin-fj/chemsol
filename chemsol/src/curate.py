"""M2 - Curation of the solubility dataset.

Converts the raw values to logS (log10 mol/L), drops unreliable records and
aggregates repeated measurements per compound (median).

Unit conversion
---------------
- ug.mL-1 : 1 ug/mL = 1e-3 g/L  ->  S[mol/L] = value * 1e-3 / MW
- nM      : 1 nM   = 1e-9 mol/L  ->  S[mol/L] = value * 1e-9

Usage:
    python src/curate.py --raw data/raw/solubility_chembl.csv \
                         --out data/processed/solubility_logS.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# Plausible physicochemical range of aqueous solubility (log mol/L).
LOGS_MIN, LOGS_MAX = -12.0, 2.0

# Assays labelled as "solubility" but carried out in media that are not water
# (proteins, culture media, surfactant formulations). They measure something
# else and are excluded to avoid biasing the model.
EXCLUDE_PATTERNS = [
    "aemoglobin", "serum albumin", "bsa", "albumin",
    "cell culture", "caco-2", "caco2", "methylcellulose", "tween",
    "broth", "middlebrook", "plasma", "serum",
]


def to_logs(value: float, unit: str, mw: float) -> float:
    """Convert a standardised solubility value to logS (log10 mol/L).

    - ug.mL-1 : 1 ug/mL = 1e-3 g/L  ->  S[mol/L] = value * 1e-3 / MW
    - nM      : 1 nM   = 1e-9 mol/L
    """
    if unit == "nM":
        return float(np.log10(value) - 9.0)
    if unit == "ug.mL-1":
        return float(np.log10(value) - 3.0 - np.log10(mw))
    raise ValueError(f"unsupported unit: {unit}")


_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def parent_mol(mol: Chem.Mol) -> Chem.Mol:
    """Parent structure: largest organic fragment with no net charges."""
    return _UNCHARGER.uncharge(_LARGEST_FRAGMENT.choose(mol))


def parent_smiles(smiles: str) -> str | None:
    """SMILES of the parent structure, or None if the SMILES is invalid.

    ChEMBL registers the same substance sometimes as a salt (e.g. diclofenac
    sodium) and sometimes as the neutral form. Normalising to the parent
    structure merges both into a single compound and keeps the counter-ion from
    contaminating the descriptors and the fingerprints.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(parent_mol(mol)) if mol is not None else None


def curate(df: pd.DataFrame, max_std: float, min_measurements: int) -> tuple[pd.DataFrame, dict]:
    """Return the curated dataset and a dict with the curation-funnel traces."""
    funnel: dict[str, int] = {"raw": len(df)}
    df = df.copy()

    # 1. Usable records
    df = df[df["standard_value"].notna()]
    df = df[df["standard_value"] > 0]
    df = df[df["standard_relation"] == "="]
    df = df[df["canonical_smiles"].notna()]
    funnel["positive_value_and_equality_relation"] = len(df)

    # 2. Drop assays run in media that are not water
    suspicious = df["assay_description"].fillna("").str.contains(
        "|".join(EXCLUDE_PATTERNS), case=False, regex=True
    )
    df = df[~suspicious]
    funnel["aqueous_context"] = len(df)

    # 3. Drop duplicates flagged by ChEMBL
    df = df[df["potential_duplicate"].fillna(0).astype(int) == 0]
    funnel["without_flagged_duplicates"] = len(df)

    # 4. Validate SMILES and compute the molar mass
    mols, mws = [], []
    for smi in df["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol)
        mws.append(Descriptors.MolWt(mol) if mol is not None else np.nan)
    df["_mol"] = mols
    df["_mw"] = mws
    df = df[df["_mol"].notna() & (df["_mw"] > 0)]
    funnel["valid_smiles"] = len(df)

    # 5. Normalise to the parent structure (desalting + neutralisation). Unit
    #    conversion still uses _mw, the mass of the registered form, because
    #    that is the form actually weighed in the laboratory.
    parent_mols = [parent_mol(m) for m in df["_mol"]]
    df["smiles"] = [Chem.MolToSmiles(m) for m in parent_mols]
    df["_parent_mw"] = [Descriptors.MolWt(m) for m in parent_mols]
    df["_was_salt"] = [
        p != o for p, o in zip(df["smiles"], (Chem.MolToSmiles(m) for m in df["_mol"]))
    ]
    funnel["desalted_entries"] = int(df["_was_salt"].sum())
    funnel["compounds_after_desalting"] = int(df["smiles"].nunique())

    # 6. Conversion to logS
    df["logS"] = [to_logs(v, u, m)
                  for v, u, m in zip(df["standard_value"], df["standard_units"], df["_mw"])]

    df = df[(df["logS"] >= LOGS_MIN) & (df["logS"] <= LOGS_MAX)]
    funnel["plausible_logS_range"] = len(df)

    # 7. Aggregate per compound: median of the available measurements
    agg = (
        df.groupby("smiles")
        .agg(
            logS=("logS", "median"),
            logS_std=("logS", "std"),
            n_measurements=("logS", "size"),
            chembl_id=("molecule_chembl_id", "first"),
            mw=("_parent_mw", "first"),
            n_salt_forms=("_was_salt", "sum"),
            first_year=("document_year", "min"),
            unit=("standard_units", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    funnel["unique_compounds"] = len(agg)

    # 8. Reliability: agreement between measurements and minimum count
    agg["logS_std"] = agg["logS_std"].fillna(0.0)
    concordant = agg[agg["logS_std"] <= max_std]
    funnel[f"agreement_logS_std<={max_std}"] = len(concordant)

    final = concordant[concordant["n_measurements"] >= min_measurements].reset_index(drop=True)
    funnel[f"at_least_{min_measurements}_measurement(s)"] = len(final)

    final["mw"] = final["mw"].round(3)
    final["logS"] = final["logS"].round(3)
    final["logS_std"] = final["logS_std"].round(3)
    final = final[["smiles", "logS", "logS_std", "n_measurements", "n_salt_forms",
                   "unit", "mw", "first_year", "chembl_id"]]
    return final, funnel


def main() -> None:
    parser = argparse.ArgumentParser(description="Curation of the solubility dataset")
    parser.add_argument("--raw", default="data/raw/solubility_chembl.csv")
    parser.add_argument("--out", default="data/processed/solubility_logS.csv")
    parser.add_argument("--max-std", type=float, default=1.5,
                        help="Maximum standard deviation between measurements of one compound")
    parser.add_argument("--min-measurements", type=int, default=1,
                        help="Minimum number of experimental measurements per compound")
    args = parser.parse_args()

    df = pd.read_csv(args.raw)
    final, funnel = curate(df, args.max_std, args.min_measurements)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)

    print("Curation funnel:")
    for step, n in funnel.items():
        print(f"  {step:<34} {n:>8}")
    print(f"\nFinal dataset: {len(final)} compounds -> {out_path}")
    print(f"logS: mean={final['logS'].mean():.2f}  "
          f"std={final['logS'].std():.2f}  "
          f"range=[{final['logS'].min():.2f}, {final['logS'].max():.2f}]")
    print(f"Source unit: {final['unit'].value_counts().to_dict()}")
    print(f"With >=2 measurements: {(final['n_measurements'] >= 2).sum()}")
    print(f"Compounds with a salt-form measurement: {(final['n_salt_forms'] > 0).sum()}")


if __name__ == "__main__":
    main()
