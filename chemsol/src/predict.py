"""M5 - logS prediction for new molecules, with applicability domain.

Usage:
    python src/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python src/predict.py --smiles-file new_molecules.csv --column smiles
    python src/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from applicability import ADConfig, ApplicabilityDomain


def parse_smiles(args) -> list[str]:
    smiles: list[str] = []
    if args.smiles:
        smiles += args.smiles
    if args.smiles_file:
        df = pd.read_csv(args.smiles_file)
        if args.column not in df.columns:
            raise SystemExit(f"Column '{args.column}' does not exist in {args.smiles_file}. "
                             f"Columns: {list(df.columns)}")
        smiles += df[args.column].dropna().tolist()
    # Allow several SMILES separated by ';' within a single argument
    expanded = [s.strip() for item in smiles for s in str(item).split(";") if s.strip()]
    if not expanded:
        raise SystemExit("No SMILES provided (use --smiles or --smiles-file)")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Aqueous solubility (logS) prediction")
    parser.add_argument("--model", default="reports/model.joblib")
    parser.add_argument("--smiles", nargs="+", help="One or more SMILES")
    parser.add_argument("--smiles-file", help="CSV with a column of SMILES")
    parser.add_argument("--column", default="smiles", help="Name of the SMILES column")
    parser.add_argument("--alpha", type=float, default=0.10, help="Conformal interval level")
    parser.add_argument("--halfwidth-tolerance", type=float, default=None,
                        help="Absolute half-width threshold (default: relative to calibration)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}. Run src/train.py first")

    artifact = joblib.load(model_path)
    ad = ApplicabilityDomain(
        artifact,
        ADConfig(alpha=args.alpha, halfwidth_tolerance=args.halfwidth_tolerance),
    )

    smiles_list = parse_smiles(args)
    verdicts = ad.evaluate_smiles(smiles_list)

    if args.json:
        print(json.dumps([v.as_dict() for v in verdicts], indent=2))
        return

    print(f"\nModel: {artifact['model_name']}  |  "
          f"AD thresholds: leverage<{ad.leverage_threshold:.4f}  "
          f"similarity>{ad.similarity_threshold:.3f}  "
          f"half-width<{ad.halfwidth_threshold:.2f}")
    header = f"{'SMILES':<45}{'logS':>7}{'90% CI':>17}{'sim':>7}{'h':>8}  status"
    print("\n" + header)
    print("-" * len(header))
    for v in verdicts:
        smi = v.smiles if len(v.smiles) <= 43 else v.smiles[:40] + "..."
        interval = f"[{v.interval_low:.2f}, {v.interval_high:.2f}]"
        print(f"{smi:<45}{v.prediction:>7.2f}{interval:>17}"
              f"{v.max_similarity:>7.2f}{v.leverage:>8.4f}  {v.status}")
    print("\nLegend: sim = maximum Tanimoto similarity to the training set; "
          "h = leverage; status = inside/borderline/outside the applicability domain.")


if __name__ == "__main__":
    main()
