"""M5 - External validation: prediction of drugs excluded from training.

Predicts the logS of the external-set molecules, evaluates the applicability
domain and compares against the experimental truth obtained from ChEMBL.

Usage:
    python src/validate_external.py --external data/external/validation_set.csv \
                                    --model reports/model.joblib --outdir reports
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from applicability import ApplicabilityDomain, ADConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation of logS")
    parser.add_argument("--external", default="data/external/validation_set.csv")
    parser.add_argument("--model", default="reports/model.joblib")
    parser.add_argument("--outdir", default="reports")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    ad = ApplicabilityDomain(artifact, ADConfig())

    ext = pd.read_csv(args.external)
    verdicts = ad.evaluate_smiles(ext["smiles"].tolist())

    ext = ext.assign(
        logS_pred=[v.prediction for v in verdicts],
        halfwidth=[v.halfwidth for v in verdicts],
        sim_max=[v.max_similarity for v in verdicts],
        leverage=[v.leverage for v in verdicts],
        status=[v.status for v in verdicts],
    )
    ext["error"] = (ext["logS_pred"] - ext["logS_experimental"]).round(3)

    print(f"Model: {artifact['model_name']}  |  external set: {len(ext)} drugs\n")
    header = (f"{'drug':<15}{'logS_exp':>9}{'logS_pred':>10}{'error':>8}"
              f"{'sim':>6}{'h':>7}  status")
    print(header)
    print("-" * len(header))
    for _, r in ext.sort_values("logS_experimental").iterrows():
        print(f"{r['name']:<15}{r['logS_experimental']:>9.2f}{r['logS_pred']:>10.2f}"
              f"{r['error']:>8.2f}{r['sim_max']:>6.2f}{r['leverage']:>7.3f}  {r['status']}")

    mae = np.abs(ext["error"]).mean()
    print(f"\nExternal MAE: {mae:.3f} log  |  mean bias: {ext['error'].mean():+.3f}")
    for status in ("inside", "borderline", "outside"):
        sub = ext[ext["status"] == status]
        if len(sub):
            print(f"  {status:<11} n={len(sub):>2}  MAE={np.abs(sub['error']).mean():.3f}")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ext.to_csv(Path(args.outdir) / "external_validation.csv", index=False)
    print(f"\nSaved to {args.outdir}/external_validation.csv")


if __name__ == "__main__":
    main()
