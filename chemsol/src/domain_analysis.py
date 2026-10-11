"""M4 - Applicability domain analysis.

Evaluates the AD on the held-out test set and on a list of molecules
deliberately placed out of domain (OOD). Produces the Williams plot
(leverage versus standardised residual) and a coverage summary.

Usage:
    python src/domain_analysis.py --model reports/model.joblib \
                                  --test reports/test_set.csv \
                                  --outdir reports
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from applicability import ApplicabilityDomain, ADConfig  # noqa: E402

# Molecules deliberately far from the training chemical space (polymers,
# peptides, surfactants, fluorinated compounds). They should fall outside the AD.
OOD_PROBES = {
    "Oligoethylene glycol (polymer)": "COCCOCCOCCOCCOCCOCCOCCOC",
    "Perfluorooctane (fluorinated)": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "Cetyl-trimethylammonium (surfactant)": "CCCCCCCCCCCCCCCC[N+](C)(C)C",
    "Tetrapeptide (H-Ala-Phe-Gly-Asn-OH)":
        "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)CNC(=O)[C@@H](N)CC(N)=O)C(=O)N[C@@H](CO)C(=O)O",
    "Cyclosporin-like (peptidic macrocycle)":
        "CC[C@H](C)[C@@H]1NC(=O)[C@H](C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](NC(=O)[C@H](CC(C)C)NC1=O)C",
    "Polyaromatic dye (hypericin-like)":
        "Cc1cc2cc3c(c2c(O)c1O)C(=O)c1c(O)c(O)c(O)c(O)c1C3=O",
}


def williams_plot(ad: ApplicabilityDomain, test: pd.DataFrame, out_path: Path) -> None:
    """Williams plot: leverage versus standardised residual."""
    residuals = test["logS"].to_numpy() - test["logS_pred"].to_numpy()
    resid_std = residuals / np.sqrt(np.mean(residuals**2))

    inside = (test["status"] != "outside").to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(3, color="grey", ls="--", lw=1)
    ax.axhline(-3, color="grey", ls="--", lw=1)
    ax.axvline(ad.leverage_threshold, color="red", ls="--", lw=1.2,
               label=f"h* = {ad.leverage_threshold:.3f}")
    ax.scatter(test.loc[inside, "leverage"], resid_std[inside], s=14, alpha=0.55,
               c="steelblue", edgecolors="none", label=f"inside/borderline (n={inside.sum()})")
    ax.scatter(test.loc[~inside, "leverage"], resid_std[~inside], s=22, alpha=0.8,
               c="crimson", edgecolors="none", label=f"outside the AD (n={(~inside).sum()})")
    ax.set_xlabel("Leverage  h")
    ax.set_ylabel("Standardised residual")
    ax.set_title("Williams plot on the test set")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Applicability domain analysis")
    parser.add_argument("--model", default="reports/model.joblib")
    parser.add_argument("--test", default="reports/test_set.csv")
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    ad = ApplicabilityDomain(artifact, ADConfig(alpha=args.alpha))

    print("=== Applicability domain thresholds ===")
    print(f"  leverage h*          : {ad.leverage_threshold:.4f}")
    print(f"  Tanimoto similarity  : {ad.similarity_threshold:.3f}")
    print(f"  calibration used     : {ad.n_calibration} molecules")
    print(f"  conformal q (90%)    : {ad.conformal_q:.4f}")
    print(f"  maximum half-width   : {ad.halfwidth_threshold:.2f} log")

    # ---------- Test set ----------
    test = pd.read_csv(args.test)
    stats = ad.evaluate_smiles(test["smiles"].tolist())
    test = test.assign(
        status=[s.status for s in stats],
        leverage=[s.leverage for s in stats],
        max_similarity=[s.max_similarity for s in stats],
        halfwidth=[s.halfwidth for s in stats],
        err=np.abs(test["logS"] - test["logS_pred"]),
    )

    print("\n=== AD on the test set ===")
    print(test["status"].value_counts().to_string())
    print("\nMean absolute error by verdict:")
    print(test.groupby("status")["err"].agg(["mean", "count"]).round(3).to_string())

    inside = test["status"] != "outside"
    if inside.any() and (~inside).any():
        print(f"\nMAE inside/borderline : {test.loc[inside, 'err'].mean():.3f} "
              f"(n={inside.sum()})")
        print(f"MAE outside           : {test.loc[~inside, 'err'].mean():.3f} "
              f"(n={(~inside).sum()})")

    # ---------- Out-of-domain molecules ----------
    print("\n=== Molecules deliberately placed out of domain ===")
    names = list(OOD_PROBES)
    probe_stats = ad.evaluate_smiles([OOD_PROBES[n] for n in names])
    for name, s in zip(names, probe_stats):
        print(f"  {name:<38} logS={s.prediction:6.2f}  "
              f"sim={s.max_similarity:4.2f}  h={s.leverage:5.2f}  -> {s.status}")

    Path(args.outdir, "figures").mkdir(parents=True, exist_ok=True)
    test.to_csv(Path(args.outdir) / "test_set_with_ad.csv", index=False)
    williams_plot(ad, test, Path(args.outdir, "figures") / "williams_plot.png")
    print(f"\nResults saved to {args.outdir}/")


if __name__ == "__main__":
    main()
