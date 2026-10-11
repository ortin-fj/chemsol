"""M3 - Training and comparison of models for logS.

Compares three model families (Ridge, RandomForest, XGBoost) through 5-fold
cross-validation and evaluates them on a held-out test set.

Two splits are produced:
  * random (fixed random_state)   -> comparable with the literature
  * temporal (compounds after --temporal-year)

Usage:
    python src/train.py --processed data/processed/solubility_logS.csv \
                        --outdir reports --random-state 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features import build_features

RANDOM_STATE = 42


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"RMSE": round(self.rmse, 4), "MAE": round(self.mae, 4), "R2": round(self.r2, 4)}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def make_models(random_state: int) -> dict[str, object]:
    """Candidate models. Ridge runs on standardised descriptors."""
    return {
        "Ridge": ("descriptors", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
        "RandomForest": ("all", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=random_state)),
        "XGBoost": ("all", XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=random_state)),
    }


def cross_validate(model, X, y, n_splits: int, random_state: int) -> Metrics:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds[val_idx] = model.predict(X[val_idx])
    return regression_metrics(y, preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solubility model training")
    parser.add_argument("--processed", default="data/processed/solubility_logS.csv")
    parser.add_argument("--outdir", default="reports")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--temporal-year", type=int, default=2015,
                        help="Cut-off year for the temporal hold-out")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--holdout-file", default=None,
                        help="CSV with molecules to EXCLUDE from training (external validation)")
    args = parser.parse_args()

    df = pd.read_csv(args.processed)

    holdout = None
    if args.holdout_file and Path(args.holdout_file).exists():
        holdout = pd.read_csv(args.holdout_file)
        held = set(holdout["smiles"].dropna())
        n_before = len(df)
        df = df[~df["smiles"].isin(held)].reset_index(drop=True)
        print(f"Excluded {n_before - len(df)} external-validation molecules "
              f"({args.holdout_file})")

    print(f"Dataset: {len(df)} compounds")

    smiles = df["smiles"].tolist()
    y = df["logS"].to_numpy(dtype=float)

    X_all, names = build_features(smiles, use_descriptors=True, use_fingerprints=True,
                                  n_bits=args.n_bits)
    X_desc = X_all[:, :len(names) - args.n_bits]
    n_desc = X_desc.shape[1]
    # Indices of compounds with finite descriptors (BalabanJ can return inf)
    valid = np.isfinite(X_desc).all(axis=1)
    if not valid.all():
        print(f"Dropped {int((~valid).sum())} compounds with non-finite descriptors")
    X_all, X_desc, y = X_all[valid], X_desc[valid], y[valid]
    smiles = [s for s, keep in zip(smiles, valid) if keep]
    years = df["first_year"].to_numpy(dtype=float)[valid]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    # ---------- Random split ----------
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_size, random_state=args.random_state
    )
    print(f"\nRandom split: {len(idx_train)} train / {len(idx_test)} test")

    for name, (scope, model) in make_models(args.random_state).items():
        Xtr = X_desc if scope == "descriptors" else X_all
        Xtrain, Xtest = Xtr[idx_train], Xtr[idx_test]

        cv = cross_validate(model, Xtrain, y[idx_train], args.folds, args.random_state)
        model.fit(Xtrain, y[idx_train])
        test = regression_metrics(y[idx_test], model.predict(Xtest))

        results[f"{name}_random"] = {"cv": cv.as_dict(), "test": test.as_dict()}
        print(f"  {name:<13} CV RMSE={cv.rmse:.3f}  R2={cv.r2:.3f} | "
              f"test RMSE={test.rmse:.3f}  MAE={test.mae:.3f}  R2={test.r2:.3f}")

    # ---------- Temporal hold-out ----------
    train_mask = years < args.temporal_year
    test_mask = years >= args.temporal_year
    print(f"\nTemporal split (<{args.temporal_year} vs >={args.temporal_year}): "
          f"{train_mask.sum()} train / {test_mask.sum()} test")

    for name, (scope, model) in make_models(args.random_state).items():
        Xtr = X_desc if scope == "descriptors" else X_all
        model.fit(Xtr[train_mask], y[train_mask])
        test = regression_metrics(y[test_mask], model.predict(Xtr[test_mask]))
        results[f"{name}_temporal"] = {"test": test.as_dict()}
        print(f"  {name:<13} test RMSE={test.rmse:.3f}  MAE={test.mae:.3f}  R2={test.r2:.3f}")

    # ---------- Final model, bias calibration and conformal set ----------
    # The train set is split into three disjoint parts: model fitting, bias
    # calibration fitting and conformal-interval calibration. Keeping them
    # separate is required for the interval to retain its coverage.
    idx_fit, idx_rest = train_test_split(
        idx_train, test_size=0.2, random_state=args.random_state
    )
    idx_map, idx_conv = train_test_split(
        idx_rest, test_size=0.5, random_state=args.random_state
    )

    xgb = make_models(args.random_state)["XGBoost"][1]
    xgb.fit(X_all[idx_fit], y[idx_fit])

    # The model compresses its predictions towards the mean (it overestimates
    # poorly soluble compounds and underestimates highly soluble ones). This is
    # corrected with an isotonic regression fitted on out-of-sample predictions
    # from idx_map.
    pred_map = xgb.predict(X_all[idx_map])
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(pred_map, y[idx_map])
    calibrator_linear = LinearRegression().fit(pred_map.reshape(-1, 1), y[idx_map])

    calib_abs_residuals = np.abs(
        y[idx_conv] - calibrator.predict(xgb.predict(X_all[idx_conv]))
    )

    raw_test = xgb.predict(X_all[idx_test])
    predictions = {
        "uncalibrated": raw_test,
        "isotonic": calibrator.predict(raw_test),
        "linear": calibrator_linear.predict(raw_test.reshape(-1, 1)),
    }
    print(f"\n=== Effect of bias calibration ({len(idx_map)} fitting molecules) ===")
    print(f"  {'variant':<14}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'std(pred)':>11}")
    print(f"  {'label':<14}{'':>8}{'':>8}{'':>8}{y[idx_test].std():>11.3f}")
    for label, preds in predictions.items():
        m = regression_metrics(y[idx_test], preds)
        results.setdefault("calibration", {})[label] = m.as_dict()
        print(f"  {label:<14}{m.rmse:>8.3f}{m.mae:>8.3f}{m.r2:>8.3f}{preds.std():>11.3f}")

    print("\n  Bias (pred - true) by solubility range:")
    for label, preds in predictions.items():
        print(f"    {label:<14} " + "  ".join(
            f"{lo:>4}{hi:>4}: {(preds[(y[idx_test] > lo) & (y[idx_test] <= hi)]
                               - y[idx_test][(y[idx_test] > lo) & (y[idx_test] <= hi)]).mean():+.2f}"
            for lo, hi in [(-12, -5), (-5, -3), (-3, 2)]
        ))

    final_test = regression_metrics(y[idx_test], predictions["isotonic"])
    results["XGBoost_final"] = {
        "n_fit": int(len(idx_fit)), "n_map": int(len(idx_map)),
        "n_conformal": int(len(idx_conv)), "test_calibrated": final_test.as_dict(),
    }
    print(f"\nFinal model: {len(idx_fit)} fit / {len(idx_map)} bias calib. / "
          f"{len(idx_conv)} conformal -> calibrated test RMSE={final_test.rmse:.3f}  "
          f"MAE={final_test.mae:.3f}  R2={final_test.r2:.3f}")

    # The AD reference set is the model's training data; the conformal set is
    # stored separately for the prediction interval.
    artifact = {
        "model": xgb,
        "model_name": "XGBoost",
        "calibrator": calibrator,
        "feature_names": names,
        "n_bits": args.n_bits,
        "n_descriptors": n_desc,
        "ref_smiles": [smiles[i] for i in idx_fit],
        "ref_fingerprints": np.packbits(X_all[idx_fit, n_desc:].astype(np.uint8), axis=1),
        "ref_descriptors": X_all[idx_fit, :n_desc],
        "ref_y": y[idx_fit],
        "calib_fingerprints": np.packbits(X_all[idx_conv, n_desc:].astype(np.uint8), axis=1),
        "calib_descriptors": X_all[idx_conv, :n_desc],
        "calib_abs_residuals": calib_abs_residuals,
        "random_state": args.random_state,
    }
    joblib.dump(artifact, outdir / "model.joblib")

    pd.DataFrame({
        "smiles": [smiles[i] for i in idx_test],
        "logS": y[idx_test],
        "logS_pred": predictions["isotonic"],
        "logS_pred_raw": predictions["uncalibrated"],
        "set": "test",
    }).to_csv(outdir / "test_set.csv", index=False)

    with open(outdir / "metrics.json", "w") as fh:
        json.dump({"n_compounds": len(y), "results": results}, fh, indent=2)

    print(f"\nFinal model and metrics saved to {outdir}/")


if __name__ == "__main__":
    main()
