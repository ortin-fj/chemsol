"""M4 - Applicability domain (AD) for the logS predictions.

Three complementary criteria are combined, following the usual QSAR
recommendations (OECD / Sahigara et al. 2012):

1. Leverage (model-based AD)
   h_i = z_i (Z'Z)^-1 z_i' over the standardised descriptors of the reference
   set (the model's training data). Warning threshold
   h* = 3 (p + 1) / n  (hat-matrix rule / Williams plot).

2. Nearest-neighbour similarity (data-based AD)
   Maximum Tanimoto similarity against the reference set in Morgan fingerprint
   space. Threshold = 5th percentile of the nearest-neighbour similarity
   distribution computed within the reference set itself.

3. Uncertainty (locally adaptive conformal prediction)
   Split-conformal normalised by the local difficulty sigma(x) = 1 - sim_max(x),
   so the interval width depends on each molecule. The threshold that defines
   "high uncertainty" is RELATIVE to the calibration itself (factor * median of
   the calibration widths), because a fixed absolute tolerance would be
   inconsistent with the model's real precision.

Calibration points nearly identical to the reference set are excluded
(similarity > calib_dedup_similarity) so they do not inflate the conformal
quantile.

The final verdict combines the three warnings: 0 -> inside the AD,
1 -> borderline, >=2 -> outside the domain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from features import compute_descriptors, morgan_fingerprints

STATUS_IN = "inside"
STATUS_BORDER = "borderline"
STATUS_OUT = "outside"


@dataclass
class ADConfig:
    alpha: float = 0.10                    # conformal interval level (90%)
    leverage_factor: float = 3.0           # h* = factor * (p + 1) / n
    similarity_percentile: float = 5.0     # percentile for the similarity threshold
    uncertainty_factor: float = 1.5        # warn if width > factor * calibration median
    halfwidth_tolerance: float | None = None  # if set, replaces the relative criterion
    sigma_floor: float = 0.25              # lower bound of the local difficulty
    calib_dedup_similarity: float = 0.95   # drops calibration points near-duplicated in reference
    similarity_sample: int = 3000          # sample size to estimate the similarity threshold


@dataclass
class ADVerdict:
    smiles: str
    prediction: float
    interval_low: float
    interval_high: float
    halfwidth: float
    difficulty: float
    leverage: float
    leverage_threshold: float
    max_similarity: float
    similarity_threshold: float
    outlier_leverage: bool
    outlier_similarity: bool
    high_uncertainty: bool
    status: str
    n_warnings: int
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _tanimoto_to_reference(query_bits: np.ndarray, ref_bits: np.ndarray,
                           chunk: int = 4096) -> np.ndarray:
    """Tanimoto similarity of every row of query_bits against every row of ref_bits."""
    query = query_bits.astype(np.float32)
    query_pop = query.sum(axis=1)[:, None]
    out = np.empty((query.shape[0], ref_bits.shape[0]), dtype=np.float32)
    for start in range(0, ref_bits.shape[0], chunk):
        block = ref_bits[start:start + chunk].astype(np.float32)
        inter = query @ block.T
        union = query_pop + block.sum(axis=1)[None, :] - inter
        out[:, start:start + chunk] = inter / np.maximum(union, 1e-9)
    return out


def _unpack_bits(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """Reverses np.packbits: from packed bytes back to a binary matrix (n, n_bits)."""
    unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1)
    return np.ascontiguousarray(unpacked[:, :n_bits])


class ApplicabilityDomain:
    """Assesses whether a molecule falls inside the model's applicability domain."""

    def __init__(self, artifact: dict, config: ADConfig | None = None):
        self.config = config or ADConfig()
        self.model = artifact["model"]
        self.calibrator = artifact.get("calibrator")
        self.n_bits = artifact["n_bits"]
        self.n_descriptors = artifact["n_descriptors"]
        self.feature_names = artifact["feature_names"]

        # Reference set = the model's training data
        self.ref_descriptors = np.asarray(artifact["ref_descriptors"], dtype=float)
        self.ref_fingerprints = _unpack_bits(artifact["ref_fingerprints"], self.n_bits)

        # Calibration set for the conformal interval
        self.calib_fingerprints = _unpack_bits(artifact["calib_fingerprints"], self.n_bits)
        self.calib_abs_residuals = np.abs(
            np.asarray(artifact["calib_abs_residuals"], dtype=float)
        )

        # --- Criterion 1: leverage ---
        self._ref_mean = self.ref_descriptors.mean(axis=0)
        self._ref_std = self.ref_descriptors.std(axis=0)
        self._ref_std[self._ref_std == 0] = 1.0
        z_ref = (self.ref_descriptors - self._ref_mean) / self._ref_std
        self._xtx_inv = np.linalg.pinv(z_ref.T @ z_ref)
        n, p = z_ref.shape
        self.leverage_threshold = self.config.leverage_factor * (p + 1) / n

        # --- Criterion 2: Tanimoto similarity ---
        self.similarity_threshold = self._reference_similarity_threshold()

        # --- Criterion 3: locally adaptive conformal ---
        calib_sim = self._max_similarity(self.calib_fingerprints)
        keep = calib_sim <= self.config.calib_dedup_similarity
        sigma_cal = np.maximum(1.0 - calib_sim[keep], self.config.sigma_floor)
        scores = self.calib_abs_residuals[keep] / sigma_cal
        self.conformal_q = float(np.quantile(scores, 1.0 - self.config.alpha))
        self.n_calibration = int(keep.sum())

        calib_halfwidths = self.conformal_q * sigma_cal
        if self.config.halfwidth_tolerance is not None:
            self.halfwidth_threshold = float(self.config.halfwidth_tolerance)
        else:
            self.halfwidth_threshold = float(
                self.config.uncertainty_factor * np.median(calib_halfwidths)
            )

    # ------------------------------------------------------------------ helpers
    def _reference_similarity_threshold(self) -> float:
        """Percentile of the nearest-neighbour similarity within the reference set."""
        rng = np.random.default_rng(0)
        n = self.ref_fingerprints.shape[0]
        idx = rng.choice(n, size=min(self.config.similarity_sample, n), replace=False)
        sims = _tanimoto_to_reference(self.ref_fingerprints[idx], self.ref_fingerprints)
        for row, i in enumerate(idx):      # exclude the molecule itself and its duplicates
            sims[row, i] = -1.0
        nn = sims.max(axis=1)
        return float(np.percentile(nn, self.config.similarity_percentile))

    def _leverage(self, descriptors: np.ndarray) -> np.ndarray:
        z = (descriptors - self._ref_mean) / self._ref_std
        return np.einsum("ij,jk,ik->i", z, self._xtx_inv, z)

    def _max_similarity(self, fingerprints: np.ndarray) -> np.ndarray:
        return _tanimoto_to_reference(fingerprints, self.ref_fingerprints).max(axis=1)

    def _difficulty(self, fingerprints: np.ndarray) -> np.ndarray:
        """Local difficulty: 0 if the molecule is identical to something in the reference set."""
        return np.maximum(1.0 - self._max_similarity(fingerprints), self.config.sigma_floor)

    def _features(self, descriptors: np.ndarray, fingerprints: np.ndarray) -> np.ndarray:
        return np.hstack([descriptors, fingerprints])

    # ------------------------------------------------------------------ public API
    def evaluate(self, smiles_list: list[str],
                 descriptors: np.ndarray, fingerprints: np.ndarray) -> list[ADVerdict]:
        descriptors = np.asarray(descriptors, dtype=float)
        fingerprints = np.asarray(fingerprints, dtype=np.uint8)

        # Non-finite descriptors (e.g. BalabanJ on pathological structures) are
        # imputed with the reference mean so a prediction is still produced, but
        # the molecule is flagged as out of domain because the model has never
        # seen it.
        finite = np.isfinite(descriptors).all(axis=1)
        if not finite.all():
            descriptors = descriptors.copy()
            descriptors[~finite] = self._ref_mean

        predictions = self.model.predict(self._features(descriptors, fingerprints))
        if self.calibrator is not None:
            predictions = self.calibrator.predict(predictions)
        leverages = self._leverage(descriptors)
        leverages[~finite] = np.inf
        similarities = self._max_similarity(fingerprints)
        difficulties = self._difficulty(fingerprints)
        halfwidths = self.conformal_q * difficulties

        verdicts = []
        for i, smi in enumerate(smiles_list):
            outlier_leverage = bool(leverages[i] > self.leverage_threshold)
            outlier_similarity = bool(similarities[i] < self.similarity_threshold)
            high_uncertainty = bool(halfwidths[i] > self.halfwidth_threshold)
            warnings = int(sum([outlier_leverage, outlier_similarity, high_uncertainty]))
            status = (STATUS_IN if warnings == 0
                      else STATUS_BORDER if warnings == 1
                      else STATUS_OUT)
            pred = float(predictions[i])
            hw = float(halfwidths[i])
            verdicts.append(ADVerdict(
                smiles=smi,
                prediction=pred,
                interval_low=pred - hw,
                interval_high=pred + hw,
                halfwidth=hw,
                difficulty=float(difficulties[i]),
                leverage=float(leverages[i]),
                leverage_threshold=float(self.leverage_threshold),
                max_similarity=float(similarities[i]),
                similarity_threshold=float(self.similarity_threshold),
                outlier_leverage=outlier_leverage,
                outlier_similarity=outlier_similarity,
                high_uncertainty=high_uncertainty,
                status=status,
                n_warnings=warnings,
                details={"alpha": self.config.alpha, "conformal_q": self.conformal_q},
            ))
        return verdicts

    def evaluate_smiles(self, smiles_list: list[str]) -> list[ADVerdict]:
        """Convenience wrapper: computes the features from the SMILES strings."""
        descriptors = compute_descriptors(smiles_list)
        fingerprints = morgan_fingerprints(smiles_list, n_bits=self.n_bits)
        return self.evaluate(smiles_list, descriptors, fingerprints)
