"""Tests for the solubility pipeline (runnable with python -m unittest)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from applicability import (  # noqa: E402
    ADConfig, ApplicabilityDomain, _tanimoto_to_reference, _unpack_bits,
)
from curate import curate, parent_smiles, to_logs  # noqa: E402
from features import DESCRIPTOR_NAMES, build_features, morgan_fingerprints  # noqa: E402


class TestUnitConversion(unittest.TestCase):
    def test_nanomolar_to_logs(self):
        # 1e6 nM = 1e-3 mol/L  ->  logS = -3
        self.assertAlmostEqual(to_logs(1e6, "nM", 200.0), -3.0, places=6)

    def test_microgram_per_ml_to_logs(self):
        # 1 ug/mL = 1e-3 g/L; with MW = 100 g/mol -> 1e-5 mol/L -> logS = -5
        self.assertAlmostEqual(to_logs(1.0, "ug.mL-1", 100.0), -5.0, places=6)

    def test_mass_conversion_is_mw_independent_in_molar_units(self):
        self.assertAlmostEqual(to_logs(1e6, "nM", 50.0), to_logs(1e6, "nM", 500.0), places=9)

    def test_unsupported_unit_raises(self):
        with self.assertRaises(ValueError):
            to_logs(1.0, "mol/L", 100.0)


class TestFeatures(unittest.TestCase):
    def test_morgan_is_binary_and_72_dim(self):
        fps = morgan_fingerprints(["CCO", "c1ccccc1"], n_bits=128)
        self.assertEqual(fps.shape, (2, 128))
        self.assertTrue(set(np.unique(fps)).issubset({0, 1}))

    def test_identical_molecules_share_fingerprint(self):
        fps = morgan_fingerprints(["CCO", "OCC"], n_bits=256)
        np.testing.assert_array_equal(fps[0], fps[1])

    def test_feature_matrix_shape(self):
        X, names = build_features(["CCO", "c1ccccc1"], n_bits=64)
        self.assertEqual(X.shape, (2, len(DESCRIPTOR_NAMES) + 64))
        self.assertEqual(len(names), X.shape[1])


class TestCuration(unittest.TestCase):
    def _raw_row(self, value, units, smiles="CCO", cid="CHEMBL1", year=2000):
        return {
            "molecule_chembl_id": cid, "canonical_smiles": smiles,
            "standard_value": value, "standard_units": units, "standard_relation": "=",
            "potential_duplicate": 0, "assay_description": "Aqueous solubility of the compound",
            "document_year": year,
        }

    def test_duplicates_are_aggregated_by_median(self):
        # MW(CCO) = 46.07; two different measurements of the same compound
        raw = pd.DataFrame([
            self._raw_row(1e6, "nM"), self._raw_row(1e7, "nM"),
        ])
        final, _ = curate(raw, max_std=5.0, min_measurements=1)
        self.assertEqual(len(final), 1)
        self.assertEqual(final.loc[0, "n_measurements"], 2)

    def test_implausible_values_are_discarded(self):
        raw = pd.DataFrame([self._raw_row(1e-6, "nM")])  # logS = -15 -> out of range
        final, _ = curate(raw, max_std=5.0, min_measurements=1)
        self.assertEqual(len(final), 0)


class TestParentStructures(unittest.TestCase):
    """Desalting: the salt and the neutral form must collapse to the same structure."""

    def test_sodium_salt_is_desalted_and_neutralised(self):
        self.assertEqual(parent_smiles("CC(=O)[O-].[Na+]"), "CC(=O)O")

    def test_largest_fragment_wins(self):
        self.assertEqual(parent_smiles("[Na+].[Cl-].c1ccccc1"), "c1ccccc1")

    def test_permanent_charge_is_preserved(self):
        # A quaternary cation cannot be neutralised: it must keep its charge
        self.assertIn("+", parent_smiles("CCCC[N+](C)(C)C.[Br-]"))

    def test_salt_and_free_form_converge(self):
        self.assertEqual(parent_smiles("CC(=O)[O-].[Na+]"), parent_smiles("CC(=O)O"))

    def test_invalid_smiles_returns_none(self):
        self.assertIsNone(parent_smiles("not-a-smiles"))


class TestApplicabilityDomainHelpers(unittest.TestCase):
    def test_packbits_roundtrip(self):
        rng = np.random.default_rng(0)
        bits = (rng.random((7, 24)) < 0.5).astype(np.uint8)
        packed = np.packbits(bits, axis=1)
        np.testing.assert_array_equal(_unpack_bits(packed, 24), bits)

    def test_tanimoto_identity_is_one(self):
        rng = np.random.default_rng(1)
        bits = (rng.random((4, 64)) < 0.3).astype(np.uint8)
        sims = _tanimoto_to_reference(bits, bits)
        np.testing.assert_allclose(np.diag(sims), 1.0, atol=1e-5)

    def test_disjoint_sets_have_zero_similarity(self):
        a = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        b = np.array([[0, 0, 1, 1]], dtype=np.uint8)
        self.assertAlmostEqual(float(_tanimoto_to_reference(a, b)[0, 0]), 0.0)


class TestApplicabilityDomainEndToEnd(unittest.TestCase):
    """Wires a minimal model into the AD to check the complete flow."""

    @classmethod
    def setUpClass(cls):
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(42)
        n_bits, n = 64, 60
        desc = rng.normal(size=(n, len(DESCRIPTOR_NAMES)))
        fps = (rng.random((n, n_bits)) < 0.3).astype(np.uint8)
        X = np.hstack([desc, fps])
        y = X[:, 0] * 0.4 + rng.normal(scale=0.1, size=n)
        model = RandomForestRegressor(n_estimators=15, random_state=0).fit(X, y)

        cls.artifact = {
            "model": model, "model_name": "RF-test", "n_bits": n_bits,
            "n_descriptors": len(DESCRIPTOR_NAMES),
            "feature_names": DESCRIPTOR_NAMES + [f"morgan_{i}" for i in range(n_bits)],
            "ref_smiles": ["C"] * 40, "ref_descriptors": desc[:40],
            "ref_fingerprints": np.packbits(fps[:40], axis=1), "ref_y": y[:40],
            "calib_fingerprints": np.packbits(fps[40:], axis=1),
            "calib_descriptors": desc[40:],
            "calib_abs_residuals": np.abs(y[40:] - model.predict(X[40:])),
        }

    def test_ad_returns_one_verdict_per_molecule_with_valid_status(self):
        ad = ApplicabilityDomain(self.artifact, ADConfig(similarity_sample=20))
        # One molecule identical to the reference set and another completely different
        desc = np.vstack([self.artifact["ref_descriptors"][0], np.full(27, 50.0)])
        fps = np.vstack([_unpack_bits(self.artifact["ref_fingerprints"], 64)[0],
                         np.zeros(64, dtype=np.uint8)])
        verdicts = ad.evaluate(["similar", "extrana"], desc, fps)
        self.assertEqual(len(verdicts), 2)
        for v in verdicts:
            self.assertIn(v.status, {"inside", "borderline", "outside"})
        similar, weird = verdicts
        self.assertGreater(similar.max_similarity, weird.max_similarity)
        self.assertLessEqual(similar.halfwidth, weird.halfwidth)
        self.assertEqual(weird.status, "outside")

    def test_calibrator_is_applied_to_predictions(self):
        """The artifact calibrator must transform the model prediction."""
        from sklearn.isotonic import IsotonicRegression

        ad_plain = ApplicabilityDomain(self.artifact, ADConfig(similarity_sample=20))
        desc = self.artifact["ref_descriptors"][:5]
        fps = _unpack_bits(self.artifact["ref_fingerprints"], 64)[:5]
        raw = np.array([v.prediction for v in ad_plain.evaluate(["a"] * 5, desc, fps)])

        shifted = dict(self.artifact)
        shifted["calibrator"] = IsotonicRegression(out_of_bounds="clip").fit(raw, raw + 2.0)
        ad_cal = ApplicabilityDomain(shifted, ADConfig(similarity_sample=20))
        calibrated = np.array([v.prediction for v in ad_cal.evaluate(["a"] * 5, desc, fps)])

        np.testing.assert_allclose(calibrated, raw + 2.0, atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
