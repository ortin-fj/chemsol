# Aqueous solubility prediction (logS) from ChEMBL [WORK IN PROGRESS]

A small-scope cheminformatics project: a physicochemical property is downloaded
from an open chemistry database (**ChEMBL**), curated, used to train a QSAR
model, and the property is then predicted for molecules **not included in
training**, together with an estimate of the **applicability domain (AD)** of
each prediction.

- **Target property:** aqueous solubility, expressed as `logS` (log₁₀ mol/L).
- **Source:** ChEMBL (public API, `standard_type = 'Solubility'`).
- **Hardware:** CPU only. The whole pipeline runs in minutes.

---

## 1. Results summary

Using the same implementation, four evaluation protocols of increasing
difficulty (details and reference figures in §7):

| Split | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| Random test (20%) | Ridge | 1.074 | 0.799 | 0.123 |
| Random test (20%) | RandomForest | 0.793 | 0.579 | 0.522 |
| Random test (20%) | **XGBoost** | **0.782** | **0.584** | **0.535** |
| Test without analogues (sim < 0.70) | XGBoost | 0.965 | 0.707 | 0.424 |
| Test on disjoint **scaffolds** | XGBoost | 0.933 | 0.680 | 0.466 |
| **Temporal** test (≥2015) | XGBoost | 0.986 | 0.736 | 0.105 |

The random split is optimistic: 70% of its molecules have a close analogue in
the training set. Honest generalisation is R² ≈ 0.42-0.47.

**Applicability domain** (test set, 7,118 molecules):

| Verdict | n | MAE |
|---|---|---|
| inside | 5,626 | 0.544 |
| borderline | 1,105 | 0.705 |
| outside | 387 | **0.906** |

The AD **correctly orders the error**: predictions outside the domain are ~67%
worse than those inside. The 90% conformal interval has an **empirical coverage
of 88.9%** and widens as molecules move away from the domain
(1.06 log inside → 2.63 log outside).

**External validation** (13 drugs excluded from training, experimental truth
from ChEMBL): MAE = **0.79 log**, mean bias **+0.03** (removed by calibration;
uncalibrated it was −0.25).

## 2. Data and curation

Download: 61,866 solubility records in the two units that hold the bulk of the
data (`ug.mL-1`: 28,114 and `nM`: 33,752).

Curation funnel:

| Step | Records |
|---|---|
| raw | 61,866 |
| positive value and `=` relation | 45,079 |
| aqueous context (excludes BSA, culture media, formulations) | 44,725 |
| without duplicates flagged by ChEMBL | 44,725 |
| valid SMILES | 44,725 |
| **desalting** (largest organic fragment, neutralised) | 44,725 (2,338 entries were salts) |
| plausible logS range (−12 … 2) | 44,716 |
| **unique compounds** (median of measurements) | **35,613** |
| with agreement between measurements (σ ≤ 1.5) | 35,613 |

Unit conversion (`src/curate.py::to_logs`):

```
ug.mL-1 :  logS = log10(value) − 3 − log10(MW)
nM      :  logS = log10(value) − 9
```

**Desalting.** ChEMBL registers the same substance sometimes as a salt and
sometimes as the neutral form. `parent_smiles` keeps the largest organic
fragment and neutralises the charges, so both forms collapse into a single
compound. Unit conversion still uses the mass of the **registered form**, which
is the one weighed in the laboratory. Effect: 36,006 → 35,613 compounds.

## 3. Applicability domain

Three criteria are combined (0 warnings → *inside*; 1 → *borderline*;
≥2 → *outside*):

1. **Leverage / Williams plot** — `h = z (ZᵀZ)⁻¹ zᵀ` over standardised
   descriptors; threshold `h* = 3(p+1)/n`.
2. **Tanimoto similarity** to the nearest neighbour in Morgan fingerprint space;
   threshold = 5th percentile of the nearest-neighbour similarity within the
   reference set itself.
3. **Locally adaptive conformal prediction** — split-conformal normalised by the
   local difficulty `σ(x) = 1 − sim_max(x)`, so the interval width depends on
   each molecule. The threshold is **relative** to the calibration
   (factor × median), because a fixed absolute tolerance would be inconsistent
   with the model's real precision.

Figure: [reports/figures/williams_plot.png](reports/figures/williams_plot.png)

---

## 4. Usage

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Download from ChEMBL (~5 min)
.venv/bin/python src/fetch_chembl.py --out data/raw/solubility_chembl.csv

# 2. Curate
.venv/bin/python src/curate.py --raw data/raw/solubility_chembl.csv \
                               --out data/processed/solubility_logS.csv

# 3. External validation set (drugs to be excluded from training)
.venv/bin/python src/make_external_set.py

# 4. Train (~5 min)
.venv/bin/python src/train.py --processed data/processed/solubility_logS.csv \
                              --outdir reports \
                              --holdout-file data/external/validation_set.csv

# 5. Applicability domain and external validation
.venv/bin/python src/domain_analysis.py --outdir reports
.venv/bin/python src/validate_external.py --outdir reports

# 6. Predict a new molecule
.venv/bin/python src/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

`predict.py` output:

```
Model: XGBoost  |  AD thresholds: leverage<0.0037  similarity>0.393  half-width<1.49

SMILES                                          logS           90% CI    sim       h  status
--------------------------------------------------------------------------------------------
CC(=O)Oc1ccccc1C(=O)O                          -1.45   [-2.73, -0.17]   0.68  0.0011  inside

Legend: sim = maximum Tanimoto similarity to the training set; h = leverage;
status = inside/borderline/outside the applicability domain.
```

Tests: `.venv/bin/python -m unittest discover -s tests`

## 5. Layout

```
src/fetch_chembl.py      paged download from the ChEMBL API
src/curate.py            unit conversion, filters and aggregation
src/features.py          RDKit 2D descriptors + Morgan fingerprints
src/train.py             5-fold CV, temporal split, final model and calibration
src/applicability.py     the three AD criteria
src/domain_analysis.py   AD on test + OOD probes + Williams plot
src/make_external_set.py external drug set with ChEMBL ground truth
src/validate_external.py external validation
src/predict.py           prediction CLI with interval and AD verdict
tests/test_pipeline.py   19 tests (conversions, features, AD)
```

## 6. Limitations

- **Shrinkage towards the mean.** The model underestimates the solubility of
  highly soluble compounds and overestimates it for poorly soluble ones. After
  calibration (§8.2) the residual bias spans **−1.11 log** (`logS > −2`) to
  **+0.76 log** (`logS < −5`); calibration only corrects 15-20% of the original
  bias. The cause is that the model **cannot rank** those molecules (prediction–
  truth correlation of 0.30 in the soluble extreme), and no recalibration can
  invent that information.
- **Compounds measured after 2015.** With a temporal split R² drops to 0.11.
  The diagnosis (§8.3) shows this is not training-set drift: that subset is
  intrinsically harder and narrower (sd(logS) 1.04 versus 1.29 for the train
  set), which mechanically depresses R².
- **Noise in the external ground truth.** ChEMBL records mix salt forms with the
  neutral form (e.g. ibuprofen with σ = 1.27 log between measurements).
  Desalting removes the identity ambiguity, but the value discrepancy persists.
  The dispersion is reported in `data/external/validation_set.csv`.
- **Tautomers.** A molecule may appear in the dataset under a different
  tautomeric form (warfarin: ChEMBL registers the open keto acid). Comparing
  SMILES *strings* is not enough; canonicalisation via InChIKey is required.
- Descriptors are 2D: neither polymorphism nor crystal packing is modelled.
- **1.8% of compounds have no year.** The temporal split drops them from both
  train and test.

---

## 7. Critical evaluation (independent validation)

The results above were checked with five additional analyses.

### 7.1 Noise floor: how much room is there for improvement?

From the compounds with ≥2 measurements in ChEMBL (5,481), the intra-compound
variability is:

| Statistic | Value |
|---|---|
| Intra-compound SD (median) | 0.286 |
| Intra-compound SD (mean) | 0.486 |
| **Intra-compound RMS (pooled)** | **0.742** |
| Compounds with SD > 0.5 log | 37% |
| Compounds with SD > 1.0 log | 16% |

With `var(logS) = 1.311`, label noise explains **42%** of the total variance,
setting a ceiling of **R² ≈ 0.58**. The model reaches 0.535 on the random split,
i.e. **92% of the attainable ceiling**. Its MAE (0.584) is of the order of the
discrepancy between two laboratories measuring the same compound (0.486).

### 7.2 Analogue leakage

| Similarity to train | n | RMSE |
|---|---|---|
| ≤ 0.4 | 341 | 1.251 |
| 0.6-0.7 | 1,071 | 0.850 |
| 0.8-0.9 | 2,071 | 0.662 |
| ≥ 0.9 | 745 | 0.623 |

The error grows monotonically with distance from the training data: the RMSE of
0.79 is held up by molecules nearly identical to training ones.

### 7.3 Comparison with the ESOL reference model

ESOL (Delaney, 2004) on the same test set:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| ESOL | 1.769 | 1.280 | −1.373 |
| **This project** | **0.787** | **0.580** | **0.531** |

The model gains 0.98 log of RMSE. Its correlation with the label is 0.74 versus
0.26 for ESOL. ESOL is biased by −0.75 log on this set: its dispersion (1.45)
exceeds that of the labels (1.15), a sign that it measures a different quantity
(intrinsic solubility of the neutral form) from what ChEMBL holds.

### 7.4 Contrast with a curated reference dataset

Same pipeline, same hyperparameters, applied to **AqSolDB** (9,878 compounds
curated from 9 sources):

| Dataset | n | SD(logS) | Random MAE | Random R² | Scaffold MAE | Scaffold R² |
|---|---|---|---|---|---|---|
| ChEMBL (this project) | 35,613 | 1.15 | **0.584** | 0.535 | **0.680** | 0.466 |
| AqSolDB (reference) | 9,878 | 2.36 | 0.639 | **0.850** | 0.954 | **0.567** |

**The absolute MAE is better on ChEMBL than on the curated dataset**, yet the R²
is far worse. The reason is that R² depends on the label variance: AqSolDB covers
a chemical space four times wider (it spans gases through pesticides), while
ChEMBL concentrates pharmaceutical compounds in the logS ≈ −5 … −2 band. With
AqSolDB's noise (RMS 0.516) its R² ceiling is 0.95 and it reaches 89% of it —
exactly the same percentage as this project.

**Conclusion: the low R² is not a model defect, it is a property of the label
set.** R² is not comparable across datasets with different variance; MAE is.

### 7.5 Re-audit of the external validation

| Drug | ChEMBL | Says ESOL | Model | Reliable reference? |
|---|---|---|---|---|
| Caffeine | −0.97 | −0.87 | −2.34 | **yes** (σ = 0.00) → genuine 1.4-log failure |
| Aspirin | −0.18 | −1.99 | −1.45 | **no** (literature ≈ −1.7) |
| Ibuprofen | −0.59 | −3.09 | −1.29 | **no** (σ = 1.27; sodium salt) |
| Warfarin | −4.52 | −4.28 | −4.22 | yes |

Excluding the two salt-contaminated references, the external MAE becomes
**0.755 log** (calibrated; 0.790 over the full set). The caffeine case is a
**genuine model failure** (shrinkage towards the mean in the soluble extreme),
confirmed by three independent sources that agree with one another.

### 7.6 What is not validated

- **The AD is not validated by the external set**: there the MAE of the
  "inside" molecules is comparable to that of the "outside" ones. With n = 13
  that is noise, but it means the usefulness of the AD is demonstrated on the
  test set (7,118 molecules) and not on the external validation.
- The external set is **13 pharmaceutical molecules**, not an independent
  blind test: they share their origin (ChEMBL) with the training data.
- **Sizes and noise**: the external noise (intra-compound σ ≈ 0.5-1.3 log) is of
  the same order as the effect being measured, so the external set only detects
  large differences.

---

## 8. Corrective actions applied

### 8.1 Desalting to the parent structure

`curate.py::parent_smiles` keeps the largest organic fragment and neutralises
the charges (`LargestFragmentChooser` + `Uncharger`).

| | Before | After |
|---|---|---|
| Multicomponent entries | 2,338 (5.2%) untreated | normalised |
| Unique compounds | 36,006 | 35,613 |
| Random test (XGBoost) | RMSE 0.779 · MAE 0.575 · R² 0.540 | RMSE 0.782 · MAE 0.584 · R² 0.535 |
| Disjoint scaffolds | R² 0.471 | R² 0.466 |

**Honest conclusion: the curation was defective but it was NOT the
bottleneck.** Desalting removes spurious duplicates and counter-ions from the
fingerprint, and makes the identity consistent with `make_external_set.py`, but
it does not improve prediction. The reason: it only affected 5% of the rows.

### 8.2 Calibration of the shrinkage bias

The train set is split into three disjoint parts (model fitting / calibration
fitting / conformal-interval calibration) and an `IsotonicRegression` is fitted
on out-of-sample predictions.

| Variant | RMSE | MAE | R² | std(pred) |
|---|---|---|---|---|
| label | — | — | — | 1.147 |
| uncalibrated | 0.798 | 0.595 | 0.516 | 0.703 |
| **isotonic** | **0.791** | **0.589** | **0.524** | 0.846 |
| linear | 0.787 | 0.587 | 0.529 | 0.837 |

The overall effect is small, but **it is not evenly distributed**: calibration
improves the extremes and slightly worsens the dense middle.

| logS range | n | MAE uncal. | MAE cal. | Bias uncal. | Bias cal. |
|---|---|---|---|---|---|
| < −5 | 1,591 | 0.910 | **0.819** | +0.89 | +0.76 |
| −5 … −4 | 2,570 | **0.343** | 0.391 | +0.11 | +0.06 |
| −4 … −3 | 2,180 | **0.489** | 0.518 | −0.39 | −0.36 |
| −3 … −2 | 465 | 0.868 | 0.845 | −0.80 | −0.64 |
| > −2 | 312 | 1.395 | **1.153** | −1.39 | −1.11 |
| **global** | 7,118 | 0.595 | **0.589** | +0.01 | −0.01 |

The benefit shows up mainly in the **external validation**, which is loaded with
highly soluble drugs: MAE 0.908 → **0.790** and bias −0.246 → **+0.025**.

**Why the correction is only partial.** The model compresses its predictions:
its range is [−8.90, −0.26] against a true range of [−10.00, 1.74], and the
prediction–truth correlation in the soluble extreme is only **0.30**. The model
does not distinguish those molecules; recalibrating can rescale, not create
order. After calibration only ~30% of the lost dispersion is recovered.

### 8.3 Diagnosis of the drop on the temporal split

Three hypotheses were ruled out with a **common test set and matched n_train**
(same test, same training size, only the selection criterion changes):

| Hypothesis | Test | Outcome |
|---|---|---|
| Less training data | same n_train in both arms | **ruled out** |
| Systematic label drift | 303 compounds measured before and after 2015 | **ruled out**: mean difference −0.03, median 0.00 |
| Training-set drift ("stale data") | pre-2015 train vs random train, same test | **real but modest**: R² 0.350 vs 0.445 |
| Change in test composition | distributions | **the main cause** |

Detail:

- **There is no label drift.** For the 303 compounds measured in both eras, the
  post − pre difference has mean −0.025 and median 0.000 log. There is no change
  in measurement convention. The **random discrepancy, however, is huge**
  (SD 1.134; 21% differ by more than 1 log).
- **The test set is narrower.** The ≥2015 subset has sd(logS) 1.04 versus 1.29
  for <2015. Since R² divides by the test variance, a narrower test gives a lower
  R² **even when the RMSE is similar**. With RMSE 0.919 and sd 1.04 the R² would
  be 0.22; with sd 1.14 it rises to 0.35.
- **Stale training data costs little.** With the test set and size matched,
  training only on pre-2015 data gives R² 0.350 versus 0.445 training on a random
  sample of the same size. That is, ageing the data costs ~0.10 of R², not 0.40.
- **The outliers are not time-specific**: dropping the worst 10% of errors
  raises R² in a similar way in both arms (0.527 vs 0.584).

**Conclusion: the "temporal drift" was largely an artefact of the test set, not
a deterioration of the model.** The correct label is that compounds registered
from 2015 onwards are chemically more diverse and have a narrower solubility
range.
