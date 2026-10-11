#!/usr/bin/env bash
# Full end-to-end pipeline. Usage: bash run_all.sh
set -euo pipefail

PY=.venv/bin/python

if [ ! -x "$PY" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
fi

echo "== 1/6 ChEMBL download =="
$PY src/fetch_chembl.py --out data/raw/solubility_chembl.csv

echo "== 2/6 Curation =="
$PY src/curate.py --raw data/raw/solubility_chembl.csv \
                  --out data/processed/solubility_logS.csv

echo "== 3/6 External validation set =="
$PY src/make_external_set.py

echo "== 4/6 Training =="
$PY src/train.py --processed data/processed/solubility_logS.csv \
                 --outdir reports \
                 --holdout-file data/external/validation_set.csv

echo "== 5/6 Applicability domain =="
$PY src/domain_analysis.py --outdir reports

echo "== 6/6 External validation =="
$PY src/validate_external.py --outdir reports

echo
echo "Tests:"
$PY -m unittest discover -s tests

echo
echo "Done. Results in reports/"
