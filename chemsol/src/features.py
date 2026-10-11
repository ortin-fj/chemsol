"""M3 - Molecular representation: RDKit 2D descriptors + Morgan fingerprints.

All descriptors are 2D (no conformer generation required), so the computation is
fast and deterministic.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

# Physicochemical and topological descriptors relevant to solubility.
# 3D descriptors (slow) and experimental ones are deliberately excluded.
DESCRIPTOR_FUNCS: dict[str, object] = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "NumSaturatedRings": Descriptors.NumSaturatedRings,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "LabuteASA": Descriptors.LabuteASA,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "Chi0v": Descriptors.Chi0v,
    "Chi1v": Descriptors.Chi1v,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "NumAmideBonds": Descriptors.NumAmideBonds,
    "NOCount": Descriptors.NOCount,
    "NHOHCount": Descriptors.NHOHCount,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
}
DESCRIPTOR_NAMES = list(DESCRIPTOR_FUNCS)


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def compute_descriptors(smiles_list: list[str]) -> np.ndarray:
    """Matrix (n_molecules, n_descriptors). Returns NaN for invalid molecules."""
    rows = []
    for smi in smiles_list:
        mol = _mol_from_smiles(smi)
        if mol is None:
            rows.append([np.nan] * len(DESCRIPTOR_FUNCS))
            continue
        rows.append([float(fn(mol)) for fn in DESCRIPTOR_FUNCS.values()])
    return np.asarray(rows, dtype=float)


def morgan_fingerprints(
    smiles_list: list[str], n_bits: int = 2048, radius: int = 2
) -> np.ndarray:
    """Binary matrix (n_molecules, n_bits) of Morgan fingerprints."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = _mol_from_smiles(smi)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, out[i])
    return out


def build_features(
    smiles_list: list[str], use_descriptors: bool = True, use_fingerprints: bool = True,
    n_bits: int = 2048, radius: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Concatenates the requested feature blocks and returns their names."""
    blocks, names = [], []
    if use_descriptors:
        blocks.append(compute_descriptors(smiles_list))
        names += DESCRIPTOR_NAMES
    if use_fingerprints:
        blocks.append(morgan_fingerprints(smiles_list, n_bits=n_bits, radius=radius))
        names += [f"morgan_{i}" for i in range(n_bits)]
    if not blocks:
        raise ValueError("At least one feature block must be enabled")
    return np.hstack(blocks), names
