"""Minimal loaders for ResDec-H3's TabPFN residual path and FiLM metadata.

Under the P5 plan, the heavy origin-tagged flat-tabular builder is NOT needed:
TabPFN consumes the flat aggregate pseudobulk (via top-2K XGBoost importance
selection per fold), and the new head's FiLM conditioning takes a small
metadata vector.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

METADATA_FIELDS = [
    "apoe_e2", "apoe_e3", "apoe_e4", "apoe_missing",
    "sex", "sex_missing",
    "age", "age_missing",
]  # total 8 dims

# Reference age stats for z-scoring (cohort-wide; fit once, frozen)
_AGE_MEAN = 86.0  # ROSMAP cohort approx
_AGE_STD = 6.5


def flatten_pseudobulk(pt_subject: dict) -> torch.Tensor:
    """Flatten pseudobulk [31, 4785] -> [148_335] as float32 tensor.

    Returned tensor is on CPU; caller moves to device as needed.
    """
    pb = pt_subject["pseudobulk"]
    if not isinstance(pb, torch.Tensor):
        pb = torch.as_tensor(pb)
    return pb.float().flatten().contiguous()


def load_metadata_vector(
    subject_id: str, meta_csv: Path
) -> tuple[torch.Tensor, list[str]]:
    """Build an 8-dim metadata vector for FiLM conditioning.

    Fields (in order): APOE one-hot presence (e2, e3, e4, missing), sex
    (val, missing), age (z-scored, missing). All missingness indicators are
    1 when the field is NaN in metadata.csv, 0 otherwise.

    Subject ID format: splits/precomputed use "R<digits>" (e.g. "R1015854");
    metadata.csv has BOTH ``projid`` (integer, unrelated to the R-prefix
    digits) AND ``ROSMAP_IndividualID`` (string, matches the precomputed
    file names exactly). Use ``ROSMAP_IndividualID`` as the join key — do
    NOT strip the "R" prefix, do NOT cast to int, and do NOT touch
    ``projid``.

    The numeric ``apoe_genotype`` column encodes allele pairs as two-digit
    integers: 22, 23, 24, 33, 34, 44 (digits 2, 3, 4 map to e2, e3, e4).

    Returns:
        (vector [8], field_names) — vector is float32 on CPU.
    """
    df = pd.read_csv(meta_csv)
    row = df.loc[df["ROSMAP_IndividualID"] == subject_id]
    vec = torch.zeros(len(METADATA_FIELDS), dtype=torch.float32)

    if len(row) == 0:
        # Subject not in metadata — all missing
        vec[3] = 1.0  # apoe_missing
        vec[5] = 1.0  # sex_missing
        vec[7] = 1.0  # age_missing
        return vec, METADATA_FIELDS

    r = row.iloc[0]

    # APOE: apoe_genotype is numeric (22, 23, 24, 33, 34, 44). Decompose into
    # digit pairs and set the presence bit for each allele observed.
    apoe = r.get("apoe_genotype")
    if pd.isna(apoe):
        vec[3] = 1.0
    else:
        code = int(apoe)
        d1, d2 = code // 10, code % 10
        for d in (d1, d2):
            if d == 2:
                vec[0] = 1.0  # e2 present
            elif d == 3:
                vec[1] = 1.0  # e3 present
            elif d == 4:
                vec[2] = 1.0  # e4 present

    # Sex
    sex = r.get("msex")
    if pd.isna(sex):
        vec[5] = 1.0
    else:
        vec[4] = float(sex)  # 0 or 1

    # Age (z-scored)
    age = r.get("age_death")
    if pd.isna(age):
        vec[7] = 1.0
    else:
        vec[6] = (float(age) - _AGE_MEAN) / _AGE_STD

    return vec, METADATA_FIELDS
