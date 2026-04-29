"""Post-generation validation — ensure no real data leaked into synthetic output."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy import stats

from .profiler import load_metadata


@dataclass
class ColumnReport:
    name: str
    col_type: str
    passed: bool
    detail: str


@dataclass
class PrivacyReport:
    columns: list[ColumnReport] = field(default_factory=list)
    row_vector_matches: int = 0
    overall_pass: bool = True

    def summary(self) -> str:
        lines = ["Privacy report:"]
        for col in self.columns:
            status = "PASS" if col.passed else "FAIL"
            lines.append(f"  {col.name}: {col.detail} ({status})")
        lines.append(
            f"  Row-vector: {self.row_vector_matches} exact matches "
            f"({'PASS' if self.row_vector_matches == 0 else 'WARN'})"
        )
        overall = "PASS" if self.overall_pass else "FAIL"
        lines.append(f"  Overall: {overall}")
        return "\n".join(lines)


def sanitise(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    metadata: dict,
    ks_threshold: float = 0.5,
) -> PrivacyReport:
    """Validate synthetic data against the real training data.

    Checks:
    1. Identifier columns: zero overlap with real values
    2. Name columns: zero overlap with real values
    3. Numeric columns: KS-test p-value above threshold
    4. Row-vector: no exact row matches

    Args:
        synthetic_df: The generated DataFrame.
        real_df: The original training DataFrame.
        metadata: Column metadata from the profiler.
        ks_threshold: Minimum KS-test p-value for numeric columns.

    Returns:
        PrivacyReport with per-column results.
    """
    report = PrivacyReport()

    for col_meta in metadata["columns"]:
        name = col_meta["name"]
        col_type = col_meta["type"]

        if name not in synthetic_df.columns or name not in real_df.columns:
            continue

        syn_vals = synthetic_df[name].dropna()
        real_vals = real_df[name].dropna()

        if col_type == "identifier":
            overlap = set(syn_vals.astype(str)) & set(real_vals.astype(str))
            n_overlap = len(overlap)
            passed = n_overlap == 0
            detail = f"{n_overlap}/{len(syn_vals)} overlap with training"
            if not passed:
                report.overall_pass = False
            report.columns.append(ColumnReport(name, col_type, passed, detail))

        elif col_type == "name":
            overlap = set(syn_vals.astype(str)) & set(real_vals.astype(str))
            n_overlap = len(overlap)
            passed = n_overlap == 0
            detail = (
                f"{n_overlap}/{len(syn_vals)} overlap with training"
                if not passed
                else f"0/{len(syn_vals)} overlap (all Faker-generated)"
            )
            if not passed:
                report.overall_pass = False
            report.columns.append(ColumnReport(name, col_type, passed, detail))

        elif col_type in ("numeric_continuous", "numeric_discrete"):
            syn_numeric = pd.to_numeric(syn_vals, errors="coerce").dropna()
            real_numeric = pd.to_numeric(real_vals, errors="coerce").dropna()

            if len(syn_numeric) > 0 and len(real_numeric) > 0:
                ks_stat, p_value = stats.ks_2samp(real_numeric, syn_numeric)
                passed = p_value >= ks_threshold
                detail = f"KS-test p={p_value:.2f}"
                report.columns.append(ColumnReport(name, col_type, passed, detail))

        elif col_type in ("categorical", "rating"):
            # Check distribution similarity
            real_counts = real_vals.value_counts(normalize=True)
            syn_counts = syn_vals.value_counts(normalize=True)
            # Align indices
            all_cats = set(real_counts.index) | set(syn_counts.index)
            real_dist = [real_counts.get(c, 0) for c in all_cats]
            syn_dist = [syn_counts.get(c, 0) for c in all_cats]
            # Simple L1 distance
            l1 = sum(abs(r - s) for r, s in zip(real_dist, syn_dist))
            passed = l1 < 1.0  # Generous threshold for POC
            detail = f"L1 distance={l1:.3f}"
            report.columns.append(ColumnReport(name, col_type, passed, detail))

    # Row-vector exact match check (on numeric columns only to avoid false positives)
    numeric_cols = [
        cm["name"]
        for cm in metadata["columns"]
        if cm["type"] in ("numeric_continuous", "numeric_discrete")
        and cm["name"] in synthetic_df.columns
        and cm["name"] in real_df.columns
    ]

    if numeric_cols:
        # Round to reduce floating point noise, then check for exact matches
        syn_rounded = synthetic_df[numeric_cols].round(2)
        real_rounded = real_df[numeric_cols].round(2)

        syn_tuples = set(syn_rounded.dropna().apply(tuple, axis=1))
        real_tuples = set(real_rounded.dropna().apply(tuple, axis=1))

        report.row_vector_matches = len(syn_tuples & real_tuples)
        if report.row_vector_matches > 0:
            report.overall_pass = False

    return report
