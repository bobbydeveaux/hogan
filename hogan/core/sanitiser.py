"""Post-generation validation — ensure no real data leaked into synthetic output."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy import stats


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


def _check_overlap(syn_vals: pd.Series, real_vals: pd.Series) -> tuple[int, int]:
    """Return (n_overlap, total) for string overlap check."""
    overlap = set(syn_vals.astype(str)) & set(real_vals.astype(str))
    return len(overlap), len(syn_vals)


def _check_identifier(name: str, syn_vals: pd.Series, real_vals: pd.Series) -> ColumnReport:
    n_overlap, total = _check_overlap(syn_vals, real_vals)
    passed = n_overlap == 0
    detail = f"{n_overlap}/{total} overlap with training"
    return ColumnReport(name, "identifier", passed, detail)


def _check_name(name: str, syn_vals: pd.Series, real_vals: pd.Series) -> ColumnReport:
    n_overlap, total = _check_overlap(syn_vals, real_vals)
    passed = n_overlap == 0
    detail = (
        f"{n_overlap}/{total} overlap with training"
        if not passed
        else f"0/{total} overlap (all Faker-generated)"
    )
    return ColumnReport(name, "name", passed, detail)


def _check_numeric(
    name: str, col_type: str, syn_vals: pd.Series, real_vals: pd.Series, ks_threshold: float
) -> ColumnReport | None:
    syn_numeric = pd.to_numeric(syn_vals, errors="coerce").dropna()
    real_numeric = pd.to_numeric(real_vals, errors="coerce").dropna()
    if len(syn_numeric) == 0 or len(real_numeric) == 0:
        return None
    _stat, p_value = stats.ks_2samp(real_numeric, syn_numeric)
    passed = p_value >= ks_threshold
    return ColumnReport(name, col_type, passed, f"KS-test p={p_value:.2f}")


def _check_categorical(
    name: str, col_type: str, syn_vals: pd.Series, real_vals: pd.Series
) -> ColumnReport:
    real_counts = real_vals.value_counts(normalize=True)
    syn_counts = syn_vals.value_counts(normalize=True)
    all_cats = set(real_counts.index) | set(syn_counts.index)
    real_dist = [real_counts.get(c, 0) for c in all_cats]
    syn_dist = [syn_counts.get(c, 0) for c in all_cats]
    l1 = sum(abs(r - s) for r, s in zip(real_dist, syn_dist))
    return ColumnReport(name, col_type, l1 < 1.0, f"L1 distance={l1:.3f}")


def _check_row_vectors(
    synthetic_df: pd.DataFrame, real_df: pd.DataFrame, metadata: dict
) -> int:
    numeric_cols = [
        cm["name"]
        for cm in metadata["columns"]
        if cm["type"] in ("numeric_continuous", "numeric_discrete")
        and cm["name"] in synthetic_df.columns
        and cm["name"] in real_df.columns
    ]
    if not numeric_cols:
        return 0
    syn_tuples = set(synthetic_df[numeric_cols].round(2).dropna().apply(tuple, axis=1))
    real_tuples = set(real_df[numeric_cols].round(2).dropna().apply(tuple, axis=1))
    return len(syn_tuples & real_tuples)


_CHECKERS = {
    "identifier": _check_identifier,
    "name": _check_name,
}

_NUMERIC_TYPES = ("numeric_continuous", "numeric_discrete")
_CATEGORICAL_TYPES = ("categorical", "rating")


def sanitise(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    metadata: dict,
    ks_threshold: float = 0.5,
) -> PrivacyReport:
    """Validate synthetic data against the real training data."""
    report = PrivacyReport()

    for col_meta in metadata["columns"]:
        name, col_type = col_meta["name"], col_meta["type"]
        if name not in synthetic_df.columns or name not in real_df.columns:
            continue

        syn_vals = synthetic_df[name].dropna()
        real_vals = real_df[name].dropna()

        col_report = _check_column(name, col_type, syn_vals, real_vals, ks_threshold)
        if col_report:
            if not col_report.passed and col_type in ("identifier", "name"):
                report.overall_pass = False
            report.columns.append(col_report)

    report.row_vector_matches = _check_row_vectors(synthetic_df, real_df, metadata)
    if report.row_vector_matches > 0:
        report.overall_pass = False

    return report


def _check_column(
    name: str, col_type: str, syn_vals: pd.Series, real_vals: pd.Series, ks_threshold: float
) -> ColumnReport | None:
    if col_type in _CHECKERS:
        return _CHECKERS[col_type](name, syn_vals, real_vals)
    if col_type in _NUMERIC_TYPES:
        return _check_numeric(name, col_type, syn_vals, real_vals, ks_threshold)
    if col_type in _CATEGORICAL_TYPES:
        return _check_categorical(name, col_type, syn_vals, real_vals)
    return None
