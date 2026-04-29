"""Auto-detect column types and build metadata for CTGAN training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _matches_any(col_name: str, patterns: list[str]) -> bool:
    col_lower = col_name.lower()
    return any(p.lower() in col_lower for p in patterns)


def _is_date_column(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().head(100)
    if sample.empty or not pd.api.types.is_string_dtype(sample):
        return False
    try:
        parsed = pd.to_datetime(sample, format="mixed", dayfirst=False)
        return parsed.notna().sum() / len(sample) > 0.8
    except (ValueError, TypeError):
        return False


def _is_rating_column(series: pd.Series, rating_values: set[str]) -> bool:
    unique = set(series.dropna().unique())
    if not unique:
        return False
    return len(unique & rating_values) / len(unique) > 0.5


def _detect_by_pattern(
    col_name: str, series: pd.Series, config: dict
) -> str | None:
    """Check name/id pattern matches. Returns type string or None."""
    profiler_cfg = config["profiler"]
    is_string = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    if _matches_any(col_name, profiler_cfg["name_patterns"]) and is_string:
        return "name"
    if _matches_any(col_name, profiler_cfg["id_patterns"]):
        return "identifier"
    return None


def _detect_numeric_type(
    series: pd.Series, n_unique: int, total_rows: int, threshold: float
) -> str:
    non_null = series.dropna()
    if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
        if n_unique < total_rows * threshold:
            return "categorical"
        return "numeric_discrete"
    return "numeric_continuous"


def _detect_string_type(n_unique: int, total_rows: int, threshold: float) -> str:
    uniqueness_ratio = n_unique / total_rows if total_rows > 0 else 0
    if uniqueness_ratio > 0.9:
        return "identifier"
    if uniqueness_ratio < threshold or n_unique < 50:
        return "categorical"
    return "text"


def _enrich_numeric(result: dict, series: pd.Series, n_unique: int, total_rows: int, threshold: float) -> None:
    non_null = series.dropna()
    result["min"] = float(non_null.min()) if len(non_null) > 0 else None
    result["max"] = float(non_null.max()) if len(non_null) > 0 else None
    result["mean"] = float(non_null.mean()) if len(non_null) > 0 else None
    result["type"] = _detect_numeric_type(series, n_unique, total_rows, threshold)
    if result["type"] == "categorical":
        result["categories"] = sorted(non_null.unique().tolist())


def _enrich_string(result: dict, series: pd.Series, n_unique: int, total_rows: int, threshold: float) -> None:
    result["type"] = _detect_string_type(n_unique, total_rows, threshold)
    if result["type"] == "categorical":
        result["categories"] = sorted(series.dropna().unique().tolist(), key=str)
    elif result["type"] in ("identifier", "text"):
        result["sample_values"] = series.dropna().unique()[:5].tolist()


def _base_result(col_name: str, series: pd.Series, total_rows: int) -> dict[str, Any]:
    n_null = int(series.isna().sum())
    return {
        "name": col_name,
        "n_unique": series.nunique(),
        "n_null": n_null,
        "null_pct": round(n_null / total_rows * 100, 1) if total_rows > 0 else 0,
    }


def _detect_column_type(
    col_name: str, series: pd.Series, total_rows: int, config: dict
) -> dict[str, Any]:
    profiler_cfg = config["profiler"]
    result = _base_result(col_name, series, total_rows)
    n_unique = result["n_unique"]
    threshold = profiler_cfg["categorical_threshold"]

    # Pattern-based detection (names, identifiers)
    pattern_type = _detect_by_pattern(col_name, series, config)
    if pattern_type:
        result["type"] = pattern_type
        if pattern_type in ("name", "identifier"):
            result["sample_values"] = series.dropna().unique()[:5].tolist()
        return result

    if _is_date_column(series):
        result["type"] = "date"
        return result

    if _is_rating_column(series, set(profiler_cfg["rating_values"])):
        result["type"] = "rating"
        result["categories"] = sorted(series.dropna().unique().tolist())
        return result

    if pd.api.types.is_numeric_dtype(series):
        _enrich_numeric(result, series, n_unique, total_rows, threshold)
        return result

    is_string = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    if is_string:
        _enrich_string(result, series, n_unique, total_rows, threshold)
        return result

    result["type"] = "categorical"
    return result


def profile(
    df: pd.DataFrame,
    sensitive_cols: list[str] | None = None,
    id_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Profile a DataFrame and return column metadata."""
    config = _load_config()
    if sensitive_cols:
        config["profiler"]["name_patterns"].extend(sensitive_cols)
    if id_cols:
        config["profiler"]["id_patterns"].extend(id_cols)

    columns = [
        _detect_column_type(col, df[col], len(df), config)
        for col in df.columns
    ]
    return {"total_rows": len(df), "total_columns": len(df.columns), "columns": columns}


def save_metadata(metadata: dict, path: Path) -> None:
    import numpy as np

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialise {type(obj)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=_convert)


def load_metadata(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
