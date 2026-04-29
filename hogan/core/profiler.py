"""Auto-detect column types and build metadata for CTGAN training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"

COLUMN_TYPES = (
    "numeric_continuous",
    "numeric_discrete",
    "categorical",
    "identifier",
    "name",
    "date",
    "rating",
    "text",
)


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _matches_any(col_name: str, patterns: list[str]) -> bool:
    """Check if a column name contains any of the given patterns (case-insensitive)."""
    col_lower = col_name.lower()
    return any(p.lower() in col_lower for p in patterns)


def _is_date_column(series: pd.Series) -> bool:
    """Check if a column looks like dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().head(100)
    if sample.empty:
        return False
    if not pd.api.types.is_string_dtype(sample):
        return False
    # Try parsing a sample
    try:
        parsed = pd.to_datetime(sample, format="mixed", dayfirst=False)
        success_rate = parsed.notna().sum() / len(sample)
        return success_rate > 0.8
    except (ValueError, TypeError):
        return False


def _is_rating_column(series: pd.Series, rating_values: set[str]) -> bool:
    """Check if a column contains credit rating values."""
    unique = set(series.dropna().unique())
    if not unique:
        return False
    overlap = unique & rating_values
    return len(overlap) / len(unique) > 0.5


def _detect_column_type(
    col_name: str,
    series: pd.Series,
    total_rows: int,
    config: dict,
) -> dict[str, Any]:
    """Detect the type of a single column and return metadata."""
    profiler_cfg = config["profiler"]
    n_unique = series.nunique()
    n_null = int(series.isna().sum())
    result: dict[str, Any] = {
        "name": col_name,
        "n_unique": n_unique,
        "n_null": n_null,
        "null_pct": round(n_null / total_rows * 100, 1) if total_rows > 0 else 0,
    }

    # 1. Check for name/label columns by pattern matching
    if _matches_any(col_name, profiler_cfg["name_patterns"]):
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            result["type"] = "name"
            result["sample_values"] = series.dropna().unique()[:5].tolist()
            return result

    # 2. Check for identifier columns by pattern matching
    if _matches_any(col_name, profiler_cfg["id_patterns"]):
        result["type"] = "identifier"
        result["sample_values"] = series.dropna().unique()[:5].tolist()
        return result

    # 3. Check for date columns
    if _is_date_column(series):
        result["type"] = "date"
        return result

    # 4. Check for rating columns
    rating_values = set(profiler_cfg["rating_values"])
    if _is_rating_column(series, rating_values):
        result["type"] = "rating"
        result["categories"] = sorted(series.dropna().unique().tolist())
        return result

    # 5. Numeric types
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        result["min"] = float(non_null.min()) if len(non_null) > 0 else None
        result["max"] = float(non_null.max()) if len(non_null) > 0 else None
        result["mean"] = float(non_null.mean()) if len(non_null) > 0 else None

        # Check if discrete (all integer values)
        if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
            if n_unique < total_rows * profiler_cfg["categorical_threshold"]:
                result["type"] = "categorical"
                result["categories"] = sorted(non_null.unique().tolist())
            else:
                result["type"] = "numeric_discrete"
        else:
            result["type"] = "numeric_continuous"
        return result

    # 6. String columns — categorical vs text vs identifier
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        uniqueness_ratio = n_unique / total_rows if total_rows > 0 else 0

        # Very high cardinality strings → likely identifiers
        if uniqueness_ratio > 0.9:
            result["type"] = "identifier"
            result["sample_values"] = series.dropna().unique()[:5].tolist()
            return result

        # Low-medium cardinality → categorical
        if uniqueness_ratio < profiler_cfg["categorical_threshold"] or n_unique < 50:
            result["type"] = "categorical"
            result["categories"] = sorted(
                series.dropna().unique().tolist(), key=str
            )
            return result

        # Medium cardinality text
        result["type"] = "text"
        result["sample_values"] = series.dropna().unique()[:5].tolist()
        return result

    result["type"] = "categorical"
    return result


def profile(
    df: pd.DataFrame,
    sensitive_cols: list[str] | None = None,
    id_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Profile a DataFrame and return column metadata.

    Returns a dict with:
      - columns: list of column metadata dicts
      - total_rows: int
      - total_columns: int
    """
    config = _load_config()
    total_rows = len(df)

    # Apply user overrides to config patterns
    if sensitive_cols:
        config["profiler"]["name_patterns"].extend(sensitive_cols)
    if id_cols:
        config["profiler"]["id_patterns"].extend(id_cols)

    columns = []
    for col_name in df.columns:
        col_meta = _detect_column_type(col_name, df[col_name], total_rows, config)
        columns.append(col_meta)

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": columns,
    }


def save_metadata(metadata: dict, path: Path) -> None:
    """Save metadata to a JSON file."""
    # Convert any numpy types for JSON serialisation
    def _convert(obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialise {type(obj)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=_convert)


def load_metadata(path: Path) -> dict:
    """Load metadata from a JSON file."""
    with open(path) as f:
        return json.load(f)
