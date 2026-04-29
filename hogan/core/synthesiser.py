"""Generate synthetic rows from a trained CTGAN model."""

from __future__ import annotations

import json
import random
import string
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from .profiler import load_metadata
from .trainer import load_model

fake = Faker()


def _random_alphanum(length: int) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def _generate_identifiers(n: int, col_name: str) -> list[str]:
    col_lower = col_name.lower()
    if "cusip" in col_lower:
        return [_random_alphanum(9) for _ in range(n)]
    if "isin" in col_lower:
        return [_random_alphanum(12) for _ in range(n)]
    if "account_id" in col_lower:
        base = random.randint(100000, 999999)
        return [str(base + i) for i in range(n)]
    return [str(uuid.uuid4())[:12].upper() for _ in range(n)]


def _generate_names(n: int, col_name: str) -> list[str]:
    col_lower = col_name.lower()
    if "client" in col_lower or "investor" in col_lower:
        return [fake.company() for _ in range(n)]
    if "fund" in col_lower:
        return [fake.company() + " Fund" for _ in range(n)]
    return [fake.company() for _ in range(n)]


def _generate_text(n: int, original_series: pd.Series) -> list[str]:
    templates = original_series.dropna().unique()
    if len(templates) > 0:
        return [random.choice(templates) for _ in range(n)]
    return [fake.text(max_nb_chars=50) for _ in range(n)]


def _ordinal_to_date(x: float) -> str:
    if pd.notna(x) and x > 0:
        return datetime.fromordinal(int(round(x))).strftime("%Y-%m-%d")
    return ""


def _process_gan_column(name: str, col_type: str, series: pd.Series) -> pd.Series:
    """Post-process a CTGAN-generated column based on its type."""
    if col_type == "date":
        return series.apply(_ordinal_to_date)
    if col_type == "numeric_discrete":
        return series.round(0).astype(int)
    if col_type in ("categorical", "rating"):
        return series.replace("_NULL_", pd.NA)
    return series


def _generate_non_gan_column(
    col_meta: dict, n_rows: int, original_df: pd.DataFrame | None
) -> pd.Series:
    """Generate values for columns not handled by CTGAN."""
    name = col_meta["name"]
    col_type = col_meta["type"]

    if col_type == "identifier":
        return pd.Series(_generate_identifiers(n_rows, name))
    if col_type == "name":
        return pd.Series(_generate_names(n_rows, name))
    if col_type == "text":
        original_series = pd.Series(dtype=str)
        if original_df is not None and name in original_df.columns:
            original_series = original_df[name]
        return pd.Series(_generate_text(n_rows, original_series))
    return pd.Series([None] * n_rows)


def synthesise(
    model_dir: Path,
    n_rows: int | None = None,
    seed: int | None = None,
    original_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate synthetic data from a trained model."""
    metadata = load_metadata(model_dir / "metadata.json")
    ctgan = load_model(model_dir)

    if n_rows is None:
        n_rows = metadata["total_rows"]

    if seed is not None:
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    synthetic = ctgan.sample(n_rows)

    result_cols: dict[str, pd.Series] = {}
    for col_meta in metadata["columns"]:
        name, col_type = col_meta["name"], col_meta["type"]
        if name in synthetic.columns:
            result_cols[name] = _process_gan_column(name, col_type, synthetic[name])
        else:
            result_cols[name] = _generate_non_gan_column(col_meta, n_rows, original_df)

    return pd.DataFrame(
        {cm["name"]: result_cols[cm["name"]] for cm in metadata["columns"]}
    )
