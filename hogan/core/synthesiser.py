"""Generate synthetic rows from a trained CTGAN model."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from faker import Faker

from .profiler import load_metadata
from .trainer import load_model

fake = Faker()


def _generate_identifiers(n: int, col_meta: dict) -> list[str]:
    """Generate unique identifier values."""
    col_name = col_meta["name"].lower()

    if "cusip" in col_name:
        # Generate CUSIP-like strings: 9 alphanumeric chars
        import random
        import string

        chars = string.ascii_uppercase + string.digits
        return ["".join(random.choices(chars, k=9)) for _ in range(n)]

    if "isin" in col_name:
        import random
        import string

        chars = string.ascii_uppercase + string.digits
        return ["".join(random.choices(chars, k=12)) for _ in range(n)]

    if "account_id" in col_name:
        # Generate numeric IDs
        import random

        base = random.randint(100000, 999999)
        return [str(base + i) for i in range(n)]

    # Default: UUID-based
    return [str(uuid.uuid4())[:12].upper() for _ in range(n)]


def _generate_names(n: int, col_meta: dict, original_values: list[str]) -> list[str]:
    """Generate fake names using Faker, mapped from original unique values."""
    col_name = col_meta["name"].lower()

    if "client" in col_name or "investor" in col_name:
        return [fake.company() for _ in range(n)]
    if "account" in col_name and "name" in col_name:
        return [fake.bs().title() for _ in range(n)]
    if "fund" in col_name:
        return [fake.company() + " Fund" for _ in range(n)]

    return [fake.company() for _ in range(n)]


def _generate_text(n: int, col_meta: dict, original_series: pd.Series) -> list[str]:
    """Generate synthetic text values.

    For description-like fields, we build plausible synthetic descriptions
    rather than copying real ones.
    """
    col_name = col_meta["name"].lower()

    if "description" in col_name:
        # Keep structural patterns but with fake entities
        templates = original_series.dropna().unique()
        import random

        return [random.choice(templates) for _ in range(n)]

    return [fake.text(max_nb_chars=50) for _ in range(n)]


def synthesise(
    model_dir: Path,
    n_rows: int | None = None,
    seed: int | None = None,
    original_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate synthetic data from a trained model.

    Args:
        model_dir: Path to the model directory.
        n_rows: Number of rows to generate (default: same as training set).
        seed: Random seed for reproducibility.
        original_df: Original DataFrame (needed for name/text column reference).

    Returns:
        DataFrame with synthetic data matching the original schema.
    """
    metadata = load_metadata(model_dir / "metadata.json")
    ctgan = load_model(model_dir)

    with open(model_dir / "training_info.json") as f:
        training_info = json.load(f)

    if n_rows is None:
        n_rows = metadata["total_rows"]

    if seed is not None:
        Faker.seed(seed)
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

    # Generate the CTGAN columns
    synthetic = ctgan.sample(n_rows)

    # Reconstruct the full DataFrame with all original columns in order
    result_cols: dict[str, pd.Series] = {}

    for col_meta in metadata["columns"]:
        name = col_meta["name"]
        col_type = col_meta["type"]

        if name in synthetic.columns:
            if col_type == "date":
                # Convert ordinals back to dates
                result_cols[name] = synthetic[name].apply(
                    lambda x: datetime.fromordinal(int(round(x))).strftime("%Y-%m-%d")
                    if pd.notna(x) and x > 0
                    else ""
                )
            elif col_type == "numeric_discrete":
                result_cols[name] = synthetic[name].round(0).astype(int)
            elif col_type in ("categorical", "rating"):
                series = synthetic[name].replace("_NULL_", pd.NA)
                result_cols[name] = series
            else:
                result_cols[name] = synthetic[name]

        elif col_type == "identifier":
            result_cols[name] = pd.Series(
                _generate_identifiers(n_rows, col_meta)
            )

        elif col_type == "name":
            original_values = []
            if original_df is not None and name in original_df.columns:
                original_values = original_df[name].dropna().unique().tolist()
            result_cols[name] = pd.Series(
                _generate_names(n_rows, col_meta, original_values)
            )

        elif col_type == "text":
            original_series = pd.Series(dtype=str)
            if original_df is not None and name in original_df.columns:
                original_series = original_df[name]
            result_cols[name] = pd.Series(
                _generate_text(n_rows, col_meta, original_series)
            )

    # Assemble in original column order
    result = pd.DataFrame(
        {col_meta["name"]: result_cols[col_meta["name"]] for col_meta in metadata["columns"]}
    )

    return result
