"""CTGAN training wrapper."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from ctgan import CTGAN

from .profiler import profile, save_metadata

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _prepare_for_ctgan(
    df: pd.DataFrame, metadata: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare the DataFrame for CTGAN training.

    Returns the cleaned DataFrame and list of discrete column names
    (columns CTGAN should treat as categorical/discrete).
    """
    discrete_columns: list[str] = []
    df_clean = df.copy()

    for col_meta in metadata["columns"]:
        name = col_meta["name"]
        col_type = col_meta["type"]

        if col_type in ("categorical", "rating"):
            discrete_columns.append(name)
            df_clean[name] = df_clean[name].astype(str).fillna("_NULL_")

        elif col_type in ("name", "identifier", "text"):
            # Drop these from CTGAN training — they'll be replaced post-synthesis
            df_clean = df_clean.drop(columns=[name], errors="ignore")

        elif col_type == "date":
            # Convert dates to ordinal for GAN training
            try:
                parsed = pd.to_datetime(df_clean[name], format="mixed", dayfirst=False)
                df_clean[name] = parsed.map(
                    lambda x: x.toordinal() if pd.notna(x) else float("nan")
                )
                # CTGAN can't handle NaN in continuous cols — fill with median
                median_val = df_clean[name].median()
                df_clean[name] = df_clean[name].fillna(median_val)
            except (ValueError, TypeError):
                df_clean = df_clean.drop(columns=[name], errors="ignore")

        elif col_type in ("numeric_continuous", "numeric_discrete"):
            df_clean[name] = pd.to_numeric(df_clean[name], errors="coerce")
            # CTGAN can't handle NaN in continuous cols — fill with median
            median_val = df_clean[name].median()
            df_clean[name] = df_clean[name].fillna(median_val if pd.notna(median_val) else 0)

    return df_clean, discrete_columns


def train(
    df: pd.DataFrame,
    metadata: dict,
    epochs: int | None = None,
    batch_size: int | None = None,
    model_dir: Path | None = None,
    model_name: str = "model",
    verbose: bool = True,
) -> Path:
    """Train a CTGAN model and save it.

    Args:
        df: The input DataFrame.
        metadata: Column metadata from the profiler.
        epochs: Training epochs (uses config default if None).
        batch_size: Batch size (uses config default if None).
        model_dir: Directory to save model artifacts.
        model_name: Name for the model.
        verbose: Print progress.

    Returns:
        Path to the model directory.
    """
    config = _load_config()
    train_cfg = config["training"]

    epochs = epochs or train_cfg["epochs"]
    batch_size = batch_size or train_cfg["batch_size"]

    df_train, discrete_columns = _prepare_for_ctgan(df, metadata)

    if verbose:
        print(f"Training CTGAN on {len(df_train)} rows, {len(df_train.columns)} columns")
        print(f"  Discrete columns: {len(discrete_columns)}")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")

    ctgan = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_dim=tuple(train_cfg["generator_dim"]),
        discriminator_dim=tuple(train_cfg["discriminator_dim"]),
        generator_lr=train_cfg["generator_lr"],
        discriminator_lr=train_cfg["discriminator_lr"],
        discriminator_steps=train_cfg["discriminator_steps"],
        log_frequency=train_cfg["log_frequency"],
        pac=train_cfg["pac"],
        verbose=verbose,
    )

    ctgan.fit(df_train, discrete_columns=discrete_columns)

    # Save artifacts
    if model_dir is None:
        model_dir = Path(".hogan") / model_name

    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(ctgan, f)

    # Save metadata alongside model
    save_metadata(metadata, model_dir / "metadata.json")

    # Save the training columns (the columns CTGAN actually trained on)
    training_info = {
        "trained_columns": df_train.columns.tolist(),
        "discrete_columns": discrete_columns,
        "epochs": epochs,
        "batch_size": batch_size,
        "training_rows": len(df_train),
    }
    with open(model_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    if verbose:
        print(f"Model saved to {model_dir}/")

    return model_dir


def load_model(model_dir: Path) -> CTGAN:
    """Load a trained CTGAN model from disk."""
    model_path = model_dir / "model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)
