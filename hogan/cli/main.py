"""Hogan CLI — train, synthesise, inspect."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="hogan")
def cli():
    """Hogan — GAN-based tabular data synthesiser."""


@cli.command()
@click.argument("input_csv", type=click.Path(exists=True, path_type=Path))
@click.option("--epochs", type=int, default=None, help="Training epochs (default: 300)")
@click.option("--batch-size", type=int, default=None, help="Batch size (default: 500)")
@click.option("--model-dir", type=click.Path(path_type=Path), default=None, help="Where to save model")
@click.option("--name", type=str, default=None, help="Model name (default: derived from filename)")
@click.option("--sensitive", type=str, default=None, help="Comma-separated columns to treat as sensitive")
@click.option("--id-cols", type=str, default=None, help="Comma-separated columns to treat as identifiers")
@click.option("--preview", is_flag=True, help="Profile only — show detected types, don't train")
def train(
    input_csv: Path,
    epochs: int | None,
    batch_size: int | None,
    model_dir: Path | None,
    name: str | None,
    sensitive: str | None,
    id_cols: str | None,
    preview: bool,
):
    """Train a CTGAN model on a CSV dataset."""
    from hogan.core.profiler import profile, save_metadata
    from hogan.core.trainer import train as do_train

    console.print(f"[bold]Loading[/bold] {input_csv}")
    df = pd.read_csv(input_csv)
    console.print(f"  {len(df)} rows, {len(df.columns)} columns\n")

    sensitive_list = [s.strip() for s in sensitive.split(",")] if sensitive else None
    id_list = [s.strip() for s in id_cols.split(",")] if id_cols else None

    metadata = profile(df, sensitive_cols=sensitive_list, id_cols=id_list)

    # Show profile
    table = Table(title="Column Profile", show_lines=False)
    table.add_column("Column", style="cyan", min_width=30)
    table.add_column("Type", style="green", min_width=20)
    table.add_column("Unique", justify="right")
    table.add_column("Nulls", justify="right")
    table.add_column("Detail", style="dim")

    for col in metadata["columns"]:
        detail = ""
        if col["type"] in ("numeric_continuous", "numeric_discrete"):
            if col.get("min") is not None:
                detail = f"range: {col['min']:.2f} - {col['max']:.2f}"
        elif col["type"] in ("categorical", "rating"):
            cats = col.get("categories", [])
            if len(cats) <= 5:
                detail = ", ".join(str(c) for c in cats)
            else:
                detail = f"{len(cats)} categories"
        elif col["type"] in ("name", "identifier"):
            samples = col.get("sample_values", [])
            if samples:
                detail = ", ".join(str(s) for s in samples[:3]) + "..."

        table.add_row(
            col["name"],
            col["type"],
            str(col["n_unique"]),
            f"{col['null_pct']}%",
            detail,
        )

    console.print(table)

    if preview:
        console.print("\n[dim]Preview mode — no training performed.[/dim]")
        return

    # Train
    model_name = name or input_csv.stem
    console.print()

    model_path = do_train(
        df=df,
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        model_dir=model_dir,
        model_name=model_name,
        verbose=True,
    )

    console.print(f"\n[bold green]Done![/bold green] Model saved to {model_path}/")


@cli.command()
@click.option("--model", "model_dir", type=click.Path(path_type=Path), default=None, help="Model directory")
@click.option("-n", "n_rows", type=int, default=None, help="Number of rows to generate")
@click.option("-o", "output", type=click.Path(path_type=Path), default=None, help="Output file (default: stdout)")
@click.option("--format", "fmt", type=click.Choice(["csv", "parquet", "json"]), default="csv")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--validate", is_flag=True, help="Run sanitiser and print privacy report")
@click.option("--training-csv", type=click.Path(exists=True, path_type=Path), default=None, help="Original training CSV (for name/text generation and validation)")
def synthesise(
    model_dir: Path | None,
    n_rows: int | None,
    output: Path | None,
    fmt: str,
    seed: int | None,
    validate: bool,
    training_csv: Path | None,
):
    """Generate synthetic rows from a trained model."""
    from hogan.core.profiler import load_metadata
    from hogan.core.sanitiser import sanitise
    from hogan.core.synthesiser import synthesise as do_synthesise

    # Find model directory
    if model_dir is None:
        hogan_dir = Path(".hogan")
        if not hogan_dir.exists():
            console.print("[red]No .hogan/ directory found. Run 'hogan train' first.[/red]")
            raise SystemExit(1)
        # Use most recent model
        subdirs = sorted(hogan_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        subdirs = [d for d in subdirs if d.is_dir() and (d / "model.pkl").exists()]
        if not subdirs:
            console.print("[red]No trained models found in .hogan/[/red]")
            raise SystemExit(1)
        model_dir = subdirs[0]

    console.print(f"[bold]Loading model from[/bold] {model_dir}/")

    original_df = None
    if training_csv is not None:
        original_df = pd.read_csv(training_csv)

    synthetic_df = do_synthesise(
        model_dir=model_dir,
        n_rows=n_rows,
        seed=seed,
        original_df=original_df,
    )

    console.print(f"[bold green]Generated {len(synthetic_df)} rows[/bold green]")

    # Validate
    if validate:
        if original_df is None:
            console.print("[yellow]Warning: --validate requires --training-csv to compare against.[/yellow]")
        else:
            metadata = load_metadata(model_dir / "metadata.json")
            report = sanitise(synthetic_df, original_df, metadata)
            console.print()
            console.print(report.summary())

    # Output
    if output is None:
        click.echo(synthetic_df.to_csv(index=False))
    else:
        if fmt == "csv":
            synthetic_df.to_csv(output, index=False)
        elif fmt == "parquet":
            synthetic_df.to_parquet(output, index=False)
        elif fmt == "json":
            synthetic_df.to_json(output, orient="records", indent=2)
        console.print(f"Written to {output}")


@cli.command()
@click.argument("synthetic_csv", type=click.Path(exists=True, path_type=Path))
@click.option("--against", type=click.Path(exists=True, path_type=Path), required=True, help="Original real CSV")
@click.option("--model", "model_dir", type=click.Path(path_type=Path), default=None, help="Model directory (for metadata)")
def inspect(synthetic_csv: Path, against: Path, model_dir: Path | None):
    """Compare synthetic data against the original dataset."""
    from hogan.core.profiler import load_metadata, profile
    from hogan.core.sanitiser import sanitise

    console.print(f"[bold]Comparing[/bold] {synthetic_csv} against {against}\n")

    synthetic_df = pd.read_csv(synthetic_csv)
    real_df = pd.read_csv(against)

    # Get metadata — either from model or by profiling the real data
    if model_dir is not None:
        metadata = load_metadata(model_dir / "metadata.json")
    else:
        metadata = profile(real_df)

    report = sanitise(synthetic_df, real_df, metadata)

    # Rich table output
    table = Table(title="Comparison Report", show_lines=False)
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Result")
    table.add_column("Detail")

    for col in report.columns:
        status_style = "green" if col.passed else "red"
        status = "PASS" if col.passed else "FAIL"
        table.add_row(col.name, col.col_type, f"[{status_style}]{status}[/{status_style}]", col.detail)

    console.print(table)

    console.print(
        f"\nRow-vector exact matches: {report.row_vector_matches} "
        f"({'[green]PASS[/green]' if report.row_vector_matches == 0 else '[red]WARN[/red]'})"
    )
    overall = "[green]PASS[/green]" if report.overall_pass else "[red]FAIL[/red]"
    console.print(f"Overall: {overall}")
