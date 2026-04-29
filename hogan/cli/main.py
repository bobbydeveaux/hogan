"""Hogan CLI — train, synthesise, inspect."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def _column_detail(col: dict) -> str:
    """Build the detail string for a column profile row."""
    col_type = col["type"]
    if col_type in ("numeric_continuous", "numeric_discrete") and col.get("min") is not None:
        return f"range: {col['min']:.2f} - {col['max']:.2f}"
    if col_type in ("categorical", "rating"):
        cats = col.get("categories", [])
        return ", ".join(str(c) for c in cats) if len(cats) <= 5 else f"{len(cats)} categories"
    if col_type in ("name", "identifier"):
        samples = col.get("sample_values", [])
        return ", ".join(str(s) for s in samples[:3]) + "..." if samples else ""
    return ""


def _show_profile(metadata: dict) -> None:
    """Render a Rich table from column metadata."""
    table = Table(title="Column Profile", show_lines=False)
    table.add_column("Column", style="cyan", min_width=30)
    table.add_column("Type", style="green", min_width=20)
    table.add_column("Unique", justify="right")
    table.add_column("Nulls", justify="right")
    table.add_column("Detail", style="dim")

    for col in metadata["columns"]:
        table.add_row(
            col["name"], col["type"],
            str(col["n_unique"]), f"{col['null_pct']}%",
            _column_detail(col),
        )
    console.print(table)


def _find_model_dir() -> Path:
    """Find the most recent model directory under .hogan/."""
    hogan_dir = Path(".hogan")
    if not hogan_dir.exists():
        console.print("[red]No .hogan/ directory found. Run 'hogan train' first.[/red]")
        raise SystemExit(1)
    subdirs = sorted(hogan_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    subdirs = [d for d in subdirs if d.is_dir() and (d / "model.pkl").exists()]
    if not subdirs:
        console.print("[red]No trained models found in .hogan/[/red]")
        raise SystemExit(1)
    return subdirs[0]


@click.group()
@click.version_option(version="0.1.0", prog_name="hogan")
def cli():
    """Hogan — GAN-based tabular data synthesiser."""


@cli.command()
@click.argument("input_csv", type=click.Path(exists=True, path_type=Path))
@click.option("--epochs", type=int, default=None, help="Training epochs (default: 300)")
@click.option("--batch-size", type=int, default=None, help="Batch size (default: 500)")
@click.option("--model-dir", type=click.Path(path_type=Path), default=None)
@click.option("--name", type=str, default=None, help="Model name")
@click.option("--sensitive", type=str, default=None, help="Comma-separated sensitive columns")
@click.option("--id-cols", type=str, default=None, help="Comma-separated identifier columns")
@click.option("--preview", is_flag=True, help="Profile only, don't train")
def train(
    input_csv: Path, epochs: int | None, batch_size: int | None,
    model_dir: Path | None, name: str | None,
    sensitive: str | None, id_cols: str | None, preview: bool,
):
    """Train a CTGAN model on a CSV dataset."""
    from hogan.core.profiler import profile
    from hogan.core.trainer import train as do_train

    console.print(f"[bold]Loading[/bold] {input_csv}")
    df = pd.read_csv(input_csv)
    console.print(f"  {len(df)} rows, {len(df.columns)} columns\n")

    sensitive_list = [s.strip() for s in sensitive.split(",")] if sensitive else None
    id_list = [s.strip() for s in id_cols.split(",")] if id_cols else None
    metadata = profile(df, sensitive_cols=sensitive_list, id_cols=id_list)
    _show_profile(metadata)

    if preview:
        console.print("\n[dim]Preview mode — no training performed.[/dim]")
        return

    model_path = do_train(
        df=df, metadata=metadata, epochs=epochs, batch_size=batch_size,
        model_dir=model_dir, model_name=name or input_csv.stem, verbose=True,
    )
    console.print(f"\n[bold green]Done![/bold green] Model saved to {model_path}/")


@cli.command()
@click.option("--model", "model_dir", type=click.Path(path_type=Path), default=None)
@click.option("-n", "n_rows", type=int, default=None, help="Number of rows")
@click.option("-o", "output", type=click.Path(path_type=Path), default=None)
@click.option("--format", "fmt", type=click.Choice(["csv", "parquet", "json"]), default="csv")
@click.option("--seed", type=int, default=None)
@click.option("--validate", is_flag=True, help="Run sanitiser")
@click.option("--training-csv", type=click.Path(exists=True, path_type=Path), default=None)
def synthesise(
    model_dir: Path | None, n_rows: int | None, output: Path | None,
    fmt: str, seed: int | None, validate: bool, training_csv: Path | None,
):
    """Generate synthetic rows from a trained model."""
    from hogan.core.synthesiser import synthesise as do_synthesise

    if model_dir is None:
        model_dir = _find_model_dir()

    console.print(f"[bold]Loading model from[/bold] {model_dir}/")
    original_df = pd.read_csv(training_csv) if training_csv else None

    synthetic_df = do_synthesise(
        model_dir=model_dir, n_rows=n_rows, seed=seed, original_df=original_df,
    )
    console.print(f"[bold green]Generated {len(synthetic_df)} rows[/bold green]")

    if validate:
        _run_validation(model_dir, synthetic_df, original_df)

    _write_output(synthetic_df, output, fmt)


def _run_validation(
    model_dir: Path, synthetic_df: pd.DataFrame, original_df: pd.DataFrame | None
) -> None:
    from hogan.core.profiler import load_metadata
    from hogan.core.sanitiser import sanitise

    if original_df is None:
        console.print("[yellow]Warning: --validate requires --training-csv.[/yellow]")
        return
    metadata = load_metadata(model_dir / "metadata.json")
    report = sanitise(synthetic_df, original_df, metadata)
    console.print()
    console.print(report.summary())


def _write_output(df: pd.DataFrame, output: Path | None, fmt: str) -> None:
    if output is None:
        click.echo(df.to_csv(index=False))
        return
    writers = {
        "csv": lambda: df.to_csv(output, index=False),
        "parquet": lambda: df.to_parquet(output, index=False),
        "json": lambda: df.to_json(output, orient="records", indent=2),
    }
    writers[fmt]()
    console.print(f"Written to {output}")


@cli.command()
@click.argument("synthetic_csv", type=click.Path(exists=True, path_type=Path))
@click.option("--against", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model", "model_dir", type=click.Path(path_type=Path), default=None)
def inspect(synthetic_csv: Path, against: Path, model_dir: Path | None):
    """Compare synthetic data against the original dataset."""
    from hogan.core.profiler import load_metadata, profile
    from hogan.core.sanitiser import sanitise

    console.print(f"[bold]Comparing[/bold] {synthetic_csv} against {against}\n")
    synthetic_df = pd.read_csv(synthetic_csv)
    real_df = pd.read_csv(against)

    metadata = load_metadata(model_dir / "metadata.json") if model_dir else profile(real_df)
    report = sanitise(synthetic_df, real_df, metadata)

    _show_inspect_report(report)


def _show_inspect_report(report: "PrivacyReport") -> None:  # noqa: F821
    from hogan.core.sanitiser import PrivacyReport as _PR  # noqa: F811

    table = Table(title="Comparison Report", show_lines=False)
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Result")
    table.add_column("Detail")

    for col in report.columns:
        style = "green" if col.passed else "red"
        status = "PASS" if col.passed else "FAIL"
        table.add_row(col.name, col.col_type, f"[{style}]{status}[/{style}]", col.detail)

    console.print(table)
    rv = report.row_vector_matches
    rv_status = "[green]PASS[/green]" if rv == 0 else "[red]WARN[/red]"
    console.print(f"\nRow-vector exact matches: {rv} ({rv_status})")
    overall = "[green]PASS[/green]" if report.overall_pass else "[red]FAIL[/red]"
    console.print(f"Overall: {overall}")
