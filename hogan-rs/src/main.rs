mod networks;
mod sampler;
mod trainer;
mod transformer;
mod types;

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use comfy_table::{Cell, Table};
use fake::faker::company::en::CompanyName;
use fake::Fake;

use sampler::DataSampler;
use trainer::CTGANTrainer;
use transformer::DataTransformer;
use types::*;

#[derive(Parser)]
#[command(name = "hogan", version = "0.1.0", about = "GAN-based tabular data synthesiser (Rust)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a CTGAN model on a CSV dataset
    Train {
        /// Input CSV file
        input: PathBuf,
        /// Training epochs
        #[arg(long, default_value_t = 150)]
        epochs: usize,
        /// Batch size
        #[arg(long, default_value_t = 500)]
        batch_size: usize,
        /// Model output directory
        #[arg(long, default_value = ".hogan-rs")]
        model_dir: PathBuf,
        /// Comma-separated discrete column names
        #[arg(long)]
        discrete: Option<String>,
        /// Preview mode (profile only)
        #[arg(long)]
        preview: bool,
    },
    /// Generate synthetic rows from a trained model
    Synthesise {
        /// Model directory
        #[arg(long, default_value = ".hogan-rs")]
        model_dir: PathBuf,
        /// Number of rows to generate
        #[arg(short, long)]
        n: Option<usize>,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn read_csv(path: &PathBuf) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut reader = csv::Reader::from_path(path)
        .context("Failed to open CSV file")?;

    let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();
    let mut data: Vec<Vec<String>> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row: Vec<String> = record.iter().map(|v| v.to_string()).collect();
        data.push(row);
    }

    Ok((headers, data))
}

fn detect_discrete_columns(headers: &[String], data: &[Vec<String>], explicit: &[String]) -> Vec<String> {
    let mut discrete = Vec::new();
    let n_rows = data.len();

    for (i, name) in headers.iter().enumerate() {
        if explicit.contains(name) {
            discrete.push(name.clone());
            continue;
        }

        // Heuristic: if column can't be parsed as float, or has low cardinality, it's discrete
        let mut n_numeric = 0;
        let mut unique_vals: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for row in data {
            let val = &row[i];
            if val.parse::<f64>().is_ok() {
                n_numeric += 1;
            }
            unique_vals.insert(val.as_str());
        }

        let numeric_ratio = n_numeric as f64 / n_rows as f64;
        let uniqueness = unique_vals.len() as f64 / n_rows as f64;

        // If mostly non-numeric, or very low cardinality relative to rows
        if numeric_ratio < 0.5 || (uniqueness < 0.05 && unique_vals.len() < 50) {
            discrete.push(name.clone());
        }
    }

    discrete
}

fn show_profile(headers: &[String], data: &[Vec<String>], discrete_cols: &[String]) {
    let mut table = Table::new();
    table.set_header(vec!["Column", "Type", "Unique", "Sample"]);

    for (i, name) in headers.iter().enumerate() {
        let is_discrete = discrete_cols.contains(name);
        let mut unique: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for row in data {
            unique.insert(row[i].as_str());
        }

        let col_type = if is_discrete { "discrete" } else { "continuous" };
        let sample: Vec<&str> = unique.iter().take(3).copied().collect();

        table.add_row(vec![
            Cell::new(name),
            Cell::new(col_type),
            Cell::new(unique.len()),
            Cell::new(sample.join(", ")),
        ]);
    }

    println!("{table}");
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            input,
            epochs,
            batch_size,
            model_dir,
            discrete,
            preview,
        } => {
            println!("Loading {}...", input.display());
            let (headers, data) = read_csv(&input)?;
            println!("  {} rows, {} columns", data.len(), headers.len());

            let explicit_discrete: Vec<String> = discrete
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().to_string())
                .collect();

            let discrete_cols = detect_discrete_columns(&headers, &data, &explicit_discrete);
            println!("  {} discrete, {} continuous columns",
                discrete_cols.len(), headers.len() - discrete_cols.len());

            show_profile(&headers, &data, &discrete_cols);

            if preview {
                println!("\nPreview mode — no training.");
                return Ok(());
            }

            // Transform data
            println!("\nFitting data transformer...");
            let start = Instant::now();
            let mut transformer = DataTransformer::new();
            transformer.fit(&headers, &data, &discrete_cols)?;
            println!("  Output dimensions: {}", transformer.meta.output_dimensions);
            println!("  Transform fit took {:.2}s", start.elapsed().as_secs_f64());

            println!("Transforming data...");
            let start = Instant::now();
            let transformed = transformer.transform(&data)?;
            println!("  Transform took {:.2}s", start.elapsed().as_secs_f64());

            // Build sampler
            let sampler = DataSampler::new(&transformed, &transformer.meta.column_transform_info);
            println!("  Conditional vector dim: {}", sampler.dim_cond_vec());

            // Train
            let config = TrainingConfig {
                epochs,
                batch_size,
                ..Default::default()
            };

            println!("\nTraining CTGAN ({} epochs, batch_size={})...", epochs, batch_size);
            let start = Instant::now();
            let trainer = CTGANTrainer::new(config.clone());
            let weights = trainer.train(&transformed, &sampler, &transformer.meta)?;
            let train_duration = start.elapsed();
            println!("Training completed in {:.2}s", train_duration.as_secs_f64());

            // Save model
            fs::create_dir_all(&model_dir)?;
            let artifact = ModelArtifact {
                config,
                transform_meta: transformer.meta,
                generator_weights: weights,
            };
            let serialized = rmp_serde::to_vec(&artifact)?;
            let model_path = model_dir.join("model.msgpack");
            fs::write(&model_path, &serialized)?;
            println!("Model saved to {} ({:.1} MB)",
                model_path.display(), serialized.len() as f64 / 1_048_576.0);
        }

        Commands::Synthesise {
            model_dir,
            n,
            output,
        } => {
            let model_path = model_dir.join("model.msgpack");
            println!("Loading model from {}...", model_path.display());
            let serialized = fs::read(&model_path)
                .context("Failed to read model file. Run 'hogan train' first.")?;
            let artifact: ModelArtifact = rmp_serde::from_slice(&serialized)?;

            let n_rows = n.unwrap_or(artifact.transform_meta.column_transform_info.iter()
                .map(|_| 10000).next().unwrap_or(10000));

            // Reconstruct transformer for inverse transform
            let transformer = DataTransformer {
                meta: artifact.transform_meta.clone(),
                ..DataTransformer::new()
            };

            // We need a sampler for conditional vectors during generation
            // For sampling, we build a minimal sampler from the transform metadata
            let dummy_data = vec![vec![0.0; artifact.transform_meta.output_dimensions]; 1];
            let sampler = DataSampler::new(&dummy_data, &artifact.transform_meta.column_transform_info);

            println!("Generating {} rows...", n_rows);
            let start = Instant::now();
            let trainer = CTGANTrainer::new(artifact.config);
            let generated = trainer.sample(
                &artifact.generator_weights,
                n_rows,
                &artifact.transform_meta,
                &sampler,
            )?;
            let gen_duration = start.elapsed();

            // Inverse transform
            let rows = transformer.inverse_transform(&generated)?;
            println!("Generated {} rows in {:.3}s ({:.0} rows/sec)",
                rows.len(), gen_duration.as_secs_f64(),
                rows.len() as f64 / gen_duration.as_secs_f64());

            // Get column names
            let col_names: Vec<String> = artifact.transform_meta.column_transform_info
                .iter()
                .map(|c| c.name.clone())
                .collect();

            // Output
            if let Some(out_path) = output {
                let mut wtr = csv::Writer::from_path(&out_path)?;
                wtr.write_record(&col_names)?;
                for row in &rows {
                    wtr.write_record(row)?;
                }
                wtr.flush()?;
                println!("Written to {}", out_path.display());
            } else {
                // Print header + first 5 rows
                let mut table = Table::new();
                table.set_header(col_names.iter().take(8).map(|n| Cell::new(n)));
                for row in rows.iter().take(5) {
                    table.add_row(row.iter().take(8).map(|v| Cell::new(v)));
                }
                println!("{table}");
            }
        }
    }

    Ok(())
}
