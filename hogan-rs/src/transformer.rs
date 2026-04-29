use anyhow::Result;
use rand::Rng;

use crate::types::*;

const MAX_CLUSTERS: usize = 10;
const WEIGHT_THRESHOLD: f64 = 0.005;
const EM_ITERATIONS: usize = 100;
const EM_TOLERANCE: f64 = 1e-3;

/// Simple 1D Gaussian Mixture Model via EM algorithm.
/// Returns (means, stds, weights) for each component.
fn fit_gmm_1d(data: &[f64], k: usize, max_iter: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    let k = k.min(n);
    if k == 0 || n == 0 {
        return (vec![], vec![], vec![]);
    }

    let mut rng = rand::thread_rng();

    // Initialize: pick k data points as means, uniform weights, empirical std
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut means: Vec<f64> = (0..k)
        .map(|i| sorted[i * n / k.max(1)])
        .collect();
    let data_std = {
        let mean = data.iter().sum::<f64>() / n as f64;
        (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt().max(1e-6)
    };
    let mut stds: Vec<f64> = vec![data_std; k];
    let mut weights: Vec<f64> = vec![1.0 / k as f64; k];

    let mut responsibilities = vec![vec![0.0f64; k]; n];

    for _iter in 0..max_iter {
        // E-step: compute responsibilities
        for i in 0..n {
            let mut total = 0.0f64;
            for j in 0..k {
                let diff = data[i] - means[j];
                let exp = (-0.5 * diff * diff / (stds[j] * stds[j])).exp();
                let r = weights[j] * exp / (stds[j] * (2.0 * std::f64::consts::PI).sqrt());
                responsibilities[i][j] = r.max(1e-300);
                total += responsibilities[i][j];
            }
            if total > 0.0 {
                for j in 0..k {
                    responsibilities[i][j] /= total;
                }
            }
        }

        // M-step: update means, stds, weights
        let mut new_means = vec![0.0; k];
        let mut new_stds = vec![0.0; k];
        let mut new_weights = vec![0.0; k];

        for j in 0..k {
            let nk: f64 = responsibilities.iter().map(|r| r[j]).sum();
            if nk < 1e-10 {
                new_means[j] = means[j];
                new_stds[j] = stds[j];
                new_weights[j] = 0.0;
                continue;
            }

            new_means[j] = responsibilities.iter()
                .zip(data.iter())
                .map(|(r, &x)| r[j] * x)
                .sum::<f64>() / nk;

            let var: f64 = responsibilities.iter()
                .zip(data.iter())
                .map(|(r, &x)| r[j] * (x - new_means[j]).powi(2))
                .sum::<f64>() / nk;

            new_stds[j] = var.sqrt().max(1e-6);
            new_weights[j] = nk / n as f64;
        }

        // Check convergence
        let mean_shift: f64 = means.iter().zip(new_means.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>() / k as f64;

        means = new_means;
        stds = new_stds;
        weights = new_weights;

        if mean_shift < EM_TOLERANCE {
            break;
        }
    }

    (means, stds, weights)
}

pub struct DataTransformer {
    pub meta: TransformMeta,
    pub raw_data: Vec<Vec<String>>,
    pub headers: Vec<String>,
}

impl DataTransformer {
    pub fn new() -> Self {
        Self {
            meta: TransformMeta {
                continuous: vec![],
                discrete: vec![],
                column_order: vec![],
                column_transform_info: vec![],
                output_dimensions: 0,
            },
            raw_data: vec![],
            headers: vec![],
        }
    }

    pub fn fit(&mut self, headers: &[String], data: &[Vec<String>], discrete_columns: &[String]) -> Result<()> {
        self.headers = headers.to_vec();
        self.raw_data = data.to_vec();
        self.meta.column_order = headers.to_vec();
        self.meta.column_transform_info.clear();
        self.meta.continuous.clear();
        self.meta.discrete.clear();
        self.meta.output_dimensions = 0;

        for (col_idx, col_name) in headers.iter().enumerate() {
            let is_discrete = discrete_columns.contains(col_name);
            let col_values: Vec<&str> = data.iter().map(|row| row[col_idx].as_str()).collect();

            if is_discrete {
                self.fit_discrete(col_name, &col_values);
            } else {
                self.fit_continuous(col_name, &col_values);
            }
        }

        Ok(())
    }

    fn fit_discrete(&mut self, name: &str, values: &[&str]) {
        let mut categories: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        categories.sort();
        categories.dedup();
        let n_cats = categories.len();

        let info = ColumnTransformInfo {
            name: name.to_string(),
            column_type: ColumnType::Discrete,
            output_info: vec![SpanInfo { dim: n_cats, activation: Activation::Softmax }],
            output_dimensions: n_cats,
        };

        self.meta.discrete.push(DiscreteTransform { name: name.to_string(), categories });
        self.meta.output_dimensions += n_cats;
        self.meta.column_transform_info.push(info);
    }

    fn fit_continuous(&mut self, name: &str, values: &[&str]) {
        let numeric: Vec<f64> = values
            .iter()
            .filter_map(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .collect();

        if numeric.is_empty() {
            self.fit_discrete(name, values);
            return;
        }

        let n_clusters = MAX_CLUSTERS.min(numeric.len());
        let (gmm_means, gmm_stds, gmm_weights) = fit_gmm_1d(&numeric, n_clusters, EM_ITERATIONS);

        let mut valid = Vec::new();
        let mut n_valid = 0usize;
        for &w in &gmm_weights {
            let is_valid = w >= WEIGHT_THRESHOLD;
            valid.push(is_valid);
            if is_valid { n_valid += 1; }
        }

        if n_valid == 0 {
            let best = gmm_weights.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            valid[best] = true;
            n_valid = 1;
        }

        let info = ColumnTransformInfo {
            name: name.to_string(),
            column_type: ColumnType::Continuous,
            output_info: vec![
                SpanInfo { dim: 1, activation: Activation::Tanh },
                SpanInfo { dim: n_valid, activation: Activation::Softmax },
            ],
            output_dimensions: 1 + n_valid,
        };

        self.meta.continuous.push(ContinuousTransform {
            name: name.to_string(),
            gmm: GmmParams { means: gmm_means, stds: gmm_stds, weights: gmm_weights, valid_components: valid, n_valid },
            n_components: n_clusters,
        });
        self.meta.output_dimensions += 1 + n_valid;
        self.meta.column_transform_info.push(info);
    }

    pub fn transform(&self, data: &[Vec<String>]) -> Result<Vec<Vec<f64>>> {
        let mut result: Vec<Vec<f64>> = vec![vec![]; data.len()];
        let mut rng = rand::thread_rng();
        let mut cont_idx = 0;
        let mut disc_idx = 0;

        for info in &self.meta.column_transform_info {
            let col_pos = self.headers.iter().position(|h| h == &info.name).unwrap();

            match info.column_type {
                ColumnType::Continuous => {
                    let ct = &self.meta.continuous[cont_idx];
                    cont_idx += 1;
                    let gmm = &ct.gmm;
                    let valid_indices: Vec<usize> = gmm.valid_components.iter()
                        .enumerate().filter(|(_, &v)| v).map(|(i, _)| i).collect();

                    for (row_idx, row) in data.iter().enumerate() {
                        let val: f64 = row[col_pos].parse().unwrap_or(0.0);
                        let mut probs: Vec<f64> = Vec::with_capacity(gmm.n_valid);
                        let mut norm_vals: Vec<f64> = Vec::with_capacity(gmm.n_valid);

                        for &k in &valid_indices {
                            let normalized = (val - gmm.means[k]) / (4.0 * gmm.stds[k]);
                            norm_vals.push(normalized.clamp(-0.99, 0.99));
                            let diff = val - gmm.means[k];
                            let prob = (-0.5 * diff * diff / (gmm.stds[k] * gmm.stds[k])).exp();
                            probs.push(prob + 1e-6);
                        }

                        let sum: f64 = probs.iter().sum();
                        for p in probs.iter_mut() { *p /= sum; }

                        let u: f64 = rng.gen();
                        let mut cumsum = 0.0;
                        let mut selected = 0;
                        for (i, &p) in probs.iter().enumerate() {
                            cumsum += p;
                            if u <= cumsum { selected = i; break; }
                        }

                        result[row_idx].push(norm_vals[selected]);
                        for i in 0..gmm.n_valid {
                            result[row_idx].push(if i == selected { 1.0 } else { 0.0 });
                        }
                    }
                }
                ColumnType::Discrete => {
                    let dt = &self.meta.discrete[disc_idx];
                    disc_idx += 1;
                    for (row_idx, row) in data.iter().enumerate() {
                        let val = &row[col_pos];
                        for cat in &dt.categories {
                            result[row_idx].push(if val == cat { 1.0 } else { 0.0 });
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    pub fn inverse_transform(&self, data: &[Vec<f64>]) -> Result<Vec<Vec<String>>> {
        let n_rows = data.len();
        let mut result: Vec<Vec<String>> = vec![vec![]; n_rows];
        let mut col_offset = 0;
        let mut cont_idx = 0;
        let mut disc_idx = 0;

        for info in &self.meta.column_transform_info {
            match info.column_type {
                ColumnType::Continuous => {
                    let ct = &self.meta.continuous[cont_idx];
                    cont_idx += 1;
                    let gmm = &ct.gmm;
                    let valid_indices: Vec<usize> = gmm.valid_components.iter()
                        .enumerate().filter(|(_, &v)| v).map(|(i, _)| i).collect();

                    for row_idx in 0..n_rows {
                        let alpha = data[row_idx][col_offset];
                        let component_probs = &data[row_idx][col_offset + 1..col_offset + 1 + gmm.n_valid];
                        let selected_local = component_probs.iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, _)| i).unwrap_or(0);
                        let k = valid_indices[selected_local];
                        let original = alpha * 4.0 * gmm.stds[k] + gmm.means[k];
                        result[row_idx].push(format!("{:.4}", original));
                    }
                    col_offset += 1 + gmm.n_valid;
                }
                ColumnType::Discrete => {
                    let dt = &self.meta.discrete[disc_idx];
                    disc_idx += 1;
                    let n_cats = dt.categories.len();
                    for row_idx in 0..n_rows {
                        let probs = &data[row_idx][col_offset..col_offset + n_cats];
                        let selected = probs.iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, _)| i).unwrap_or(0);
                        result[row_idx].push(dt.categories[selected].clone());
                    }
                    col_offset += n_cats;
                }
            }
        }
        Ok(result)
    }
}
