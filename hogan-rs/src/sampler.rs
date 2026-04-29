use rand::Rng;

use crate::types::*;

pub struct DataSampler {
    /// For each discrete column: for each category, list of row indices
    rid_by_cat_cols: Vec<Vec<Vec<usize>>>,
    /// Category probabilities (log-frequency) per discrete column
    category_probs: Vec<Vec<f64>>,
    /// Number of discrete columns
    n_discrete_columns: usize,
    /// Total number of categories across all discrete columns
    n_categories: usize,
    /// Category counts per discrete column
    n_categories_per_col: Vec<usize>,
}

pub struct CondVec {
    pub cond: Vec<Vec<f64>>,
    pub mask: Vec<Vec<f64>>,
    pub col_ids: Vec<usize>,
    pub opt_ids: Vec<usize>,
}

impl DataSampler {
    pub fn new(
        data: &[Vec<f64>],
        transform_info: &[ColumnTransformInfo],
    ) -> Self {
        // Find discrete columns and their positions in the transformed data
        let mut discrete_col_info: Vec<(usize, usize)> = Vec::new(); // (offset, n_categories)
        let mut offset = 0;

        for info in transform_info {
            if let ColumnType::Discrete = info.column_type {
                let n_cats = info.output_dimensions;
                discrete_col_info.push((offset, n_cats));
            }
            offset += info.output_dimensions;
        }

        let n_discrete = discrete_col_info.len();
        let n_categories: usize = discrete_col_info.iter().map(|(_, n)| n).sum();
        let n_cats_per_col: Vec<usize> = discrete_col_info.iter().map(|(_, n)| *n).collect();

        // Build row index lookup per category
        let mut rid_by_cat_cols: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n_discrete);
        let mut category_probs: Vec<Vec<f64>> = Vec::with_capacity(n_discrete);

        for (col_offset, n_cats) in &discrete_col_info {
            let mut cat_rows: Vec<Vec<usize>> = vec![vec![]; *n_cats];
            let mut cat_counts: Vec<f64> = vec![0.0; *n_cats];

            for (row_idx, row) in data.iter().enumerate() {
                // Find which category this row has (argmax of one-hot)
                let cat_slice = &row[*col_offset..*col_offset + *n_cats];
                let cat_idx = cat_slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                cat_rows[cat_idx].push(row_idx);
                cat_counts[cat_idx] += 1.0;
            }

            // Log-frequency probabilities
            let log_freqs: Vec<f64> = cat_counts.iter().map(|&c| (c + 1.0).ln()).collect();
            let sum: f64 = log_freqs.iter().sum();
            let probs: Vec<f64> = log_freqs.iter().map(|&f| f / sum).collect();

            rid_by_cat_cols.push(cat_rows);
            category_probs.push(probs);
        }

        Self {
            rid_by_cat_cols,
            category_probs,
            n_discrete_columns: n_discrete,
            n_categories,
            n_categories_per_col: n_cats_per_col,
        }
    }

    pub fn dim_cond_vec(&self) -> usize {
        self.n_categories
    }

    pub fn has_discrete(&self) -> bool {
        self.n_discrete_columns > 0
    }

    pub fn sample_condvec(&self, batch_size: usize) -> Option<CondVec> {
        if self.n_discrete_columns == 0 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut cond = vec![vec![0.0f64; self.n_categories]; batch_size];
        let mut mask = vec![vec![0.0f64; self.n_discrete_columns]; batch_size];
        let mut col_ids = vec![0usize; batch_size];
        let mut opt_ids = vec![0usize; batch_size];

        for i in 0..batch_size {
            // Pick a random discrete column
            let col = rng.gen_range(0..self.n_discrete_columns);
            col_ids[i] = col;
            mask[i][col] = 1.0;

            // Sample a category using log-frequency probabilities
            let probs = &self.category_probs[col];
            let u: f64 = rng.gen();
            let mut cumsum = 0.0;
            let mut selected_cat = 0;
            for (j, &p) in probs.iter().enumerate() {
                cumsum += p;
                if u <= cumsum {
                    selected_cat = j;
                    break;
                }
            }
            opt_ids[i] = selected_cat;

            // Set the one-hot in the conditional vector
            let global_offset: usize = self.n_categories_per_col[..col].iter().sum();
            cond[i][global_offset + selected_cat] = 1.0;
        }

        Some(CondVec {
            cond,
            mask,
            col_ids,
            opt_ids,
        })
    }

    pub fn sample_original_condvec(&self, batch_size: usize) -> Option<Vec<Vec<f64>>> {
        if self.n_discrete_columns == 0 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut cond = vec![vec![0.0f64; self.n_categories]; batch_size];

        for i in 0..batch_size {
            let col = rng.gen_range(0..self.n_discrete_columns);
            let n_cats = self.n_categories_per_col[col];

            // Pick a random row and use its actual category
            let total_rows: usize = self.rid_by_cat_cols[col].iter().map(|v| v.len()).sum();
            if total_rows == 0 {
                continue;
            }
            let random_row = rng.gen_range(0..total_rows);
            let mut cumulative = 0;
            let mut selected_cat = 0;
            for (cat, rows) in self.rid_by_cat_cols[col].iter().enumerate() {
                cumulative += rows.len();
                if random_row < cumulative {
                    selected_cat = cat;
                    break;
                }
            }

            let global_offset: usize = self.n_categories_per_col[..col].iter().sum();
            cond[i][global_offset + selected_cat] = 1.0;
        }

        Some(cond)
    }

    pub fn sample_data(
        &self,
        data: &[Vec<f64>],
        batch_size: usize,
        col_ids: &[usize],
        opt_ids: &[usize],
    ) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut result = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let col = col_ids[i];
            let opt = opt_ids[i];
            let rows = &self.rid_by_cat_cols[col][opt];

            if rows.is_empty() {
                // Fallback: random row
                let idx = rng.gen_range(0..data.len());
                result.push(data[idx].clone());
            } else {
                let idx = rows[rng.gen_range(0..rows.len())];
                result.push(data[idx].clone());
            }
        }

        result
    }
}
