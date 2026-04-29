use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::networks::{Discriminator, Generator};
use crate::sampler::DataSampler;
use crate::types::*;

/// Apply activation functions to generator output.
fn apply_activate(data: &Tensor, output_info: &[ColumnTransformInfo]) -> Tensor {
    let mut parts: Vec<Tensor> = Vec::new();
    let mut st: i64 = 0;

    for info in output_info {
        for span in &info.output_info {
            let ed = st + span.dim as i64;
            let slice = data.narrow(1, st, span.dim as i64);
            match span.activation {
                Activation::Tanh => {
                    parts.push(slice.tanh());
                }
                Activation::Softmax => {
                    parts.push(gumbel_softmax(&slice, 0.2));
                }
            }
            st = ed;
        }
    }

    Tensor::cat(&parts, 1)
}

/// Gumbel-Softmax with temperature tau.
fn gumbel_softmax(logits: &Tensor, tau: f64) -> Tensor {
    // Retry up to 10 times if NaN occurs
    for _ in 0..10 {
        let gumbel_noise = -(-Tensor::rand_like(logits)
            .clamp(1e-20, 1.0)
            .log())
        .clamp(1e-20, 1.0)
        .log();
        let y = ((logits + gumbel_noise) / tau).softmax(1, Kind::Float);
        if !bool::try_from(y.isnan().any()).unwrap_or(true) {
            return y;
        }
    }
    // Fallback: regular softmax
    (logits / tau).softmax(1, Kind::Float)
}

/// Conditional loss: cross-entropy on discrete columns only.
fn cond_loss(
    data: &Tensor,
    c: &Tensor,
    m: &Tensor,
    output_info: &[ColumnTransformInfo],
) -> Tensor {
    let mut losses: Vec<Tensor> = Vec::new();
    let mut st: i64 = 0;
    let mut st_c: i64 = 0;

    for info in output_info {
        // Only pure discrete columns (single softmax span)
        let is_discrete = info.output_info.len() == 1
            && info.output_info[0].activation == Activation::Softmax;

        if !is_discrete {
            let total_dim: i64 = info.output_info.iter().map(|s| s.dim as i64).sum();
            st += total_dim;
            continue;
        }

        let dim = info.output_info[0].dim as i64;
        let logits = data.narrow(1, st, dim);
        let targets = c.narrow(1, st_c, dim).argmax(1, false);

        let loss = logits.cross_entropy_loss::<Tensor>(
            &targets,
            None,
            tch::Reduction::None,
            -100,
            0.0,
        );
        losses.push(loss);

        st += dim;
        st_c += dim;
    }

    if losses.is_empty() {
        return Tensor::zeros([], (Kind::Float, data.device()));
    }

    let stacked = Tensor::stack(&losses, 1);
    let batch_size = data.size()[0] as f64;
    (&stacked * m).sum(Kind::Float) / batch_size
}

pub struct CTGANTrainer {
    config: TrainingConfig,
    device: Device,
}

impl CTGANTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        let device = Device::Cpu; // CPU for POC, CUDA later
        Self { config, device }
    }

    pub fn train(
        &self,
        transformed_data: &[Vec<f64>],
        sampler: &DataSampler,
        transform_meta: &TransformMeta,
    ) -> Result<Vec<u8>> {
        let data_dim = transform_meta.output_dimensions as i64;
        let cond_dim = sampler.dim_cond_vec() as i64;
        let embedding_dim = self.config.embedding_dim as i64;
        let batch_size = self.config.batch_size;
        let pac = self.config.pac as i64;
        let gen_dim: Vec<i64> = self.config.generator_dim.iter().map(|&d| d as i64).collect();
        let disc_dim: Vec<i64> = self.config.discriminator_dim.iter().map(|&d| d as i64).collect();

        // Generator: input = noise + cond_vec
        let gen_vs = nn::VarStore::new(self.device);
        let generator = Generator::new(
            &gen_vs.root(),
            embedding_dim + cond_dim,
            &gen_dim,
            data_dim,
        );

        // Discriminator: input = data + cond_vec
        let disc_vs = nn::VarStore::new(self.device);
        let discriminator = Discriminator::new(
            &disc_vs.root(),
            data_dim + cond_dim,
            &disc_dim,
            pac,
        );

        // Optimizers: Adam with betas=(0.5, 0.9)
        let mut opt_g = nn::adam(0.5, 0.9, self.config.generator_decay)
            .build(&gen_vs, self.config.generator_lr)?;

        let mut opt_d = nn::adam(0.5, 0.9, self.config.discriminator_decay)
            .build(&disc_vs, self.config.discriminator_lr)?;

        // Convert training data to tensor
        let n_rows = transformed_data.len();
        let flat: Vec<f32> = transformed_data
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();
        let train_tensor = Tensor::from_slice(&flat)
            .view([n_rows as i64, data_dim])
            .to_device(self.device);

        let steps_per_epoch = (n_rows / batch_size).max(1);

        // Progress bar
        let pb = ProgressBar::new(self.config.epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} epochs ({eta})")
                .unwrap()
                .progress_chars("=>-"),
        );

        let output_info = &transform_meta.column_transform_info;

        for epoch in 0..self.config.epochs {
            for _step in 0..steps_per_epoch {
                // === DISCRIMINATOR TRAINING ===
                for _d_step in 0..self.config.discriminator_steps {
                    // Sample noise
                    let fakez = Tensor::randn(
                        [batch_size as i64, embedding_dim],
                        (Kind::Float, self.device),
                    );

                    // Sample conditional vector + real data
                    let (fakez, real_cat, fake_c1) = if sampler.has_discrete() {
                        let cv = sampler.sample_condvec(batch_size).unwrap();
                        let c1_tensor = self.vec2d_to_tensor(&cv.cond);
                        let fakez = Tensor::cat(&[&fakez, &c1_tensor], 1);

                        // Permuted indices for real data
                        let mut perm: Vec<usize> = (0..batch_size).collect();
                        perm.shuffle(&mut rand::thread_rng());

                        let perm_col: Vec<usize> = perm.iter().map(|&i| cv.col_ids[i]).collect();
                        let perm_opt: Vec<usize> = perm.iter().map(|&i| cv.opt_ids[i]).collect();
                        let real_rows = sampler.sample_data(
                            transformed_data, batch_size, &perm_col, &perm_opt,
                        );
                        let real_tensor = self.vec2d_to_tensor(&real_rows);

                        // Permuted conditional vector
                        let c2_rows: Vec<Vec<f64>> = perm.iter().map(|&i| cv.cond[i].clone()).collect();
                        let c2_tensor = self.vec2d_to_tensor(&c2_rows);
                        let real_cat = Tensor::cat(&[&real_tensor, &c2_tensor], 1);

                        (fakez, real_cat, Some(c1_tensor))
                    } else {
                        // No discrete columns — random sample
                        let indices: Vec<usize> = (0..batch_size)
                            .map(|_| rand::random::<usize>() % n_rows)
                            .collect();
                        let real_rows: Vec<Vec<f64>> =
                            indices.iter().map(|&i| transformed_data[i].clone()).collect();
                        let real_tensor = self.vec2d_to_tensor(&real_rows);
                        (fakez, real_tensor, None)
                    };

                    // Generate fake data
                    let fake_raw = generator.forward_t(&fakez, true);
                    let fakeact = apply_activate(&fake_raw, output_info);

                    let fake_cat = if let Some(ref c1) = fake_c1 {
                        Tensor::cat(&[&fakeact, c1], 1)
                    } else {
                        fakeact
                    };

                    // Discriminator scores
                    let y_fake = discriminator.forward_t(&fake_cat.detach(), true);
                    let y_real = discriminator.forward_t(&real_cat, true);

                    // Gradient penalty
                    let pen = discriminator.calc_gradient_penalty(
                        &real_cat, &fake_cat.detach(), self.device, 10.0,
                    );

                    // WGAN loss: maximize E[D(real)] - E[D(fake)]
                    let loss_d = -(y_real.mean(Kind::Float) - y_fake.mean(Kind::Float)) + pen;

                    opt_d.zero_grad();
                    loss_d.backward();
                    opt_d.step();
                }

                // === GENERATOR TRAINING ===
                let fakez = Tensor::randn(
                    [batch_size as i64, embedding_dim],
                    (Kind::Float, self.device),
                );

                let (fakez, g_c1, g_mask) = if sampler.has_discrete() {
                    let cv = sampler.sample_condvec(batch_size).unwrap();
                    let c1_tensor = self.vec2d_to_tensor(&cv.cond);
                    let m_tensor = self.vec2d_to_tensor(&cv.mask);
                    let fakez = Tensor::cat(&[&fakez, &c1_tensor], 1);
                    (fakez, Some(c1_tensor), Some(m_tensor))
                } else {
                    (fakez, None, None)
                };

                let fake_raw = generator.forward_t(&fakez, true);
                let fakeact = apply_activate(&fake_raw, output_info);

                let fake_cat = if let Some(ref c1) = g_c1 {
                    Tensor::cat(&[&fakeact, c1], 1)
                } else {
                    fakeact
                };

                let y_fake = discriminator.forward_t(&fake_cat, true);

                // Conditional loss
                let cross_entropy = if let (Some(ref c1), Some(ref m)) = (&g_c1, &g_mask) {
                    cond_loss(&fake_raw, c1, m, output_info)
                } else {
                    Tensor::zeros([], (Kind::Float, self.device))
                };

                let loss_g = -y_fake.mean(Kind::Float) + cross_entropy;

                opt_g.zero_grad();
                loss_g.backward();
                opt_g.step();
            }

            pb.inc(1);
        }

        pb.finish_with_message("Training complete");

        // Save generator weights
        let mut buf = Vec::new();
        gen_vs.save_to_stream(&mut buf)?;
        Ok(buf)
    }

    pub fn sample(
        &self,
        generator_weights: &[u8],
        n_rows: usize,
        transform_meta: &TransformMeta,
        sampler: &DataSampler,
    ) -> Result<Vec<Vec<f64>>> {
        let data_dim = transform_meta.output_dimensions as i64;
        let cond_dim = sampler.dim_cond_vec() as i64;
        let embedding_dim = self.config.embedding_dim as i64;
        let gen_dim: Vec<i64> = self.config.generator_dim.iter().map(|&d| d as i64).collect();
        let batch_size = self.config.batch_size;

        // Recreate generator and load weights
        let mut gen_vs = nn::VarStore::new(self.device);
        let generator = Generator::new(
            &gen_vs.root(),
            embedding_dim + cond_dim,
            &gen_dim,
            data_dim,
        );
        gen_vs.load_from_stream(std::io::Cursor::new(generator_weights))?;

        let output_info = &transform_meta.column_transform_info;
        let steps = (n_rows + batch_size - 1) / batch_size;
        let mut all_data: Vec<Vec<f64>> = Vec::new();

        for _step in 0..steps {
            let current_batch = batch_size.min(n_rows - all_data.len());
            let fakez = Tensor::randn(
                [current_batch as i64, embedding_dim],
                (Kind::Float, self.device),
            );

            let fakez = if sampler.has_discrete() {
                let condvec = sampler.sample_original_condvec(current_batch).unwrap();
                let c_tensor = self.vec2d_to_tensor(&condvec);
                Tensor::cat(&[&fakez, &c_tensor], 1)
            } else {
                fakez
            };

            let fake_raw = generator.forward_t(&fakez, false);
            let fakeact = apply_activate(&fake_raw, output_info);

            // Convert to Vec<Vec<f64>>
            let flat: Vec<f32> = Vec::try_from(fakeact.flatten(0, -1))?;
            for i in 0..current_batch {
                let start = i * data_dim as usize;
                let end = start + data_dim as usize;
                let row: Vec<f64> = flat[start..end].iter().map(|&v| v as f64).collect();
                all_data.push(row);
            }

            if all_data.len() >= n_rows {
                break;
            }
        }

        all_data.truncate(n_rows);
        Ok(all_data)
    }

    fn vec2d_to_tensor(&self, data: &[Vec<f64>]) -> Tensor {
        let rows = data.len() as i64;
        let cols = data[0].len() as i64;
        let flat: Vec<f32> = data.iter().flat_map(|row| row.iter().map(|&v| v as f32)).collect();
        Tensor::from_slice(&flat).view([rows, cols]).to_device(self.device)
    }
}
