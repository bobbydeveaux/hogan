use tch::{nn, nn::Module, nn::ModuleT, Kind, Tensor};

/// Residual block: Linear -> BatchNorm -> ReLU, then CONCATENATE with input.
/// Dimensions grow: output_dim = input_dim + layer_dim
#[derive(Debug)]
pub struct Residual {
    fc: nn::Linear,
    bn: nn::BatchNorm,
}

impl Residual {
    pub fn new(vs: &nn::Path, input_dim: i64, output_dim: i64) -> Self {
        let fc = nn::linear(vs / "fc", input_dim, output_dim, Default::default());
        let bn = nn::batch_norm1d(vs / "bn", output_dim, Default::default());
        Self { fc, bn }
    }
}

impl nn::ModuleT for Residual {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let out = xs.apply(&self.fc);
        let out = out.apply_t(&self.bn, train);
        let out = out.relu();
        Tensor::cat(&[&out, xs], 1) // Concatenate, not add
    }
}

/// CTGAN Generator with concatenation-based residual blocks.
pub struct Generator {
    residuals: Vec<Residual>,
    final_linear: nn::Linear,
}

impl Generator {
    pub fn new(
        vs: &nn::Path,
        embedding_dim: i64,
        generator_dim: &[i64],
        data_dim: i64,
    ) -> Self {
        let mut residuals = Vec::new();
        let mut input_dim = embedding_dim;

        for (i, &dim) in generator_dim.iter().enumerate() {
            let res = Residual::new(&(vs / format!("res_{}", i)), input_dim, dim);
            residuals.push(res);
            input_dim += dim; // Concatenation grows the dimension
        }

        let final_linear = nn::linear(vs / "final", input_dim, data_dim, Default::default());

        Self {
            residuals,
            final_linear,
        }
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut out = xs.shallow_clone();
        for res in &self.residuals {
            out = res.forward_t(&out, train);
        }
        out.apply(&self.final_linear)
    }
}

/// CTGAN Discriminator with PacGAN grouping.
pub struct Discriminator {
    layers: Vec<nn::Linear>,
    pac: i64,
    pacdim: i64,
}

impl Discriminator {
    pub fn new(
        vs: &nn::Path,
        input_dim: i64,
        discriminator_dim: &[i64],
        pac: i64,
    ) -> Self {
        let pacdim = input_dim * pac;
        let mut layers = Vec::new();
        let mut dim = pacdim;

        for (i, &out_dim) in discriminator_dim.iter().enumerate() {
            layers.push(nn::linear(
                vs / format!("layer_{}", i),
                dim,
                out_dim,
                Default::default(),
            ));
            dim = out_dim;
        }

        // Final layer: output 1 score (no activation - Wasserstein critic)
        layers.push(nn::linear(vs / "final", dim, 1, Default::default()));

        Self { layers, pac, pacdim }
    }

    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch_size = xs.size()[0];
        let mut out = xs.view([batch_size / self.pac, self.pacdim]);

        for (i, layer) in self.layers.iter().enumerate() {
            out = out.apply(layer);
            // All layers except the last get LeakyReLU + Dropout
            if i < self.layers.len() - 1 {
                out = out.leaky_relu();
                out = out.dropout(0.5, train);
            }
        }
        out
    }

    pub fn calc_gradient_penalty(
        &self,
        real_data: &Tensor,
        fake_data: &Tensor,
        device: tch::Device,
        lambda: f64,
    ) -> Tensor {
        let batch_size = real_data.size()[0];
        let data_dim = real_data.size()[1];

        // Alpha: one per pac group, expanded to all samples in group
        let n_groups = batch_size / self.pac;
        let alpha = Tensor::rand([n_groups, 1, 1], (Kind::Float, device))
            .expand([n_groups, self.pac, data_dim], false)
            .contiguous()
            .view([batch_size, data_dim]);

        let interpolated = (&alpha * real_data) + (&(1.0 - &alpha) * fake_data);
        let interpolated = interpolated.set_requires_grad(true);

        let disc_out = self.forward_t(&interpolated, true);

        let gradients = Tensor::run_backward(
            &[&disc_out],
            &[&interpolated],
            true,  // retain_graph
            true,  // create_graph
        );
        let gradients = &gradients[0];

        let gradients_flat = gradients.view([n_groups, self.pac * data_dim]);
        let gradient_norm = gradients_flat.norm_scalaropt_dim(2, [1], false);
        let penalty = (gradient_norm - 1.0).pow_tensor_scalar(2).mean(Kind::Float) * lambda;

        penalty
    }
}
