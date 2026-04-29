use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnType {
    Continuous,
    Discrete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanInfo {
    pub dim: usize,
    pub activation: Activation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    Tanh,
    Softmax,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnTransformInfo {
    pub name: String,
    pub column_type: ColumnType,
    pub output_info: Vec<SpanInfo>,
    pub output_dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmParams {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub weights: Vec<f64>,
    pub valid_components: Vec<bool>,
    pub n_valid: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousTransform {
    pub name: String,
    pub gmm: GmmParams,
    pub n_components: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteTransform {
    pub name: String,
    pub categories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformMeta {
    pub continuous: Vec<ContinuousTransform>,
    pub discrete: Vec<DiscreteTransform>,
    pub column_order: Vec<String>,
    pub column_transform_info: Vec<ColumnTransformInfo>,
    pub output_dimensions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub embedding_dim: usize,
    pub generator_dim: Vec<usize>,
    pub discriminator_dim: Vec<usize>,
    pub generator_lr: f64,
    pub discriminator_lr: f64,
    pub generator_decay: f64,
    pub discriminator_decay: f64,
    pub batch_size: usize,
    pub discriminator_steps: usize,
    pub epochs: usize,
    pub pac: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            generator_dim: vec![256, 256],
            discriminator_dim: vec![256, 256],
            generator_lr: 2e-4,
            discriminator_lr: 2e-4,
            generator_decay: 1e-6,
            discriminator_decay: 1e-6,
            batch_size: 500,
            discriminator_steps: 1,
            epochs: 300,
            pac: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub config: TrainingConfig,
    pub transform_meta: TransformMeta,
    pub generator_weights: Vec<u8>,
}
