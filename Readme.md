# QuantumAnimals: Equivariant Deep Learning for Molecular Dynamics

An implementation of equivariant neural networks for learning molecular dynamics using geometric deep learning and stochastic interpolants. This project implements the EquiJump framework for protein trajectory modeling and prediction.

## Overview

QuantumAnimals combines:
- **Equivariant Neural Networks (E3NN)**: Geometric neural networks that respect SE(3) symmetry
- **Tensor Cloud Representations**: Efficient residue-based encoding of protein structures
- **Stochastic Interpolants**: Principled approach to modeling continuous molecular dynamics
- **JAX & Flax**: High-performance functional programming for ML research

The framework learns to model protein dynamics by:
1. Encoding protein conformations as tensor clouds (position + equivariant features)
2. Predicting geometric drift and noise through equivariant networks
3. Simulating dynamics via stochastic interpolation endpoints

## Key Features

- **SE(3)-Equivariant Architecture**: Fully equivariant to rotations and translations
- **Tensor Cloud Representation**: Efficient encoding using C-alpha positions + local atomic features
- **Three Core Mechanisms**:
  - **Self-Interaction**: Intra-residue feature updates via tensor products
  - **Spatial Convolution**: Message passing between residues
  - **Full Model**: Combined architecture for end-to-end predictions
- **Stochastic Interpolant Framework**: Learning endpoint-conditioned dynamics
- **Multiple Output Targets**: Position and feature drift/noise predictions
- **JAX Integration**: Composable, differentiable transformations

## Architecture

### Data Representation

**Tensor Cloud**: Set of tuples {(V_i, P_i)} for each residue:
- **V_i**: 13×3 matrix of L=1 irreducible representations (relative atomic positions)
- **P_i**: 3D position of C-alpha atom
- **R_i**: One-hot encoded residue label (amino acid type)

```
Residue i = {R_i (one-hot), P_i (position), V_i (local structure)}
                                ↓
                          Canonical Heavy Atoms
                          (13 features max, padded)
```

### Network Modules

#### 1. Self-Interaction Layer
Updates tensor cloud features independently via tensor products:

```
V_i_out = MLP_Gate(features) ⊙ Linear(V_i ⊗ V_i + V_i)
```

- Computes V_i ⊗ V_i (quadratic feature interactions)
- Filters by degree (l_max)
- Gated mechanism for feature modulation
- Preserves equivariance through tensor products

#### 2. Spatial Convolution Layer
Message passing between residues within radius:

```
For each residue i:
  aggregated = SUM over neighbors j (V_j, relative_position)
  V_i_out = self_interaction(V_i) + Linear(aggregated)
```

- Radius-based neighborhood aggregation
- Equivariant aggregation operations
- Residual connections for stable training

#### 3. Full EquiJump Model
Combines mechanisms for complete dynamics prediction:

```
Input: Tensor Cloud at time t
  ↓
[Self-Interaction + Spatial Conv] × N layers
  ↓
Output: Predicted geometric features
  ↓
Added to tensor cloud at interpolant time τ
  ↓
Four Networks (position drift, position noise, feature drift, feature noise)
  ↓
Endpoints for stochastic interpolant dynamics
```

## Installation

### Prerequisites
- Python 3.9+
- JAX 0.3.0+
- CUDA 11.0+ (for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
cd QuantumAnimals
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install jax jaxlib flax optax haiku jraph
pip install e3nn-jax MDAnalysis mdtraj
pip install numpy jupyter ipykernel pyevtk python-dotenv
pip install nglview  # For interactive visualization
```

4. **GPU Setup (Optional)**
```bash
pip install --upgrade jax jaxlib==0.3.0 -c jax-releases
pip install jax[cuda11_cudnn82]  # Adjust CUDA version as needed
```

5. **Configure environment**
Create `.env` file in project root:
```bash
DATASET_PATH=/path/to/molecular/datasets
DATA_OUTPUT_DIR=./data/processed
CHECKPOINT_DIR=./checkpoints
```

## Quick Start

### Explore Data Representation

```python
# Notebook: notebooks/xtc_visualize.ipynb
# Visualize molecular dynamics trajectory
# Understand tensor cloud encoding
# Inspect residue representations
```

### Load Molecular Data

```python
from src.preprocessing.trajectory_loader import TrajectoryLoader

loader = TrajectoryLoader()
trajectory = loader.load_xtc(
    topology_file="protein.pdb",
    trajectory_file="trajectory.xtc"
)
```

### Create Tensor Cloud

```python
from src.preprocessing.tensor_cloud import TensorCloudBuilder

builder = TensorCloudBuilder(l_max=1)
tensor_cloud = builder.from_trajectory(
    trajectory,
    residue_selection="protein",
    heavy_atoms_only=True
)
```

### Train Model

```bash
python scripts/train.py --config conf/config.yaml
```

## Configuration

### Configuration Structure

`conf/config.yaml`:
```yaml
data:
  trajectory_dir: "./data/trajectories"
  output_dir: "./data/processed"
  subseq_len: 50
  stride: 10
  val_split: 0.2
  test_split: 0.1

model:
  l_max: 1                    # Max irrep degree
  num_layers: 3               # Spatial conv layers
  hidden_dim: 64
  radius: 10.0                # Neighborhood radius (Å)
  
training:
  learning_rate: 1e-3
  batch_size: 32
  num_epochs: 100
  checkpoint_interval: 10
  device: "cuda"
  precision: "float32"
```

### Model Configurations

Pre-defined configs in `conf/model/`:
- `small.yaml`: 1-layer model for quick testing
- `medium.yaml`: 3-layer production model
- `large.yaml`: 6-layer for high-accuracy predictions

Override from command line:
```bash
python scripts/train.py model=large data.batch_size=64
```

## Project Structure

```
QuantumAnimals/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .env                             # Environment configuration
├── checkpoints/                     # Saved model weights
├── conf/
│   ├── config.yaml                  # Main configuration
│   ├── data/
│   │   └── default.yaml             # Data pipeline config
│   └── model/
│       ├── small.yaml
│       ├── medium.yaml
│       └── large.yaml
├── data/
│   ├── raw/                         # Original trajectories
│   └── processed/                   # Tensor clouds, preprocessed
├── images/                          # Documentation images
│   ├── representation/              # Data encoding visualizations
│   └── computing/                   # Architecture diagrams
├── notebooks/
│   └── xtc_visualize.ipynb          # Interactive exploration
├── scripts/
│   ├── train.py                     # Training entry point
│   ├── evaluate.py                  # Model evaluation
│   ├── generate_trajectories.py     # Generate predictions
│   └── preprocess_data.py           # Data pipeline
├── src/
│   ├── __init__.py
│   ├── io/                          # File I/O utilities
│   ├── modules/
│   │   ├── geometric_models.py      # Core E3NN modules
│   │   ├── LayerNorm.py             # Equivariant normalization
│   │   └── __init__.py
│   ├── preprocessing/               # Data preparation
│   │   ├── trajectory_loader.py
│   │   ├── tensor_cloud.py
│   │   └── __init__.py
│   ├── training/                    # Training utilities
│   │   ├── engine.py                # Training loop
│   │   ├── losses.py                # Loss functions
│   │   └── __init__.py
│   ├── visualizations/              # Plotting utilities
│   │   ├── trajectory_viz.py
│   │   ├── feature_viz.py
│   │   └── __init__.py
│   └── tests/                       # Unit tests
│       ├── test_modules.py
│       ├── test_preprocessing.py
│       └── __init__.py
├── data_visualization/              # Generated visualizations
└── old/                             # Archive
```

## Core Components

### Self-Interaction Module
```python
from src.modules.geometric_models import SelfInteraction

si = SelfInteraction(target_irreps="0e + 1o + 2e")
output = si(node_features)  # Computes V_i ⊗ V_i
```

**Features:**
- Tensor product of features with themselves
- Degree filtering (l_max)
- Gated mechanism for modulation
- Equivariance-preserving

### Spatial Convolution
```python
from src.modules.geometric_models import SpatialConvolution

conv = SpatialConvolution(
    target_irreps="0e + 1o",
    radius=10.0
)
output = conv(node_features, positions, graph)
```

**Features:**
- Neighbor aggregation within radius
- Equivariant message passing
- Relative position encoding

### Full Model
```python
from src.modules.geometric_models import EquiJumpNetwork

model = EquiJumpNetwork(
    num_layers=3,
    target_irreps="0e + 1o + 2e",
    l_max=1
)
```

## Training

### Basic Training Loop

```python
from src.training.engine import Trainer
from src.preprocessing.tensor_cloud import load_data

train_loader, val_loader = load_data(config)
model = EquiJumpNetwork(config)
trainer = Trainer(model, config)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    checkpoint_dir="./checkpoints"
)
```

### Loss Functions

| Component | Purpose |
|---|---|
| **Reconstruction Loss** | Position and feature prediction accuracy |
| **Drift Loss** | Predicted drift accuracy |
| **Noise Loss** | Predicted noise (score matching) |
| **Regularization** | Feature norm constraints |

### Training Metrics

- **train_loss**: Combined training loss
- **val_loss**: Validation loss
- **rmse_position**: Position prediction RMSE (Å)
- **rmse_features**: Feature prediction RMSE
- **drift_mae**: Mean absolute drift error
- **noise_mae**: Mean absolute noise error

## Inference & Prediction

### Generate Trajectory

```python
from src.training.engine import Trainer

trainer = Trainer.load(checkpoint_path="checkpoints/model_best.ckpt")

# Initial frame
initial_frame = tensor_cloud_sequence[0]

# Predict next N frames
predictions = trainer.predict(
    initial_state=initial_frame,
    num_steps=100,
    dt=0.01  # Timestep in ps
)

# Save trajectory
trainer.save_trajectory(predictions, output_path="prediction.xtc")
```

### Visualization

```python
from src.visualizations.trajectory_viz import TrajectoryVisualizer

viz = TrajectoryVisualizer()

# Compare predicted vs actual
viz.compare_trajectories(
    actual=actual_trajectory,
    predicted=predicted_trajectory,
    output_file="comparison.html"
)

# Feature evolution
viz.plot_feature_evolution(tensor_cloud_sequence)
```

## Stochastic Interpolants Framework

The model learns dynamics as endpoint-conditioned interpolation:

```
Interpolant: γ(τ) x_0 + (1 - γ(τ)) x_1
```

Where:
- **x_0**: Starting configuration (time t)
- **x_1**: Ending configuration (time t+Δt)
- **τ**: Interpolation parameter [0, 1]
- **γ(τ)**: Schedule (linear, polynomial, etc.)

**Network Outputs:**
- **μ_drift**: Position velocity estimate
- **σ_noise**: Uncertainty/noise estimate
- **θ_drift**: Feature drift components
- **ζ_noise**: Feature noise components

## Performance Tips

1. **Batch Size**: Larger batches (64-128) improve stability
2. **Learning Rate**: Start with 1e-3, decay by 0.5 every 20 epochs
3. **Data Augmentation**: Random rotations preserve equivariance
4. **GPU Memory**: Use mixed precision (float16/float32) for large proteins
5. **Checkpointing**: Save best model by validation loss
6. **Early Stopping**: Monitor validation loss, stop if no improvement

## Advanced Usage

### Custom Residue Selection

```python
builder = TensorCloudBuilder()
tensor_cloud = builder.from_trajectory(
    trajectory,
    residue_selection="residue 10 to 100 and not HOH",  # MDAnalysis syntax
    include_hydrogens=False
)
```

### Variable Output Targets

```yaml
model:
  outputs:
    - name: "position_drift"
      irreps: "1o"  # Vectors only
    - name: "position_noise"
      irreps: "0e"  # Scalars only
    - name: "feature_drift"
      irreps: "0e + 1o + 2e"
```

### Multi-Head Predictions

```python
# Predict multiple timeframes simultaneously
model.predict(
    initial_state=frame,
    forecast_horizons=[1, 5, 10, 20],  # Steps ahead
    ensemble_size=10  # Averaging
)
```

## Validation & Evaluation

### Metrics Suite

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/model_best.ckpt \
  --test_data data/processed/test.h5 \
  --metrics "rmse,mae,correlation"
```

### Trajectory Quality Assessment

- **RMSE**: Position prediction accuracy
- **Correlation**: Feature correlation over time
- **Stability**: Long-term trajectory divergence
- **Energy Conservation**: Physical plausibility (if available)

## Extension Points

### Adding New Irreps

```python
class CustomModel(EquiJumpNetwork):
    def __init__(self, config):
        super().__init__(config)
        # Higher degree irreps
        self.high_order = SelfInteraction("0e + 1o + 2e + 3o")
```

### Custom Loss Functions

```python
from src.training.losses import BaseLoss

class CustomLoss(BaseLoss):
    def compute(self, pred, target):
        # Implement custom loss
        return loss_value
```

### New Representations

Extend `src/preprocessing/tensor_cloud.py` for:
- All-atom representations
- Coarse-grained models
- Non-protein molecules

## References

- **E3NN**: [Geiger & Smidt, "e3nn: Euclidean Neural Networks"](https://github.com/e3nn/e3nn-jax)
- **EquiJump**: [Midgley et al., "Equivariant Diffusion for Protein Sequences"](https://arxiv.org/abs/2301.13802)
- **Geometric DL**: [Bronstein et al., "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"](https://arxiv.org/abs/2104.13478)
- **Stochastic Interpolants**: [Albergo et al., "Stochastic Interpolants"](https://arxiv.org/abs/2303.08797)
- **JAX**: [Bradbury et al., "JAX: composable transformations of Python+NumPy programs"](https://jax.readthedocs.io/)

## Citation

If you use QuantumAnimals in your research, please cite:

```bibtex
@software{quantum_animals,
  title={QuantumAnimals: Equivariant Deep Learning for Molecular Dynamics},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/QuantumAnimals}
}
```

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Areas for improvement:
- Additional molecular representations
- Performance optimizations for large proteins
- New loss functions for improved training
- Documentation and examples
- Benchmark datasets

Please submit issues and pull requests.

## Contact

For questions about the project, open an issue on GitHub or contact the maintainers.

---

**Status**: Active Development  
**Last Updated**: May 2024  
**Framework Versions**: JAX 0.3+, E3NN-JAX, Flax 0.5+
