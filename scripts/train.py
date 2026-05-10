import os
import jax
import jax.numpy as jnp
import optax
import e3nn_jax as e3nn
import pickle
from dotenv import load_dotenv
from tqdm import tqdm

# Custom project imports
from src.preprocessing.transform import TrajectoryTransformer
from src.preprocessing.residue_registry import AtomRegistry
from src.training.trainer import EquiJumpTrainer
from src.training.interpolants import linear_interpolant, sine_noise_schedule

# --- 1. SETUP & CONFIG ---
load_dotenv()
MD_DATA_PATH = os.getenv("MDSIM_DATASET_PATH")
RESIDUE_MAP = os.getenv("RESIDUE_MAP_PATH")

# File paths
xtc_path = "chignolin_trajectories/filtered/e1s1_chignolin_50ns_0/e1s1_chignolin_50ns_0-ADRIA_CHIG_ADAPTIVE_crystal_ss_contacts_50_chignolin_0-0-1-RND3469_9.filtered.xtc"
pdb_path = "chignolin_trajectories/filtered/filtered.pdb"

# Hyperparameters
LR = 5e-5           # Lowered for stability since we aren't clipping
COORD_SCALE = 0.1   # Normalizes Angstroms to near-unit range
EPOCHS = 10
MAX_SLOTS = 15
LATENT_IRREPS = "1x0e + 1x0o + 1x1e + 1x1o"

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# --- 2. NORMALIZED DATA PIPELINE ---
def process_frame(res_types, ca_pos, local_vecs):
    """Casts to JAX, centers at origin, and scales coordinates."""
    # 1. Cast to JAX arrays
    res_types = jnp.array(res_types)
    ca_pos = jnp.array(ca_pos)
    local_vecs = jnp.array(local_vecs)

    # 2. Normalize: Center the protein and scale
    # Subtracting the mean (centroid) removes global translation noise
    center = jnp.mean(ca_pos, axis=0)
    ca_pos_norm = (ca_pos - center) * COORD_SCALE
    local_vecs_norm = local_vecs * COORD_SCALE

    # 3. Wrap in Irreps
    R = e3nn.IrrepsArray("21x0e", jax.nn.one_hot(res_types, 21))
    P = e3nn.IrrepsArray("1x1o", ca_pos_norm)
    
    # Flatten local vectors: (N, 15, 3) -> (N, 45)
    v_flat = local_vecs_norm[:, :MAX_SLOTS, :].reshape(local_vecs_norm.shape[0], -1)
    V = e3nn.IrrepsArray(f"{MAX_SLOTS}x1o", v_flat)
    
    return R, P, V

def get_batch_iterator(cloud_obj):
    """Yields adjacent frames as training samples."""
    num_frames = len(cloud_obj.ca_positions)
    res_types = jnp.array(cloud_obj.residue_types)
    
    for t in range(num_frames - 1):
        R, P0, V0 = process_frame(res_types, cloud_obj.ca_positions[t], cloud_obj.local_vectors[t])
        _, P1, V1 = process_frame(res_types, cloud_obj.ca_positions[t+1], cloud_obj.local_vectors[t+1])
        
        yield {
            'residues': R,
            'X_init': (P0, V0),
            'X_end': (P1, V1)
        }

# --- 3. INITIALIZATION ---
input_irreps = f"1x0e + 21x0e + 1x1o + {MAX_SLOTS}x1o + {LATENT_IRREPS}"

trainer = EquiJumpTrainer(
    latent_irreps=LATENT_IRREPS,
    input_irreps=input_irreps,
    target_irreps_p="1x1o",
    target_irreps_v=f"{MAX_SLOTS}x1o",
    interpolant_fn=linear_interpolant,
    noise_fn=sine_noise_schedule,
    num_layers=2, 
    lr=LR
)

# Load data structures
registry = AtomRegistry(RESIDUE_MAP)
engine = TrajectoryTransformer(registry, max_slots=MAX_SLOTS)
cloud = engine.transform(
    pdb_path=os.path.join(MD_DATA_PATH, pdb_path), 
    xtc_path=os.path.join(MD_DATA_PATH, xtc_path)
)

# Setup Optimizer & JIT
rng = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(rng)

# Initialize parameters
sample_batch = next(get_batch_iterator(cloud))
params = trainer.init_params(k1, sample_batch)

# Explicitly define optimizer
optimizer = optax.adam(LR)
opt_state = optimizer.init(params)
trainer.optimizer = optimizer # Set trainer internal optimizer

@jax.jit
def train_step(p, opt, key, b):
    return trainer.step(p, opt, key, b)

# --- 4. MAIN TRAINING LOOP ---
print(f"Dataset Ready: {len(cloud.ca_positions)} frames.")

for epoch in range(EPOCHS):
    total_loss = 0.0
    steps = 0
    
    # Initialize generator for this epoch
    data_gen = get_batch_iterator(cloud)
    pbar = tqdm(data_gen, total=len(cloud.ca_positions)-1, desc=f"Epoch {epoch+1}")

    for batch in pbar:
        k2, step_key = jax.random.split(k2)
        
        params, opt_state, loss = train_step(params, opt_state, step_key, batch)
        
        # Stop if things explode
        if jnp.isnan(loss):
            print("\n[!] NaN Loss detected. Stopping.")
            exit()

        total_loss += float(loss)
        steps += 1
        
        if steps % 10 == 0:
            pbar.set_postfix({"loss": f"{loss:.4f}"})

    avg_loss = total_loss / steps
    print(f"Epoch {epoch+1} Finished | Avg Loss: {avg_loss:.6f}")

    # Save Checkpoint
    with open(f"checkpoints/equijump_epoch_{epoch+1}.pkl", "wb") as f:
        pickle.dump(params, f)

print("Training finished successfully.")