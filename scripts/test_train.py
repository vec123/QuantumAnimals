
import os
from dotenv import load_dotenv
import jax
import optax
import e3nn_jax as e3nn
import jax.numpy as jnp

from src.preprocessing.transform import TrajectoryTransformer
from src.preprocessing.residue_registry import AtomRegistry
from src.visualizations.visualizer import CloudVisualizer
from src.training.trainer import EquiJumpTrainer
from src.training.interpolants import linear_interpolant, sine_noise_schedule


# Prepare the Data Batch
# take two adjacent frames from the 'cloud' object to represent Xt and Xt+1
def get_test_batch(cloud_obj):
    # 1. One-hot encode residues (N, 21)
    res_types = jnp.array(cloud_obj.residue_types)
    ca_pos = jnp.array(cloud_obj.ca_positions)
    local_vecs = jnp.array(cloud_obj.local_vectors)

    residue_one_hot = jax.nn.one_hot(res_types, 21)
    R = e3nn.IrrepsArray("21x0e", residue_one_hot)
    
    # 2. Position Anchors (N, 3)
    P0 = e3nn.IrrepsArray("1x1o", ca_pos[0])
    P1 = e3nn.IrrepsArray("1x1o", ca_pos[1])
    
    # 3. Local Vectors (N, 15, 3) -> (N, 45)
    # Your data has 15 slots, so we take the first 13.
    def prepare_vectors(v_frame):
        # Slice to N slots: (10, 15, 3) -> (10, N, 3)
        v_sliced = v_frame[:, :15, :]
        # Reshape to flatten vectors: (10, N, 3) -> (10, N*3)
        v_flat = v_sliced.reshape(v_sliced.shape[0], -1)
        return e3nn.IrrepsArray("15x1o", v_flat)

    V0 = prepare_vectors(local_vecs[0])
    V1 = prepare_vectors(local_vecs[1])
    
    return {
        'residues': R,
        'X_init': (P0, V0),
        'X_end': (P1, V1)
    }
load_dotenv()

project_path = os.getenv("PROJECT_PATH")
resiude_map = os.getenv("RESIDUE_MAP_PATH")
md_data_path = os.getenv("MDSIM_DATASET_PATH")
tc_data_path = os.getenv("TENSOR_CLOUD_STORAGE")
vtk_dir = os.getenv("VTK_DIR")

xtc_trajectory = "chignolin_trajectories/filtered/e1s1_chignolin_50ns_0\e1s1_chignolin_50ns_0-ADRIA_CHIG_ADAPTIVE_crystal_ss_contacts_50_chignolin_0-0-1-RND3469_9.filtered.xtc"
tc_trajectory = "chignolin_trajectories/filtered/e1s1_chignolin_50ns_0\e1s1_chignolin_50ns_0-ADRIA_CHIG_ADAPTIVE_crystal_ss_contacts_50_chignolin_0-0-1-RND3469_9.filtered.pkl"
pdb_file = "chignolin_trajectories/filtered/filtered.pdb"


registry = AtomRegistry(resiude_map)
engine = TrajectoryTransformer(registry, max_slots=15)

# Execute
cloud = engine.transform(pdb_path = os.path.join(md_data_path,pdb_file), xtc_path = os.path.join(md_data_path,xtc_trajectory))

# the output irreps 
# these are a design choice
latent_irreps = "1x0e + 1x0o + 1x1e + 1x1o"

# the input irreps of the drift and noise networks
# scalar for time, 21 scalar field for residual field, odd vector for node position, 13 odd vectors for heavy atoms vectors
input_irreps = "1x0e + 21x0e + 1x1o + 13x1o +" + latent_irreps

# the output irreps of the drift and noise networks
# odd vector for node position drift, 13 odd vectors for heavy atoms vecto drift
target_irreps_p = "1x1o "
target_irreps_v = "15x1o"

trainer = EquiJumpTrainer(
    latent_irreps=latent_irreps,
    input_irreps=input_irreps,
    target_irreps_p=target_irreps_p,
    target_irreps_v = target_irreps_v,
    interpolant_fn=linear_interpolant,
    noise_fn=sine_noise_schedule,
    num_layers=1,
    lr = 1e-6,
    verbose = False
)

# Set the optimizer (required for the .step() method)
trainer.optimizer = optax.adam(1e-4)

batch = get_test_batch(cloud)

# Initialization
rng = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(rng)

params = trainer.init_params(k1, batch)
opt_state = trainer.optimizer.init(params)

# Run 2 Updates
print(f"{'Step':<10} | {'Loss':<15} | {'Param Delta':<15}")
print("-" * 45)

curr_params, curr_opt_state = params, opt_state

for i in range(2000):
    k2, step_key = jax.random.split(k2)
    
    # Execute gradient step
    next_params, next_opt_state, loss = trainer.step(
        curr_params, curr_opt_state, step_key, batch
    )
    
    # Check if parameters actually changed
    # flatten the pytree to calculate the Euclidean distance between weights
    flat_old, _ = jax.flatten_util.ravel_pytree(curr_params)
    flat_new, _ = jax.flatten_util.ravel_pytree(next_params)
    delta = jnp.linalg.norm(flat_new - flat_old)
    
    print(f"{i+1:<10} | {float(loss):<15.6f} | {float(delta):<15.8e}")
    
    curr_params, curr_opt_state = next_params, next_opt_state