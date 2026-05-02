import pickle
import jax.numpy as jnp
import jax
import jraph
import haiku as hk
import e3nn_jax as e3nn
from geometric_models import EquiJumpDeepNetwork

def prepare_data_for_model(data, frame_idx=0):
    pos = jnp.array(data['P_i'][frame_idx]) 
    cloud_raw = jnp.array(data['V_ij'][frame_idx]) 
    res_raw = jnp.array(data['R_i']) 
    num_nodes = pos.shape[0]

    # One-Hot Encoding
    res_indices = jnp.where(res_raw == -1, 20, res_raw)
    res_one_hot = jax.nn.one_hot(res_indices, num_classes=21)

    # Degree logic is now handled by the graph connectivity in the model, 
    # but if you want it as a node feature, keep it here:
    # (Optional: If the model builds the graph, degree can be computed inside too)
    
    res_irreps = e3nn.IrrepsArray("21x0e", res_one_hot)
    cloud_flattened = cloud_raw.reshape(num_nodes, -1)
    cloud_irreps = e3nn.IrrepsArray("13x1o", cloud_flattened)
    
    node_features = e3nn.concatenate([res_irreps, cloud_irreps], axis=-1)
    
    return node_features, pos

# --- EXECUTION BLOCK ---
file_path = 'data/tensor_cloud_representations/e1s1_chignolin_50ns_0-ADRIA_CHIG_ADAPTIVE_crystal_ss_contacts_50_chignolin_0-0-1-RND3469_9.filtered_cloud.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)


# STEP 0: Peek inside the data to fix KeyErrors
print(f"Successfully loaded pickle. Keys found: {list(data.keys())}")
residuals =  data["R_i"]
positions = data["P_i"]
tensors =  data["V_ij"]
print("residuals.shape: ", residuals.shape)
print("resiudals: ", residuals)
print("positions.shape: ", positions.shape)
print("tensors.shape: ", tensors.shape)
# STEP Prepare data (This will now raise a helpful error if keys are wrong)
sample_graph, sample_pos = prepare_data_for_model(data)

# STEP Model Setup
output_irreps =  "14x1e"
input_irreps = "32x0e + 16x1o"
internal_irreps = "32x0e +  124x1o + 10x2e"
output_irreps="16x0e"
model_def = lambda g, p: EquiJumpDeepNetwork(L=2,
                                            input_irreps = input_irreps,
                                            internal_irreps = internal_irreps,
                                            output_irreps = output_irreps)(g, p)
model = hk.without_apply_rng(hk.transform(model_def))

print("Initializing parameters...")
key = jax.random.PRNGKey(42)
params = model.init(key, sample_graph, sample_pos)

# STEP 3: Run Output
output_cloud = model.apply(params, sample_graph, sample_pos)
print("output_cloud.irreps: ", output_cloud.irreps)
print(f"Success! Output cloud shape: {output_cloud.shape}")