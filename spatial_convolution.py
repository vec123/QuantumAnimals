import time
import numpy as np
import jax
import jax.numpy as jnp
import jraph
import optax
from tqdm.auto import tqdm

import e3nn_jax as e3nn

import pickle

# Replace 'data.pkl' with your actual file path
file_path = 'data/tensor_cloud_representations/e1s1_chignolin_50ns_0-ADRIA_CHIG_ADAPTIVE_crystal_ss_contacts_50_chignolin_0-0-1-RND3469_9.filtered_cloud.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Quick look at the type and top-level structure
print(f"Type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")

# 1. Check R_i: Expecting a string (or an array of strings/types)
r_i = data.get('R_i')
print(f"R_i: type={type(r_i)}")
if isinstance(r_i, (list, jnp.ndarray, str)):
    # If it's a batch/list of strings, check the first element
    sample = r_i[0] if hasattr(r_i, '__len__') and not isinstance(r_i, str) else r_i
    print(f"  Sample element type: {type(sample)}")

# 2. Check P_i: Expecting 3D positions
p_i = data.get('P_i')
if p_i is not None:
    # Ensure it's a JNP/NP array and check the last dimension is 3
    shape = jnp.shape(p_i)
    print(f"P_i: shape={shape}, valid_3d={shape[-1] == 3}")

# 3. Check V_ij: Expecting a list/array of vectors
v_ij = data.get('V_ij')
if v_ij is not None:
    # In graph contexts, V_ij is often (num_edges, 3)
    shape_v = jnp.shape(v_ij)
    print(f"V_ij: shape={shape_v}, valid_vectors={shape_v[-1] == 3}")

# 4. Check Mask
mask = data.get('mask')
if mask is not None:
    print(f"Mask: shape={jnp.shape(mask)}, dtype={mask.dtype}")

def load_to_jraph(data):
    # 1. Handle Node Types (R_i)
    # Assuming R_i is shape (12,) - the types don't change over time
    R_i_raw = np.array(data['R_i'])
    unique_types = np.unique(R_i_raw)
    type_to_int = {t: i for i, t in enumerate(unique_types)}
    num_classes = len(unique_types)
    
    R_i_ids = np.vectorize(type_to_int.get)(R_i_raw)
    # Shape: (12, num_classes)
    R_i_one_hot = jax.nn.one_hot(R_i_ids, num_classes)

    # 2. Extract Trajectory Data
    P_t = data['P_i']    # Shape: (500, 12, 3)
    V_t = data['V_ij']   # Shape: (500, 12, 13, 3)
    mask = data['mask']  # Shape: (12, 13)

    num_steps = P_t.shape[0]
    num_nodes = P_t.shape[1]

    # 3. Topology (remains constant over time)
    i_idx, j_idx = jnp.indices(mask.shape)
    senders = i_idx[mask]
    receivers = j_idx[mask] 

    graphs = []
    for t in range(num_steps):
        # Concatenate: (12, num_classes) + (12, 3) -> (12, num_classes + 3)
        # Note: We use the same R_i_one_hot for every time step t
        node_feats = jnp.concatenate([R_i_one_hot, P_t[t]], axis=-1)
        
        # Edge features for this time step
        edge_feats = V_t[t][mask]

        graphs.append(
            jraph.GraphsTuple(
                nodes=node_feats,
                edges=edge_feats,
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([num_nodes]),
                n_edge=jnp.array([len(senders)]),
                globals=None
            )
        )

    return jraph.batch(graphs), unique_types

batched_graph, unique_types = load_to_jraph(data)

print(f"Node features shape: {batched_graph.nodes.shape}") # (6000, num_classes + 3)

import random

# 1. Pick a random timestep
t = random.randint(0, 499)
num_nodes_per_graph = 12

# 2. Extract Node Features for this frame
# Nodes are at indices [t*12 : (t+1)*12]
start_node = t * num_nodes_per_graph
end_node = (t + 1) * num_nodes_per_graph
frame_nodes = batched_graph.nodes[start_node:end_node]

# 3. Extract Edges for this frame
# n_edge tells us how many edges each graph has
# Since your mask is fixed (12, 13), every frame has the same number of edges
edges_per_graph = batched_graph.n_edge[0] 
start_edge = t * edges_per_graph
end_edge = (t + 1) * edges_per_graph
frame_edges = batched_graph.edges[start_edge:end_edge]

print(f"--- Inspection of Timestep {t} ---")
print(f"Node Features Shape: {frame_nodes.shape}")  # (12, 11)
print(f"Edge Features Shape: {frame_edges.shape}")  # (num_edges, 3)

# 4. Verification
# Check if One-Hot is identical to frame 0
first_frame_types = batched_graph.nodes[0:12, :8]
current_frame_types = frame_nodes[:, :8]
is_identical = jnp.allclose(first_frame_types, current_frame_types)

# Check if Positions are different
first_frame_pos = batched_graph.nodes[0:12, 8:]
current_frame_pos = frame_nodes[:, 8:]
is_pos_different = not jnp.allclose(first_frame_pos, current_frame_pos)

print(f"Identical One-Hot Encodings? {is_identical}")
print(f"Positions changed since Frame 0? {is_pos_different}")

first_frame_edges = batched_graph.edges[0:82] # First 82 edges
current_frame_edges = frame_edges             # Random frame 304 edges

is_edge_geom_different = not jnp.allclose(first_frame_edges, current_frame_edges)

# Check Edge Topology (Indices)
# Since the mask is (12, 13) and static, senders/receivers are constant
first_frame_senders = batched_graph.senders[0:82]
current_frame_senders = batched_graph.senders[start_edge:end_edge]
is_topology_different = not jnp.allclose(first_frame_senders, current_frame_senders)

print(f"Edge vectors (V_ij) changed? {is_edge_geom_different}")
print(f"Edge connectivity (Topology) changed? {is_topology_different}")