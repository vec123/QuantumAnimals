import numpy as np
import os
import configparser
import jax.numpy as jnp
import jraph
from scipy.spatial import cKDTree
import pickle

# --- SETTINGS ---
config = configparser.ConfigParser()
config.read('path_cfg.cfg')
numpy_folder = "numpy_trajs"
output_folder = "graph_trajs"
R_CUTOFF = 5.0
BOX_SIZE = 100.0

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- FUNCTIONS ---
def unwrap_coords(coords, box_size=100.0):
    reference_atom = coords[:, 0:1, :] 
    diff = coords - reference_atom
    diff = diff - box_size * np.round(diff / box_size)
    return reference_atom + diff

def center_coords(coords):
    return coords - np.mean(coords, axis=1, keepdims=True)

def build_graph_scaled(positions, types, type_to_idx, num_classes, r_cutoff=12.0):
    n_atoms = positions.shape[0]
    tree = cKDTree(positions)
    adj_list = tree.query_pairs(r_cutoff, output_type='ndarray')
    
    if len(adj_list) == 0: return None

    s_half, r_half = adj_list[:, 0], adj_list[:, 1]
    senders = np.concatenate([s_half, r_half])
    receivers = np.concatenate([r_half, s_half])
    
    dist = np.linalg.norm(positions[senders] - positions[receivers], axis=-1)
    
    indices = np.array([type_to_idx[t] for t in types])
    node_feats = np.eye(num_classes)[indices]
    
    return jraph.GraphsTuple(
        nodes=jnp.array(node_feats),
        edges=jnp.array(dist)[:, None],
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=jnp.array([n_atoms]),
        n_edge=jnp.array([len(senders)]),
        globals=None
    )

def build_graph_geometric(positions, types, type_to_idx, num_classes, r_cutoff=12.0):
    n_atoms = positions.shape[0]
    tree = cKDTree(positions)
    adj_list = tree.query_pairs(r_cutoff, output_type='ndarray')
    
    if len(adj_list) == 0: return None

    s_half, r_half = adj_list[:, 0], adj_list[:, 1]
    senders = np.concatenate([s_half, r_half])
    receivers = np.concatenate([r_half, s_half])
    
    dist = np.linalg.norm(positions[senders] - positions[receivers], axis=-1)
    
    # --- GEOMETRIC NODE FEATURES ---
    # One-hot: [1, 0, 0] (for 3 classes)
    one_hot = np.eye(num_classes)[np.array([type_to_idx[t] for t in types])]
    
    # Concatenate: [1, 0, 0, x, y, z] -> Shape (N, num_classes + 3)
    node_features = np.hstack([one_hot, positions])
    
    return jraph.GraphsTuple(
        nodes=jnp.array(node_features), 
        edges=jnp.array(dist)[:, None],
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=jnp.array([n_atoms]),
        n_edge=jnp.array([len(senders)]),
        globals=None
    )


# --- MAIN LOOP ---
def main():
    files = [f for f in os.listdir(numpy_folder) if f.endswith('.npz')]
    if not files: exit("No files found.")

    # 1. Global Type Mapping (Must be consistent across all trajectories!)
    # We peek at the first file to establish types. 
    # If different files have different atoms, you'll need to pre-scan all files.
    with np.load(os.path.join(numpy_folder, files[0]), allow_pickle=True) as data:
        atom_types_unique = np.unique(data['types'])
        type_to_idx = {t: i for i, t in enumerate(atom_types_unique)}
        num_classes = len(atom_types_unique)

    print(f"Found {len(files)} trajectories.")
    print(f"Mapping: {type_to_idx}")

    for file_name in files:
        print(f"\nProcessing {file_name}...")
        file_path = os.path.join(numpy_folder, file_name)
        
        with np.load(file_path, allow_pickle=True) as data:
            coords = data['coords']
            types = data['types']

        # Pre-process
        coords = unwrap_coords(coords, box_size=BOX_SIZE)
        coords = center_coords(coords)

        # Iterate frames and store in a list (one list per trajectory)
        traj_graphs = []
        for i in range(len(coords)):
            graph = build_graph_geometric(coords[i], types, type_to_idx, num_classes, R_CUTOFF)
            if graph:
                traj_graphs.append(graph)
            
            if (i + 1) % 100 == 0:
                print(f"  Frame {i+1}/{len(coords)}...")

        # Save the trajectory of graphs
        out_name = file_name.replace('.npz', '_graphs.pkl')
        out_path = os.path.join(output_folder, out_name)
        
        with open(out_path, 'wb') as f:
            pickle.dump(traj_graphs, f)
        
        print(f"Saved trajectory to {out_path}")

if __name__ == "__main__":
    main()