import numpy as np
import os
import argparse
import jax.numpy as jnp
import jraph
from scipy.spatial import cKDTree
import pickle

def center_coords(coords):
    """Centers the trajectory at the origin for each frame."""
    return coords - np.mean(coords, axis=1, keepdims=True)

def build_graph_geometric(positions, types, type_to_idx, num_classes, r_cutoff=5.0):
    """Builds a jraph GraphsTuple with geometric node features [one-hot, x, y, z]."""
    n_atoms = positions.shape[0]
    tree = cKDTree(positions)
    # Efficient fixed-radius neighbor search
    adj_list = tree.query_pairs(r_cutoff, output_type='ndarray')
    
    if len(adj_list) == 0: 
        return None

    # Create bi-directional edges (A->B and B->A)
    s_half, r_half = adj_list[:, 0], adj_list[:, 1]
    senders = np.concatenate([s_half, r_half])
    receivers = np.concatenate([r_half, s_half])
    
    # Edge features: Euclidean distance
    dist = np.linalg.norm(positions[senders] - positions[receivers], axis=-1)
    
    # Node features: Concatenate One-hot encoding of atom types with XYZ coords
    indices = np.array([type_to_idx[t] for t in types])
    one_hot = np.eye(num_classes)[indices]
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

def main():
    parser = argparse.ArgumentParser(description="Convert NPZ trajectory to Jraph Pickle.")
    
    parser.add_argument("--input", type=str, required=True, help="Path to the .npz file")
    parser.add_argument("--out", type=str, default="graph_trajs", help="Output folder")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Radius cutoff for edges (Å)")
    
    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    os.makedirs(args.out, exist_ok=True)

    # 2. Load Data
    print(f"Loading {args.input}...")
    data = np.load(args.input, allow_pickle=True)
    coords = data['coords']
    types = data['types']

    # 3. Global Type Mapping
    atom_types_unique = np.unique(types)
    type_to_idx = {t: i for i, t in enumerate(atom_types_unique)}
    num_classes = len(atom_types_unique)
    
    print(f"Mapping: {type_to_idx}")

    # 4. Optional: Re-center
    # Even if centered before, ensures origin is (0,0,0) based on current frame
    #coords = center_coords(coords)

    # 5. Build Graphs
    traj_graphs = []
    edge_counts = []
    total_frames = len(coords)
    print(f"Building graphs for {total_frames} frames (Cutoff: {args.cutoff} Å)...")

    for i in range(total_frames):
        graph = build_graph_geometric(coords[i], types, type_to_idx, num_classes, args.cutoff)
        if graph:
            traj_graphs.append(graph)
            # Calculate degree for this frame
            edge_counts.append(len(graph.senders) / int(graph.n_node[0]))
        
        if (i + 1) % 100 == 0 or i == total_frames - 1:
            print(f"  Processed {i+1}/{total_frames} frames", end="\r")

    # 6. Final Stats and Save
    avg_edges = np.mean(edge_counts) if edge_counts else 0
    file_basename = os.path.basename(args.input).replace('.npz', '_graphs.pkl')
    out_path = os.path.join(args.out, file_basename)
    
    with open(out_path, 'wb') as f:
        pickle.dump(traj_graphs, f)
    
    print(f"\n\nDone!")
    print(f"Saved {len(traj_graphs)} graphs to: {out_path}")
    print(f"Average edges per atom: {avg_edges:.2f}")

if __name__ == "__main__":
    main()