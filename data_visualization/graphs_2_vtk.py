import os
import pickle
import numpy as np
import argparse

def graphs_to_vtk(traj_file_path, output_dir="paraview_graphs"):
    """
    Converts a jraph graph pickle to VTK format for ParaView.
    Correctly extracts coordinates from the geometric node features.
    """
    if not os.path.exists(traj_file_path):
        print(f"Error: File {traj_file_path} not found.")
        return

    # Create output directory for this specific trajectory
    traj_name = os.path.basename(traj_file_path).replace('.pkl', '')
    save_path = os.path.join(output_dir, traj_name)
    os.makedirs(save_path, exist_ok=True)

    # Load the graphs
    with open(traj_file_path, 'rb') as f:
        graphs = pickle.load(f)

    print(f"Converting {len(graphs)} frames from {traj_file_path}...")

    for frame_idx, graph in enumerate(graphs):
        # 1. Extract Data
        # Nodes contain: [One-Hot-Types (N columns), X, Y, Z]
        nodes = np.array(graph.nodes)
        senders = np.array(graph.senders)
        receivers = np.array(graph.receivers)
        
        # Slicing the last 3 columns to get the X, Y, Z positions
        pos = nodes[:, -3:] 
        # The atom type index is found in the columns before the last 3
        type_one_hot = nodes[:, :-3]
        type_indices = np.argmax(type_one_hot, axis=1)

        vtk_file = os.path.join(save_path, f"frame_{frame_idx:04d}.vtk")
        
        with open(vtk_file, 'w') as f:
            # VTK Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Protein Graph Trajectory\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # 2. POINTS (The positions)
            f.write(f"POINTS {len(pos)} float\n")
            for p in pos:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

            # 3. LINES (The Graph Edges)
            num_edges = len(senders)
            # VTK needs: (number of lines) (total integers in list: 3 per line [count, s, r])
            f.write(f"LINES {num_edges} {num_edges * 3}\n")
            for s, r in zip(senders, receivers):
                f.write(f"2 {s} {r}\n")

            # 4. POINT_DATA (Scalars for coloring)
            # This MUST match the number of points defined in step 2
            f.write(f"POINT_DATA {len(pos)}\n")
            f.write("SCALARS AtomType int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for t in type_indices:
                f.write(f"{t}\n")

    print(f"Successfully saved {len(graphs)} VTK files to: {os.path.abspath(save_path)}")

def main():
    parser = argparse.ArgumentParser(description="Convert Graph Pickle to VTK for ParaView.")
    parser.add_argument("--input", type=str, required=True, help="Path to the _graphs.pkl file")
    parser.add_argument("--out", type=str, default="paraview_graphs", help="Base output folder")
    
    args = parser.parse_args()
    graphs_to_vtk(args.input, args.out)

if __name__ == "__main__":
    main()