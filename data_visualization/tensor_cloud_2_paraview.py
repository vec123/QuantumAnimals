import numpy as np
import os
import argparse
import pickle
from pyevtk.hl import unstructuredGridToVTK

def export_tensor_lines_to_vtp(output_path, R_i, P_i_frame, V_ij_frame, mask):
    all_coords = []
    all_res_types = []
    all_canonical_indices = []
    connectivity = []
    offsets = []
    
    current_idx = 0
    LINE_ID = 3 # VTK identifier for a Line

    for i in range(P_i_frame.shape[0]):
        p_alpha = P_i_frame[i]
        res_type = R_i[i]
        
        for j in range(V_ij_frame.shape[1]):
            # Only draw a line if the atom exists in the tensor
            if mask[i, j]:
                # 1. The Root (C-alpha)
                root_pos = p_alpha
                # 2. The Tip (Position relative to C-alpha)
                tip_pos = p_alpha + V_ij_frame[i, j]
                
                all_coords.extend([root_pos, tip_pos])
                
                # Metadata for the points (for coloring)
                all_res_types.extend([res_type, res_type])
                all_canonical_indices.extend([j, j])
                
                # Connectivity: line from root to tip
                connectivity.extend([current_idx, current_idx + 1])
                current_idx += 2
                offsets.append(current_idx)

    coords = np.array(all_coords, dtype='float32')
    
    # Export to VTU (Unstructured Grid)
    unstructuredGridToVTK(
        output_path,
        np.ascontiguousarray(coords[:, 0]),
        np.ascontiguousarray(coords[:, 1]),
        np.ascontiguousarray(coords[:, 2]),
        connectivity=np.array(connectivity, dtype='int32'),
        offsets=np.array(offsets, dtype='int32'),
        cell_types=np.full(len(offsets), LINE_ID, dtype='uint8'),
        pointData={
            "ResidueType": np.array(all_res_types, dtype='int32'),
            "CanonicalIndex": np.array(all_canonical_indices, dtype='int32')
        },
        cellData={
            "CanonicalIndex": np.array(all_canonical_indices, dtype='int32')
        }
    )

def main():
    parser = argparse.ArgumentParser(description="Visualize Tensor Cloud as Star Lines")
    parser.add_argument("--input", type=str, required=True, help="Path to cloud_data.pkl")
    parser.add_argument("--out", type=str, default="paraview_lines")
    parser.add_argument("--step", type=int, default=1, help="Subsample every Nth frame")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File {args.input} not found.")
        return

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # Output directory named after input file
    file_basename = os.path.basename(args.input).replace(".pkl", "")
    output_dir = os.path.join(args.out, file_basename)
    os.makedirs(output_dir, exist_ok=True)

    n_frames = len(data['P_i'])
    print(f"Exporting every {args.step} frame(s) to {output_dir}...")

    for f_idx in range(0, n_frames, args.step):
        export_tensor_lines_to_vtp(
            os.path.join(output_dir, f"star_lines_{f_idx:04d}"),
            data['R_i'], data['P_i'][f_idx], data['V_ij'][f_idx], data['mask']
        )
        if f_idx % 10 == 0:
            print(f" Frame {f_idx}/{n_frames} done", end="\r")

    print(f"\nFinished. Files saved in {output_dir}")

if __name__ == "__main__":
    main()