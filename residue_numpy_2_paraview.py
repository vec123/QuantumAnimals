import numpy as np
import os
import argparse
import pickle
from pyevtk.hl import unstructuredGridToVTK, pointsToVTK
try:
    from pyevtk.vtk import VtkLine
    LINE_ID = VtkLine.tid
except ImportError:
    LINE_ID = 3 # Fallback to VTK standard integer for Line

def export_residue_to_vtp(output_path, frame_data, include_edges=True):
    """
    Saves a frame of residue data. 
    If include_edges is True, saves as UnstructuredGrid with CA-atom lines.
    If include_edges is False, saves only the points.
    """
    all_coords = []
    res_type_id = []
    is_ca_flag = [] 
    
    # Map residue names to ints
    unique_res_names = sorted(list(set(res['label'] for res in frame_data)))
    label_map = {name: i for i, name in enumerate(unique_res_names)}

    if include_edges:
        connectivity = []
        offsets = []
        current_atom_idx = 0

        for res in frame_data:
            p_alpha = res['p_alpha']
            rel_atoms = res['relative_positions']
            label_idx = label_map[res['label']]
            abs_atoms = rel_atoms + p_alpha
            
            # Find CA (assumed origin at 0,0,0)
            ca_local_idx = np.argmin(np.linalg.norm(rel_atoms, axis=1))
            global_ca_idx = current_atom_idx + ca_local_idx

            for i, pos in enumerate(abs_atoms):
                all_coords.append(pos)
                res_type_id.append(label_idx)
                is_ca_flag.append(1 if i == ca_local_idx else 0)
                
                if i != ca_local_idx:
                    connectivity.extend([global_ca_idx, current_atom_idx + i])
                    offsets.append(len(connectivity))
            current_atom_idx += len(abs_atoms)

        coords = np.array(all_coords, dtype='float32')
        cell_types = np.full(len(offsets), LINE_ID, dtype='uint8')
        
        unstructuredGridToVTK(
            output_path, 
            np.ascontiguousarray(coords[:, 0]), 
            np.ascontiguousarray(coords[:, 1]), 
            np.ascontiguousarray(coords[:, 2]), 
            connectivity=np.array(connectivity, dtype='int32'), 
            offsets=np.array(offsets, dtype='int32'), 
            cell_types=cell_types, 
            pointData={
                "ResidueType": np.array(res_type_id, dtype='int32'),
                "IsCAlpha": np.array(is_ca_flag, dtype='int32')
            }
        )
    else:
        # Just points mode
        for res in frame_data:
            abs_atoms = res['relative_positions'] + res['p_alpha']
            ca_local_idx = np.argmin(np.linalg.norm(res['relative_positions'], axis=1))
            for i, pos in enumerate(abs_atoms):
                all_coords.append(pos)
                res_type_id.append(label_map[res['label']])
                is_ca_flag.append(1 if i == ca_local_idx else 0)

        coords = np.array(all_coords, dtype='float32')
        pointsToVTK(
            output_path,
            np.ascontiguousarray(coords[:, 0]), 
            np.ascontiguousarray(coords[:, 1]), 
            np.ascontiguousarray(coords[:, 2]),
            data={
                "ResidueType": np.array(res_type_id, dtype='int32'),
                "IsCAlpha": np.array(is_ca_flag, dtype='int32')
            }
        )

def main():
    parser = argparse.ArgumentParser(description="Export Residue data to VTK.")
    parser.add_argument("--input", type=str, required=True, help="Path to .pkl")
    parser.add_argument("--out", type=str, default="paraview_residues")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--no_edges", action="store_true", help="Toggle off the star connections")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        return

    with open(args.input, 'rb') as f:
        traj_data = pickle.load(f)

    file_basename = os.path.basename(args.input).replace(".pkl", "")
    traj_output_dir = os.path.join(args.out, file_basename)
    os.makedirs(traj_output_dir, exist_ok=True)

    print(f"Exporting (Edges: {not args.no_edges})...")

    for frame_idx in range(0, len(traj_data), args.step):
        export_residue_to_vtp(
            os.path.join(traj_output_dir, f"frame_{frame_idx:04d}"), 
            traj_data[frame_idx],
            include_edges=not args.no_edges
        )
        if frame_idx % 10 == 0:
            print(f" Frame {frame_idx} done", end="\r")

    print(f"\nDone! Files saved to {traj_output_dir}")

if __name__ == "__main__":
    main()