import numpy as np
import os
import argparse
from pyevtk.hl import pointsToVTK

def export_to_vtp(output_path, coords, types):
    """
    Saves a single frame to a VTK PolyData (.vtu) file.
    """
    # Create unique integer IDs for atom types
    unique_types = np.unique(types)
    type_map = {t: i for i, t in enumerate(unique_types)}
    type_ids = np.array([type_map[t] for t in types], dtype='int32')

    # Ensure coordinates are contiguous float32 for EVTK compatibility
    x = np.ascontiguousarray(coords[:, 0], dtype='float32')
    y = np.ascontiguousarray(coords[:, 1], dtype='float32')
    z = np.ascontiguousarray(coords[:, 2], dtype='float32')

    # Save using evtk (automatically appends .vtu)
    pointsToVTK(output_path, x, y, z, data={"AtomType": type_ids})

def main():
    parser = argparse.ArgumentParser(description="Convert a specific NPZ trajectory to VTK frames.")
    
    # Arguments
    parser.add_argument("--input", type=str, required=True, help="Full path to the .npz file")
    parser.add_argument("--out", type=str, default="paraview_exports", help="Base output directory")
    parser.add_argument("--step", type=int, default=1, help="Export every Nth frame (default: 10)")

    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' not found.")
        return

    # 2. Load Data
    try:
        data = np.load(args.input, allow_pickle=True)
        coords = data['coords']  # Shape: (frames, atoms, 3)
        types = data['types']    # Shape: (atoms,)
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return

    # 3. Setup Output Directory
    # Uses the filename (minus .npz) as the subfolder name
    file_basename = os.path.basename(args.input).replace(".npz", "")
    traj_output_dir = os.path.join(args.out, file_basename)
    os.makedirs(traj_output_dir, exist_ok=True)

    print(f"Processing: {file_basename}")
    print(f"Total frames: {len(coords)} | Exporting every {args.step} frame(s)")

    # 4. Export Loop
    count = 0
    for frame_idx in range(0, len(coords), args.step):
        frame_coords = coords[frame_idx]
        # Format filename as frame_0000, frame_0010, etc.
        file_path = os.path.join(traj_output_dir, f"frame_{frame_idx:04d}")
        
        export_to_vtp(file_path, frame_coords, types)
        count += 1

    print(f"Done! {count} frames exported to: {os.path.abspath(traj_output_dir)}")

if __name__ == "__main__":
    main()