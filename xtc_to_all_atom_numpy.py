import mdtraj as md
import numpy as np
import os
import argparse

def convert_xtc_to_numpy_mdtraj(pdb_path, xtc_path):
    """Converts xtc to numpy using MDTraj logic."""
    try:
        # 1. Load Trajectory
        t = md.load(xtc_path, top=pdb_path)
        
        # 2. Fix Periodic Boundary Conditions
        t.topology.create_standard_bonds()
        molecules = t.topology.find_molecules()
        t.image_molecules(inplace=True, anchor_molecules=[molecules[0]])
        
        # 3. Center at (0,0,0)
        t.center_coordinates()
        
        # 4. Extract and Convert to Angstroms
        coords_angstroms = t.xyz * 10.0
        atom_types = [a.element.symbol for a in t.topology.atoms]
        
        # 5. Calculate Max Diameter
        frame_0 = coords_angstroms[0]
        dist_matrix = np.linalg.norm(frame_0[:, None, :] - frame_0[None, :, :], axis=-1)
        max_d = np.max(dist_matrix)
        
        return coords_angstroms, atom_types, max_d
        
    except Exception as e:
        print(f"\n[Error] MDTraj failed: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Convert XTC to NPZ with relative PDB path.")
    
    parser.add_argument("--dir", type=str, required=True, help="Base directory")
    parser.add_argument("--xtc", type=str, required=True, help="XTC path relative to --dir")
    parser.add_argument("--pdb", type=str, required=True, help="PDB filename relative to --dir")
    parser.add_argument("--out", type=str, default="numpy_trajs", help="Output directory")

    args = parser.parse_args()

    # Construct paths relative to args.dir
    full_xtc_path = os.path.join(args.dir, args.xtc)
    full_pdb_path = os.path.join(args.dir, args.pdb)
    
    # Validation
    if not os.path.exists(full_xtc_path):
        print(f"Error: XTC file not found at: {full_xtc_path}")
        return
    if not os.path.exists(full_pdb_path):
        print(f"Error: PDB file not found at: {full_pdb_path}")
        return

    os.makedirs(args.out, exist_ok=True)

    # Flatten filename for output (replace slashes to avoid subfolder issues in output)
    clean_name = args.xtc.replace(os.sep, '_').replace('/', '_').replace('.xtc', '.npz')
    output_path = os.path.join(args.out, clean_name)

    print(f"Processing: {args.xtc}...", end=" ", flush=True)
    coords, types, max_d = convert_xtc_to_numpy_mdtraj(full_pdb_path, full_xtc_path)
    
    if coords is not None:
        np.savez_compressed(output_path, coords=coords, types=types)
        print(f"Done! (Max Diam: {max_d:.2f} Å)")
    else:
        print("Failed.")

if __name__ == "__main__":
    main()