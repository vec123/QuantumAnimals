import mdtraj as md
import numpy as np
import os
import argparse
import pickle

def convert_xtc_to_residue_numpy(pdb_path, xtc_path):
    """
    Converts xtc to residue-based representation:
    Returns:
        - res_data: List of frames, where each frame is a list of 
                    (C_alpha_pos, relative_positions, res_label)
        - max_d: Maximum diameter of the protein in the first frame.
    """
    try:
        # 1. Load Trajectory
        t = md.load(xtc_path, top=pdb_path)
        
        # 2. Fix Periodic Boundary Conditions
        t.topology.create_standard_bonds()
        molecules = t.topology.find_molecules()
        if len(molecules) > 0:
            t.image_molecules(inplace=True, anchor_molecules=[molecules[0]])
        
        # 3. Center at (0,0,0)
        t.center_coordinates()
        
        # 4. Extract and Convert to Angstroms
        coords_angstroms = t.xyz * 10.0
        topology = t.topology
        
        # 5. Calculate Max Diameter (Frame 0)
        frame_0 = coords_angstroms[0]
        dist_matrix = np.linalg.norm(frame_0[:, None, :] - frame_0[None, :, :], axis=-1)
        max_d = np.max(dist_matrix)

        # 6. Build Residue Representation
        # Structure: [frames][residues] -> {dict of features}
        traj_res_representation = []
        
        for f in range(len(coords_angstroms)):
            frame_data = []
            for res in topology.residues:
                # Find CA atom
                ca_atoms = [a for a in res.atoms if a.name == 'CA']
                if not ca_atoms:
                    continue # Skip residues without CA (e.g. water, ions)
                
                ca_idx = ca_atoms[0].index
                p_alpha = coords_angstroms[f, ca_idx]
                
                # Get all atom positions for this residue
                res_atom_indices = [a.index for a in res.atoms]
                res_coords = coords_angstroms[f, res_atom_indices]
                
                # Calculate relative positions P_i = P_atom - P_alpha
                relative_p = res_coords - p_alpha
                
                frame_data.append({
                    'p_alpha': p_alpha,               # Shape (3,)
                    'relative_positions': relative_p, # Shape (N_atoms_in_res, 3)
                    'label': res.name                 # e.g., 'ALA'
                })
            traj_res_representation.append(frame_data)
        
        return traj_res_representation, max_d
        
    except Exception as e:
        print(f"\n[Error] MDTraj failed: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Convert XTC to Residue-level Pickle.")
    
    parser.add_argument("--dir", type=str, required=True, help="Base directory")
    parser.add_argument("--xtc", type=str, required=True, help="XTC path relative to --dir")
    parser.add_argument("--pdb", type=str, required=True, help="PDB filename relative to --dir")
    parser.add_argument("--out", type=str, default="residue_trajs", help="Output directory")

    args = parser.parse_args()

    full_xtc_path = os.path.join(args.dir, args.xtc)
    full_pdb_path = os.path.join(args.dir, args.pdb)
    
    if not os.path.exists(full_xtc_path) or not os.path.exists(full_pdb_path):
        print(f"Error: Files not found.")
        return

    os.makedirs(args.out, exist_ok=True)

    # Use .pkl extension for structured residue data
    clean_name = args.xtc.replace(os.sep, '_').replace('/', '_').replace('.xtc', '_res.pkl')
    output_path = os.path.join(args.out, clean_name)

    print(f"Processing: {args.xtc}...", end=" ", flush=True)
    res_data, max_d = convert_xtc_to_residue_numpy(full_pdb_path, full_xtc_path)
    
    if res_data is not None:
        with open(output_path, 'wb') as f:
            pickle.dump(res_data, f)
        print(f"Done! (Max Diam: {max_d:.2f} Å)")
    else:
        print("Failed.")

if __name__ == "__main__":
    main()