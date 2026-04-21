import mdtraj as md
import numpy as np
import os
import argparse
import pickle

# Global Canonical Order - DO NOT CHANGE THIS ORDER
# This ensures Slot 0 is ALWAYS 'N', Slot 4 is ALWAYS 'CB', etc.
CANONICAL_ATOMS = [
    'N', 'C', 'O', 'OXT', 'CB', 'CG', 'CG1', 'CG2', 
    'CD', 'CD1', 'CD2', 'NE', 'NE1', 'NE2', 'CE', 
    'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', 
    'ND1', 'ND2', 'NH1', 'NH2', 'NZ', 'OD1', 'OD2', 
    'OG', 'OG1', 'OH', 'SD', 'SG'
]

RES_MAP = {res: i for i, res in enumerate([
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
])}

def convert_to_tensor_cloud(pdb_path, xtc_path, max_v_slots=34, heavy_only=True):
    """
    Fixed version using direct mapping to canonical indices.
    Note: max_v_slots should ideally be len(CANONICAL_ATOMS) to avoid truncation.
    """
    try:
        t = md.load(xtc_path, top=pdb_path)
        t.topology.create_standard_bonds()
        
        t.center_coordinates()
        coords = t.xyz * 10.0 # Convert nm to Angstroms
        
        topology = t.topology
        n_frames = t.n_frames
        n_residues = topology.n_residues

        R_i = np.zeros(n_residues, dtype=int)
        P_i = np.zeros((n_frames, n_residues, 3))
        V_ij = np.zeros((n_frames, n_residues, max_v_slots, 3))
        Mask_i = np.zeros((n_residues, max_v_slots), dtype=bool)

        for r_idx, res in enumerate(topology.residues):
            R_i[r_idx] = RES_MAP.get(res.name, -1)
            
            # 1. Identify CA for origin
            ca_atoms = [a for a in res.atoms if a.name == 'CA']
            if not ca_atoms: continue
            ca_idx = ca_atoms[0].index
            
            # 2. Create a fast lookup for atoms in this residue
            # We exclude CA because it is the origin (0,0,0)
            res_atom_map = {a.name: a.index for a in res.atoms if a.name != 'CA'}

            # 3. Reserved Seating Logic
            # Loop through the global list, NOT the residue's atom list
            for j, atom_name in enumerate(CANONICAL_ATOMS):
                if j >= max_v_slots: break
                
                if atom_name in res_atom_map:
                    global_atom_idx = res_atom_map[atom_name]
                    
                    # Filter hydrogens if requested
                    if heavy_only and topology.atom(global_atom_idx).element.symbol == 'H':
                        continue

                    for f in range(n_frames):
                        p_alpha = coords[f, ca_idx]
                        P_i[f, r_idx] = p_alpha # Backbone position
                        
                        # Relative vector to the specific reserved slot
                        V_ij[f, r_idx, j] = coords[f, global_atom_idx] - p_alpha
                        if f == 0: Mask_i[r_idx, j] = True
                        
        return {"R_i": R_i, "P_i": P_i, "V_ij": V_ij, "mask": Mask_i}
        
    except Exception as e:
        print(f"\n[Error]: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert XTC to Fixed Tensor Cloud")
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--xtc", type=str, required=True)
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--out", type=str, default="tensor_clouds")
    # Set default to 34 to accommodate the full canonical list
    parser.add_argument("--max_atoms", type=int, default=34)
    parser.add_argument("--heavy_only", action="store_true", default=True)

    args = parser.parse_args()

    full_xtc_path = os.path.join(args.dir, args.xtc)
    full_pdb_path = os.path.join(args.dir, args.pdb)
    
    if not os.path.exists(full_xtc_path) or not os.path.exists(full_pdb_path):
        print(f"Error: Files not found.")
        return

    os.makedirs(args.out, exist_ok=True)
    clean_name = args.xtc.replace(os.sep, '_').replace('/', '_').replace('.xtc', '_cloud.pkl')
    output_path = os.path.join(args.out, clean_name)

    print(f"Processing: {args.xtc} with fixed mapping...")
    data = convert_to_tensor_cloud(full_pdb_path, full_xtc_path, max_v_slots=args.max_atoms, heavy_only=args.heavy_only)
    
    if data:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Done! Shapes: P_i {data['P_i'].shape}, V_ij {data['V_ij'].shape}")

if __name__ == "__main__":
    main()