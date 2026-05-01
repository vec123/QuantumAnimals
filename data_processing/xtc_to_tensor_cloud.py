import mdtraj as md
import numpy as np
import os
import argparse
import pickle


RESIDUE_ATOM_MAPS = {
    'ALA': ['N', 'C', 'O', 'CB', 'OXT'],
    'ARG': ['N', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', 'OXT'],
    'ASN': ['N', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', 'OXT'],
    'ASP': ['N', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', 'OXT'],
    'CYS': ['N', 'C', 'O', 'CB', 'SG', 'OXT'],
    'GLN': ['N', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'OXT'],
    'GLU': ['N', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'OXT'],
    'GLY': ['N', 'C', 'O', 'OXT'],
    'HIS': ['N', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', 'OXT'],
    'ILE': ['N', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', 'OXT'],
    'LEU': ['N', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'OXT'],
    'LYS': ['N', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', 'OXT'],
    'MET': ['N', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'OXT'],
    'PHE': ['N', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OXT'],
    'PRO': ['N', 'C', 'O', 'CB', 'CG', 'CD', 'OXT'],
    'SER': ['N', 'C', 'O', 'CB', 'OG', 'OXT'],
    'THR': ['N', 'C', 'O', 'CB', 'OG1', 'CG2', 'OXT'],
    'TRP': ['N', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT'],
    'TYR': ['N', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'OXT'],
    'VAL': ['N', 'C', 'O', 'CB', 'CG1', 'CG2', 'OXT']
}

RES_MAP = {res: i for i, res in enumerate(sorted(RESIDUE_ATOM_MAPS.keys()))}

def convert_to_tensor_cloud(pdb_path, xtc_path, max_v_slots=13):
    try:
        t = md.load(xtc_path, top=pdb_path)
        t.center_coordinates()
        coords = t.xyz * 10.0 # Angstroms
        
        topology = t.topology
        n_frames = t.n_frames
        n_residues = topology.n_residues

        R_i = np.zeros(n_residues, dtype=int)
        P_i = np.zeros((n_frames, n_residues, 3))
        V_ij = np.zeros((n_frames, n_residues, max_v_slots, 3))
        Mask_i = np.zeros((n_residues, max_v_slots), dtype=bool)

        for r_idx, res in enumerate(topology.residues):
            R_i[r_idx] = RES_MAP.get(res.name, -1)
            
            ca_atoms = [a for a in res.atoms if a.name == 'CA']
            if not ca_atoms: continue
            ca_idx = ca_atoms[0].index
            
            res_atoms_dict = {a.name: a.index for a in res.atoms if a.name != 'CA'}
            # Get the unique seat-chart for this specific residue type
            compact_order = RESIDUE_ATOM_MAPS.get(res.name, [])

            for j, target_name in enumerate(compact_order):
                if j >= max_v_slots: break
                
                if target_name in res_atoms_dict:
                    atom_idx = res_atoms_dict[target_name]
                    for f in range(n_frames):
                        p_alpha = coords[f, ca_idx]
                        P_i[f, r_idx] = p_alpha
                        V_ij[f, r_idx, j] = coords[f, atom_idx] - p_alpha
                        if f == 0: Mask_i[r_idx, j] = True
                        
        return {"R_i": R_i, "P_i": P_i, "V_ij": V_ij, "mask": Mask_i}
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--xtc", type=str, required=True)
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--out", type=str, default="tensor_clouds_c15")
    args = parser.parse_args()

    full_xtc = os.path.join(args.dir, args.xtc)
    full_pdb = os.path.join(args.dir, args.pdb)
    
    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, os.path.basename(args.xtc).replace('.xtc', '_cloud.pkl'))

    print(f"Compressing {os.path.basename(args.xtc)} into 15 slots...")
    data = convert_to_tensor_cloud(full_pdb, full_xtc)
    
    if data:
        with open(out_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved to {out_file}. V_ij shape: {data['V_ij'].shape}")

if __name__ == "__main__":
    main()