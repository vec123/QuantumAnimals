import pickle
import argparse
import numpy as np

# This MUST match the dictionary used in your conversion script
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

RES_LIST = sorted(RESIDUE_ATOM_MAPS.keys())

def print_multiple_residues(data_path, res_indices, frame_idx=0):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    max_slots = data['V_ij'].shape[2]

    for res_idx in res_indices:
        if res_idx >= len(data['R_i']): continue
        
        res_type_idx = data['R_i'][res_idx]
        res_name = RES_LIST[res_type_idx]
        
        # Get the specific mapping for this residue type
        mapping = RESIDUE_ATOM_MAPS.get(res_name, [])
        
        print(f"\n{'='*20} RESIDUE {res_idx}: {res_name} {'='*20}")
        print(f"{'Slot':<5} {'Label':<6} {'Vector (x, y, z)':<30} {'Status':<8}")
        
        for j in range(max_slots):
            vec = data['V_ij'][frame_idx, res_idx, j]
            is_present = data['mask'][res_idx, j]
            
            # Look up what this slot represents for THIS residue type
            slot_label = mapping[j] if j < len(mapping) else "-"
            
            status = "PRESENT" if is_present else "-"
            vec_str = f"[{vec[0]:6.3f}, {vec[1]:6.3f}, {vec[2]:6.3f}]" if is_present else "[  0.000,   0.000,   0.000]"
            
            print(f"{j:<5} {slot_label:<6} {vec_str:<30} {status}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    
    print_multiple_residues(args.input, args.indices)

if __name__ == "__main__":
    main()