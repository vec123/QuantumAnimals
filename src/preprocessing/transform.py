import mdtraj as md
import numpy as np
from .schema import TensorCloud
from .residue_registry import AtomRegistry

class TrajectoryTransformer:
    def __init__(self, registry: AtomRegistry, max_slots: int = 14):
        self.registry = registry
        self.max_slots = max_slots

    def _get_local_geometry(self, residue, frame_coords):
        """Logic for calculating relative vectors for a single residue."""
        ca_idx = [a.index for a in residue.atoms if a.name == 'CA']
        if not ca_idx:
            return None, None
            
        ca_pos = frame_coords[:, ca_idx[0], :]
        
        # Pre-allocate local vectors for this residue
        v_ij = np.zeros((frame_coords.shape[0], self.max_slots, 3), dtype=np.float32)
        mask = np.zeros(self.max_slots, dtype=bool)
        
        target_names = self.registry.map.get(residue.name, [])
        atom_dict = {a.name: a.index for a in residue.atoms if a.name != 'CA'}
        
        for i, name in enumerate(target_names[:self.max_slots]):
            if name in atom_dict:
                v_ij[:, i, :] = frame_coords[:, atom_dict[name], :] - ca_pos
                mask[i] = True
                
        return ca_pos, v_ij, mask

    def transform(self, pdb_path: str, xtc_path: str) -> TensorCloud:
        traj = md.load(xtc_path, top=pdb_path)
        traj = traj.atom_slice(traj.topology.select("protein")).center_coordinates()
        
        coords = traj.xyz.astype(np.float32) * 10.0 # Convert to Angstroms
        n_res = traj.topology.n_residues
        
        # Initialize containers
        r_i = np.zeros(n_res, dtype=np.int32)
        p_i = np.zeros((traj.n_frames, n_res, 3), dtype=np.float32)
        v_ij = np.zeros((traj.n_frames, n_res, self.max_slots, 3), dtype=np.float32)
        m_i = np.zeros((n_res, self.max_slots), dtype=bool)

        for idx, res in enumerate(traj.topology.residues):
            r_i[idx] = self.registry.get_res_id(res.name)
            ca_pos, local_v, mask = self._get_local_geometry(res, coords)
            
            if ca_pos is not None:
                p_i[:, idx, :] = ca_pos
                v_ij[:, idx, :, :] = local_v
                m_i[idx, :] = mask

        return TensorCloud(r_i, p_i, v_ij, m_i)