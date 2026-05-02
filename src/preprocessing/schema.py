import pickle
from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TensorCloud:
    residue_types: np.ndarray  # (N_res,)
    ca_positions: np.ndarray   # (N_frames, N_res, 3)
    local_vectors: np.ndarray  # (N_frames, N_res, N_slots, 3)
    mask: np.ndarray           # (N_res, N_slots)

    @property
    def n_frames(self) -> int:
        return self.ca_positions.shape[0]
    
    def save(self, path: str | Path):
        """Serializes the tensor cloud to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                "R_i": self.residue_types,
                "P_i": self.ca_positions,
                "V_ij": self.local_vectors,
                "mask": self.mask
            }, f)
        print(f"Successfully saved TensorCloud to {path}")