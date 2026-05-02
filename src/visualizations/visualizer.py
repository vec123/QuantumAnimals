import numpy as np
import os
from pathlib import Path
from typing import Optional
from pyevtk.hl import unstructuredGridToVTK
from src.preprocessing.schema import TensorCloud

class CloudVisualizer:
    """Handles exporting TensorCloud data to VTK formats for ParaView."""
    
    VTK_LINE_ID = 3  # Cell type for Poly Line / Line

    def __init__(self, output_dir: str | Path = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_frame(self, cloud: TensorCloud, frame_idx: int, filename: Optional[str] = None):
        """Exports a single frame of the TensorCloud as a star-line VTU file."""
        if frame_idx >= cloud.n_frames:
            raise IndexError(f"Frame index {frame_idx} out of range for cloud with {cloud.n_frames} frames.")

        # Filter valid atoms using the mask
        # cloud.mask shape: (N_res, N_slots)
        res_indices, slot_indices = np.where(cloud.mask)
        
        # Extract coordinates
        # p_alpha: (N_valid, 3)
        p_alpha = cloud.ca_positions[frame_idx, res_indices]
        # v_ij: (N_valid, 3)
        v_ij = cloud.local_vectors[frame_idx, res_indices, slot_indices]
        
        root_positions = p_alpha
        tip_positions = p_alpha + v_ij

        # Interleave roots and tips for VTK line connectivity
        # Result shape: (N_valid * 2, 3)
        coords = np.empty((len(res_indices) * 2, 3), dtype=np.float32)
        coords[0::2] = root_positions
        coords[1::2] = tip_positions

        # Setup connectivity: [0, 1, 2, 3, 4, 5...]
        n_lines = len(res_indices)
        connectivity = np.arange(n_lines * 2, dtype=np.int32)
        offsets = np.arange(2, n_lines * 2 + 1, 2, dtype=np.int32)
        cell_types = np.full(n_lines, self.VTK_LINE_ID, dtype=np.uint8)

        # Prepare point and cell data
        # Point data needs to be duplicated for root and tip
        res_types = np.repeat(cloud.residue_types[res_indices], 2)
        canonical_indices = np.repeat(slot_indices, 2)

        # Define path
        if filename is None:
            filename = f"cloud_frame_{frame_idx:04d}"
        
        full_path = str(self.output_dir / filename)

        unstructuredGridToVTK(
            full_path,
            np.ascontiguousarray(coords[:, 0]),
            np.ascontiguousarray(coords[:, 1]),
            np.ascontiguousarray(coords[:, 2]),
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            pointData={
                "ResidueType": res_types.astype(np.int32),
                "SlotIndex": canonical_indices.astype(np.int32)
            },
            cellData={
                "SlotIndex": slot_indices.astype(np.int32)
            }
        )

    def export_sequence(self, cloud: TensorCloud, step: int = 1, prefix: str = "frame"):
        """Exports the entire trajectory or a subsampled version."""
        print(f"Exporting sequence to {self.output_dir}...")
        for f_idx in range(0, cloud.n_frames, step):
            self.export_frame(cloud, f_idx, filename=f"{prefix}_{f_idx:04d}")
            if f_idx % 10 == 0:
                print(f"  Processed frame {f_idx}/{cloud.n_frames}", end="\r")
        print(f"\nExport complete.")