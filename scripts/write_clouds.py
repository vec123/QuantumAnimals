
import os
from dotenv import load_dotenv

from src.preprocessing.transform import TrajectoryTransformer
from src.preprocessing.residue_registry import AtomRegistry
from src.visualizations.visualizer import CloudVisualizer
load_dotenv()

project_path = os.getenv("PROJECT_PATH")
resiude_map = os.getenv("RESIDUE_MAP_PATH")
md_data_path = os.getenv("MDSIM_DATASET_PATH")
tc_data_path = os.getenv("TENSOR_CLOUD_STORAGE")
vtk_dir = os.getenv("VTK_DIR")

xtc_trajectory = "chignolin_trajectories/filtered/e1s1_chignolin_50ns_0\e1s1_chignolin_50ns_0-ADRIA_CHIG_ADAPTIVE_crystal_ss_contacts_50_chignolin_0-0-1-RND3469_9.filtered.xtc"
tc_trajectory = "chignolin_trajectories/filtered/e1s1_chignolin_50ns_0\e1s1_chignolin_50ns_0-ADRIA_CHIG_ADAPTIVE_crystal_ss_contacts_50_chignolin_0-0-1-RND3469_9.filtered.pkl"
pdb_file = "chignolin_trajectories/filtered/filtered.pdb"


registry = AtomRegistry(resiude_map)
engine = TrajectoryTransformer(registry, max_slots=15)

# Execute
cloud = engine.transform(pdb_path = os.path.join(md_data_path,pdb_file), xtc_path = os.path.join(md_data_path,xtc_trajectory))

test_protein = os.path.join(tc_data_path, tc_trajectory)
cloud.save(test_protein)

visualizer = CloudVisualizer(output_dir=vtk_dir)
visualizer.export_sequence(cloud, step=1, prefix="chignolin_star_cloud")