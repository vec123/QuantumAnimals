
Examining Data of Molecular Simulations


Reading Papers such as EquiJump has lead to the discovery of a new dataset. Since EquiJump does not provide a git implementation, lets see if we can make one.
The necessary compute will be a hurdle, but to code it up and see if it can at least overfit one can use the smallest protein in the dataset.

Step 1: Understand the Data and choose a Representation 

Here a snapshot of the molecular dynamics simulation (see the notebook script):
<p align="center">
  <img src="images/mol_traj_visualization.png" alt="MD Visualization">
</p>

For Neural Networks a different data representation will be required. Numpy will serve as intermediary since it is relatively slim and easily integrated into pytorch or jax computations. Paraview will serve as visualization platform to ensure plausibility along each transformation. A possibility would be to use graph neural networks on an all-atom radius graph. The following image shows all atoms as a point cloud and the corresponding  graph with a five Angstrom radius:
<p align="center">
  <img src="images/Numpy_Points_Paraview.png" alt="Atom Point Cloud" height="300">
  <img src="images/All_Atom_Graph_Paraview.png" alt="All Atom Radius Graph" height="300">
</p>


This representation does not scale well. Other papers, such as EquiJump and Ophiuchus use a residue representation. Each Amino-Acid is represented by its label, the position of its C_alpha atom and the positions of other atoms relative to the C_alpha atom. A protein can be understood as a one-dimensional sequence of these residue descriptions. This description scales better. The following image shows the residues and the corresponding atoms.

<p align="center">
  <img src="images/Residue_Representation_Paraview.png" alt="Resiude Representation" height="300">
</p>


