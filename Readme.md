
Examining Data of Molecular Simulations


Reading papers such as EquiJump has lead to the discovery of a new dataset. Since EquiJump does not provide a git implementation, lets see if we can make one.
The necessary compute will be a hurdle (they train on 2-4 A100 GPUs), but to code it up and see if it can at least overfit, one can simply use the smallest protein in the dataset.

Let us start with Step 1: Understand the Data and choose a representation. 
This is one of the most important steps in artificial intelligence and machine learning which is sometimes overlooked. 
The representation decides what information is available, how it can be processed, which biases are introduced. 
It is a crucial element.

Let us start with the raw data. 
Here a snapshot of the molecular dynamics simulation (see the notebook script):
<p align="center">
  <img src="images/representation/mol_traj_visualization.png" alt="MD Visualization">
</p>
One can see bonds and atom types as well as arrows representing the backbone. 
Any chemist beyond high-school will admit that these drawn edges, i.e. the bonds, are not really physical. 
<br>
Let us instead assume that the molecular dynamics are determined completely by the positions of all atoms. This assumption is also known as the Born-Oppenheimer or the adaibatic approximation. The electron time-scales are much smaller, justifying the simplifying (and often, but not always, good) assumption that the electron wave-function instantaneously settles into the ground state. The atom positions are easily represented as a labeled point cloud. Numpy will serve as intermediary since it is relatively slim and easily integrated into pytorch or jax computations. Paraview will serve as visualization platform to ensure plausibility along each transformation. A possibility would be to use graph neural networks on an all-atom radius graph. The following image shows all atoms as a point cloud and the corresponding  graph with a five Angstrom radius:
<p align="center">
  <img src="images/representation/Numpy_Points_Paraview.png" alt="Atom Point Cloud" height="300">
  <img src="images/representation/All_Atom_Graph_Paraview.png" alt="All Atom Radius Graph" height="300">
</p>


This representation does not scale well. Other papers, such as EquiJump and Ophiuchus use a coarse residue representation. Each Amino-Acid is represented by its label, the position of its C_alpha atom and the positions of other atoms relative to the C_alpha atom. A protein can be understood as a one-dimensional sequence of these residue descriptions. This description scales better. The following image shows the residues and the corresponding atoms.
Aparently it is common to consider only the heavy atoms. Then a residual has maximally 13 such connections.


<p align="center">
  <img src="images/representation/Residue_Representation_Paraview.png" alt="Atom Point Cloud" height="300">
  <img src="images/representation/Heavy_Residue_Representation_Paraview.png" alt="All Atom Radius Graph" height="300">
</p>


In EquiJump a protein is represented as a Tensor Cloud. 
A Tensor Cloud is set, whose elements are tuples. Each tuple is a tensor V_i associated to a 3D position P_i, in essence (V_i, P_i), where V_i is a tensor of irreducible representations with degree cutoff l_max.


The i-th resiudal will be represented as {R_i, P_i, V_i} with R_i being its label, P_i being the position of the C_alpha atom and V_i being a 13x3 matrix of  l=1 representations with multiplicity 13. These features represent the relative distances of the heavy atoms. If a residual has less than 13 heavy atoms, the representation is padded. The ordering of the 13 features requires a canoncial ordering of the heavy atoms.

<p align="center">
  <img src="images/representation/TC_Representation.png" alt="Atom Point Cloud" height="300">
  <img src="images/representation/Residue_Mapping.png" alt="All Atom Radius Graph" height="300">
</p>

<p align="center">
  <img src="images/representation/Residue_1_2_TYR.png" alt="Atom Point Cloud" height="300">
  <img src="images/representation/Residue_3_4_ASP_PRO.png" alt="All Atom Radius Graph" height="300">
</p>



Step 2: Building the NN modules

Three mechanisms are implemented.

The Self-Interaction Mechanism (updates the tensor cloud of each residual independently by performing tensor-products with its own features) 
<p align="center">
  <img src="images/computing/SelfInteraction_pseudocode.png" alt="SelfInteraction Pseudocode" height="300">
</p>
the Spatial Convolution (essentially message passing between residual representations to update the i-th residual)
<p align="center">
  <img src="images/computing/SpatialConvolution_pseudocode.png" alt="SpatialConvolution Pseudocode" height="300">
</p>
the full model, combines Self-Interaction with Spatial Convolutions to outputs the target irrep arrays for each residual (node).
<p align="center">
  <img src="images/computing/EquiJumpDeepNetwork_pseudocode.png" alt="SpatialConvolution Pseudocode" height="300">
</p>
This full network will be the backbone for the training. It receives a tensor-cloud as input and outputs the specified geometric irreps.
The irreps of each node can be specified independently. 
<br>
This enables the implementation of a specific information flow in which one computes
<br>
1: latent geometric features, conditioned on a scalar field of one-hot encodings related to the the residual label (i.e. the residual field) and the tensor cloud at time t initialized with V_ij as relative distances from the C_alpha atom
<br>
2: The output features can be added to the tensor cloud at time tau, which is obtained by stochastic interpolation. Additionally, a scalar field representing the time tau is added. This is the input tensor cloud to the following four networks, each tasked with the approximation of the position and feature drift and noise respectively.

<p align="center">
  <img src="images/computing/EquiJump_full.png" alt="SpatialConvolution Pseudocode" height="300">>
</p>

Lets us test the information flow:
<p align="center">
  <img src="images/computing/module_test_1.png" alt="Atom Point Cloud" height="300">
  <img src="images/computing/module_test_2.png" alt="All Atom Radius Graph" height="300">
</p>
The input was a scalar and 14 3d vectors for each residual and the output was 14 3d vectord for each residual. 
The intermediate information flow consists in gated equivariant updates with irreps up to order 3. 

Step 3: 

The training, based on stochastic interpolants, interprets the model output as drift and scores of an end-point fixed focker-plank density evolution. By choosing the interpolant (here a linear interpolant) between endpoints, a differentiable objective can be formulated for both quantities. 

The state of the SDE is X = (P,V), where P is a 3D position and V a tensor product of irreps up to order l_max. By choosing a basis for the representations and treating the tensor as high-dimensional vector in that basis, the SDE on the tensorial part of X can be written as dV = b(X,V)dt + sigma(X,V)dW_t, under the condition that b(X,V) and sigma(X,V) be equivariant. By working in this basis, the linear stochastic interpolaton V_t = t V_1 + (1-t)V_0 + sigma(t)z_t remains valid, i.e. at each time t, V_t represents the irreps.

(ToDo)


Step 4: 

Validation by projecting the energy landscapes of molecular trajectories onto the 2 principal TiCA components.

(ToDo)