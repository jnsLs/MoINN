import ase
import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from schnetpack import Properties
from moinn.nn.neighbors import AdjMatrix, PairwiseDistances
from schnetpack.nn.cutoff import HardCutoff
import schnetpack as spk
from moinn.evaluation.visualization import vis_type_ass_on_molecule
import os
from torch import nn


smiles_dict = {
    "alanine_dipeptide": "CC(=O)NC(C)C(=O)NC",
    "decaalanine": "CC(C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)O)N",
}



########################################################################################################################
# main
########################################################################################################################

from schnetpack.data.atoms import AtomsConverter
from schnetpack.environment import SimpleEnvironmentProvider


mdir = "/home/jonas/Desktop/moinn_pretrained"
mpath = os.path.join(mdir, "best_model")
device = torch.device('cuda')

atoms_converter = AtomsConverter(
    environment_provider=SimpleEnvironmentProvider(),
    collect_triples=False,
    device=device,
)


# get model
model = torch.load(mpath, map_location=device)


# define 3d molecule structure from smiles
smiles = "CC(=O)NC(C)C(=O)NC"   # alanine-dipeptide
rdkmol = Chem.MolFromSmiles(smiles)
rdkmol = Chem.AddHs(rdkmol)
AllChem.EmbedMolecule(rdkmol, randomSeed=42)

# optimize structure
AllChem.MMFFOptimizeMolecule(rdkmol)
# get positions
n_nodes = rdkmol.GetNumAtoms()
pos = []
for i in range(n_nodes):
    conformer_pos = rdkmol.GetConformer().GetAtomPosition(i)
    pos.append([conformer_pos.x, conformer_pos.y, conformer_pos.z])
positions_3d = torch.tensor(pos)
# get atomic numbers
atomic_numbers = []
for atom in rdkmol.GetAtoms():
    atomic_numbers.append(atom.GetAtomicNum())
atomic_numbers = torch.tensor(atomic_numbers)

con_mat = Chem.GetAdjacencyMatrix(rdkmol, useBO=False)
con_mat = con_mat + np.eye(con_mat.shape[0], con_mat.shape[1])
con_mat = np.expand_dims(con_mat, axis=0)
con_mat = torch.tensor(con_mat, dtype=torch.float32).to(device)
print("con_mat: ", con_mat)
print("pos: ", positions_3d)
print("at_nums: ", atomic_numbers)

at = ase.Atoms(positions=positions_3d, numbers=atomic_numbers)
sample = atoms_converter(at)

print(sample)

sample["con_mat"] = con_mat

result = model(sample)
type_ass = result["type_assignments"][0]

# get valid clusters and map colors to valid clusters only (for entire validation set)
# this is done because it is hard to find more than 20 distinguishable colors
non_empty_clusters = torch.load(os.path.join(mdir, "filled_clusters"))
print(non_empty_clusters)

for idx in range(30):
    if idx in torch.argmax(type_ass, dim=-1):
        non_empty_clusters[idx] = True

# filter out non-used type assignments
n_atoms = type_ass.shape[0]
n_used_types = non_empty_clusters.sum().item()
ass = torch.empty(size=(n_atoms, n_used_types))
for at_idx in range(n_atoms):
    ass[at_idx] = type_ass[at_idx][non_empty_clusters]

# get list of node colors (integers)
node_colors = {}
percent = torch.zeros_like(ass).int()
for node_idx, node_ass in enumerate(ass):
    percent[node_idx] = (node_ass * 20).int()
    node_colors[node_idx] = (node_ass > 0.05).nonzero()[:, 0].tolist()

# apply color map
cmap = plt.get_cmap("tab20")
for k, values in node_colors.items():
    node_colors_tmp = []
    for v in values:
        for _ in range(percent[k, v].item()):
            node_colors_tmp.append(tuple(cmap(v)))
    node_colors[k] = node_colors_tmp

print(torch.max(type_ass, dim=-1))
print(torch.argmax(type_ass, dim=-1))

fig_name = '/home/jonas/Documents/tmp/type_assignments'
vis_type_ass_on_molecule(rdkmol, fig_name, node_colors)



"""        
        
        # define sample (has the shape of batch with size 1)
        sample = batch_defining(positions_3d, atomic_numbers)
        if con_mat is None:
            # load molecule properties
            positions = sample[Properties.R]
            atom_mask = sample[Properties.atom_mask]
            # calculate adjacency matrix
            distances = PairwiseDistances()
            adjacency_min = AdjMatrix(HardCutoff, [1.6], normalize=False, zero_diag=True)
            r_ij = distances(positions)
            con_mat = adjacency_min(atom_mask=atom_mask, r_ij=r_ij)
        else:
            con_mat = torch.tensor(con_mat).unsqueeze(0).float()
        sample["con_mat"] = con_mat



# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--load", help="name or dataset", required=True, type=str)
parser.add_argument("--molecule_id", help="trivial name", required=True, type=str)
parser.add_argument("--modeldir", help="directory/of/model", required=True, type=str)
args = parser.parse_args()

# get model
modelpath = os.path.join(args.modeldir, "best_model")
schnet_model = torch.load(modelpath, map_location=torch.device('cpu'))

# get molecule
sample, mol = load_sample(args)
# get type assignments
result = schnet_model(sample)
type_ass = result["type_assignments"][0]

# get valid clusters and map colors to valid clusters only (for entire validation set)
# this is done because it is hard to find more than 20 distinguishable colors
non_empty_clusters = torch.load(os.path.join(args.modeldir, "filled_clusters"))
print(non_empty_clusters)

for idx in range(30):
    if idx in torch.argmax(type_ass, dim=-1):
        non_empty_clusters[idx] = True

# filter out non-used type assignments
n_atoms = type_ass.shape[0]
n_used_types = non_empty_clusters.sum().item()
ass = torch.empty(size=(n_atoms, n_used_types))
for at_idx in range(n_atoms):
    ass[at_idx] = type_ass[at_idx][non_empty_clusters]

# get list of node colors (integers)
node_colors = {}
percent = torch.zeros_like(ass).int()
for node_idx, node_ass in enumerate(ass):
    percent[node_idx] = (node_ass * 20).int()
    node_colors[node_idx] = (node_ass > 0.05).nonzero()[:, 0].tolist()

# apply color map
cmap = plt.get_cmap("tab20")
for k, values in node_colors.items():
    node_colors_tmp = []
    for v in values:
        for _ in range(percent[k, v].item()):
            node_colors_tmp.append(tuple(cmap(v)))
    node_colors[k] = node_colors_tmp

print(torch.max(type_ass, dim=-1))
print(torch.argmax(type_ass, dim=-1))

fig_name = '/home/jonas/Documents/tmp/type_assignments'
vis_type_ass_on_molecule(mol, fig_name, node_colors)
"""