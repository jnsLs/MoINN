import ase
import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from schnetpack import Properties
from schnetpack.clustering.neighbors import AdjMatrix, PairwiseDistances
from schnetpack.nn.cutoff import HardCutoff
import schnetpack as spk
from scripts.xyz2mol import convert
from schnetpack.clustering.utils.visualization import vis_type_ass_on_molecule
import os
from torch import nn


smiles_dict = {
    "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "adenine": "C1=NC2=NC=NC(=C2N1)N",
    "ethanol": "CCO",
    "aspirin": "CC(OC1=C(C(=O)O)C=CC=C1)=O CC(=O)OC1C=CC=CC=1C(O)=O",
    "malondialdehyde": "C(C=O)C=O",
    "uracil": "C1=CNC(=O)NC1=O",
    "L-Alanyl-L-alanine": "CC(C(=O)NC(C)C(=O)O)N",
    "chignolin": "CC(C(C(=O)NC(CC1=CNC2=CC=CC=C21)C(=O)NCC(=O)O)NC(=O)CNC(=O)C(C(C)O)"
                 "NC(=O)C(CCC(=O)O)NC(=O)C3CCCN3C(=O)C(CC(=O)O)NC(=O)C(CC4=CC=C(C=C4)O)NC(=O)CN)O",
    "alanine_dipeptide": "CC(=O)NC(C)C(=O)NC",
    "water": "O",
    "methyl": "[CH3]",
    "methane": "C",
    "methyl_formate": "COC=O",
    "methanol": "CO",
    "formyl_radical": "[CH]=O",
    "benzaldyhde": "C1=CC=C(C=C1)C=O",
    "acetaldehyde": "CC=O",
    "formic_acid": "C(=O)CN",
    "propionaldehyde": "CCC=O",
    "mda_prot_trans": "O=CC=CO",
    "cl": "CNC(C)C",
    "p1": "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O",
    "p2": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2COC3C2CCO3)O)S(=O)(=O)C4=CC5=C(C=C4)N=C(S5)NC6CCN(CC6)C7CCCC7",
    "p3": "CC(C)CC(C(=O)O)NC(=O)C(CC1=CC=C(C=C1)O)NC(=O)C(CCCCN)NC(=O)C(CO)NC(=O)C(CCCN=C(N)N)NC(=O)C(CCC(=O)N)NC(=O)C(CO)NC(=O)C2CCCN2C(=O)C(CCCN=C(N)N)NC(=O)C(CCCCN)NC(=O)C(CCC(=O)N)N",
    "azithromycin": "CCC1C(C(C(N(CC(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O",
    "b12": "CC1=CC2=C(C=C1C)N(C=N2)C3C(C(C(O3)CO)OP(=O)([O-])OC(C)CNC(=O)CCC4(C(C5C6(C(C(C(=C(C7=NC(=CC8=NC(=C(C4=N5)C)C(C8(C)C)CCC(=O)N)C(C7(C)CC(=O)N)CCC(=O)N)C)[N-]6)CCC(=O)N)(C)CC(=O)N)C)CC(=O)N)C)O.[C-]#N.[Co+3]",
    "cholesterin": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "insulin": "CCC(C)C1C(=O)NC2CSSCC(C(=O)NC(CSSCC(C(=O)NCC(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(C(=O)NC(CSSCC(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC(=O)C(NC2=O)CO)CC(C)C)CC3=CC=C(C=C3)O)CCC(=O)N)CC(C)C)CCC(=O)O)CC(=O)N)CC4=CC=C(C=C4)O)C(=O)NC(CC(=O)N)C(=O)O)C(=O)NCC(=O)NC(CCC(=O)O)C(=O)NC(CCCNC(=N)N)C(=O)NCC(=O)NC(CC5=CC=CC=C5)C(=O)NC(CC6=CC=CC=C6)C(=O)NC(CC7=CC=C(C=C7)O)C(=O)NC(C(C)O)C(=O)N8CCCC8C(=O)NC(CCCCN)C(=O)NC(C(C)O)C(=O)O)C(C)C)CC(C)C)CC9=CC=C(C=C9)O)CC(C)C)C)CCC(=O)O)C(C)C)CC(C)C)CC2=CNC=N2)CO)NC(=O)C(CC(C)C)NC(=O)C(CC2=CNC=N2)NC(=O)C(CCC(=O)N)NC(=O)C(CC(=O)N)NC(=O)C(C(C)C)NC(=O)C(CC2=CC=CC=C2)N)C(=O)NC(C(=O)NC(C(=O)N1)CO)C(C)O)NC(=O)C(CCC(=O)N)NC(=O)C(CCC(=O)O)NC(=O)C(C(C)C)NC(=O)C(C(C)CC)NC(=O)CN",
    "trypsin": "CC1=CC(=CC2=C1C=CC(=O)O2)NC(=O)C(CCCN=C(N)N)NC(=O)C(CO)NC(=O)C(CC3=CC=CC=C3)NC(=O)OC(C)(C)C.CC(=O)O",
    "trypsin_2": "CCC(C)C(C(=O)O)NC(=O)C(CC1=CC=CC=C1)NC(=O)C(C)NC(=O)C(CCCN=C(N)N)NC(=O)C(CS)NC(=O)C2CCCN2C(=O)CNC(=O)C(CCCN=C(N)N)N",
    "covid_1": "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl",
    "covid_2": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2COC3C2CCO3)O)S(=O)(=O)C4=CC5=C(C=C4)N=C(S5)NC6CCN(CC6)C7CCCC7",
    "fingolimod": "CCCCCCCCC1=CC=C(C=C1)CCC(CO)(CO)N",
    "resolvin": "CCC=CCC(C=CC=CC=CC=CC(C(CC=CCCC(=O)O)O)O)O",
    "lypo": "CCCCCC(C=CC=CCC=CCC=CCCCC(=O)[O-])O",
    "asp1": "CCC=CCC(C=CC=CC=CC(CC=CCC=CCCC(=O)[O-])O)O",
    "decaalanine": "CC(C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)O)N",
    "rdkit": "C/C=C/CC(C)C(O)C1C(=O)NC(CC)C(=O)N(C)CC(=O)N(C)C(CC(C)C)C(=O)NC(C(C)C)C(=O)N(C)C(CC(C)C)C(=O)NC(C)C(=O)NC(C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(C(C)C)C(=O)N1C"
}


def batch_defining(positions, atomic_numbers):
    """defines batch in schnet format"""
    # get number of nodes
    n_nodes = atomic_numbers.shape[0]
    # create batch
    batch = {
        "_positions": positions.unsqueeze(0),
        "_atomic_numbers": atomic_numbers.unsqueeze(0),
        "_neighbor_mask": torch.ones((1, n_nodes, n_nodes - 1), dtype=torch.float),
        "_atom_mask": torch.ones((1, n_nodes), dtype=torch.float),
        "_cell_offset": torch.zeros((1, n_nodes, n_nodes - 1, 3), dtype=torch.float),
        "_cell": torch.zeros((1, 3, 3), dtype=torch.float)
    }
    # neighbors
    neighborhood_idx = np.tile(
        np.arange(n_nodes, dtype=np.float32)[np.newaxis], (n_nodes, 1)
    )
    neighborhood_idx = neighborhood_idx[
        ~np.eye(n_nodes, dtype=bool)
    ].reshape(n_nodes, n_nodes - 1)
    batch["_neighbors"] = torch.tensor(neighborhood_idx, dtype=torch.long).unsqueeze(0)
    return batch


def load_sample(args):
    """
    Returns batch of size 1 for respective molecule (optimized structure) and conformer of the molecule (rdkit object).
    The conformer corresponds to the energy optimized structure of the molecule.
    """
    def _rdkmol2batch(rdkmol, con_mat):
        # optimize structure
        #AllChem.UFFOptimizeMolecule(rdkmol)
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
        return sample

    #def _idx_loader():
    #    """Define pseudo data loader. Returns rdkit molecule object for chosen data sample."""
    #    # select molecule from dataset
    #    datapath = "/home/jonas/Documents/datasets/schnetpack/md17/{}.db".format(args.molecule_id)
    #    dataset = spk.datasets.MD17(datapath, args.molecule_id)
    #    # get molecule properties
    #    atoms = dataset[10000]["_atomic_numbers"].tolist()
    #    xyz_coordinates = dataset[10000]["_positions"].tolist()
    #    charge = 0.
    #    # return rdkit mol object and connectivity matrix
    #    return convert(atoms, xyz_coordinates, charge)

    def _idx_loader():
        """Define pseudo data loader. Returns rdkit molecule object for chosen data sample."""
        # select molecule from dataset
        dpath = "/home/jonas/Documents/datasets/schnetpack/qm9/qm9.db"
        dataset = spk.AtomsData(dpath)

        # get molecule properties

        split_path = os.path.join(args.modeldir, "split.npz")
        data_train, data_val, data_test = spk.data.train_test_split(
            dataset, split_file=split_path
        )

        atoms = data_test[1395]["_atomic_numbers"].tolist()
        xyz_coordinates = data_test[1395]["_positions"].tolist()
        charge = 0.
        # return rdkit mol object and connectivity matrix
        return convert(atoms, xyz_coordinates, charge)

    def _name_loader():
        """
        Define pseudo data loader for a single molecule based on its trivial name. Provided the particular molecule
        is included in our trivial name dictionary, this function generates a conformer for the respective molecule.
        """
        # define 3d molecule structure from smiles
        smiles = smiles_dict[args.molecule_id]
        rdkmol = Chem.MolFromSmiles(smiles)
        rdkmol = Chem.AddHs(rdkmol)
        AllChem.EmbedMolecule(rdkmol, randomSeed=42)
        # return rdkit mol object
        return rdkmol, None

    # define appropriate data loader
    if args.load == "dataset":
        mol_loader = _idx_loader
    elif args.load == "name":
        mol_loader = _name_loader
    else:
        raise NotImplementedError("data loader not implemented, yet")

    rdkmol, con_mat = mol_loader()
    sample = _rdkmol2batch(rdkmol, con_mat)
    return sample, rdkmol


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