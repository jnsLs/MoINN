import os
import ase
import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

from schnetpack.data.atoms import AtomsConverter
from schnetpack.environment import SimpleEnvironmentProvider

from moinn.evaluation.visualization import vis_type_ass_on_molecule


def get_model_input_from_smiles(smiles, device):

    # define converter ase.Atoms --> MoINN input
    atoms_converter = AtomsConverter(
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        device=device,
    )

    # define 3d molecule structure from smiles
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

    # get connectivity matrix
    con_mat = Chem.GetAdjacencyMatrix(rdkmol, useBO=False)
    con_mat = con_mat + np.eye(con_mat.shape[0], con_mat.shape[1])
    con_mat = np.expand_dims(con_mat, axis=0)
    con_mat = torch.tensor(con_mat, dtype=torch.float32).to(device)

    # define ase object which is converted to MoINN input
    at = ase.Atoms(positions=positions_3d, numbers=atomic_numbers)
    sample = atoms_converter(at)

    # add connectivity matrix
    sample["con_mat"] = con_mat

    return sample, rdkmol


smiles_dict = {
    "alanine_dipeptide": "CC(=O)NC(C)C(=O)NC",
    "decaalanine": "CC(C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)NC(C)C(=O)O)N",
}

if __name__ == "__main__":

    mdir = "/home/jonas/Documents/1-graph_pooling/moinn/trained_models/moinn_pretrained_10000"
    mpath = os.path.join(mdir, "best_model")
    device = torch.device('cuda')
    smiles = "CC(=O)NC(C)C(=O)NC"   # alanine-dipeptide
    cmap = plt.get_cmap("tab10")
    eval_dir = os.path.join(mdir, "eval")

    # get model
    model = torch.load(mpath, map_location=device)

    # get model input
    sample, rdkmol = get_model_input_from_smiles(smiles, device)

    # get model output
    result = model(sample)
    type_ass = result["type_assignments"][0]

    # get valid clusters and map colors to valid clusters only (for entire validation set)
    # this is done because it is hard to find more than 20 distinguishable colors
    non_empty_clusters = torch.load(os.path.join(eval_dir, "filled_clusters"))

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
    for k, values in node_colors.items():
        node_colors_tmp = []
        for v in values:
            for _ in range(percent[k, v].item()):
                node_colors_tmp.append(tuple(cmap(v)))
        node_colors[k] = node_colors_tmp

    fig_name = os.path.join(eval_dir, 'type_assignments')
    vis_type_ass_on_molecule(rdkmol, fig_name, node_colors)
