import os
import torch
from PIL import Image
from shutil import rmtree
import matplotlib.pyplot as plt
import numpy as np
import ase
from ase.io import write
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from tqdm import tqdm
import schnetpack as spk
import argparse

from moinn.evaluation import EnvironmentTypes
from moinn.evaluation.moieties import get_topk_moieties, spatial_assignments
from moinn.evaluation.visualization import vis_type_ass_on_molecule


def plot_table(figname, filled_clusters, cellText, rows, colors_text=None):

    fig = plt.figure(figsize=(25, 14))

    # bar plot of frequency of cluster types
    if colors_text is not None:
        colors = [cmap(idx) for idx in range(filled_clusters.shape[0])]
        plt.bar(range(filled_clusters.shape[0]), filled_clusters.numpy(), 0.4, color=colors)
    else:
        plt.bar(range(filled_clusters.shape[0]), filled_clusters.numpy(), 0.4)
    # Add a table at the bottom of the axes
    if colors_text is not None:
        the_table = plt.table(cellText=cellText,
                              rowLabels=rows,
                              cellColours=colors_text,
                              loc='bottom',
                              cellLoc="right")
    else:
        the_table = plt.table(cellText=cellText,
                              rowLabels=rows,
                              loc='bottom',
                              cellLoc="center")
    the_table.scale(0.93, 2.6)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.title('Environment Type Usage')
    plt.ylabel('#atoms assigned to certain type')
    plt.xticks([])
    fig.savefig(figname)


def get_node_colors(type_ass, non_empty_clusters):
    """
    non_empty_clusters is used to get valid clusters and map colors to valid clusters only (for entire validation set)
    this is done because it is hard to find more than 20 distinguishable colors
    """

    valid_cl = {}
    n_valid = 0
    for idx, entry in enumerate(non_empty_clusters):
        if entry == 1.:
            valid_cl[idx] = n_valid
            n_valid += 1

    # get assignment matrix for pentacene
    ass = type_ass

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
                node_colors_tmp.append(tuple(cmap(valid_cl[v])))
        node_colors[k] = node_colors_tmp

    return node_colors


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--modeldir", type=str)
    parser.add_argument("--eval_set_size", type=int, default=1000)
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    batch_size = 1  # batch size greater than 1 does not work yet for the sample analysis
    topk = 3
    cmap = plt.get_cmap("tab10")

    atom_names_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}

    eval_dir = os.path.join(args.modeldir, "eval")

    # create directory
    tmp_dir = os.path.join(eval_dir, "tmp")
    if os.path.exists(tmp_dir):
        print("existing evaluation results will be overwritten...")
        rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    # load dataset splits
    dataset = spk.AtomsData(args.datapath)
    split_path = os.path.join(args.modeldir, "split.npz")
    data_train, data_val, data_test = spk.data.train_test_split(
        dataset, split_file=split_path
    )
    data_test = data_test.create_subset([_ for _ in range(args.eval_set_size)])

    # chose loader
    test_loader = spk.data.AtomsLoader(data_test, batch_size=batch_size, num_workers=0, pin_memory=True)

    # load model
    modelpath = os.path.join(args.modeldir, "best_model")
    model = torch.load(modelpath, map_location=device)

    print("get used environment types ...")
    environment_types = EnvironmentTypes(test_loader, model, device)
    used_types, filled_clusters = environment_types.get_used_types()

    print("get moieties ...")
    substruc_indices, count_values, all_substructures_dense = get_topk_moieties(
        test_loader, model, used_types, topk, device
    )

    n_used_types = substruc_indices.shape[1]

    colors_dict = torch.unique(substruc_indices).tolist()
    colors_dict = {struc_idx: cmap(color_idx) for color_idx, struc_idx in enumerate(colors_dict)}

    substruc_indices = substruc_indices.tolist()

    colors = [[] for _ in range(len(substruc_indices))]
    for row_idx, row in enumerate(substruc_indices):
        row_tmp = []
        for col_idx, struc_idx in enumerate(row):
            if count_values[row_idx][col_idx].item() > 1.:
                row_tmp.append(all_substructures_dense[struc_idx]+", %d %%" % (count_values[row_idx][col_idx].item()))
                colors[row_idx].append(colors_dict[struc_idx])
            else:
                row_tmp.append("")
                colors[row_idx].append("w")
        substruc_indices[row_idx] = row_tmp

    # rows
    rows = ["env.-type"]
    for _ in range(topk):
        rows.append("moiety {}".format(_))
    # table entries
    cellText = [[str(_) for _ in range(n_used_types)]]
    cellText += substruc_indices
    # table cell coloring
    colors_text = [['w' for _ in range(n_used_types)]]
    colors_text += colors

    # in case we do not want to color all substructures
    n_colored_rows = 0
    for idx in range(n_colored_rows + 1, len(colors_text)):
        colors_text[idx] = ['w' for _ in range(n_used_types)]

    ####################################################################################################################
    # plot
    ####################################################################################################################

    # show most common substructures
    plot_table(
        figname=os.path.join(eval_dir, "common_moieties.pdf"),
        filled_clusters=filled_clusters[used_types],
        cellText=cellText,
        rows=rows,
        colors_text=colors_text
    )

    torch.save(used_types, os.path.join(eval_dir, "filled_clusters"))


    ####################################################################################################################
    # sample analysis
    ####################################################################################################################
    print("visualize env. types and moieties on exemplary molecules ...")
    for batch_idx, batch in tqdm(enumerate(test_loader)):

        if batch_idx >= args.n_examples:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # define rdkit molecule object
        atoms = batch["_atomic_numbers"][0].tolist()
        xyz_coordinates = batch["_positions"][0].tolist()

        at = ase.Atoms(positions=xyz_coordinates, numbers=atoms)
        at_tmp_path = os.path.join(eval_dir, "tmp", "at.xyz")
        write(at_tmp_path, at)

        try:
            raw_mol = Chem.MolFromXYZFile(at_tmp_path)
            mol = Chem.Mol(raw_mol)
            rdDetermineBonds.DetermineBonds(mol, charge=0)
        except:
            continue

        # get type assignments
        result = model(batch)
        type_ass = result["type_assignments"][0]

        con_mat = Chem.GetAdjacencyMatrix(mol, useBO=False)
        con_mat = con_mat + np.eye(con_mat.shape[0], con_mat.shape[1])
        con_mat = np.expand_dims(con_mat, axis=0)
        con_mat = torch.tensor(con_mat, dtype=torch.float32).to(device)
        result["graph"] = con_mat

        bead_ass = spatial_assignments(
            result["type_assignments"],
            result["graph"],
            batch["_atom_mask"]
        ).detach()[0]

        colors = []
        for row in bead_ass:
            colors.append([cmap(row.nonzero().tolist()[0][0])])  # could be buggy
        node_colors = {}
        for at_idx, color in enumerate(colors):
            node_colors[at_idx] = color

        fig_name = os.path.join(eval_dir, "tmp", "bead_assignments_{}".format(batch_idx))
        vis_type_ass_on_molecule(mol, fig_name, node_colors)

        node_colors = get_node_colors(type_ass.cpu(), used_types.float().tolist())
        fig_name = os.path.join(eval_dir, "tmp", "type_assignments_{}".format(batch_idx))
        vis_type_ass_on_molecule(mol, fig_name, node_colors)

        # store in one file
        images = [Image.open(x) for x in [os.path.join(eval_dir, "tmp", "type_assignments_{}.png".format(batch_idx)),
                                          os.path.join(eval_dir, "tmp", "bead_assignments_{}.png".format(batch_idx))]]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        new_im.save(os.path.join(eval_dir, "assignments_{}.png".format(batch_idx)))
