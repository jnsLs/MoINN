from scripts.xyz2mol import convert
import os
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from shutil import rmtree
from matplotlib import pyplot as plt
from schnetpack.nn.cutoff import HardCutoff
from schnetpack.utils.spk_utils import read_from_json

from schnetpack.clustering.neighbors import PairwiseDistances, AdjMatrix
from schnetpack.clustering.utils.visualization import UsedBeadTypes, ClusteredSubstructures, vis_type_ass_on_molecule
from schnetpack.clustering.utils.coarse_graining import spatial_assignments
from schnetpack.clustering.utils.data import get_loader

from torch import nn

plt.rcParams.update({'font.size': 25})


class EndToEndModel(nn.Module):
    """
    Join a representation model with clustering module.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_module (e.g. clustering modul): Output block of the model. Needed for predicting properties.

    Returns:
         dict: property predictions
    """

    def __init__(self, representation, output_module):
        super(EndToEndModel, self).__init__()
        self.representation = representation
        self.output_module = output_module

    def forward(self, inputs):
        """
        Forward representation output through output modules.
        """
        inputs["representation"] = self.representation(inputs)
        outs = self.output_module(inputs)
        return outs


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
    print(valid_cl)

    # get assignment matrix for pentacene
    ass = type_ass

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
                node_colors_tmp.append(tuple(cmap(valid_cl[v])))
                #node_colors_tmp.append(tuple(cmap(v)))
        node_colors[k] = node_colors_tmp

    return node_colors


def do_statistics(dataloader, clustering_model):
    # get non empty clusters
    used_clusters = UsedBeadTypes(args.max_clusters)
    contained_atom_types = ClusteredSubstructures()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(torch.device('cuda')) for k, v in batch.items()}

        # get type assignments
        result = clustering_model(batch)

        # define rdkit molecule object
        atoms = batch["_atomic_numbers"][0].tolist()
        xyz_coordinates = (batch["_positions"][0]).tolist()#*0.529177).tolist()
        charge = 0.
        mol, _ = convert(atoms, xyz_coordinates, charge)

        adjacency = AdjMatrix(HardCutoff, [1.6], normalize=False, zero_diag=False).to(torch.device('cuda'))
        distances = PairwiseDistances().to(torch.device('cuda'))
        r_ij = distances(batch["_positions"])#*0.529177)
        result["graph"] = adjacency(batch["_atom_mask"], r_ij)

        #con_mat = Chem.GetAdjacencyMatrix(mol, useBO=False)
        #con_mat = con_mat + np.eye(con_mat.shape[0], con_mat.shape[1])
        #con_mat = np.expand_dims(con_mat, axis=0)
        #con_mat = torch.tensor(con_mat, dtype=torch.float32).to(torch.device("cuda"))
        #result["graph"] = con_mat

        contained_atom_types.add_batch(batch, result)
        used_clusters.add_batch(result)

        if batch_idx + 2 > 1000:
            break

    tot_ass = used_clusters.tot_ass
    filled_clusters = tot_ass.sum(dim=0)

    _, used_substructures, all_substructures_dense, count_values = contained_atom_types.plot()

    mask = filled_clusters > 1.

    torch.save(mask, os.path.join(args.modeldir, "filled_clusters"))

    used_substructures = used_substructures.transpose(0, 1)[mask]
    used_substructures = used_substructures.transpose(0, 1)

    count_values = count_values.transpose(0, 1)[mask]
    count_values = count_values.transpose(0, 1)
    count_values = count_values / count_values.sum(0) * 100.

    cmap = plt.get_cmap("tab20")
    colors_dict = torch.unique(used_substructures).tolist()
    colors_dict = {struc_idx: cmap(color_idx) for color_idx, struc_idx in enumerate(colors_dict)}

    n_non_empty_clusters = used_substructures.shape[1]
    used_substructures = used_substructures.tolist()

    colors = [[] for _ in range(len(used_substructures))]
    for row_idx, row in enumerate(used_substructures):
        row_tmp = []
        for col_idx, struc_idx in enumerate(row):
            if count_values[row_idx][col_idx].item() > 1.:
                row_tmp.append(all_substructures_dense[struc_idx]+", %d %%" % (count_values[row_idx][col_idx].item()))
                colors[row_idx].append(colors_dict[struc_idx])
            else:
                row_tmp.append("")
                colors[row_idx].append("w")
        used_substructures[row_idx] = row_tmp

    return filled_clusters, mask, used_substructures, n_non_empty_clusters, colors


def plot_table(figname, filled_clusters, cellText, rows, colors_text=None):

    fig = plt.figure(figsize=(25, 14))

    # bar plot of frequency of cluster types
    if colors_text is not None:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(idx) for idx in range(filled_clusters.shape[0])]
        plt.bar(range(filled_clusters.shape[0]), filled_clusters.numpy(), 0.4, color=colors)
    else:
        plt.bar(range(filled_clusters.shape[0]), filled_clusters.numpy(), 0.4)
    # Add a table at the bottom of the axes
    if colors_text is not None:
        the_table = plt.table(cellText=cellText,
                              #rowLabels=rows,
                              cellColours=colors_text,
                              loc='bottom',
                              cellLoc="right")
    else:
        the_table = plt.table(cellText=cellText,
                              #rowLabels=rows,
                              loc='bottom',
                              cellLoc="center")
    the_table.scale(0.93, 2.6)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.title('Cluster Type Usage')
    plt.ylabel('#atoms assigned to certain type')
    plt.xticks([])
    fig.savefig(figname)


########################################################################################################################
# setup
########################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--modeldir", help="directory/of/clustering/model", type=str, required=True)
parser.add_argument("--datapath_local", help="path/to/dataset", type=str, required=True)
args = parser.parse_args()

jsonpath = os.path.join(args.modeldir, "args.json")
args = argparse.Namespace(**vars(args), **vars(read_from_json(jsonpath)))

modelpath = os.path.join(args.modeldir, "best_model")
eval_dir = os.path.join(args.modeldir, "eval_images")

# create directory
tmp_dir = os.path.join(eval_dir, "tmp")
if os.path.exists(tmp_dir):
    print("existing evaluation results will be overwritten...")
    rmtree(tmp_dir)
os.makedirs(tmp_dir)

dataloader = get_loader(args.datapath_local, args.modeldir, "test")
model = torch.load(modelpath, map_location=torch.device('cuda'))


########################################################################################################################
# statistics
########################################################################################################################
filled_clusters, mask, used_substructures, n_non_empty_clusters, colors = do_statistics(dataloader, model)

# rows
rows = ["cluster-type"]
for _ in range(5):
    rows.append("sub-struc. {}".format(_))
# table entries
cellText = [[str(_) for _ in range(n_non_empty_clusters)]]
cellText += used_substructures
# table cell coloring
colors_text = [['w' for _ in range(n_non_empty_clusters)]]
colors_text += colors

# in case we do not want to color all substructures
n_colored_rows = 0
for idx in range(n_colored_rows+1, len(colors_text)):
    colors_text[idx] = ['w' for _ in range(n_non_empty_clusters)]

########################################################################################################################
# plot
########################################################################################################################

# show most common substructures
plot_table(
    figname=os.path.join(eval_dir, "common_substructures.pdf"),
    filled_clusters=filled_clusters[mask],
    cellText=cellText,
    rows=rows,
    colors_text=colors_text
)


########################################################################################################################
# sample analysis
########################################################################################################################
for batch_idx, batch in enumerate(dataloader):
    batch = {k: v.to(torch.device('cuda')) for k, v in batch.items()}

    # define rdkit molecule object
    atoms = batch["_atomic_numbers"][0].tolist()
    xyz_coordinates = batch["_positions"][0].tolist()
    charge = 0.
    mol, _ = convert(atoms, xyz_coordinates, charge)

    # get type assignments
    result = model(batch)
    type_ass = result["type_assignments"][0]

    adjacency = AdjMatrix(HardCutoff, [1.6], normalize=False, zero_diag=False).to(torch.device('cuda'))
    distances = PairwiseDistances().to(torch.device('cuda'))
    r_ij = distances(batch["_positions"])
    result["graph"] = adjacency(batch["_atom_mask"], r_ij)

    #con_mat = Chem.GetAdjacencyMatrix(mol, useBO=False)
    #con_mat = con_mat + np.eye(con_mat.shape[0], con_mat.shape[1])
    #con_mat = np.expand_dims(con_mat, axis=0)
    #con_mat = torch.tensor(con_mat, dtype=torch.float32).to(torch.device("cuda"))
    #result["graph"] = con_mat

    bead_ass = spatial_assignments(
        result["type_assignments"],
        result["graph"],
        batch["_atom_mask"]
    ).detach()[0]
    cmap = plt.get_cmap("tab20")
    colors = []
    for row in bead_ass:
        colors.append([cmap(row.nonzero().tolist()[0][0])])  # could be buggy
    node_colors = {}
    for at_idx, color in enumerate(colors):
        node_colors[at_idx] = color

    fig_name = os.path.join(eval_dir, "tmp", "bead_assignments_{}".format(batch_idx))
    vis_type_ass_on_molecule(mol, fig_name, node_colors)

    node_colors = get_node_colors(type_ass.cpu(), mask.float().tolist())
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

    if batch_idx + 2 > 20:
        break

