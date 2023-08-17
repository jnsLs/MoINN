import os
import schnetpack as spk
import torch
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
from shutil import rmtree

from moinn.clustering.utils.coarse_graining import spatial_assignments, get_hard_assignments
from moinn.clustering.neighbors import PairwiseDistances, AdjMatrix
from schnetpack.nn.cutoff import HardCutoff


def get_used_types(loader, model, device=torch.device('cuda')):

    for batch_idx, batch in tqdm(enumerate(loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        results = model(batch)

        type_ass = results["type_assignments"].detach()

        if batch_idx == 0:
            max_n_clusters = type_ass.to(torch.device("cpu")).shape[-1]
            tot_ass = torch.zeros((30, max_n_clusters))

        # sum up type assignments
        n_nodes = type_ass.shape[1]
        for ass in type_ass.cpu():
            tot_ass[:n_nodes, :] += ass

        max_in_sample = torch.max(type_ass, dim=1)[0]
        max_in_batch = torch.max(max_in_sample, dim=0, keepdim=True)[0]

        if batch_idx == 0:
            max_in_batch_prev = torch.zeros_like(max_in_batch)

        comp = torch.concat((max_in_batch, max_in_batch_prev), dim=0)
        max_in_batch = torch.max(comp, dim=0, keepdim=True)[0]
        max_in_batch_prev = max_in_batch.clone()

        mask = max_in_batch.cpu() > 0.05

        filled_clusters = tot_ass.sum(dim=0)

    return mask[0], filled_clusters


def get_moieties(loader, model, mask, device=torch.device('cuda')):

    for batch_idx, batch in tqdm(enumerate(loader)):

        batch = {k: v.to(device) for k, v in batch.items()}
        results = model(batch)

        if batch_idx == 0:
            max_n_clusters = results["type_assignments"].detach().to(torch.device("cpu")).shape[-1]
            type_dict = {idx: [] for idx in range(max_n_clusters)}

        # get connectivity matrix
        adjacency = AdjMatrix(HardCutoff, [1.6], normalize=False, zero_diag=False).to(device)
        distances = PairwiseDistances().to(device)
        r_ij = distances(batch["_positions"])
        results["graph"] = adjacency(batch["_atom_mask"], r_ij)

        # get bead assignments
        bead_ass = spatial_assignments(
            results["type_assignments"],
            results["graph"],
            batch["_atom_mask"]
        ).detach().transpose(1, 2)

        # get hard type assignments
        assignments = results["type_assignments"].detach()
        assignments = get_hard_assignments(assignments)

        # get atomic numbers
        atomic_numbers = batch["_atomic_numbers"].detach()

        # for each cluster type get the respective beads
        for spl_bead_ass, spl_ass, at_num in zip(bead_ass, assignments, atomic_numbers):
            # convert hard assignment matrix to dictionary
            mapping = dict(spl_ass.nonzero().tolist())

            # generate bead string that describes all atoms that are present in a certain bead
            for at_idx, row in enumerate(spl_bead_ass):
                # collect all atoms contained in the bead
                indices = row.nonzero()[:, 0]
                contained_atom_types = at_num[indices].tolist()
                contained_atom_types.sort()
                # convert to string
                bead_str = ""
                for at in contained_atom_types:
                    bead_str += atom_names_dict[at]
                # ignore empty beads
                if bead_str != "":
                    # map back to cluster type
                    cl_idx = mapping[at_idx]
                    # save bead string that belongs to certain cluster type
                    type_dict[cl_idx].append(bead_str)

    # get list of all substructures
    all_substructures = []
    for substructures in type_dict.values():
        all_substructures += substructures
    # make unique and sort (each substructure only appears once, sorted from small to large size)
    all_substructures = set(all_substructures)
    all_substructures = sorted(all_substructures, key=len)
    # count appearance of substructure for each cluster
    counts = torch.zeros((len(all_substructures), max_n_clusters))
    for cl_idx in type_dict.keys():
        for bead_idx, bead_name in enumerate(all_substructures):
            count = type_dict[cl_idx].count(bead_name)
            counts[bead_idx][cl_idx] = count

    # change bead names from HHCCC --> H2C3
    all_substructures_dense = []
    for bead_name in all_substructures:
        atom_dict = collections.Counter(bead_name)
        bead_name_tmp = ""
        for k, v in atom_dict.items():
            if v == 1:
                bead_name_tmp += k
            else:
                bead_name_tmp += k + str(v)
        all_substructures_dense.append(bead_name_tmp)
    # get topk substructures for each cluster
    topk = 5
    count_values, substruc_indices = counts.topk(k=topk, dim=0)

    substruc_indices = substruc_indices.transpose(0, 1)[mask]
    substruc_indices = substruc_indices.transpose(0, 1)

    count_values = count_values.transpose(0, 1)[mask]
    count_values = count_values.transpose(0, 1)
    count_values = count_values / count_values.sum(0) * 100.

    return substruc_indices, count_values, all_substructures_dense


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

    plt.title('Cluster Type Usage')
    plt.ylabel('#atoms assigned to certain type')
    plt.xticks([])
    fig.savefig(figname)


if __name__ == "__main__":
    dpath = "/home/jonas/Documents/datasets/qm9_old/qm9.db"
    mdir = "/home/jonas/Desktop/moinn_pretrained"
    eval_set_size = 100
    device = torch.device('cuda')
    batch_size = 1

    atom_names_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}

    eval_dir = os.path.join(mdir, "eval_images")

    # create directory
    tmp_dir = os.path.join(eval_dir, "tmp")
    if os.path.exists(tmp_dir):
        print("existing evaluation results will be overwritten...")
        rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    # load dataset splits
    dataset = spk.AtomsData(dpath)
    split_path = os.path.join(mdir, "split.npz")
    data_train, data_val, data_test = spk.data.train_test_split(
        dataset, split_file=split_path
    )
    data_test = data_test.create_subset([_ for _ in range(eval_set_size)])

    # chose loader
    test_loader = spk.data.AtomsLoader(data_test, batch_size=batch_size, num_workers=0, pin_memory=True)

    # load model
    modelpath = os.path.join(mdir, "best_model")
    model = torch.load(modelpath, map_location=device)

    used_types, filled_clusters = get_used_types(test_loader, model, device)

    substruc_indices, count_values, all_substructures_dense = get_moieties(test_loader, model, used_types, device)

    n_used_types = substruc_indices.shape[1]

    cmap = plt.get_cmap("tab20")
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
    rows = ["cluster-type"]
    for _ in range(5):
        rows.append("sub-struc. {}".format(_))
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

    ########################################################################################################################
    # plot
    ########################################################################################################################

    # show most common substructures
    plot_table(
        figname=os.path.join(eval_dir, "common_substructures.pdf"),
        filled_clusters=filled_clusters[used_types],
        cellText=cellText,
        rows=rows,
        colors_text=colors_text
    )

    print()