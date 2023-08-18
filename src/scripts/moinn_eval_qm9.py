import os
import schnetpack as spk
import torch
import matplotlib.pyplot as plt
from shutil import rmtree

from moinn.evaluation import EnvironmentTypes
from moinn.evaluation import get_topk_moieties


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

    plt.title('Environment Type Usage')
    plt.ylabel('#atoms assigned to certain type')
    plt.xticks([])
    fig.savefig(figname)


if __name__ == "__main__":
    dpath = "/home/jonas/Documents/datasets/qm9_old/qm9.db"
    mdir = "/home/jonas/Desktop/moinn_pretrained"
    eval_set_size = 100
    device = torch.device('cuda')
    batch_size = 1
    topk = 5

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

    environment_types = EnvironmentTypes(test_loader, model, device)
    used_types, filled_clusters = environment_types.get_used_types()

    substruc_indices, count_values, all_substructures_dense = get_topk_moieties(
        test_loader, model, used_types, topk, device
    )

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
    rows = ["env.-type"]
    for _ in range(5):
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

    ########################################################################################################################
    # plot
    ########################################################################################################################

    # show most common substructures
    plot_table(
        figname=os.path.join(eval_dir, "common_moieties.pdf"),
        filled_clusters=filled_clusters[used_types],
        cellText=cellText,
        rows=rows,
        colors_text=colors_text
    )

    print()