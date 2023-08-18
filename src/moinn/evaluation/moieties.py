import torch
from tqdm import tqdm
import collections

from moinn.clustering.utils.coarse_graining import spatial_assignments, get_hard_assignments
from moinn.clustering.neighbors import PairwiseDistances, AdjMatrix
from schnetpack.nn.cutoff import HardCutoff


def get_topk_moieties(loader, model, mask, topk=5, device=torch.device('cuda')):

    atom_names_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}

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

    count_values, substruc_indices = counts.topk(k=topk, dim=0)

    substruc_indices = substruc_indices.transpose(0, 1)[mask]
    substruc_indices = substruc_indices.transpose(0, 1)

    count_values = count_values.transpose(0, 1)[mask]
    count_values = count_values.transpose(0, 1)
    count_values = count_values / count_values.sum(0) * 100.

    return substruc_indices, count_values, all_substructures_dense