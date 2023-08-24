import torch
from tqdm import tqdm
import collections
from schnetpack.nn.cutoff import HardCutoff
from moinn.nn.neighbors import PairwiseDistances, AdjMatrix


def get_hard_assignments(s):
    # get most likely cluster for each atom
    max_idx = s.argmax(dim=-1).flatten()
    # list of spl indices and atom indices
    c1 = torch.arange(s.size(0)).unsqueeze(-1).repeat(1, s.size(1)).flatten()
    c2 = torch.arange(s.size(1)).repeat(s.size(0))
    # define new hard assignment matrix (indicates most likely cl. for each atom)
    s_hard = torch.zeros_like(s)
    s_hard[c1, c2, max_idx] = 1
    return s_hard


def spatial_assignments(type_ass, adj, atom_mask):
    # get hard cluster type similarity matrix
    s_hard = get_hard_assignments(type_ass)
    type_sim_hard = torch.matmul(s_hard, s_hard.transpose(1, 2))
    type_sim_hard_d = type_sim_hard * adj

    # assignment smoothing (inspired by laplacian smoothing)
    spatial_ass = type_sim_hard_d.clone()
    spatial_ass_tmp = type_sim_hard.clone()
    while torch.norm(spatial_ass_tmp - spatial_ass, p=1) > 1e-5:
        spatial_ass_tmp = spatial_ass.clone()
        spatial_ass = torch.matmul(type_sim_hard_d, spatial_ass)
        spatial_ass = (spatial_ass > 0.99).type(torch.FloatTensor).cuda()

    # eliminate duplicates in spatial assignment matrix
    spatial_ass_tmp = spatial_ass.clone()
    mask = torch.ones_like(spatial_ass)
    for sample_idx, spl_ass_red in enumerate(spatial_ass_tmp):
        for row_idx, row in enumerate(spl_ass_red):
            for column_idx, sim_value in enumerate(row):
                if (column_idx != row_idx) and (sim_value > 0.99):
                    spatial_ass_tmp[sample_idx][column_idx] *= 0
                    mask[sample_idx][column_idx] *= 0
    # transpose --> n_atoms x n_clusters
    mask = mask.transpose(1, 2)
    # apply mask
    spatial_ass = spatial_ass * mask

    # mask out empty nodes
    node_dim = atom_mask.shape[1]
    atom_mask = atom_mask.unsqueeze(-1).repeat(1, 1, node_dim)  # define mask
    spatial_ass = spatial_ass * atom_mask * atom_mask.transpose(1, 2)  # apply mask

    return spatial_ass


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
