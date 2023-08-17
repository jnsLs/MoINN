import os
import torch
from shutil import rmtree
from moinn.clustering.utils.coarse_graining import spatial_assignments


# get bead assignments
bead_ass = spatial_assignments(
    result["type_assignments"],
    result["graph"],
    batch["_atom_mask"]
).detach().transpose(1, 2)

# get hard type assignments
assignments = result["type_assignments"].detach()
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
            self.type_dict[cl_idx].append(bead_str)


if __name__ == "__main__":
    dpath = "/home/jonas/Documents/datasets/qm9_old/qm9.db"
    mdir = "/home/jonas/Desktop/moinn_pretrained"
    eval_set_size = 1000
    device = torch.device('cuda')
    batch_size = 10

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

    used_types = get_used_types(test_loader, model, device)
    print(used_types)