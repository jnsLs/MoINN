import os
import schnetpack as spk


def get_loader(dpath, mpath, split="val"):
    """This function is used to define a dataloader of the selected data split ("val", "train, or "test")."""
    # define dataset
    dataset = spk.AtomsData(dpath)
    #dataset = spk.datasets.MD17(dpath, "aspirin")
    #dataset = spk.datasets.QM9(
    #    dpath,
    #    load_only=["energy_U0"],
    #)
    # load dataset splits
    split_path = os.path.join(mpath, "split.npz")
    data_train, data_val, data_test = spk.data.train_test_split(
        dataset, split_file=split_path
    )
    # chose loader
    data = {"train": data_train, "val": data_val, "test": data_test}
    loader = spk.data.AtomsLoader(data[split], batch_size=1, num_workers=2, pin_memory=True)
    return loader
