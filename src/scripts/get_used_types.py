import os
import schnetpack as spk
from shutil import rmtree
import torch
from tqdm import tqdm


def get_used_types(dpath, mdir, device=torch.device('cuda'), batch_size=10, eval_set_size=1000):
    # create eval directory
    eval_dir = os.path.join(mdir, "eval_images")
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

    for batch_idx, batch in tqdm(enumerate(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        results = model(batch)

        type_ass = results["type_assignments"].detach()

        max_in_sample = torch.max(type_ass, dim=1)[0]
        max_in_batch = torch.max(max_in_sample, dim=0, keepdim=True)[0]

        if batch_idx == 0:
            max_in_batch_prev = torch.zeros_like(max_in_batch)

        comp = torch.concat((max_in_batch, max_in_batch_prev), dim=0)
        max_in_batch = torch.max(comp, dim=0, keepdim=True)[0]
        max_in_batch_prev = max_in_batch.clone()

    return max_in_batch > 0.05


if __name__ == "__main__":
    dpath = "/home/jonas/Documents/datasets/qm9_old/qm9.db"
    mdir = "/home/jonas/Desktop/moinn_pretrained"

    used_types = get_used_types(dpath, mdir)
    print(used_types)

