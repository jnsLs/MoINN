import os
import torch
from tqdm import tqdm
import schnetpack as spk


class EnvironmentTypes:
    def __init__(self, loader, model, device=torch.device('cuda')):
        self.loader = loader
        self.model = model
        self.device = device
        self.max_n_atoms = 30

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            results = model(batch)
            type_ass = results["type_assignments"].detach()

            # initialize tot_ass which contains the total atom assignments to the respective environment types
            max_n_clusters = type_ass.cpu().shape[-1]
            self.tot_ass = torch.zeros((self.max_n_atoms, max_n_clusters))

            # initialize max_in_batch_prev that contains the max assignment values of the last iteration
            max_in_sample = torch.max(type_ass, dim=1)[0]
            max_in_batch = torch.max(max_in_sample, dim=0, keepdim=True)[0]
            self.max_in_batch_prev = torch.zeros_like(max_in_batch)

            break

    def get_used_types(self):

        for batch_idx, batch in tqdm(enumerate(self.loader)):

            # get environment type assignments for the batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            results = self.model(batch)
            type_ass = results["type_assignments"].detach()

            # get matrix that contains the summed up atom assignments to the respective types
            # sum up type assignments
            n_nodes = type_ass.shape[1]
            for ass in type_ass.cpu():
                self.tot_ass[:n_nodes, :] += ass

            # get max assignment values for all environment types
            max_in_sample = torch.max(type_ass, dim=1)[0]
            max_in_batch = torch.max(max_in_sample, dim=0, keepdim=True)[0]
            comp = torch.concat((max_in_batch, self.max_in_batch_prev), dim=0)
            max_in_batch = torch.max(comp, dim=0, keepdim=True)[0]
            self.max_in_batch_prev = max_in_batch.clone()

            # get mask defining the used types based on the max. assignment value
            mask = max_in_batch.cpu() > 0.05

        # sum over all atoms
        filled_clusters = self.tot_ass.sum(dim=0)

        return mask[0], filled_clusters


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

    environment_types = EnvironmentTypes(test_loader, model, device)
    used_types, filled_clusters = environment_types.get_used_types()

    print(used_types)
    print(filled_clusters)