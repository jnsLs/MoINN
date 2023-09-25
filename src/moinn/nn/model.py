import torch
from torch import nn
from schnetpack.nn import shifted_softplus, Dense


class EndToEndModel(nn.Module):
    """
    Join a representation model with nn module.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_module (e.g. nn modul): Output block of the model. Needed for predicting properties.

    Returns:
         dict: property predictions
    """

    def __init__(self, representation, output_module, clustering_mode):
        super(EndToEndModel, self).__init__()
        self.representation = representation
        self.output_module = output_module
        self.clustering_mode = clustering_mode

    def forward(self, inputs):
        """
        Forward representation output through output modules.
        """

        if self.clustering_mode == "pretrained":
            inputs["representation"] = self.representation(inputs).detach()
        elif self.clustering_mode == "end_to_end":
            inputs["representation"] = self.representation(inputs)
        else:
            raise NotImplementedError("clustering mode not implemented, yet. Choose \"pretrained\" or \"end_to_end\"")
        outs = self.output_module(inputs)
        return outs


class MLP(nn.Module):
    """
    Linear model to map cluster finger prints to molecular properties
    """

    def __init__(self, cluster_model, mean, stddev, device, args):
        super(MLP, self).__init__()

        self.property = args.property
        self.feat_type = args.feature_type

        self.cluster_model = cluster_model
        self.mean = mean[self.property].to(device)
        self.stddev = stddev[self.property].to(device)

        self.network = nn.Sequential(
            Dense(args.feature_dim, 1, activation=None)
        )

    def forward(self, x):
        if self.feat_type == "type_feat":
            pred = self.network(x["type_features"])
        elif self.feat_type == "group_feat":
            pred = self.network(x["groups_full"])
        elif self.feat_type == "group_feat_weighted":
            feat = x["groups_full"] / (x["groups_full"].sum(1, keepdim=True)+1e-6) * x["_atom_mask"].sum(1, keepdim=True)
            pred = self.network(feat)
        elif self.feat_type == "rdkit_feat":
            pred = self.network(x["rdkit_features"])
        elif self.feat_type == "morgan_features":
            pred = self.network(x["morgan_features"])
        elif self.feat_type == "dressed_atoms":
            pred = self.network(x["dressed_atoms"])
        else:
            results = self.cluster_model(x)
            type_ass = results["type_assignments"]
            feat = type_ass.sum(dim=1).detach()
            pred = self.network(feat)

        # scale shift
        if self.property in ["energy_U0", "energy_U", "enthalpy_H", "free_energy"]:
            pred = pred * self.stddev + self.mean
        return {self.property: pred}
