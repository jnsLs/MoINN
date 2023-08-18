import torch
from torch import nn
from torch.nn.init import xavier_uniform_, orthogonal_

from schnetpack import Properties
from schnetpack.nn.base import Dense
from schnetpack.nn.cutoff import HardCutoff, CosineCutoff
from moinn.nn.activations import softmax, swish
from moinn.nn.neighbors import AdjMatrix, PairwiseDistances


class TypeMapping(nn.Module):
    r"""SchNet type mapping layer. An assignment matrix is learned which
    maps atoms to their respective clusters (atom-group types).

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        max_clusters (int): maximum number of atom-group types.
    """
    def __init__(self, n_atom_basis, max_clusters):
        super(TypeMapping, self).__init__()
        self.n_clusters = max_clusters
        self.assignment_network = nn.Sequential(
            Dense(n_atom_basis, max_clusters, activation=swish, weight_init=xavier_uniform_),
            Dense(max_clusters, max_clusters, activation=softmax, weight_init=orthogonal_)
        )
        
    def _mask_assignments(self, assignments, atom_mask):
        """
        Apply atom mask to assignment matrix. Essential to conserve number of
        atoms for pooling. Otherwise cluster sizes computed in the standardize 
        layer in output_module.py would contain a random contribution from non
        existing atoms.
        """
        atom_mask = atom_mask.unsqueeze(-1).repeat(1, 1, self.n_clusters)
        clean_assignments = torch.zeros_like(assignments)
        clean_assignments[atom_mask != 0] = assignments[atom_mask != 0]
        return clean_assignments
    
    def forward(self, x, atom_mask):
        # learn type assignments
        assignments = self.assignment_network(x)       
        assignments = self._mask_assignments(assignments, atom_mask)
        return assignments
    
    
class Clustering(nn.Module):
    r"""Clustering block. This block contains the TypeMapping-layer which is used to learn the atom-cluster assignments.
    Furthermore, this block provides all quantities necessary for evaluating the nn loss function.
    Args:
        features (int): number of features to describe atomic environments.
        max_clusters (int): maximum number of atom-group types.
        mincut_cutoff_function (torch.nn.Module): cutoff function to define MinCut adjacency matrix. This matrix
            determines the graph connectivity in the MinCut loss. If set to None, the connectivity matrix is loaded
            directly from the dataset.
        mincut_cutoff_radius (list of floats): cutoff radius of mincut_cutoff_function.
        normalize_mincut_adj (bool): if set to True, the MinCut adjacency matrix is symmetrically normalized.
        bead_cutoff_function (torch.nn.Module): cutoff function to determine distance-dependent type similarity matrix.
        bead_cutoff_parameters (list of floats): If len(bead_cutoff_parameters)==1, this represents the cutoff radius
            of the bead_cutoff_function. If len(bead_cutoff_parameters)==2, this represents the cuton and cutoff radii
            of the bead_cutoff_function.
    """
    def __init__(
        self,
        features=24,
        max_clusters=30,
        mincut_cutoff_function=HardCutoff,
        mincut_cutoff_radius=[2.5],
        normalize_mincut_adj=False,
        bead_cutoff_function=CosineCutoff,
        bead_cutoff_parameters=[4.0],
    ):
        super(Clustering, self).__init__()

        # layer for computing interatomic distances
        self.distances = PairwiseDistances()
        # layers for computing adjacency matrices and molecule graphs
        self.adjacency_min = AdjMatrix(mincut_cutoff_function,
                                       mincut_cutoff_radius,
                                       normalize=normalize_mincut_adj,
                                       zero_diag=True)
        self.adjacency_max = AdjMatrix(bead_cutoff_function,
                                       bead_cutoff_parameters,
                                       normalize=False,
                                       zero_diag=False)
        # layer for learning the type assignment matrix
        self.type_mapping = TypeMapping(features, max_clusters)

    def forward(self, inputs):  
        # get tensors from input dictionary
        positions = inputs[Properties.R]
        atom_mask = inputs[Properties.atom_mask]
        # get representations from pretrained model
        x = inputs["representation"]
        # compute interatomic distance of every atom to its neighbors
        r_ij = self.distances(positions)
        
        # get adj_min for minCUT-loss, and graph for visualizing the molecule
        adj_min = self.adjacency_min(atom_mask=atom_mask, r_ij=r_ij)

        # get adj_max to introduce distance dependency in cluster type similarity
        adj_max = self.adjacency_max(atom_mask=atom_mask, r_ij=r_ij)

        # cluster type assignment matrix
        type_ass = self.type_mapping(x, atom_mask)
        # compute (distance dependent) cluster type similarity between all atoms
        type_sim = torch.matmul(type_ass, type_ass.transpose(1, 2))
        type_sim_d = type_sim * adj_max

        return {"type_assignments": type_ass,
                "adjacency_min": adj_min,
                "adjacency_max": adj_max,
                "type_similarity": type_sim,
                "distance_dependent_type_similarity": type_sim_d,
                }
