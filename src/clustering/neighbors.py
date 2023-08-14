import torch
from torch import nn
from schnetpack.nn.cutoff import HardCutoff 

import os

import pybel
#import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np


class PairwiseDistances(nn.Module):
    """
    Layer for computing the pairwise distance matrix. Matrix is quadratic 
    and symmetric with entries d_ij = r_i - r_j. All possible atom pairs are
    considered. This Layer is particularly used in the clustering module.
       
    Returns:
        torch.Tensor: Pairwise distances (Nbatch x Nat x Nat)
     
    """
    def __init__(self):
        super(PairwiseDistances, self).__init__()

    def forward(self, positions):
        n_nodes = positions.shape[1]
        # compute tensor distance tensor with entries d_ij = r_i - r_j 
        positions_i = positions.unsqueeze(2).repeat(1, 1, n_nodes, 1)
        positions_j = positions.repeat(1, n_nodes, 1).view(-1, n_nodes, n_nodes, 3)
        dist_vecs = positions_i - positions_j
        r_ij = dist_vecs.norm(dim=-1, p=2)
        return r_ij

    
class AdjMatrix(nn.Module):
    """
    Layer for computing the adjacency matrix of the molecular graph. It is 
    obtained by applying a cutoff function to the tensor of pairwise distances or by loading
    a connectivity matrix from the input dictionary.
    Args:
        cutoff_network (torch.nn.Module): cut-off function (HardCutoff, CosineCutoff) 
        cutoff (float): cut-off radius
        normalize (boolean): if True, adjacency matrix is normalized (default=True) 
        visualize (boolean): if True, forward pass returns two tensors. The first 
            tensor is the adjacency matrix associated with the specified parameters.
            The second tensor represents the undirected molecular graph (binary 
            adjacency matrix). (default=True)
        representation_term (boolean): if True, the returned tensor is given by
            the elementwise product between specified adjacency matrix and a
            representation similarity term. This term is usefull when performing
            spectral clustering. (default=False)
        zero_diag (boolean): if True, self loops in the adjacency matrix are 
            eliminated (default=True)
    Returns:
        torch.Tensor or list of tensors: adjacency matrix (Nbatch x Nat x Nat)
    """    
    def __init__(
        self,
        cutoff_network,
        cutoff_parameters,
        normalize=True, 
        zero_diag=True,
    ):
        super(AdjMatrix, self).__init__()
        if len(cutoff_parameters) == 1:
            self.cutoff_network = cutoff_network(cutoff_parameters[0])
        elif len(cutoff_parameters) == 2:
            self.cutoff_network = cutoff_network(cutoff_parameters[0],
                                                 cutoff_parameters[1])
        self.normalize = normalize
        self.zero_diag = zero_diag

    def _normalize(self, adj):
        """
        Normalizes the given adjacency matrix using the degree matrix as 
        \(D^{-1/2}AD^{-1/2}\) (symmetric normalization)
        :param adj: rank 2 tensor or sparse matrix;
        :return: the normalized adjacency matrix.
        """
        batch_size = adj.shape[0]
        degrees_power = torch.zeros_like(adj)
        for spl_idx in range(batch_size):
            batch_degrees_power = adj[spl_idx].sum(dim=1) ** -0.5
            batch_degrees_power[torch.isinf(batch_degrees_power)] = 0
            degrees_power[spl_idx] = torch.diag(batch_degrees_power)
        adj_asym_norm = torch.matmul(adj,degrees_power)
        adj_sym_norm = torch.matmul(degrees_power, adj_asym_norm)
        return adj_sym_norm

    def _mask_empty(self, adj, atom_mask):
        """
        Apply atom mask to adjacency matrix. Used for consistency. Thus, the
        connectivity of non-existing nodes is set to be 0. This is needed
        for datasets containing samples with varying number of nodes.
        """
        # define mask
        node_dim = atom_mask.shape[1]
        atom_mask = atom_mask.unsqueeze(-1).repeat(1, 1, node_dim)
        # apply mask
        clean_adj = adj * atom_mask * atom_mask.transpose(1, 2)
        return clean_adj

    def forward(self, atom_mask, r_ij):
        batch_size, n_nodes, _ = r_ij.shape
        device = torch.device("cuda" if r_ij.is_cuda else "cpu")
        # apply cut-off function
        adj = self.cutoff_network(r_ij)
        # remove self loops in adj matrix 
        if self.zero_diag:
            self_loops_mask = torch.eye(n_nodes, n_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            adj -= self_loops_mask  
        # mask out non-existing nodes
        adj = self._mask_empty(adj, atom_mask)
        # normalize
        if self.normalize:
            adj = self._normalize(adj)
        return adj


class Molecule:
    """
    pos: (dim: n_at x 3)
    at_num: (dim: n_at)
    """
    
    def __init__(self, pos, at_num, at_mask):
        n_nodes = int(at_mask.sum().item())

        self.pos = pos[:n_nodes]
        self.at_num = at_num[:n_nodes]
        self._define_obmol()
        self._define_pymol()
        self._define_rdkmol()
        self.n_nodes = n_nodes
        self.graph_dim = at_num.shape[0] # also counts zero-atom-nodes
        ##self.n_nodes = self.rdkmol.GetNumAtoms()
    
    def _define_obmol(self):
        # use openbabel to infer bonds and bond order:
        obmol = ob.OBMol()
        obmol.BeginModify()
        # set positions and atomic numbers of all atoms in the molecule
        for p, n in zip(self.pos, self.at_num):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        # finish
        obmol.EndModify()
        self.obmol = obmol
        
    def _define_pymol(self):
        pymol = pybel.Molecule(self.obmol)
        self.pymol = pymol
    
    def _define_rdkmol(self):
        # conversion openbabel --> rdkit (by storing and loading mol file)
        #self.pymol.removeh()
        self.pymol.write(format="mol", filename="tmp.mol", overwrite=False)
        rdkmol = Chem.MolFromMolFile("tmp.mol")
        os.remove("tmp.mol")
        # add hydrogen atoms
        rdkmol = Chem.AddHs(rdkmol)
        self.rdkmol = rdkmol
        
    def embed_in_2d(self):
        # compute 2D embedding
        AllChem.Compute2DCoords(self.rdkmol)
        # compute 2D positions
        pos = []
        for i in range(self.n_nodes):
            conformer_pos = self.rdkmol.GetConformer().GetAtomPosition(i)
            pos.append([conformer_pos.x,conformer_pos.y])
        pos = np.array(pos)
        return(pos)
    
    def get_bond_graph(self):
        graph = np.zeros((self.graph_dim, self.graph_dim), dtype=int)
        for obatom in ob.OBMolAtomIter(self.obmol):
            at_idx = obatom.GetIdx()-1
            for neighbour_atom in ob.OBAtomAtomIter(obatom):
                nb_idx = neighbour_atom.GetIdx()-1
                graph[at_idx][nb_idx] = 1
        return(graph)
    
    def get_canonical_smiles(self):
        can = pybel.Molecule(self.obmol).write("can")
        ##rdkmol = Chem.MolFromSmiles(_can)
        return(can)
