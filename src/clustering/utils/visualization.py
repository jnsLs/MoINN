import torch
import numpy as np
import matplotlib.pyplot as plt
import collections
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from schnetpack.clustering.utils.coarse_graining import (
    pool_positions, pool_representations, spatial_assignments, get_hard_assignments
)

plt.ioff() # prevent figures from popping up

cpu = torch.device("cpu")
atom_names_dict = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}
atom_colors_dict = {1: "green", 6: "black", 7: "blue", 8: "red", 9: "purple"}


def vis_type_ass_on_molecule(mol, fig_name, node_colors):
    # define 2d embedding of molecule
    AllChem.Compute2DCoords(mol)
    mol = Draw.PrepareMolForDrawing(mol)

    # draw molecule and store as png
    d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    d.drawOptions().fillHighlights = False
    d.DrawMoleculeWithHighlights(mol, '', node_colors, {}, {}, {}, -1)
    d.FinishDrawing()
    d.WriteDrawingText(fig_name + ".png")

    # draw molecule and store as svg
    d = rdMolDraw2D.MolDraw2DSVG(5000, 5000)
    d.drawOptions().fillHighlights = False
    d.DrawMoleculeWithHighlights(mol, '', node_colors, {}, {}, {}, -1)
    d.FinishDrawing()
    svg = d.GetDrawingText().replace('svg:', '')

    # store figure
    struc_form_pic = open(fig_name + ".svg", "w")
    struc_form_pic.write(svg)
    struc_form_pic.close()


class UsedBeadTypes:
    """Class to obtain all used cluster types (non-empty rows in assignment matrix)"""
    def __init__(self, max_clusters):
        self.tot_ass = torch.zeros((30, max_clusters))

    def add_batch(self, result):
        assignments = result["type_assignments"].detach().to(cpu)
        n_nodes = assignments.shape[1]
        for ass in assignments:
            self.tot_ass[:n_nodes, :] += ass

    #def get_used_types(self, threshold=1.):
    #    used_types = (self.tot_ass.sum(dim=0) > threshold).float()
    #    n_clusters = used_types.sum().long().item()
    #    return n_clusters, used_types, self.tot_ass


class ClusteredSubstructures:
    """Class to obtain all combinations of atom types belonging to a certain cluster type"""
    def __init__(self):
        self.first_step = True

    def add_batch(self, batch, result):
        # initialize substructure dictionary <type_dict>
        if self.first_step:
            self.max_n_clusters = result["type_assignments"].detach().to(cpu).shape[-1]
            self.type_dict = {idx: [] for idx in range(self.max_n_clusters)}
            self.first_step = False

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

    def plot(self):
        # get list of all substructures
        all_substructures = []
        for substructures in self.type_dict.values():
            all_substructures += substructures
        # make unique and sort (each substructure only appears once, sorted from small to large size)
        all_substructures = set(all_substructures)
        all_substructures = sorted(all_substructures, key=len)
        # count appearance of substructure for each cluster
        counts = torch.zeros((len(all_substructures), self.max_n_clusters))
        for cl_idx in self.type_dict.keys():
            for bead_idx, bead_name in enumerate(all_substructures):
                count = self.type_dict[cl_idx].count(bead_name)
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
        count_values, substruc_indices = counts.topk(k=1, dim=0)
        # plot
        fig = plt.figure(figsize=(5, 15))
        ax = plt.subplot(1, 1, 1)
        fig = heatmap(fig, counts)
        plt.ylabel('bead')
        plt.xlabel('cluster')
        plt.yticks(ticks=range(len(all_substructures_dense)), labels=all_substructures_dense)#, rotation='vertical')
        #plt.gcf().subplots_adjust(bottom=0.30)
        return fig, substruc_indices, all_substructures_dense, count_values


class ClustersOverAtoms:
    def __init__(self, max_clusters):
        self.clus_over_atoms = torch.zeros(30, max_clusters+1)

    def add_batch(self, batch, result, threshold=0.5):
        assignments = result["type_assignments"].detach().to(cpu)
        # get number of cluster types
        cluster_sizes = assignments.sum(dim=1)
        n_cl = cluster_sizes > threshold
        n_cl = n_cl.sum(dim=1).data.tolist()
        # get molecule sizes
        n_atoms = batch["_atom_mask"].sum(dim=1).detach().to(cpu).long().tolist()

        # update n_clusters-n_atoms matrix
        for idx_pair in zip(n_atoms, n_cl):
            self.clus_over_atoms[idx_pair] += 1

    def plot(self, cutoff):
        #fig = plt.figure()
        fig, ax = plt.subplots()
        fig = heatmap(fig, self.clus_over_atoms[:, :cutoff].transpose(0,1))
        ax.invert_yaxis()
        plt.ylabel('n_cluster_types')
        plt.xlabel('n_atoms')
        return fig


def heatmap(fig, matrix):
    """
    plot heatmap of matrix.
    
    Args:
        matrix (2D torch Tensor): input Tensor 
    Returns:
        fig
    """
    # detach input Tensor and send them to cpu
    matrix = matrix.to(cpu).detach().numpy()
    # visualize    
    ##fig = plt.figure()
    ##fig.add_subplot(1,1,1)
    plt.imshow(matrix)
    plt.colorbar()
    ##fig.canvas.draw()
    return fig
        



