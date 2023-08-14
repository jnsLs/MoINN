import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


###############################################################################
# pooling

def pool_positions(initial_positions, assignments):
    """
    equivalent to calculating the center of mass of all nodes contained in a 
    cluster. The cluster membership assignment (assignmentmatix entry) of each
    node is the analogue to its respective mass
    
    Args: 
        initial_positions (2D or 3D torch Tensor): initial node positions
            can be either 2D (single sample) or 3D (single batch)
        assignments (2D or 3D torch Tensor): assignment matrix
    Return:
        positions (torch Tensor): pooled positions
    """
    # transpose assignment matrix
    if initial_positions.dim() == 3:
        assignments = assignments.transpose(1, 2)
    elif initial_positions.dim() == 2:
        assignments = assignments.transpose(0, 1)
    # pool
    positions = torch.matmul(assignments, initial_positions)
    # normalize
    norm = torch.matmul(assignments, torch.ones_like(initial_positions))
    positions = positions / norm
    return positions


def pool_representations(x, assignments):
    x = torch.matmul(assignments.transpose(1, 2), x)
    return x


###############################################################################
# cluster assignments
    
# this function is only needed for debugging (check if positions are pooled correclty)
def plot_3D(positions_at, positions_cl):
    positions_np = positions_at.detach().cpu().numpy()
    positions_cl_np = positions_cl.detach().cpu().numpy()
    # 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions_cl_np[0, :, 0], positions_cl_np[0, :, 1], positions_cl_np[0, :, 2], color="r")
    ax.scatter(positions_np[0, :, 0], positions_np[0, :, 1], positions_np[0,:,2], color="b")
    plt.show() 
    
    
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
