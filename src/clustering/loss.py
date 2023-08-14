import torch
import schnetpack.clustering.utils.batch_tensor_operations as bo
import math


def clustering_loss_fn(args):
    def loss(batch, result, epoch):

        # define tradeoff warmup parameters
        cut_warmup, entropy_warmup = args.tradeoff_warmup_epochs
        cut_warmup -= 70
        entropy_warmup -= 70

        # clustering loss
        cut = cut_loss(batch, result) / (1 + math.exp((-epoch + cut_warmup) / 10))# * (1 - args.clustering_tradeoff[1])
        ortho = ortho_loss(batch, result, args) * args.clustering_tradeoff[0]# * (1 - args.clustering_tradeoff[1])
        ent = entropy_loss(batch, result) * args.clustering_tradeoff[1] / (1 + math.exp((-epoch + entropy_warmup) / 10))
        err = torch.mean(cut + ortho + ent)

        return err
    return loss


def cut_loss(batch, result, args=None):
    r"""Compute mincut loss.

    .. math::
        L_\text{cut} = -\frac{
            Tr\left( \bm{S}^T \bm{AS} \right)
        }{
            Tr\left( \bm{S}^T \bm{DS} \right)
        }.

    Args:
        A (torch.Tensor): adjacency matrix.
        S (torch.Tensor): assignment matrix.

    Returns:
        cut_loss (scalar torch.Tensor): mincut loss.

    """
    adj_min = result["adjacency_min"]
    SS_t_dd = result["distance_dependent_type_similarity"]
    # trace(S^T A S)
    adj_pool = torch.matmul(SS_t_dd.transpose(1, 2), torch.matmul(adj_min, SS_t_dd))
    num = bo.batch_trace(adj_pool)
    # trace(S^T D S)
    degree = adj_min.sum(dim=-1)
    degree = bo.batch_diag(degree)
    degree_pool = torch.matmul(torch.transpose(SS_t_dd, 1, 2), torch.matmul(degree, SS_t_dd))
    den = bo.batch_trace(degree_pool)
    return (-1) * num / den


def ortho_loss(batch, result, args):
    device = torch.device("cuda" if args.cuda else "cpu")

    r"""Compute orthogonality loss.

    .. math:: \left\Vert 
    \frac{\bm{S}^T\bm{S}}{\left\Vert\bm{S}^T\bm{S}\right\Vert_F} 
    - \frac{\bm{I}_K}{\sqrt{K}}
    \right\Vert_F

    Args: 
        S (torch.Tensor): assignment matrix.

    Returns:
        ortho_loss (scalar torch.Tensor): orthogonality loss.

    """
    SS_t = result["type_similarity"]
    n_nodes = SS_t.shape[1]

    # define identity matrix
    I_S = torch.eye(n_nodes)
    I_S = I_S.to(device)
    # calculate orthogonality loss term
    diff = bo.batch_div(SS_t, bo.batch_norm(SS_t)) - I_S / torch.norm(I_S)
    diff = bo.batch_norm(diff)
    return diff


def entropy_loss(batch, result, args=None):
    type_ass = result["type_assignments"]
    EPS = 1e-15
    ent = (-type_ass * torch.log(type_ass + EPS)).sum(dim=-1).mean(dim=-1)  # logarithmic
    #ent = torch.exp((-type_ass * torch.log(type_ass + EPS)).sum(dim=-1).mean())     # linear
    #ent = (torch.exp((-type_ass * torch.log(type_ass + EPS)).sum(dim=-1).mean())) ** 2     # quadratic
    return ent
