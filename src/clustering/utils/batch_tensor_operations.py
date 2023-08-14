import torch

__all__ = ["batch_trace", "batch_norm", "batch_diag", "batch_div"]


def batch_operation(operation, tensor_a, tensor_b=None):
    """
    This function allows to extend tensor operations provided in pytorch to 
    batch tensors. As an input, it takes the pytorch operation and the tensors 
    on which the operation is performed.
    
    Depending on wether a single tensor operation or a double tensor operation
    is performed, an output tensor is defined. Subsequently, the specified oper-
    ation is iterated over the number of samples contained in the batch.
    """
    batch_size, sample_size, *_ = tensor_a.shape
    # single tensor operation
    if tensor_b is None: 
        # define output dimensionality
        if _ == []:
            out = torch.zeros(batch_size, sample_size, sample_size).cuda()
        else:    
            out = torch.zeros(batch_size).cuda() 
        # run operation
        for spl_idx in range(batch_size):
            out[spl_idx] = operation(tensor_a[spl_idx])
    # double tensor operation
    else: 
        # define output dimensionality
        out = torch.zeros_like(tensor_a)
        # run operation
        for spl_idx in range(batch_size):
            out[spl_idx] = operation(tensor_a[spl_idx], tensor_b[spl_idx])
    return out    


def batch_trace(tensor):
    """
    Batch extension of torch.trace() function 
    Trace calculation for all samples included in the mini batch
    
    Args: 3D tensor
    Returns: 1D tensor containing trace values 
    """
    return batch_operation(torch.trace, tensor)


def batch_norm(tensor):
    """
    Batch extension of torch.norm() function 
    Calculation of frobenius norm for all samples included in the mini batch
    
    Args: 3D tensor
    Returns: 1D tensor containing norm values 
    """
    return batch_operation(torch.norm, tensor)


def batch_diag(tensor):
    """
    Batch extension of torch.diag() function.
    
    For each sample included in the batch, creates 2D matrix with elements of 
    the input tensor on the diagonal.
        
    Args: 2D tensor
    Returns: 3D tensor containing trace values 
    """
    return batch_operation(torch.diag, tensor)


def batch_div(tensor_a, tensor_b):
    """
    Batch extension of torch.div() function.
    
    For each sample included in the batch, the respective matrix in tensor_a is
    divided by the scalar element of tensor_b
        
    Args: 
        tensor_a: 3D tensor
        tensor_b: 1D tensor
    Returns: 3D tensor containing trace values 
    """
    return batch_operation(torch.div, tensor_a, tensor_b)
