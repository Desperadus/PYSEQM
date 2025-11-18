import torch

def sparse_eye(size, device=None, dtype=None, layout=torch.sparse_coo):
    """
    Creates a sparse identity matrix.
    """
    indices = torch.arange(size, device=device)
    indices = torch.stack((indices, indices), dim=0)
    values = torch.ones(size, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, (size, size), device=device, dtype=dtype).to(layout)

def sparse_diagonal(sparse_tensor):
    """
    Extracts the diagonal of a sparse tensor.
    Assumes COO or CSR format.
    """
    if sparse_tensor.layout == torch.sparse_coo:
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        mask = indices[0] == indices[1]
        # This gives values on diagonal, but we need to map them to their positions
        # to return a full diagonal vector (with zeros where missing)
        diag_indices = indices[0][mask]
        diag_values = values[mask]
        
        diag = torch.zeros(sparse_tensor.shape[0], device=sparse_tensor.device, dtype=sparse_tensor.dtype)
        diag[diag_indices] = diag_values
        return diag
    elif sparse_tensor.layout == torch.sparse_csr:
        # Convert to COO for simplicity as CSR diagonal extraction is non-trivial without custom kernel
        return sparse_diagonal(sparse_tensor.to_sparse_coo())
    else:
        raise NotImplementedError(f"Layout {sparse_tensor.layout} not supported")

def truncate_sparse(sparse_tensor, threshold):
    """
    Truncates values in a sparse tensor below a threshold.
    """
    if sparse_tensor.layout == torch.sparse_coo:
        sparse_tensor = sparse_tensor.coalesce()
        mask = sparse_tensor.values().abs() > threshold
        new_indices = sparse_tensor.indices()[:, mask]
        new_values = sparse_tensor.values()[mask]
        return torch.sparse_coo_tensor(new_indices, new_values, sparse_tensor.shape, device=sparse_tensor.device, dtype=sparse_tensor.dtype)
    elif sparse_tensor.layout == torch.sparse_csr:
        # For CSR, it's often easier to convert to COO, filter, and back, 
        # unless we manipulate crow_indices/col_indices directly.
        # Given PyTorch's current state, round-trip is safest.
        return truncate_sparse(sparse_tensor.to_sparse_coo(), threshold).to_sparse_csr()
    else:
        raise NotImplementedError(f"Layout {sparse_tensor.layout} not supported")
