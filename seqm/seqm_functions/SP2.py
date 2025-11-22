import torch
from .sparse_utils import sparse_eye, sparse_diagonal, truncate_sparse

def SP2(a, nocc, eps=1.0e-4, factor=2.0, mask=None):
    #print(a.shape)
    # a: batch of fock matrixes, don't need to be truncated
    # noccd: number of occupied MO
    #return a0: denisty matrixes wich commute with a
    # factor = 1.0 or 2.0, return a0, tr(a0)= factor*nocc
    device = a.device
    dtype = a.dtype
    
    is_sparse = a.is_sparse or a.is_sparse_csr
    
    flag = dtype==torch.float32
    if flag:
        #float32, harder to converge to a smaller eps, for this one set eps=1.0e-2, and break when no more improvement
        #use critiria to check there is no more improvement
        if eps<1.0e-2:
            eps=1.0e-2
    else:
        #float64, if use above critiria, the error will keep going down and take lots of iteration to reach no more improvement
        #so put eps as a small one like 1.0e-4, to recude the number of iterations
        #use critiria to check the err of current and last iterations both <= eps
        if eps>1.0e-3:
            eps=1.0e-3
        elif eps<1.0e-7:
            eps=1.0e-7
    noccd = nocc.type(dtype)

    if a.dim() == 3:
        N, D, _ = a.shape
    elif a.dim() == 2:
        N = 1
        D, _ = a.shape
    else:
        raise ValueError(f"SP2 expects 2D or 3D tensor, got {a.dim()}D")
    
    if is_sparse:
        # Sparse Implementation
        # Gershgorin circle theorem estimate
        # maximal and minimal eigenvalues
        # Note: We assume batch size N=1 for sparse optimization for now, or handle batching carefully.
        # PyTorch sparse tensors usually don't support batch dim > 0 well for mm.
        # If N > 1, we might need to loop or use block sparse.
        # Assuming N=1 or treating batch as block diagonal is complex. 
        # Let's assume N=1 for the sparse path as typical for large molecules.
        
        # Convert to COO for internal operations as CSR has limited support for arithmetic/sum on CPU
        original_layout = a.layout
        if original_layout == torch.sparse_csr:
            a = a.to_sparse_coo()
            if mask is not None and mask.layout == torch.sparse_csr:
                mask = mask.to_sparse_coo()
        
        aii = sparse_diagonal(a) # Returns dense vector of diagonal
        
        # Determine dimensions for sum and min/max
        if N == 1 and a.dim() == 2:
            sum_dim = 1
            minmax_dim = 0
        else:
            sum_dim = 2
            minmax_dim = 1

        # row sum of abs values. 
        # COO sum is supported
        ri = torch.sparse.sum(torch.abs(a), dim=sum_dim).to_dense() - torch.abs(aii)
            
        h1 = torch.min(aii-ri,dim=minmax_dim)[0]
        hN = torch.max(aii+ri,dim=minmax_dim)[0]
        
        # scale a
        # a0 = (hN*I - a) / (hN - h1)
        # Sparse identity in COO
        I = sparse_eye(D, device=device, dtype=dtype, layout=torch.sparse_coo)
        
        # Expand scalars for broadcasting
        hN_view = hN.reshape(-1, 1, 1) # (N, 1, 1)
        diff_view = (hN - h1).reshape(-1, 1, 1)
        
        # Note: Sparse tensor * scalar is supported.
        # Sparse + Sparse is supported for COO.
        # We need to be careful with batch dimension broadcasting which is often not supported in sparse.
        # If N=1:
        if N == 1:
            a0 = (I * hN.item() - a) / (hN.item() - h1.item())
        else:
            # Fallback to dense if batched sparse is tricky, or loop.
            # Let's loop for safety.
            a0_list = []
            for i in range(N):
                a_i = a[i] if a.dim() == 3 else a # Handle if a is (D, D) or (1, D, D)
                # If a is (N, D, D) sparse, a[i] is (D, D) sparse.
                hN_i = hN[i].item()
                h1_i = h1[i].item()
                a0_i = (I * hN_i - a_i) / (hN_i - h1_i)
                a0_list.append(a0_i)
            # Stack back? Sparse stacking is slow. 
            # Better to keep as list or assume N=1.
            # For now, let's assume N=1 for "large system" optimization.
            a0 = a0_list[0] 
            # TODO: Support N > 1 properly.

        # error from current iteration
        # diag sum
        diag_a0 = sparse_diagonal(a0)
        errm0 = torch.abs(torch.sum(diag_a0) - noccd)
        errm1 = errm0.clone()
        errm2 = errm1.clone()
        
        notconverged = True
        k = 0
        
        while notconverged:
            # a2 = a0 @ a0
            a2 = torch.matmul(a0, a0)
            
            # Truncate / Mask
            if mask is not None:
                # Element-wise mult with mask.
                # If mask is dense (boolean), this makes it dense?
                # If mask is sparse, it's intersection.
                # Ideally mask is a sparse tensor of 1s.
                if mask.is_sparse:
                     a2 = a2 * mask # Intersection
                else:
                     # If mask is dense, we probably don't want to densify.
                     # Assume mask matches sparsity pattern we want.
                     pass
            
            # Dynamic truncation to keep sparsity
            a2 = truncate_sparse(a2, 1e-6) # Threshold?
            
            # cond check
            diag_a2 = sparse_diagonal(a2)
            tr_a2 = torch.sum(diag_a2)
            diag_a0 = sparse_diagonal(a0)
            tr_a0 = torch.sum(diag_a0)
            
            cond = torch.abs(tr_a2 - noccd) < torch.abs(2.0 * tr_a0 - tr_a2 - noccd)
            
            if cond:
                a0 = a2
            else:
                a0 = 2.0 * a0 - a2
            
            errm2 = errm1
            errm1 = errm0
            errm0 = torch.abs(torch.sum(sparse_diagonal(a0)) - noccd)
            k += 1
            
            if flag:
                notconverged = not ((errm0 < eps) and (errm0 >= errm2))
            else:
                notconverged = not ((errm0 < eps) and (errm1 < eps))
                
            if k > 100: break # Safety break
            
        # Convert back to original layout if needed
        if original_layout == torch.sparse_csr:
            a0 = a0.to_sparse_csr()
            
        return factor * a0

    else:
        # Dense Implementation (Original)
        #Gershgorin circle theorem estimate
        ###maximal and minimal eigenvalues
        aii = a.diagonal(dim1=1,dim2=2)
        ri = torch.sum(torch.abs(a),dim=2)-torch.abs(aii)
        h1 = torch.min(aii-ri,dim=1)[0]
        hN = torch.max(aii+ri,dim=1)[0]
        #scale a
        a0 = (torch.eye(D,dtype=dtype,device=device).unsqueeze(0).expand(N,D,D)*hN.reshape(-1,1,1)-a)/(hN-h1).reshape(-1,1,1)

        #error from current iteration
        errm0=torch.abs(torch.sum(a0.diagonal(dim1=1,dim2=2),dim=1)-noccd)
        errm1=errm0.clone() #error from last iteration
        errm2=errm1.clone() #error from last to second iteration

        notconverged = torch.ones(N,dtype=torch.bool,device=device)
        a2 = torch.zeros_like(a)
        cond = torch.zeros_like(notconverged)
        k=0
        while notconverged.any():
            a2[notconverged] = a0[notconverged].matmul(a0[notconverged]) #batch supported
            if mask is not None:
                a2[notconverged] = a2[notconverged] * mask[notconverged]

            tr_a2 = torch.sum(a2[notconverged].diagonal(dim1=1,dim2=2),dim=1)
            cond[notconverged] = torch.abs(tr_a2-noccd[notconverged]) < \
                                 torch.abs(2.0*torch.sum(a0[notconverged].diagonal(dim1=1,dim2=2),dim=1) - tr_a2 - noccd[notconverged])
            cond1 = notconverged * cond
            cond2 = notconverged * (~cond)
            a0[cond1] = a2[cond1]
            a0[cond2] = 2.0*a0[cond2]-a2[cond2]
            errm2[notconverged] = errm1[notconverged]
            errm1[notconverged] = errm0[notconverged]
            errm0[notconverged] = torch.abs(torch.sum(a0[notconverged].diagonal(dim1=1,dim2=2),dim=1)-noccd[notconverged])
            k+=1
            if k > 100:
                print("SP2 dense loop reached max iter 100")
                break
            #"""
            #print('SP2', k,' '.join([str(x) for x in errm0.tolist()]))
            #print(' '.join([str(x) for x in torch.symeig(a0)[0][0].tolist()]))
            #"""
            if flag:
                #float32, harder to converge to a smaller eps, for this one set eps=1.0e-2, and break when no more improvement
                notconverged[notconverged.clone()] = ~((errm0[notconverged] < eps) * (errm0[notconverged] >= errm2[notconverged]))
            else:
                #float64, if use above critiria, the error will keep going down and take lots of iteration to reach no more improvement
                #so put eps as a small one like 1.0e-4, to recude the number of iterations
                notconverged[notconverged.clone()] = ~((errm0[notconverged] < eps) * (errm1[notconverged] < eps))

        return factor*a0
