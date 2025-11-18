import torch
import time
import gc

def benchmark_sparse_vs_dense(size, sparsity, device='cpu'):
    print(f"--- Benchmark: Size={size}, Sparsity={sparsity}, Device={device} ---")
    
    # Create random sparse mask
    mask = (torch.rand(size, size, device=device) > sparsity).float()
    nnz = mask.sum().item()
    print(f"Non-zeros: {nnz} ({nnz/(size*size)*100:.2f}%)")
    
    # Dense Matrix
    A_dense = torch.randn(size, size, device=device)
    A_dense = A_dense * mask # Apply sparsity pattern
    
    # Sparse Matrix (COO might be better supported on non-MKL CPU builds)
    A_sparse = A_dense.to_sparse()
    
    # --- Benchmark Dense + Mask ---
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(10):
        C_dense = torch.matmul(A_dense, A_dense)
        C_dense = C_dense * mask # Enforce sparsity
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    time_dense = (end - start) / 10
    print(f"Dense + Mask Time: {time_dense*1000:.2f} ms")
    
    # --- Benchmark Sparse ---
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(10):
        C_sparse = torch.matmul(A_sparse, A_sparse)
        # Note: Result of sparse x sparse is sparse, but structure might expand.
        # In SP2, we want to keep the original sparsity pattern (or close to it).
        # torch.sparse doesn't support element-wise mult with another sparse matrix easily if indices differ.
        # But let's just measure the matmul for now, as that's the heavy lifter.
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    time_sparse = (end - start) / 10
    print(f"Sparse Time:      {time_sparse*1000:.2f} ms")
    
    print(f"Speedup: {time_dense / time_sparse:.2f}x")
    print("-" * 40)

if __name__ == "__main__":
    # Test with sizes relevant to linear scaling (where N is large)
    # Sparsity should be high (e.g., 95% zeros)
    sizes = [500, 1000, 2000, 4000]
    sparsity = 0.95 # 95% zeros
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for size in sizes:
        benchmark_sparse_vs_dense(size, sparsity, device)
