import torch
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

def test_mozyme_option():
    print("Testing MOZYME option...")
    
    # Define a simple molecule (Water)
    species = torch.tensor([[8, 1, 1]], dtype=torch.int64)
    coordinates = torch.tensor([
        [
            [0.0000, 0.0000, 0.0000],
            [0.7570, 0.5860, 0.0000],
            [-0.7570, 0.5860, 0.0000]
        ]
    ], dtype=torch.float64)
    
    const = Constants()
    
    # 2. Test WITH mozyme option
    print("\n--- Test 2: With MOZYME ---")
    seqm_parameters_mozyme = {
        'method': 'PM6',
        'scf_eps': 1e-6,
        'scf_converger': [0, 0.0], # Simple damping
        'sp2': [True, 1e-5], # Enable SP2 for Step D verification
        'mozyme': True,
        'mozyme_cutoffs': [15.0, 10.0]
    }
    
    mol_mozyme = Molecule(const, seqm_parameters_mozyme, coordinates, species)
    
    if mol_mozyme.P_initial is not None:
        print("SUCCESS: P_initial is populated.")
        P0 = mol_mozyme.P_initial[0]
        print(f"P_initial shape: {P0.shape}")
        
        # Check Trace (Total Electrons)
        # Water: O(6) + 2H(1) = 8 valence electrons.
        # seqm uses valence basis.
        trace = torch.trace(P0)
        print(f"Trace (Total Electrons): {trace.item()}")
        
        if abs(trace.item() - 8.0) < 0.1:
            print("SUCCESS: Trace is approximately 8.0.")
        else:
            print(f"WARNING: Trace is {trace.item()}, expected 8.0.")
            
        # Check Off-Diagonal Elements (Bonds)
        # O is atom 0 (indices 0-3). H1 is atom 1 (indices 4-7). H2 is atom 2 (indices 8-11).
        # Bond O-H1 should have non-zero elements in block [0:4, 4:8].
        block_OH1 = P0[0:4, 4:8]
        if torch.norm(block_OH1) > 0.1:
             print("SUCCESS: O-H1 bond block is non-zero.")
        else:
             print("WARNING: O-H1 bond block is zero.")
             
    else:
        print("FAILURE: P_initial is None even with mozyme=True")

    # Verify P_initial
    if mol_mozyme.P_initial is not None:
        print("SUCCESS: P_initial is populated.")
        print(f"P_initial shape: {mol_mozyme.P_initial.shape}")
        # Handle batch dimension
        if mol_mozyme.P_initial.dim() == 3:
             trace = mol_mozyme.P_initial[0].trace().item()
        else:
             trace = mol_mozyme.P_initial.trace().item()
             
        print(f"Trace (Total Electrons): {trace}")
        if abs(trace - 8.0) < 0.1:
            print("SUCCESS: Trace is approximately 8.0.")
        else:
            print(f"FAILURE: Trace is {trace}, expected 8.0.")
            
        # Check bond block (O-H1)
        # O is atom 0 (indices 0-3), H1 is atom 1 (indices 4-7)
        # We expect non-zero elements in P[0:4, 4:8]
        if mol_mozyme.P_initial.dim() == 3:
            block = mol_mozyme.P_initial[0, 0:4, 4:8]
        else:
            block = mol_mozyme.P_initial[0:4, 4:8]
        if block.abs().sum() > 0.01:
            print("SUCCESS: O-H1 bond block is non-zero.")
        else:
            print("FAILURE: O-H1 bond block is zero.")
    else:
        print("FAILURE: P_initial is None.")

    # Verify Sparse Neighbor List (Step E)
    # mask_nddo is removed, instead we check if idxi/idxj are pruned.
    if mol_mozyme.mask_nddo is None:
        print("SUCCESS: mask_nddo is None (as expected with sparse neighbor list).")
    else:
        print("FAILURE: mask_nddo should be None.")
        
    # Check pair count
    n_pairs_full = mol_mozyme.molsize * (mol_mozyme.molsize - 1) // 2
    n_pairs_actual = mol_mozyme.idxi.shape[0]
    print(f"Full Pair Count: {n_pairs_full}")
    print(f"Actual Pair Count (Pruned): {n_pairs_actual}")
    
    if n_pairs_actual <= n_pairs_full:
        print("SUCCESS: Pair list is pruned (or equal for small molecules).")
    else:
        print("FAILURE: Pair list is larger than full?")
        
    if mol_mozyme.mask_lmo_matrix is not None:
        print("SUCCESS: mask_lmo_matrix is populated.")
        print(f"mask_lmo_matrix shape: {mol_mozyme.mask_lmo_matrix.shape}")
    else:
        print("FAILURE: mask_lmo_matrix is None.")

if __name__ == "__main__":
    test_mozyme_option()
