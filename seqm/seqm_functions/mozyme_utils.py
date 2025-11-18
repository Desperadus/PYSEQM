import torch
import warnings
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Valence electrons for common elements (Group number)
VALENCE_ELECTRONS = {
    1: 1, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
    11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7,
    35: 7, 53: 7
}

def build_lewis_structure(molecule, seqm_parameters):
    """
    Builds the Lewis structure (connectivity, bond orders, formal charges, lone pairs).
    """
    if RDKIT_AVAILABLE:
        try:
            return _build_lewis_structure_rdkit(molecule)
        except Exception as e:
            warnings.warn(f"RDKit failed to build Lewis structure: {e}. Falling back to distance check.")
            return _build_lewis_structure_distance(molecule)
    else:
        warnings.warn("RDKit not found. Using simple distance-based connectivity for MOZYME Lewis structure.")
        return _build_lewis_structure_distance(molecule)

def _build_lewis_structure_rdkit(molecule):
    """
    Uses RDKit to determine connectivity, bond orders, charges, and lone pairs.
    """
    species = molecule.species.cpu().numpy()
    coords = molecule.coordinates.detach().cpu().numpy()
    
    nmol = species.shape[0]
    molsize = species.shape[1]
    device = molecule.coordinates.device
    
    adj_matrices = torch.zeros((nmol, molsize, molsize), dtype=torch.int8, device=device)
    bond_orders = torch.zeros((nmol, molsize, molsize), dtype=torch.float32, device=device)
    formal_charges = torch.zeros((nmol, molsize), dtype=torch.int8, device=device)
    lone_pairs = torch.zeros((nmol, molsize), dtype=torch.int8, device=device)
    
    for i in range(nmol):
        mol = Chem.RWMol()
        atom_indices = []
        
        # Add atoms
        for j in range(molsize):
            atomic_num = int(species[i, j])
            if atomic_num > 0:
                a = Chem.Atom(atomic_num)
                idx = mol.AddAtom(a)
                atom_indices.append(idx)
            else:
                atom_indices.append(-1) # Padding atom
                
        # Add conformer for 3D coordinates
        conf = Chem.Conformer(len(atom_indices))
        valid_atom_count = 0
        for j in range(molsize):
            if atom_indices[j] != -1:
                conf.SetAtomPosition(valid_atom_count, (float(coords[i, j, 0]), float(coords[i, j, 1]), float(coords[i, j, 2])))
                valid_atom_count += 1
        mol.AddConformer(conf)
        
        # Determine connectivity and bond orders
        try:
            rdDetermineBonds.DetermineConnectivity(mol)
            rdDetermineBonds.DetermineBondOrders(mol, charge=int(molecule.tot_charge[i].item()))
        except Exception as e:
            raise e

        # Map valid_idx to original_idx
        valid_to_orig = {}
        for orig_idx, valid_idx in enumerate(atom_indices):
            if valid_idx != -1:
                valid_to_orig[valid_idx] = orig_idx

        # Extract Bond Info
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            orig_begin = valid_to_orig[begin_idx]
            orig_end = valid_to_orig[end_idx]
            
            btype = bond.GetBondType()
            bo = 1.0
            if btype == Chem.BondType.DOUBLE: bo = 2.0
            elif btype == Chem.BondType.TRIPLE: bo = 3.0
            elif btype == Chem.BondType.AROMATIC: bo = 1.5
            
            adj_matrices[i, orig_begin, orig_end] = 1
            adj_matrices[i, orig_end, orig_begin] = 1
            bond_orders[i, orig_begin, orig_end] = bo
            bond_orders[i, orig_end, orig_begin] = bo

        # Extract Atom Info (Charges, Lone Pairs)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            orig_idx = valid_to_orig[idx]
            
            fcharge = atom.GetFormalCharge()
            formal_charges[i, orig_idx] = fcharge
            
            atomic_num = atom.GetAtomicNum()
            valence_e = VALENCE_ELECTRONS.get(atomic_num, 0)
            
            bonded_e = 0.0
            for bond in atom.GetBonds():
                bonded_e += bond.GetBondTypeAsDouble()
            
            lp = (valence_e - bonded_e - fcharge) / 2.0
            if lp < 0: lp = 0
            lone_pairs[i, orig_idx] = int(lp)

    return {
        'adj_matrix': adj_matrices,
        'bond_orders': bond_orders,
        'formal_charges': formal_charges,
        'lone_pairs': lone_pairs
    }

def _build_lewis_structure_distance(molecule):
    """
    Simple distance-based connectivity.
    """
    covalent_radii = {
        1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
        15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
    }
    default_radius = 1.5
    
    species = molecule.species
    coords = molecule.coordinates
    
    nmol, molsize = species.shape
    device = coords.device
    
    adj_matrices = torch.zeros((nmol, molsize, molsize), dtype=torch.int8, device=device)
    bond_orders = torch.zeros((nmol, molsize, molsize), dtype=torch.float32, device=device)
    formal_charges = torch.zeros((nmol, molsize), dtype=torch.int8, device=device)
    lone_pairs = torch.zeros((nmol, molsize), dtype=torch.int8, device=device)
    
    if hasattr(molecule, 'rij') and molecule.rij is not None:
        
        rij = molecule.rij
        idxi = molecule.idxi
        idxj = molecule.idxj
        pair_molid = molecule.pair_molid
        
        if hasattr(molecule, 'const') and hasattr(molecule.const, 'length_conversion_factor'):
            bohr_conv = molecule.const.length_conversion_factor
        else:
            bohr_conv = 1.8897259886
            
        local_i = idxi % molsize
        local_j = idxj % molsize
        
        Z = molecule.Z
        Zi = Z[idxi]
        Zj = Z[idxj]
        
        max_z = int(torch.max(Z).item()) + 1
        radii_tensor = torch.full((max_z,), default_radius, device=coords.device)
        for z, r in covalent_radii.items():
            if z < max_z:
                radii_tensor[z] = r
                
        Ri = radii_tensor[Zi] * bohr_conv
        Rj = radii_tensor[Zj] * bohr_conv
        
        tolerance = 0.4 * bohr_conv
        is_bonded = rij < (Ri + Rj + tolerance)
        
        bonded_indices = is_bonded.nonzero().squeeze()
        
        if bonded_indices.numel() > 0:
            if bonded_indices.dim() == 0:
                 bonded_indices = bonded_indices.unsqueeze(0)
            mols = pair_molid[bonded_indices]
            is_local = local_i[bonded_indices]
            js_local = local_j[bonded_indices]
            
            adj_matrices[mols, is_local, js_local] = 1
            adj_matrices[mols, js_local, is_local] = 1
            
            bond_orders[mols, is_local, js_local] = 1.0
            bond_orders[mols, js_local, is_local] = 1.0
            
    return {
        'adj_matrix': adj_matrices,
        'bond_orders': bond_orders,
        'formal_charges': formal_charges,
        'lone_pairs': lone_pairs
    }

def construct_initial_density(molecule, nmol, molsize):
    """
    Constructs the initial density matrix P0 based on the Lewis structure.
    
    Args:
        molecule: The Molecule object with Lewis structure info.
        nmol: Number of molecules.
        molsize: Max molecule size.
        
    Returns:
        P0: Initial density matrix (nmol, norb, norb).
    """
    # nmol = molecule.nmol # Not yet attached
    # molsize = molecule.molsize # Not yet attached
    # norb = molecule.norb # Not yet attached.
    # Assuming standard padding of 4 orbitals per atom (s, px, py, pz)
    # Even for H, we allocate a block, though only s is used.
    
    max_orb = molsize * 4
    P0 = torch.zeros((nmol, max_orb, max_orb), dtype=molecule.coordinates.dtype, device=molecule.coordinates.device)
    
    # We need to iterate through molecules and atoms.
    # Vectorization is hard here due to complex logic. Loop over molecules is fine for now.
    
    species = molecule.species
    coords = molecule.coordinates
    bond_orders = molecule.bond_orders
    lone_pairs = molecule.lone_pairs
    adj = molecule.adj_matrix
    
    for i in range(nmol):
        # Per molecule
        atom_hybridization = {} # Map atom_idx -> 'sp3', 'sp2', 'sp'
        atom_coords = coords[i]
        
        # 1. Determine Hybridization
        for j in range(molsize):
            atomic_num = int(species[i, j])
            if atomic_num <= 0: continue
            if atomic_num == 1:
                atom_hybridization[j] = 's'
                continue
                
            # Count neighbors and bond orders
            neighbors = []
            bo_sum = 0.0
            for k in range(molsize):
                if adj[i, j, k] == 1:
                    neighbors.append(k)
                    bo_sum += bond_orders[i, j, k].item()
            
            coord_num = len(neighbors)
            lp_count = int(lone_pairs[i, j].item())
            steric_num = coord_num + lp_count
            
            if steric_num >= 4:
                atom_hybridization[j] = 'sp3'
            elif steric_num == 3:
                atom_hybridization[j] = 'sp2'
            elif steric_num == 2:
                atom_hybridization[j] = 'sp'
            else:
                atom_hybridization[j] = 'sp3' # Default
        
        # 2. Construct LMOs and fill P0
        # We need to track used orbitals/electrons to ensure we don't double count?
        # Or just iterate through bonds and lone pairs.
        
        # Helper to get orbital indices for atom j
        def get_orb_indices(atom_idx):
            start = atom_idx * 4
            return [start, start+1, start+2, start+3] # s, px, py, pz
            
        # Helper to get hybrid orbital coefficients (s, px, py, pz)
        # pointing from atom A to atom B
        def get_hybrid_coeffs(idx_A, idx_B, hyb_type):
            if hyb_type == 's':
                return torch.tensor([1.0, 0.0, 0.0, 0.0], device=molecule.coordinates.device)
            
            vec = atom_coords[idx_B] - atom_coords[idx_A]
            dist = torch.norm(vec)
            if dist < 1e-6: return torch.tensor([1.0, 0.0, 0.0, 0.0], device=molecule.coordinates.device)
            vec = vec / dist
            
            # s, px, py, pz
            # sp3: 1/2 s + sqrt(3)/2 p_sigma
            # sp2: 1/sqrt(3) s + sqrt(2)/3 p_sigma
            # sp: 1/sqrt(2) s + 1/sqrt(2) p_sigma
            
            s_coeff = 0.0
            p_coeff = 0.0
            
            if hyb_type == 'sp3':
                s_coeff = 0.5
                p_coeff = np.sqrt(3.0)/2.0
            elif hyb_type == 'sp2':
                s_coeff = 1.0/np.sqrt(3.0)
                p_coeff = np.sqrt(2.0/3.0)
            elif hyb_type == 'sp':
                s_coeff = 1.0/np.sqrt(2.0)
                p_coeff = 1.0/np.sqrt(2.0)
            
            # p_sigma = vec_x * px + vec_y * py + vec_z * pz
            return torch.stack([torch.tensor(s_coeff, device=vec.device), 
                                p_coeff*vec[0], 
                                p_coeff*vec[1], 
                                p_coeff*vec[2]])

        # A. Bonds
        # Iterate upper triangle of adj
        for j in range(molsize):
            for k in range(j+1, molsize):
                if adj[i, j, k] == 1:
                    bo = bond_orders[i, j, k].item()
                    
                    # Sigma Bond
                    hyb_j = atom_hybridization.get(j, 'sp3')
                    hyb_k = atom_hybridization.get(k, 'sp3')
                    
                    phi_j = get_hybrid_coeffs(j, k, hyb_j)
                    phi_k = get_hybrid_coeffs(k, j, hyb_k)
                    
                    # Construct Bond Orbital (simplified homonuclear approximation for coeffs)
                    # psi = 1/sqrt(2) (phi_j + phi_k)
                    # P contribution = 2 * psi * psi.T
                    # P_uv = 2 * c_u * c_v
                    
                    # Indices
                    idx_j = get_orb_indices(j)
                    idx_k = get_orb_indices(k)
                    
                    # We add to P0 blocks: JJ, KK, JK, KJ
                    # P_JJ += 2 * (1/sqrt(2))^2 * phi_j * phi_j.T = phi_j * phi_j.T
                    # P_JK += 2 * (1/sqrt(2))^2 * phi_j * phi_k.T = phi_j * phi_k.T
                    
                    # Wait, normalization.
                    # psi = c1 phi1 + c2 phi2.
                    # For homopolar: c1=c2=1/sqrt(2).
                    # Density D = 2 * |psi><psi|
                    # D = 2 * 0.5 * (|phi1><phi1| + |phi1><phi2| + |phi2><phi1| + |phi2><phi2|)
                    # D = |phi1><phi1| + ...
                    
                    # Add to P0
                    # Block JJ
                    P0[i, idx_j[0]:idx_j[3]+1, idx_j[0]:idx_j[3]+1] += torch.outer(phi_j, phi_j)
                    # Block KK
                    P0[i, idx_k[0]:idx_k[3]+1, idx_k[0]:idx_k[3]+1] += torch.outer(phi_k, phi_k)
                    # Block JK
                    P0[i, idx_j[0]:idx_j[3]+1, idx_k[0]:idx_k[3]+1] += torch.outer(phi_j, phi_k)
                    # Block KJ
                    P0[i, idx_k[0]:idx_k[3]+1, idx_j[0]:idx_j[3]+1] += torch.outer(phi_k, phi_j)
                    
                    # Pi Bonds (Double/Triple)
                    if bo >= 2.0:
                        # Need to find p-orbitals orthogonal to sigma bond vector
                        vec = atom_coords[k] - atom_coords[j]
                        vec = vec / torch.norm(vec)
                        
                        # Arbitrary perpendicular vector
                        if abs(vec[0]) < 0.9:
                            aux = torch.tensor([1.0, 0.0, 0.0], device=molecule.coordinates.device)
                        else:
                            aux = torch.tensor([0.0, 1.0, 0.0], device=molecule.coordinates.device)
                            
                        pi1 = torch.cross(vec, aux)
                        pi1 = pi1 / torch.norm(pi1)
                        
                        # Construct pi orbitals (pure p)
                        # phi_pi_j = 0*s + pi1_x*px + ...
                        phi_pi_j = torch.tensor([0.0, pi1[0], pi1[1], pi1[2]], device=molecule.coordinates.device)
                        phi_pi_k = torch.tensor([0.0, pi1[0], pi1[1], pi1[2]], device=molecule.coordinates.device) # Parallel alignment
                        
                        # Add density (same logic as sigma)
                        P0[i, idx_j[0]:idx_j[3]+1, idx_j[0]:idx_j[3]+1] += torch.outer(phi_pi_j, phi_pi_j)
                        P0[i, idx_k[0]:idx_k[3]+1, idx_k[0]:idx_k[3]+1] += torch.outer(phi_pi_k, phi_pi_k)
                        P0[i, idx_j[0]:idx_j[3]+1, idx_k[0]:idx_k[3]+1] += torch.outer(phi_pi_j, phi_pi_k)
                        P0[i, idx_k[0]:idx_k[3]+1, idx_j[0]:idx_j[3]+1] += torch.outer(phi_pi_k, phi_pi_j)
                        
                    if bo >= 3.0:
                        # Second pi bond
                        pi2 = torch.cross(vec, pi1)
                        pi2 = pi2 / torch.norm(pi2)
                        
                        phi_pi2_j = torch.tensor([0.0, pi2[0], pi2[1], pi2[2]], device=molecule.coordinates.device)
                        phi_pi2_k = torch.tensor([0.0, pi2[0], pi2[1], pi2[2]], device=molecule.coordinates.device)
                        
                        P0[i, idx_j[0]:idx_j[3]+1, idx_j[0]:idx_j[3]+1] += torch.outer(phi_pi2_j, phi_pi2_j)
                        P0[i, idx_k[0]:idx_k[3]+1, idx_k[0]:idx_k[3]+1] += torch.outer(phi_pi2_k, phi_pi2_k)
                        P0[i, idx_j[0]:idx_j[3]+1, idx_k[0]:idx_k[3]+1] += torch.outer(phi_pi2_j, phi_pi2_k)
                        P0[i, idx_k[0]:idx_k[3]+1, idx_j[0]:idx_j[3]+1] += torch.outer(phi_pi2_k, phi_pi2_j)

        # B. Lone Pairs
        for j in range(molsize):
            lp_count = int(lone_pairs[i, j].item())
            if lp_count > 0:
                # We need to assign lone pairs to remaining hybrid orbitals.
                # This is tricky without explicit geometry of hybrids.
                # Simplified: Just fill diagonal p-character not used by bonds?
                # Or construct explicit hybrids pointing "away" from bonds.
                
                # For now, let's just add density to the atom's block to conserve charge.
                # If we don't direct them, we can just fill s/p isotropically or fill remaining capacity.
                
                # Better approximation:
                # If sp3 and 2 bonds (Water), we have 2 lone pairs.
                # They occupy the other 2 sp3 lobes.
                # We can just add 2.0 to the diagonal population of the atom, distributed among s/p.
                # But P0 needs off-diagonals (hybridization) to be effective.
                
                # Let's try to construct a "dummy" hybrid direction?
                # Or just fill the diagonal for now to ensure electron count is correct.
                # P_ss += 2 * s_character
                # P_pp += 2 * p_character
                
                # For sp3: s=0.25, p=0.75.
                # Each LP adds 2 electrons.
                # 2 * 0.25 = 0.5 to s.
                # 2 * 0.75 = 1.5 to p (0.5 to px, py, pz).
                
                hyb = atom_hybridization.get(j, 'sp3')
                s_frac = 0.0
                p_frac = 0.0
                if hyb == 'sp3': s_frac, p_frac = 0.25, 0.75
                elif hyb == 'sp2': s_frac, p_frac = 0.33, 0.67
                elif hyb == 'sp': s_frac, p_frac = 0.5, 0.5
                elif hyb == 's': s_frac, p_frac = 1.0, 0.0
                
                idx_j = get_orb_indices(j)
                
                for _ in range(lp_count):
                    # Add 2 electrons distributed according to hybridization
                    # Note: This ignores directionality of lone pairs (off-diagonals between s/p),
                    # but ensures correct atomic population.
                    P0[i, idx_j[0], idx_j[0]] += 2.0 * s_frac
                    P0[i, idx_j[1], idx_j[1]] += 2.0 * p_frac / 3.0
                    P0[i, idx_j[2], idx_j[2]] += 2.0 * p_frac / 3.0
                    P0[i, idx_j[3], idx_j[3]] += 2.0 * p_frac / 3.0
                    
                    # To add directionality (hybridization), we would need P_sp terms.
                    # P_sp terms come from the cross terms in (as + bp)^2.
                    # Without direction, we assume average.

    return P0
