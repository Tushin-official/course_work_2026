# ising_mapping.py
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from typing import Dict, Tuple

QuadKey = Tuple[int, int]  # (u,v) u<v

def qubo_to_ising(lin: np.ndarray, quad: Dict[QuadKey, float]):
    """
    QUBO:
      E(z) = sum_i lin[i] z_i + sum_{i<j} quad[i,j] z_i z_j

    Map z_i=(s_i+1)/2, s_i in {-1,+1}:
      E(s) = const + sum_i a_i s_i + sum_{i<j} b_ij s_i s_j

    For SB implementation expecting energy:
      E_SB(s) = - sum_{i<j} J_ij s_i s_j - sum_i h_i s_i (+const)
    choose:
      J_ij = -b_ij
      h_i  = -a_i
    """
    lin = np.asarray(lin, dtype=float)
    M = lin.shape[0]

    a = np.zeros(M, dtype=float)
    const = 0.0

    # linear: lin_i * (s_i+1)/2
    a += lin / 2.0
    const += float(np.sum(lin) / 2.0)

    # quadratic: q_ij * (s_i s_j + s_i + s_j + 1)/4
    rows = []
    cols = []
    data = []

    for (i, j), q in quad.items():
        q = float(q)
        if q == 0.0:
            continue

        # b_ij term
        b = q / 4.0

        # contributes to a_i, a_j and const
        a[i] += q / 4.0
        a[j] += q / 4.0
        const += q / 4.0

        # SB wants J = -b
        Jij = -b
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([Jij, Jij])

    J = csr_matrix(coo_matrix((data, (rows, cols)), shape=(M, M)))
    h = (-a).reshape(-1, 1)  # SBBase expects (M,1)
    return J, h, const