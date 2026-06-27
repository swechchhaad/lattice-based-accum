from __future__ import annotations

from typing import List
from .ring import RingElement
from .linalg import Matrix, Vector, zero_matrix
from .params import LatticeParams


def build_gadget_matrix(params: LatticeParams) -> Matrix:
    """G = I_n (x) g^T, an n x m' matrix over R_q."""
    qtilde = params.qtilde
    q, d, n = params.q, params.d, params.n
    g_row = [RingElement.from_int(params.base ** i, q, d) for i in range(qtilde)]

    G = zero_matrix(n, n * qtilde, q, d)
    for i in range(n):
        for j, ge in enumerate(g_row):
            G[i][i * qtilde + j] = ge
    return G


def _decompose_element(u: RingElement, base: int, qtilde: int) -> List[RingElement]:
    """
    Gadget-decompose a single ring element against g^T = (1, b, ..., b^{qtilde-1}).

    Returns qtilde ring elements v_0, ..., v_{qtilde-1} with small (in [0, base))
    coefficients such that sum_j base^j * v_j = u  (exactly, over R_q).
    Digit j collects the j-th base-b digit of every coefficient of u.
    """
    q, d = u.q, u.d
    digit_coeffs = [[0] * d for _ in range(qtilde)]
    for c_idx, c in enumerate(u.coeffs):  # c in [0, q)
        val = c
        for j in range(qtilde):
            digit_coeffs[j][c_idx] = val % base
            val //= base
    return [RingElement(digit_coeffs[j], q, d) for j in range(qtilde)]


def g_inv_vector(u: Vector, params: LatticeParams) -> Vector:
    """
    G^{-1} applied to a target vector u in R_q^n.
    Returns v in R_q^{m'} (m' = n*qtilde) with G * v = u.
    Entry i of u expands into qtilde digit elements at positions i*qtilde .. i*qtilde+qtilde-1.
    """
    if len(u) != params.n:
        raise ValueError(f"g_inv_vector expects length n={params.n}, got {len(u)}")
    out: Vector = []
    for entry in u:
        out.extend(_decompose_element(entry, params.base, params.qtilde))
    return out


def g_inv_matrix(M: Matrix, params: LatticeParams) -> Matrix:
    """
    G^{-1} applied column-wise to an n x t matrix, giving an m' x t matrix
    with G * G^{-1}(M) = M.
    """
    if len(M) != params.n:
        raise ValueError(f"g_inv_matrix expects n={params.n} rows, got {len(M)}")
    cols = len(M[0])
    decomposed_cols = []
    for j in range(cols):
        col = [M[i][j] for i in range(params.n)]
        decomposed_cols.append(g_inv_vector(col, params))
    # reassemble into m' x t
    mp = params.m_prime
    out = zero_matrix(mp, cols, params.q, params.d)
    for j in range(cols):
        for i in range(mp):
            out[i][j] = decomposed_cols[j][i]
    return out
