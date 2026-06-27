from __future__ import annotations

from typing import List
from .ring import RingElement
from .linalg import (
    Matrix,
    matrix_mul,
    matrix_sub,
    scalar_matrix_mul,
    identity_matrix,
    zero_matrix,
    vcat,
)
from .gadget import build_gadget_matrix, g_inv_matrix
from .params import LatticeParams


def _block(B: Matrix, i: int, mp: int) -> Matrix:
    """Return the i-th n x m' gadget block of B = [B_0 | ... | B_{ell-1}]."""
    return [row[i * mp : (i + 1) * mp] for row in B]


def _D_matrices(B: Matrix, y_bits: List[int], params: LatticeParams) -> List[Matrix]:
    """
    D_i = B_i           if y_i == 1
        = G - B_i       if y_i == 0
    (Corollary 2.2.1, with the indicator defined w.r.t. y.)
    """
    G = build_gadget_matrix(params)
    mp = params.m_prime
    Ds = []
    for i, yi in enumerate(y_bits):
        Bi = _block(B, i, mp)
        Ds.append(Bi if yi == 1 else matrix_sub(G, Bi))
    return Ds


def _nested_T(Ds: List[Matrix], params: LatticeParams) -> List[Matrix]:
    """
    T_{ell-1} = D_{ell-1}
    T_i       = D_i * G^{-1}(T_{i+1})      for i = ell-2 .. 0
    Returns [T_0, ..., T_{ell-1}], each n x m'.  T_0 = EvalF(1_y, B).
    """
    ell = len(Ds)
    T: List[Matrix] = [None] * ell  # type: ignore
    T[ell - 1] = Ds[ell - 1]
    for i in range(ell - 2, -1, -1):
        T[i] = matrix_mul(Ds[i], g_inv_matrix(T[i + 1], params))
    return T


def eval_f_indicator(B: Matrix, y_bits: List[int], params: LatticeParams) -> Matrix:
    """
    EvalF(1_y, B) = B_{1_y}, an n x m' matrix (the A_f of Theorem 2.2 / Eq. 2.1).
    """
    Ds = _D_matrices(B, y_bits, params)
    return _nested_T(Ds, params)[0]


def eval_fx_indicator(
    B: Matrix,
    y_bits: List[int],
    x_bits: List[int],
    params: LatticeParams,
) -> Matrix:
    """
    EvalFx(1_y, B, x) = H_{1_y, x}, an (ell*m') x m' matrix (Corollary 2.2.1).

    Row i block (each m' x m'):
        i < ell-1 :  (-1)^(1-y_i) * (prod_{j<i} delta_{y_j,x_j}) * G^{-1}(T_{i+1})
        i = ell-1 :  (-1)^(1-y_i) * (prod_{j<i} delta_{y_j,x_j}) * I_{m'}
    where delta_{y_j,x_j} = 1 iff y_j == x_j.
    """
    ell = len(y_bits)
    mp = params.m_prime
    q, d = params.q, params.d

    Ds = _D_matrices(B, y_bits, params)
    T = _nested_T(Ds, params)

    pos_one = RingElement.one(q, d)
    neg_one = -RingElement.one(q, d)
    I_mp = identity_matrix(mp, q, d)

    blocks: List[Matrix] = []
    delta_prefix = 1  # product of delta_{y_j,x_j} for j < i
    for i in range(ell):
        sign = pos_one if y_bits[i] == 1 else neg_one  # (-1)^(1-y_i)
        if delta_prefix == 0:
            blocks.append(zero_matrix(mp, mp, q, d))
        else:
            base_block = I_mp if i == ell - 1 else g_inv_matrix(T[i + 1], params)
            blocks.append(scalar_matrix_mul(sign, base_block))
        if x_bits[i] != y_bits[i]:
            delta_prefix = 0

    return vcat(*blocks)


def check_eval_relation(
    B: Matrix,
    y_bits: List[int],
    x_bits: List[int],
    params: LatticeParams,
) -> bool:
    """
    Self-check of Eq. 2.1:
        (B - x^T (x) G) * H_{1_y,x} == B_{1_y} - 1_y(x) * G.
    """
    from .linalg import kron_bit_gadget_row

    G = build_gadget_matrix(params)
    Af = eval_f_indicator(B, y_bits, params)
    H = eval_fx_indicator(B, y_bits, x_bits, params)

    lhs = matrix_mul(matrix_sub(B, kron_bit_gadget_row(x_bits, G)), H)
    indicator = 1 if list(x_bits) == list(y_bits) else 0
    rhs = Af if indicator == 0 else matrix_sub(Af, G)
    return lhs == rhs
