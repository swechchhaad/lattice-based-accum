from __future__ import annotations

from dataclasses import dataclass

from .linalg import (
    Matrix,
    Vector,
    hcat,
    matrix_add,
    matrix_sub,
    matrix_vector_mul,
    random_matrix,
    random_small_matrix,
    concat_vectors,
    vector_add,
)
from .gadget import build_gadget_matrix, g_inv_vector
from .params import LatticeParams
from .ring import RingElement


@dataclass
class Trapdoor:
    """
    MP12 gadget trapdoor for A = [A_bar | G - A_bar R].
        A_bar : n x m_bar  (the uniform part)
        R     : m_bar x m' (short secret), satisfying A * [R; I_{m'}] = G
    """

    A_bar: Matrix
    R: Matrix


def trapgen(params: LatticeParams) -> tuple[Matrix, Trapdoor]:
    n, m_bar, mp = params.n, params.m_bar, params.m_prime
    q, d = params.q, params.d

    A_bar = random_matrix(n, m_bar, q, d)
    R = random_small_matrix(m_bar, mp, q, d, bound=1)  # short secret
    G = build_gadget_matrix(params)

    A_bar_R = _mat_mul(A_bar, R)             # n x m'
    second = matrix_sub(G, A_bar_R)          # G - A_bar R
    A = hcat(A_bar, second)                  # n x m
    return A, Trapdoor(A_bar=A_bar, R=R)


def _mat_mul(A: Matrix, B: Matrix) -> Matrix:
    from .linalg import matrix_mul

    return matrix_mul(A, B)


def _sample_pre_A(A_trap: Trapdoor, params: LatticeParams, target: Vector) -> Vector:
    """
    Short preimage of `target` under A alone:  A * s0 = target, s0 in R_q^m.

    Using A = [A_bar | G - A_bar R] and A * [R; I] = G:
        let w = G^{-1}(target)  (short, gadget digits in [0, base))
        then s0 = [R*w ; w] satisfies
            A * s0 = A_bar (R w) + (G - A_bar R) w = G w = target.
    """
    w = g_inv_vector(target, params)          # m' vector, short
    Rw = matrix_vector_mul(A_trap.R, w)       # m_bar vector
    return concat_vectors(Rw, w)              # length m_bar + m' = m


def sample_left(
    A: Matrix,
    A_trap: Trapdoor,
    M: Matrix,
    target: Vector,
    params: LatticeParams,
    mask_bound: int = 1,
) -> Vector:

    m_cols = len(M[0])
    q, d = params.q, params.d

    s1 = [RingElement.random_small(q, d, mask_bound) for _ in range(m_cols)]
    residual = _sub_vec(target, matrix_vector_mul(M, s1))
    s0 = _sample_pre_A(A_trap, params, residual)
    return concat_vectors(s0, s1)


def _sub_vec(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


# sanity self-check helpers (used by tests / demo

def verify_trapdoor(A: Matrix, A_trap: Trapdoor, params: LatticeParams) -> bool:
    """Check that A * [R; I] = G, i.e. R is a valid gadget trapdoor for A."""
    from .linalg import identity_matrix, vcat, matrix_mul

    G = build_gadget_matrix(params)
    RI = vcat(A_trap.R, identity_matrix(params.m_prime, params.q, params.d))
    return matrix_mul(A, RI) == G
