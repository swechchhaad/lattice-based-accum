from __future__ import annotations

from typing import List
from .ring import RingElement

Vector = List[RingElement]
Matrix = List[List[RingElement]]


def zero_vector(length: int, q: int, d: int) -> Vector:
    return [RingElement.zero(q, d) for _ in range(length)]


def zero_matrix(rows: int, cols: int, q: int, d: int) -> Matrix:
    return [[RingElement.zero(q, d) for _ in range(cols)] for _ in range(rows)]


def identity_matrix(n: int, q: int, d: int) -> Matrix:
    M = zero_matrix(n, n, q, d)
    for i in range(n):
        M[i][i] = RingElement.one(q, d)
    return M


def random_matrix(rows: int, cols: int, q: int, d: int) -> Matrix:
    return [[RingElement.random(q, d) for _ in range(cols)] for _ in range(rows)]


def random_small_matrix(rows: int, cols: int, q: int, d: int, bound: int = 1) -> Matrix:
    return [
        [RingElement.random_small(q, d, bound) for _ in range(cols)]
        for _ in range(rows)
    ]


def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    rows, cols = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]


def matrix_sub(A: Matrix, B: Matrix) -> Matrix:
    rows, cols = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(cols)] for i in range(rows)]


def matrix_vector_mul(A: Matrix, x: Vector) -> Vector:
    rows, cols = len(A), len(A[0])
    if len(x) != cols:
        raise ValueError(f"Dimension mismatch in matrix_vector_mul: {cols} vs {len(x)}")
    q, d = x[0].q, x[0].d
    out = []
    for i in range(rows):
        acc = RingElement.zero(q, d)
        for j in range(cols):
            acc = acc + (A[i][j] * x[j])
        out.append(acc)
    return out


def matrix_mul(A: Matrix, B: Matrix) -> Matrix:
    rows, mid = len(A), len(A[0])
    if len(B) != mid:
        raise ValueError("Dimension mismatch in matrix_mul")
    cols = len(B[0])
    q, d = A[0][0].q, A[0][0].d
    out = zero_matrix(rows, cols, q, d)
    for i in range(rows):
        for k in range(mid):
            a = A[i][k]
            if a.is_zero():
                continue
            for j in range(cols):
                out[i][j] = out[i][j] + (a * B[k][j])
    return out


def scalar_matrix_mul(s: RingElement, A: Matrix) -> Matrix:
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def hcat(*blocks: Matrix) -> Matrix:
    rows = len(blocks[0])
    out: Matrix = []
    for i in range(rows):
        row: List[RingElement] = []
        for blk in blocks:
            row.extend(blk[i])
        out.append(row)
    return out


def vcat(*blocks: Matrix) -> Matrix:
    out: Matrix = []
    for blk in blocks:
        out.extend([list(row) for row in blk])
    return out


def transpose(A: Matrix) -> Matrix:
    return [list(row) for row in zip(*A)]


def concat_vectors(*parts: Vector) -> Vector:
    out: Vector = []
    for p in parts:
        out.extend(p)
    return out


def vector_add(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Dimension mismatch in vector_add")
    return [x + y for x, y in zip(a, b)]


def vector_norm_inf(v: Vector) -> int:
    return max((e.norm_inf() for e in v), default=0)


def kron_bit_gadget_row(x_bits: List[int], G: Matrix) -> Matrix:
    """
    Compute x^T (x) G, where x is a bit vector of length ell and G is n x m',
    giving an n x (ell*m') matrix. The k-th gadget block is multiplied by x[k].
    """
    n = len(G)
    mp = len(G[0])
    q, d = G[0][0].q, G[0][0].d
    out = zero_matrix(n, len(x_bits) * mp, q, d)
    for k, bit in enumerate(x_bits):
        if bit == 1:
            for i in range(n):
                for j in range(mp):
                    out[i][k * mp + j] = G[i][j]
    return out
