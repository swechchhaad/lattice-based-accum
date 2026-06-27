from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple

from .params import LatticeParams
from .ring import RingElement
from .linalg import (
    Matrix,
    Vector,
    hcat,
    matrix_sub,
    matrix_vector_mul,
    random_matrix,
    zero_vector,
    concat_vectors,
    vector_add,
    vector_norm_inf,
    kron_bit_gadget_row,
)
from .gadget import build_gadget_matrix, g_inv_vector
from .trapdoor import Trapdoor, trapgen, sample_left
from .homomorphic import eval_f_indicator, eval_fx_indicator


@dataclass
class PublicParams:
    A: Matrix          # n x m
    B: Matrix          # n x (ell * m')
    u: Vector          # n-vector
    params: LatticeParams


@dataclass
class SecretKey:
    TA: Trapdoor


@dataclass
class Witness:
    s: Vector          # length witness_dim = m + ell*m'

    def norm_inf(self) -> int:
        return vector_norm_inf(self.s)


@dataclass
class UpdateMessage:
    op: str                 # "Add" or "Del"
    y_bits: List[int]       # element deleted (empty for Add)


@dataclass
class AccumulatorState:
    c: Vector
    active: Set[Tuple[int, ...]] = field(default_factory=set)
    ever_added: Set[Tuple[int, ...]] = field(default_factory=set)


class LatticePRFA:
    """Positive replacement-free accumulator (Construction 4.1)."""

    def __init__(self, params: LatticeParams):
        self.params = params
        self.G = build_gadget_matrix(params)

    # gen

    def gen(self) -> Tuple[PublicParams, SecretKey, AccumulatorState]:
        p = self.params
        A, TA = trapgen(p)
        B = random_matrix(p.n, p.ell * p.m_prime, p.q, p.d)
        u = [RingElement.random(p.q, p.d) for _ in range(p.n)]
        c0 = [RingElement.random(p.q, p.d) for _ in range(p.n)]

        pp = PublicParams(A=A, B=B, u=u, params=p)
        sk = SecretKey(TA=TA)
        st = AccumulatorState(c=c0)
        return pp, sk, st

    # helpers

    def _element_key(self, x_bits: List[int]) -> Tuple[int, ...]:
        if len(x_bits) != self.params.ell or any(b not in (0, 1) for b in x_bits):
            raise ValueError("Element must be a bit vector of length ell")
        return tuple(x_bits)

    def _membership_matrix(self, pp: PublicParams, x_bits: List[int]) -> Matrix:
        """[A | B - x^T (x) G]."""
        x_kron_G = kron_bit_gadget_row(x_bits, self.G)
        return hcat(pp.A, matrix_sub(pp.B, x_kron_G))

    # add

    def add(
        self,
        pp: PublicParams,
        sk: SecretKey,
        st: AccumulatorState,
        x_bits: List[int],
    ) -> Tuple[Witness, UpdateMessage]:
        """
        Sign x against the current c (free addition: c is unchanged).
        Returns (witness, upmsg). upmsg carries no update info for Add.
        """
        key = self._element_key(x_bits)
        if key in st.ever_added:
            raise ValueError("PRFA forbids re-adding an element that was ever added before.")

        M_block = matrix_sub(pp.B, kron_bit_gadget_row(x_bits, self.G))  # B - x^T (x) G
        sx = sample_left(pp.A, sk.TA, M_block, st.c, self.params)

        if len(sx) != self.params.witness_dim:
            raise ValueError("sample_left returned wrong witness dimension")

        st.active.add(key)
        st.ever_added.add(key)
        return Witness(s=sx), UpdateMessage(op="Add", y_bits=[])

    # del

    def delete(
        self,
        pp: PublicParams,
        st: AccumulatorState,
        y_bits: List[int],
    ) -> UpdateMessage:
        """
        c' = c + B_{1_y} * G^{-1}(u). Publishes (Del, y).
        """
        key = self._element_key(y_bits)
        if key not in st.active:
            raise ValueError("Cannot delete element that is not active")

        B_1y = eval_f_indicator(pp.B, y_bits, self.params)   # n x m'
        ginv_u = g_inv_vector(pp.u, self.params)             # m' vector
        delta = matrix_vector_mul(B_1y, ginv_u)              # n vector

        st.c = vector_add(st.c, delta)
        st.active.remove(key)
        return UpdateMessage(op="Del", y_bits=list(y_bits))

    # memwitup

    def memwit_update(
        self,
        pp: PublicParams,
        x_bits: List[int],
        witness: Witness,
        upmsg: UpdateMessage,
    ) -> Witness:
        """
        Add  -> witness unchanged.
        Del  -> s_x' = s_x + [0_m ; H_{1_y,x} * G^{-1}(u)]   (requires x != y).
        """
        self._element_key(x_bits)
        if upmsg.op == "Add":
            return Witness(s=list(witness.s))

        if tuple(x_bits) == tuple(upmsg.y_bits):
            raise ValueError("Deleted element's witness should not be updated")

        H = eval_fx_indicator(pp.B, upmsg.y_bits, x_bits, self.params)  # (ell*m') x m'
        ginv_u = g_inv_vector(pp.u, self.params)                       # m' vector
        H_part = matrix_vector_mul(H, ginv_u)                          # ell*m' vector
        zero_part = zero_vector(self.params.m, self.params.q, self.params.d)
        delta = concat_vectors(zero_part, H_part)
        return Witness(s=vector_add(witness.s, delta))

    # memver

    def memver(
        self,
        pp: PublicParams,
        c: Vector,
        x_bits: List[int],
        witness: Witness,
        beta: int | None = None,
    ) -> bool:
        """
        Check [A | B - x^T (x) G] * s_x == c, and (optionally) ||s_x||_inf <= beta.
        """
        Mx = self._membership_matrix(pp, x_bits)
        if matrix_vector_mul(Mx, witness.s) != c:
            return False
        if beta is not None and witness.norm_inf() > beta:
            return False
        return True
