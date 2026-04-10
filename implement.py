from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable, Set
import math
import secrets

class RingElement:
    """
    Element of R_q = Z_q[X] / (X^d + 1).

    Representation:
        coeffs[i] is the coefficient of X^i, for 0 <= i < d.
    """

    def __init__(self, coeffs: List[int], q: int, d: int):
        self.q = q
        self.d = d
        coeffs = coeffs[:]
        if len(coeffs) < d:
            coeffs += [0] * (d - len(coeffs))
        elif len(coeffs) > d:
            coeffs = self._reduce(coeffs, q, d)
        self.coeffs = [c % q for c in coeffs[:d]]

    @staticmethod
    def zero(q: int, d: int) -> "RingElement":
        return RingElement([0] * d, q, d)

    @staticmethod
    def one(q: int, d: int) -> "RingElement":
        coeffs = [0] * d
        coeffs[0] = 1
        return RingElement(coeffs, q, d)

    @staticmethod
    def random(q: int, d: int) -> "RingElement":
        return RingElement([secrets.randbelow(q) for _ in range(d)], q, d)

    @staticmethod
    def from_int(v: int, q: int, d: int) -> "RingElement":
        coeffs = [0] * d
        coeffs[0] = v % q
        return RingElement(coeffs, q, d)

    @staticmethod
    def _reduce(poly: List[int], q: int, d: int) -> List[int]:
        """
        Reduce modulo X^d + 1.
        """
        poly = poly[:]
        while len(poly) > d:
            c = poly.pop()
            idx = len(poly) - d
            poly[idx] -= c
        return [x % q for x in poly] + [0] * max(0, d - len(poly))

    def __add__(self, other: "RingElement") -> "RingElement":
        self._check_same_ring(other)
        return RingElement(
            [(a + b) % self.q for a, b in zip(self.coeffs, other.coeffs)],
            self.q,
            self.d,
        )

    def __sub__(self, other: "RingElement") -> "RingElement":
        self._check_same_ring(other)
        return RingElement(
            [(a - b) % self.q for a, b in zip(self.coeffs, other.coeffs)],
            self.q,
            self.d,
        )

    def __neg__(self) -> "RingElement":
        return RingElement([(-a) % self.q for a in self.coeffs], self.q, self.d)

    def __mul__(self, other: "RingElement") -> "RingElement":
        self._check_same_ring(other)
        raw = [0] * (2 * self.d - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                raw[i + j] += a * b
        return RingElement(self._reduce(raw, self.q, self.d), self.q, self.d)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RingElement):
            return False
        return (
            self.q == other.q and
            self.d == other.d and
            self.coeffs == other.coeffs
        )

    def lift_centered(self) -> List[int]:
        """
        Return coefficients in centered representation in (-q/2, q/2].
        Useful for toy norm checks / debugging.
        """
        out = []
        half = self.q // 2
        for c in self.coeffs:
            if c > half:
                out.append(c - self.q)
            else:
                out.append(c)
        return out

    def norm_inf(self) -> int:
        return max(abs(x) for x in self.lift_centered())

    def copy(self) -> "RingElement":
        return RingElement(self.coeffs[:], self.q, self.d)

    def _check_same_ring(self, other: "RingElement") -> None:
        if self.q != other.q or self.d != other.d:
            raise ValueError("Ring mismatch")

    def __repr__(self) -> str:
        return f"R({self.coeffs})"


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


def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]


def matrix_sub(A: Matrix, B: Matrix) -> Matrix:
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] - B[i][j] for j in range(cols)] for i in range(rows)]


def matrix_vector_mul(A: Matrix, x: Vector) -> Vector:
    rows = len(A)
    cols = len(A[0])
    if len(x) != cols:
        raise ValueError("Dimension mismatch in matrix_vector_mul")
    out = []
    for i in range(rows):
        acc = RingElement.zero(x[0].q, x[0].d)
        for j in range(cols):
            acc = acc + (A[i][j] * x[j])
        out.append(acc)
    return out


def matrix_mul(A: Matrix, B: Matrix) -> Matrix:
    rows = len(A)
    mid = len(A[0])
    if len(B) != mid:
        raise ValueError("Dimension mismatch in matrix_mul")
    cols = len(B[0])
    q = A[0][0].q
    d = A[0][0].d
    out = zero_matrix(rows, cols, q, d)
    for i in range(rows):
        for j in range(cols):
            acc = RingElement.zero(q, d)
            for k in range(mid):
                acc = acc + (A[i][k] * B[k][j])
            out[i][j] = acc
    return out


def hcat(*blocks: Matrix) -> Matrix:
    rows = len(blocks[0])
    out: Matrix = []
    for i in range(rows):
        row: List[RingElement] = []
        for blk in blocks:
            row.extend(blk[i])
        out.append(row)
    return out


def transpose(A: Matrix) -> Matrix:
    return [list(row) for row in zip(*A)]


def kron_bit_gadget_row(x_bits: List[int], G: Matrix) -> Matrix:
    """
    Compute x^T ⊗ G where x is a bit vector of length ell and
    G is n x m', resulting in n x (ell*m').
    """
    ell = len(x_bits)
    n = len(G)
    mp = len(G[0])
    q = G[0][0].q
    d = G[0][0].d
    out = zero_matrix(n, ell * mp, q, d)
    for k, bit in enumerate(x_bits):
        for i in range(n):
            for j in range(mp):
                out[i][k * mp + j] = G[i][j] if bit == 1 else RingElement.zero(q, d)
    return out


def concat_vectors(*parts: Vector) -> Vector:
    out: Vector = []
    for p in parts:
        out.extend(p)
    return out


def vector_add(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Dimension mismatch in vector_add")
    return [x + y for x, y in zip(a, b)]

@dataclass
class LatticeParams:
    """
    Mirrors the paper's parameter style.

    q: modulus
    d: cyclotomic degree, ring is Z_q[X]/(X^d + 1)
    n: row dimension
    m: A-width
    ell: bit-length of accumulated element x
    base: gadget base b
    """
    q: int
    d: int
    n: int
    m: int
    ell: int
    base: int = 2

    @property
    def m_prime(self) -> int:
        return self.n * math.ceil(math.log(self.q, self.base))

    @property
    def witness_dim(self) -> int:
        return self.m + self.ell * self.m_prime


@dataclass
class Trapdoor:
    """
    Placeholder for the MP12 / GPV trapdoor for A.
    In a real implementation this would carry the structured short basis/trapdoor data.
    """
    raw: object


@dataclass
class PublicParams:
    """
    Compressed vector-form public parameters from the paper:
        pp = (A, B, u)

    A : n x m
    B : n x (ell * m')
    u : n-vector
    """
    A: Matrix
    B: Matrix
    u: Vector
    params: LatticeParams


@dataclass
class SecretKey:
    """
    Secret trapdoor for A.
    """
    TA: Trapdoor


@dataclass
class Witness:
    """
    Membership witness s_x in R_q^{m + ell*m'}.
    """
    s: Vector


@dataclass
class UpdateMessage:
    """
    Deletion update message:
        upmsg = (c_new, y)
    """
    c_new: Vector
    y_bits: List[int]


@dataclass
class AccumulatorState:
    """
    Maintains the current accumulator value and the set of active / ever-added elements.
    """
    c: Vector
    active: Set[Tuple[int, ...]] = field(default_factory=set)
    ever_added: Set[Tuple[int, ...]] = field(default_factory=set)

def trapgen(params: LatticeParams) -> Tuple[Matrix, Trapdoor]:
    """
    Placeholder for MP12 TrapGen.

    Paper-faithful intent:
      output A in R_q^{n x m} together with a trapdoor TA that enables
      short preimage sampling for targets under A.

    In a real implementation this must be replaced with a proper lattice trapdoor generator.
    """
    A = random_matrix(params.n, params.m, params.q, params.d)
    TA = Trapdoor(raw={"note": "TODO: implement MP12/GPV trapdoor data"})
    return A, TA


def sample_pre(
    Abar: Matrix,
    TA: Trapdoor,
    target: Vector,
    params: LatticeParams,
) -> Vector:
    """
    Placeholder for SamplePre.

    Paper-faithful intent:
      given a matrix Abar = [A | B_x] and target c,
      sample a *short* preimage s such that Abar * s = c.

    In a real implementation this must be trapdoor-based discrete Gaussian sampling.
    """
    raise NotImplementedError(
        "SamplePre must be implemented with a real lattice trapdoor sampler."
    )


def build_gadget_matrix(params: LatticeParams) -> Matrix:
    """
    G = I_n ⊗ g^T, where g^T = (1, b, b^2, ..., b^{qtilde-1}).
    """
    qtilde = math.ceil(math.log(params.q, params.base))
    q = params.q
    d = params.d
    n = params.n

    g_row = [RingElement.from_int(params.base ** i, q, d) for i in range(qtilde)]

    G = zero_matrix(n, n * qtilde, q, d)
    for i in range(n):
        for j, ge in enumerate(g_row):
            G[i][i * qtilde + j] = ge
    return G


def G_inv(u: Vector, params: LatticeParams) -> Vector:
    """
    Placeholder for gadget decomposition.

    Paper-faithful intent:
      return v in R_q^{m'} such that G * v = u.

    If b = 2 and R_q = Z_q, this is bit decomposition; over R_q it is coefficient-wise gadget decomposition.
    """
    raise NotImplementedError("G_inv requires gadget decomposition over R_q.")


def compute_indicator_Af(
    B: Matrix,
    y_bits: List[int],
    params: LatticeParams,
) -> Matrix:
    """
    Return B_{1_y} from the paper's homomorphic machinery, i.e. A_f for f = 1_y.

    Paper-faithful meaning:
      for indicator function 1_y, produce A_f so that
          (B - x^T ⊗ G) H_{1_y, x} = B_{1_y} - 1_y(x) * G.

    This is not a generic matrix slice. It comes from GSW/BGG-style homomorphic evaluation.
    """
    raise NotImplementedError("Need real homomorphic evaluation for indicator function 1_y.")


def compute_H_fx(
    B: Matrix,
    y_bits: List[int],
    x_bits: List[int],
    params: LatticeParams,
) -> Matrix:
    """
    Return H_{1_y, x} from the paper.

    Dimensions:
      H_{f,x} should be (ell*m') x m' in the compressed/vector view.

    It must satisfy:
      (B - x^T ⊗ G) * H_{1_y, x} = B_{1_y} - 1_y(x) * G
    and for x != y one expects ||H_{1_y, x}||_∞ = 1 in the base-2 gadget setting.
    """
    raise NotImplementedError("Need real computation of H_{1_y, x} from the homomorphic construction.")

class LatticePRFA:
    """
    Positive replacement-free accumulator, compressed witness / compressed accumulator form.

    This follows the paper's compressed vector presentation:
      pp = (A, B, u)
      witness s_x satisfies [A | B - x^T ⊗ G] * s_x = c

    The main algorithms are:
      Gen
      Add
      Delete
      MemWitUp
      MemVer
    """

    def __init__(self, params: LatticeParams):
        self.params = params
        self.G = build_gadget_matrix(params)

    def gen(self) -> Tuple[PublicParams, SecretKey, AccumulatorState]:
        """
        Gen:
          - generate A with trapdoor
          - sample B uniformly
          - sample u uniformly
          - initialize accumulator value c = A0 (paper uses C / c as initial accumulator value)

        The paper's overview presents pp = (A, B, u), sk = TA, and initial c. 
        """
        A, TA = trapgen(self.params)
        B = random_matrix(
            self.params.n,
            self.params.ell * self.params.m_prime,
            self.params.q,
            self.params.d,
        )
        u = zero_vector(self.params.n, self.params.q, self.params.d)
        for i in range(self.params.n):
            u[i] = RingElement.random(self.params.q, self.params.d)

        # c should be chosen exactly as in the paper's compressed public paradigm.
        c = [RingElement.random(self.params.q, self.params.d) for _ in range(self.params.n)]

        pp = PublicParams(A=A, B=B, u=u, params=self.params)
        sk = SecretKey(TA=TA)
        st = AccumulatorState(c=c)

        return pp, sk, st

    def _element_key(self, x_bits: List[int]) -> Tuple[int, ...]:
        if len(x_bits) != self.params.ell or any(b not in (0, 1) for b in x_bits):
            raise ValueError("Element must be a bit vector of length ell")
        return tuple(x_bits)

    def _membership_matrix(self, pp: PublicParams, x_bits: List[int]) -> Matrix:
        """
        Return [A | B - x^T ⊗ G].
        """
        x_kron_G = kron_bit_gadget_row(x_bits, self.G)
        right = matrix_sub(pp.B, x_kron_G)
        return hcat(pp.A, right)

    def add(
        self,
        pp: PublicParams,
        sk: SecretKey,
        st: AccumulatorState,
        x_bits: List[int],
    ) -> Witness:
        """
        Add(pp, sk, st, c, x) -> witness

        Paper-faithful meaning:
          using trapdoor TA, sample a short witness s_x satisfying
              [A | B - x^T ⊗ G] s_x = c

        Free addition:
          c does NOT change when x is added.
        """
        key = self._element_key(x_bits)
        if key in st.ever_added:
            raise ValueError("PRFA forbids re-adding an element that was ever added before.")

        Mx = self._membership_matrix(pp, x_bits)

        sx = sample_pre(
            Abar=Mx,
            TA=sk.TA,
            target=st.c,
            params=self.params,
        )

        if len(sx) != self.params.witness_dim:
            raise ValueError("sample_pre returned wrong witness dimension")

        st.active.add(key)
        st.ever_added.add(key)
        return Witness(s=sx)

    def delete(
        self,
        pp: PublicParams,
        st: AccumulatorState,
        y_bits: List[int],
    ) -> UpdateMessage:
        """
        Delete(pp, st, c, y) -> upmsg

        Paper-faithful compressed-vector update:
            c' = c + B_{1_y} * G^{-1}(u)

        The paper publishes the new accumulator value plus y.
        """
        key = self._element_key(y_bits)
        if key not in st.active:
            raise ValueError("Cannot delete element that is not active")

        B_1y = compute_indicator_Af(pp.B, y_bits, self.params)
        ginv_u = G_inv(pp.u, self.params)  # vector in R_q^{m'}

        delta = matrix_vector_mul(B_1y, ginv_u)
        c_new = vector_add(st.c, delta)

        st.c = c_new
        st.active.remove(key)

        return UpdateMessage(c_new=c_new, y_bits=y_bits[:])

    def memwit_update(
        self,
        pp: PublicParams,
        x_bits: List[int],
        witness: Witness,
        upmsg: UpdateMessage,
    ) -> Witness:
        """
        MemWitUp(pp, x, s_x, upmsg) -> s_x'

        If x != y:
            s_x' = s_x + [0 ; H_{1_y, x}^T] * G^{-1}(u)

        If x == y, the witness should not remain valid after deletion.
        """
        self._element_key(x_bits)
        if tuple(x_bits) == tuple(upmsg.y_bits):
            raise ValueError("Deleted element's witness should not be updated")

        H = compute_H_fx(pp.B, upmsg.y_bits, x_bits, self.params)   # (ell*m') x m'
        ginv_u = G_inv(pp.u, self.params)                           # m' vector

        H_part = matrix_vector_mul(H, ginv_u)                       # (ell*m') vector
        zero_part = zero_vector(self.params.m, self.params.q, self.params.d)
        delta = concat_vectors(zero_part, H_part)

        return Witness(s=vector_add(witness.s, delta))

    def memver(
        self,
        pp: PublicParams,
        c: Vector,
        x_bits: List[int],
        witness: Witness,
    ) -> bool:
        """
        MemVer(pp, c, x, s_x):
            check [A | B - x^T ⊗ G] * s_x == c
        """
        Mx = self._membership_matrix(pp, x_bits)
        lhs = matrix_vector_mul(Mx, witness.s)
        return lhs == c

class AdaptiveWrapper:
    """
    Sketch only.

    The paper says the selectively secure PRFA can be compiled to an adaptively secure positive dynamic
    accumulator by combining it with an adaptively secure digital signature / Baldimtsi et al. style compiler.

    This wrapper just shows the intended place where the compiler would sit.
    """
    def __init__(self, base_prfa: LatticePRFA):
        self.base = base_prfa

    def gen(self):
        """
        Would generate:
          - base PRFA parameters
          - digital signature keys for authenticated unique tags / compiler metadata
        """
        raise NotImplementedError("Compiler layer not implemented in this skeleton.")

    def add(self, *args, **kwargs):
        raise NotImplementedError

    def delete(self, *args, **kwargs):
        raise NotImplementedError

    def memver(self, *args, **kwargs):
        raise NotImplementedError

def demo_shape() -> None:
    """
    This is intentionally not runnable until the hard subroutines are filled in.
    """
    params = LatticeParams(
        q=12289,   # placeholder; use the paper's actual chosen parameter set
        d=256,     # placeholder
        n=4,       # placeholder
        m=16,      # placeholder; must satisfy m > m'
        ell=32,    # revocation handle length from the paper's example
        base=2,
    )

    prfa = LatticePRFA(params)
    pp, sk, st = prfa.gen()

    x = [secrets.randbelow(2) for _ in range(params.ell)]
    y = [secrets.randbelow(2) for _ in range(params.ell)]
    while y == x:
        y = [secrets.randbelow(2) for _ in range(params.ell)]

    wx = prfa.add(pp, sk, st, x)
    wy = prfa.add(pp, sk, st, y)

    assert prfa.memver(pp, st.c, x, wx)
    assert prfa.memver(pp, st.c, y, wy)

    upmsg = prfa.delete(pp, st, y)

    wx_updated = prfa.memwit_update(pp, x, wx, upmsg)
    assert prfa.memver(pp, st.c, x, wx_updated)

    assert not prfa.memver(pp, st.c, y, wy)