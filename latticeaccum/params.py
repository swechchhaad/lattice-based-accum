from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class LatticeParams:
    """
    q    : modulus
    d    : cyclotomic degree, ring is Z_q[X]/(X^d + 1)  (power of two)
    n    : row dimension (height of A, B, and the gadget G)
    m    : width of A. Must satisfy m > m' so that A admits a gadget trapdoor.
    ell  : bit-length of an accumulated element x in {0,1}^ell
    base : gadget base b (b = 2 gives binary decomposition)

    Derived:
    m_prime  m' = n * ceil(log_b q)   (width of one gadget block)
    m_bar    m - m'                   (width of the random part of A)
    witness_dim  m + ell*m'           (length of a membership witness s_x)
    """

    q: int
    d: int
    n: int
    m: int
    ell: int
    base: int = 2

    def __post_init__(self) -> None:
        if self.d & (self.d - 1) != 0:
            raise ValueError("d must be a power of two")
        if self.m <= self.m_prime:
            raise ValueError(
                f"need m > m' for a gadget trapdoor, got m={self.m}, m'={self.m_prime}"
            )

    @property
    def qtilde(self) -> int:
        return math.ceil(math.log(self.q, self.base))

    @property
    def m_prime(self) -> int:
        return self.n * self.qtilde

    @property
    def m_bar(self) -> int:
        return self.m - self.m_prime

    @property
    def witness_dim(self) -> int:
        return self.m + self.ell * self.m_prime
    