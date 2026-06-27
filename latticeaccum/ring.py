from __future__ import annotations

from typing import List
import secrets


class RingElement:
    """
    Element of R_q = Z_q[X] / (X^d + 1).

    Representation:
        coeffs[i] is the coefficient of X^i, for 0 <= i < d.
    Coefficients are stored in [0, q).
    """

    __slots__ = ("q", "d", "coeffs")

    def __init__(self, coeffs: List[int], q: int, d: int):
        self.q = q
        self.d = d
        coeffs = list(coeffs)
        if len(coeffs) < d:
            coeffs += [0] * (d - len(coeffs))
        elif len(coeffs) > d:
            coeffs = self._reduce(coeffs, q, d)
        self.coeffs = [c % q for c in coeffs[:d]]

    # constructors

    @staticmethod
    def zero(q: int, d: int) -> "RingElement":
        return RingElement([0] * d, q, d)

    @staticmethod
    def one(q: int, d: int) -> "RingElement":
        coeffs = [0] * d
        coeffs[0] = 1 % q
        return RingElement(coeffs, q, d)

    @staticmethod
    def random(q: int, d: int) -> "RingElement":
        return RingElement([secrets.randbelow(q) for _ in range(d)], q, d)

    @staticmethod
    def random_small(q: int, d: int, bound: int = 1) -> "RingElement":
        """
        Sample coefficients uniformly from {-bound, ..., bound} (centered),
        stored reduced mod q. Used for short/binary secrets.
        """
        span = 2 * bound + 1
        return RingElement([secrets.randbelow(span) - bound for _ in range(d)], q, d)

    @staticmethod
    def from_int(v: int, q: int, d: int) -> "RingElement":
        coeffs = [0] * d
        coeffs[0] = v % q
        return RingElement(coeffs, q, d)

    # reduction mod X^d + 1

    @staticmethod
    def _reduce(poly: List[int], q: int, d: int) -> List[int]:
        """
        Reduce a raw polynomial modulo X^d + 1, i.e. X^d == -1.
        """
        poly = list(poly)
        while len(poly) > d:
            c = poly.pop()
            idx = len(poly) - d
            poly[idx] -= c
        return [x % q for x in poly] + [0] * max(0, d - len(poly))

    # arithmetic

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
            if a == 0:
                continue
            for j, b in enumerate(other.coeffs):
                if b:
                    raw[i + j] += a * b
        return RingElement(self._reduce(raw, self.q, self.d), self.q, self.d)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RingElement):
            return False
        return self.q == other.q and self.d == other.d and self.coeffs == other.coeffs

    def __hash__(self) -> int:
        return hash((self.q, self.d, tuple(self.coeffs)))

    # norms / debugging

    def lift_centered(self) -> List[int]:
        """Coefficients in centered representation in (-q/2, q/2]."""
        out = []
        half = self.q // 2
        for c in self.coeffs:
            out.append(c - self.q if c > half else c)
        return out

    def norm_inf(self) -> int:
        return max((abs(x) for x in self.lift_centered()), default=0)

    def is_zero(self) -> bool:
        return all(c == 0 for c in self.coeffs)

    def copy(self) -> "RingElement":
        return RingElement(self.coeffs[:], self.q, self.d)

    def _check_same_ring(self, other: "RingElement") -> None:
        if self.q != other.q or self.d != other.d:
            raise ValueError("Ring mismatch")

    def __repr__(self) -> str:
        return f"R({self.lift_centered()})"
