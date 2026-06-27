from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import hashlib
import secrets

from .params import LatticeParams
from .prfa import LatticePRFA, PublicParams, SecretKey, AccumulatorState, Witness, UpdateMessage


def _mac(key: bytes, x_bits: List[int], r_bits: List[int]) -> bytes:
    msg = bytes(x_bits) + b"|" + bytes(r_bits)
    return hashlib.sha256(key + b"||" + msg).digest()


@dataclass
class AdaptiveWitness:
    r_bits: List[int]
    sigma: bytes
    w: Witness


class AdaptiveAccumulator:
    """Adaptively-secure positive dynamic accumulator"""

    def __init__(self, params: LatticeParams):
        self.params = params
        self.prfa = LatticePRFA(params)

    def gen(self) -> Tuple[PublicParams, SecretKey, AccumulatorState, bytes]:
        pp, sk, st = self.prfa.gen()
        sig_key = secrets.token_bytes(32)
        return pp, sk, st, sig_key

    def _fresh_handle(self, st: AccumulatorState) -> List[int]:
        while True:
            r = [secrets.randbelow(2) for _ in range(self.params.ell)]
            if tuple(r) not in st.ever_added:
                return r

    def add(
        self,
        pp: PublicParams,
        sk: SecretKey,
        st: AccumulatorState,
        sig_key: bytes,
        x_bits: List[int],
        handles: Dict[Tuple[int, ...], List[int]],
    ) -> AdaptiveWitness:
        r_bits = self._fresh_handle(st)
        w, _ = self.prfa.add(pp, sk, st, r_bits)
        sigma = _mac(sig_key, x_bits, r_bits)
        handles[tuple(x_bits)] = r_bits
        return AdaptiveWitness(r_bits=r_bits, sigma=sigma, w=w)

    def delete(
        self,
        pp: PublicParams,
        st: AccumulatorState,
        wit: AdaptiveWitness,
    ) -> UpdateMessage:
        return self.prfa.delete(pp, st, wit.r_bits)

    def memwit_update(
        self,
        pp: PublicParams,
        wit: AdaptiveWitness,
        upmsg: UpdateMessage,
    ) -> AdaptiveWitness:
        new_w = self.prfa.memwit_update(pp, wit.r_bits, wit.w, upmsg)
        return AdaptiveWitness(r_bits=wit.r_bits, sigma=wit.sigma, w=new_w)

    def memver(
        self,
        pp: PublicParams,
        c,
        sig_key: bytes,
        x_bits: List[int],
        wit: AdaptiveWitness,
        beta: int | None = None,
    ) -> bool:
        if _mac(sig_key, x_bits, wit.r_bits) != wit.sigma:
            return False
        return self.prfa.memver(pp, c, wit.r_bits, wit.w, beta=beta)
