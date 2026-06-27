"""
Minimal correctness tests for the lattice PRFA. Run with `python test_accumulator.py`
(or `pytest test_accumulator.py`).
"""

from __future__ import annotations

import secrets

from latticeaccum import LatticeParams, LatticePRFA, AdaptiveAccumulator
from latticeaccum.trapdoor import trapgen, verify_trapdoor
from latticeaccum.gadget import build_gadget_matrix, g_inv_vector
from latticeaccum.homomorphic import check_eval_relation
from latticeaccum.linalg import matrix_vector_mul, random_matrix
from latticeaccum.ring import RingElement

P = LatticeParams(q=12289, d=16, n=2, m=36, ell=8, base=2)


def _handle():
    return [secrets.randbelow(2) for _ in range(P.ell)]


def test_gadget_decomposition_exact():
    G = build_gadget_matrix(P)
    u = [RingElement.random(P.q, P.d) for _ in range(P.n)]
    v = g_inv_vector(u, P)
    assert matrix_vector_mul(G, v) == u


def test_trapdoor_is_valid():
    A, TA = trapgen(P)
    assert verify_trapdoor(A, TA, P)


def test_homomorphic_relation():
    B = random_matrix(P.n, P.ell * P.m_prime, P.q, P.d)
    y, x = _handle(), _handle()
    while x == y:
        x = _handle()
    assert check_eval_relation(B, y, x, P)   # x != y -> indicator 0
    assert check_eval_relation(B, y, y, P)   # x == y -> indicator 1


def test_add_and_verify():
    acc = LatticePRFA(P)
    pp, sk, st = acc.gen()
    x = _handle()
    w, _ = acc.add(pp, sk, st, x)
    assert acc.memver(pp, st.c, x, w)


def test_delete_and_update():
    acc = LatticePRFA(P)
    pp, sk, st = acc.gen()
    x, y = _handle(), _handle()
    while y == x:
        y = _handle()
    wx, _ = acc.add(pp, sk, st, x)
    wy, _ = acc.add(pp, sk, st, y)
    upd = acc.delete(pp, st, y)
    wx2 = acc.memwit_update(pp, x, wx, upd)
    assert acc.memver(pp, st.c, x, wx2)      # Alice still valid
    assert not acc.memver(pp, st.c, y, wy)   # Bob revoked


def test_replacement_free():
    acc = LatticePRFA(P)
    pp, sk, st = acc.gen()
    y = _handle()
    acc.add(pp, sk, st, y)
    acc.delete(pp, st, y)
    try:
        acc.add(pp, sk, st, y)
        assert False, "re-add should be rejected"
    except ValueError:
        pass


def test_adaptive_compiler():
    acc = AdaptiveAccumulator(P)
    pp, sk, st, key = acc.gen()
    handles = {}
    alice, bob = [1, 0, 1], [0, 1, 1]
    wa = acc.add(pp, sk, st, key, alice, handles)
    wb = acc.add(pp, sk, st, key, bob, handles)
    assert acc.memver(pp, st.c, key, alice, wa)
    upd = acc.delete(pp, st, wb)
    wa = acc.memwit_update(pp, wa, upd)
    assert acc.memver(pp, st.c, key, alice, wa)
    assert not acc.memver(pp, st.c, key, bob, wb)
    assert not acc.memver(pp, st.c, key, [1, 1, 1], wa)  # wrong identity


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [ok] {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
