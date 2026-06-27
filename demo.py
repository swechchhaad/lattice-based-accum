"""
End-to-end demo of the lattice PRFA (Construction 4.1) and the adaptive
compiler (Construction 4.2).

Run:
    python demo.py
"""

from __future__ import annotations

import secrets

from latticeaccum import LatticeParams, LatticePRFA, AdaptiveAccumulator
from latticeaccum.trapdoor import trapgen, verify_trapdoor
from latticeaccum.homomorphic import check_eval_relation


# toy parameters; these are NOT secure
TOY = LatticeParams(q=12289, d=16, n=2, m=36, ell=8, base=2)


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def rand_handle(ell: int) -> list[int]:
    return [secrets.randbelow(2) for _ in range(ell)]


def self_checks(p: LatticeParams) -> None:
    banner("0. Subroutine self-checks")
    A, TA = trapgen(p)
    print(f"  params: q={p.q} d={p.d} n={p.n} m={p.m} m'={p.m_prime} ell={p.ell}")
    print(f"  witness dimension m + ell*m' = {p.witness_dim} ring elements")
    assert verify_trapdoor(A, TA, p), "gadget trapdoor A*[R;I]=G failed"
    print("  [ok] gadget trapdoor:  A * [R ; I] = G")

    y = rand_handle(p.ell)
    x = rand_handle(p.ell)
    while x == y:
        x = rand_handle(p.ell)
    # need a B to test the homomorphic relation
    from latticeaccum.linalg import random_matrix
    B = random_matrix(p.n, p.ell * p.m_prime, p.q, p.d)
    assert check_eval_relation(B, y, x, p), "EvalF/EvalFx relation failed (x != y)"
    assert check_eval_relation(B, y, y, p), "EvalF/EvalFx relation failed (x == y)"
    print("  [ok] homomorphic relation:  (B - x^T(x)G) H_{1_y,x} = B_{1_y} - 1_y(x) G")


def prfa_demo(p: LatticeParams) -> None:
    banner("1. PRFA: add / delete / witness-update (Construction 4.1)")
    acc = LatticePRFA(p)
    pp, sk, st = acc.gen()

    x = rand_handle(p.ell)   # Alice's revocation handle
    y = rand_handle(p.ell)   # Bob's revocation handle
    while y == x:
        y = rand_handle(p.ell)

    wx, _ = acc.add(pp, sk, st, x)
    wy, _ = acc.add(pp, sk, st, y)
    print(f"  added Alice (x) and Bob (y); accumulator value c unchanged (free addition)")
    print(f"  |w_x|_inf = {wx.norm_inf()},  |w_y|_inf = {wy.norm_inf()}")

    assert acc.memver(pp, st.c, x, wx), "Alice should verify"
    assert acc.memver(pp, st.c, y, wy), "Bob should verify"
    print("  [ok] both witnesses verify against c")

    upmsg = acc.delete(pp, st, y)        # revoke Bob
    print(f"  revoked Bob -> new accumulator value c' published (upmsg op={upmsg.op})")

    wx2 = acc.memwit_update(pp, x, wx, upmsg)
    assert acc.memver(pp, st.c, x, wx2), "Alice should still verify after update"
    print(f"  [ok] Alice updates her witness and still verifies; |w_x'|_inf = {wx2.norm_inf()}")

    assert not acc.memver(pp, st.c, y, wy), "Bob must no longer verify"
    print("  [ok] Bob's old witness no longer verifies against c'")

    # replacement-free: cannot re-add Bob
    try:
        acc.add(pp, sk, st, y)
        raise SystemExit("  [FAIL] re-adding Bob should have been rejected")
    except ValueError:
        print("  [ok] replacement-free: re-adding a deleted handle is rejected")

    # witness norm grows with deletions
    banner("2. Witness norm growth across many deletions")
    acc2 = LatticePRFA(p)
    pp2, sk2, st2 = acc2.gen()
    alice = rand_handle(p.ell)
    wa, _ = acc2.add(pp2, sk2, st2, alice)
    print(f"  start: |w_alice|_inf = {wa.norm_inf()}")
    deletions = 6
    for k in range(deletions):
        victim = rand_handle(p.ell)
        while tuple(victim) in st2.ever_added or victim == alice:
            victim = rand_handle(p.ell)
        acc2.add(pp2, sk2, st2, victim)
        upd = acc2.delete(pp2, st2, victim)
        wa = acc2.memwit_update(pp2, alice, wa, upd)
        ok = acc2.memver(pp2, st2.c, alice, wa)
        print(f"  after {k+1} deletion(s): |w_alice|_inf = {wa.norm_inf():>4}   verifies={ok}")
        assert ok


def adaptive_demo(p: LatticeParams) -> None:
    banner("3. Adaptive compiler: credential identities (Construction 4.2)")
    acc = AdaptiveAccumulator(p)
    pp, sk, st, sig_key = acc.gen()
    handles: dict = {}

    alice = [1, 0, 1, 1]      # arbitrary identity bits (any length)
    bob = [0, 0, 1, 0]

    wa = acc.add(pp, sk, st, sig_key, alice, handles)
    wb = acc.add(pp, sk, st, sig_key, bob, handles)
    print(f"  issued credentials to Alice and Bob (fresh handles bound by signature)")
    assert acc.memver(pp, st.c, sig_key, alice, wa)
    assert acc.memver(pp, st.c, sig_key, bob, wb)
    print("  [ok] both credentials verify (signature + accumulator membership)")

    upmsg = acc.delete(pp, st, wb)       # revoke Bob's credential
    wa = acc.memwit_update(pp, wa, upmsg)
    assert acc.memver(pp, st.c, sig_key, alice, wa)
    assert not acc.memver(pp, st.c, sig_key, bob, wb)
    print("  [ok] after revoking Bob: Alice still valid, Bob rejected")

    # reusing Alice's witness under a different identity fails the signature check
    assert not acc.memver(pp, st.c, sig_key, [1, 1, 1, 1], wa)
    print("  [ok] reusing Alice's witness for a different identity fails (signature check)")


def main() -> None:
    self_checks(TOY)
    prfa_demo(TOY)
    adaptive_demo(TOY)
    banner("All demo checks passed.")


if __name__ == "__main__":
    main()
