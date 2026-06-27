# Lattice-Based Accumulator (toy implementation)

A small, runnable Python implementation of the M-SIS based *positive
replacement-free accumulator* (PRFA) and its adaptive compiler from [this paper](https://eprint.iacr.org/2025/1099).

We implement **Construction 4.1** and
**Construction 4.2** from the paper, including 
a gadget (MP12) trapdoor, gadget decomposition, and the homomorphic
evaluation of the indicator function.

> ⚠️ **This is a toy implementation only, not meant for production use.** Parameters are tiny, and
> preimage sampling uses a small uniform mask + deterministic gadget preimage
> instead of discrete-Gaussian sampling. The linear relations and
> norm behaviour of the paper are exact; the distributions needed for the
> security proof are not reproduced.

## Quick start

```bash
python demo.py             # full end-to-end demo
python implement.py        # same demo (thin entry point / API re-export)
python test_accumulator.py # correctness tests
```

No dependencies beyond the Python standard library (Python 3.10+).
<!-- 
## What the demo shows

1. **Self-checks** — the gadget trapdoor `A·[R;I] = G` and the homomorphic
   identity `(B − xᵀ⊗G)·H_{1_y,x} = B_{1_y} − 1_y(x)·G`.
2. **PRFA** — add two handles (accumulator value unchanged → *free addition*),
   both verify; delete one; the survivor updates its witness and still verifies;
   the revoked witness no longer verifies; re-adding a deleted handle is rejected
   (*replacement-free*).
3. **Witness-norm growth** — witnesses stay short and grow slowly across
   deletions (the paper's `‖w′‖ ≤ ‖w‖ + …` behaviour).
4. **Adaptive compiler** — credentials for arbitrary identities, each bound to a
   fresh random revocation handle by a signature; revoke one, the other survives,
   and a witness can't be reused under a different identity. -->

<!-- ## How the code maps to the paper

| Module | Paper | Contents |
|---|---|---|
| `latticeaccum/ring.py` | §2 Notations | `R_q = Z_q[X]/(X^d+1)` arithmetic |
| `latticeaccum/linalg.py` | — | vectors/matrices over `R_q`, the `xᵀ⊗G` block |
| `latticeaccum/params.py` | Construction 4.1 | `LatticeParams` (`q,d,n,m,ℓ,b`; derives `m' = n⌈log_b q⌉`) |
| `latticeaccum/gadget.py` | §2 Gadget Matrix [MP12] | `G = I_n⊗gᵀ` and `G^{-1}` (base-`b` decomposition) |
| `latticeaccum/trapdoor.py` | Thm 2.1 / Lemma 2.6 | gadget trapdoor `A=[Ā ∣ G−ĀR]`, `SampleLeft` |
| `latticeaccum/homomorphic.py` | Thm 2.2 / Cor 2.2.1 | `EvalF`/`EvalFx` for the indicator `1_y` |
| `latticeaccum/prfa.py` | **Construction 4.1** | `Gen, Add, Delete, MemWitUp, MemVer` |
| `latticeaccum/adaptive.py` | **Construction 4.2** | adaptive compiler (signature + fresh handles) |
| `demo.py`, `test_accumulator.py` | §5.2 application | end-to-end usage |

### The core relation

A membership witness `s_x` for an element `x ∈ {0,1}^ℓ` satisfies

```
[ A | B − xᵀ⊗G ] · s_x = c        (the accumulator value)
```

- **Add(x):** use the trapdoor for `A` to sample a short `s_x` with the above
  equation. `c` does not change → free addition.
- **Delete(y):** `c' = c + B_{1_y}·G^{-1}(u)`, where `B_{1_y} = EvalF(1_y, B)`.
- **MemWitUp (x ≠ y):** `s_x' = s_x + [0 ; H_{1_y,x}·G^{-1}(u)]`, where
  `H_{1_y,x} = EvalFx(1_y, B, x)`. Correctness follows from
  `(B − xᵀ⊗G)·H_{1_y,x} = B_{1_y}` (since `1_y(x)=0`), giving
  `[A | B − xᵀ⊗G]·s_x' = c'`.

## Parameters

`demo.py` uses `q=12289, d=16, n=2, m=36, ℓ=8` — sub-second to run, with every
relation exact. The only hard constraint is `m > m'` (so `A` admits a gadget
trapdoor). You can scale `ℓ` (revocation-handle length) and the lattice
dimensions up; cost grows because polynomial multiplication here is schoolbook
`O(d²)` and `EvalF/EvalFx` nest `ℓ` matrix products.

## What is intentionally omitted

- discrete-Gaussian / perturbation sampling (we use a short uniform mask);
- the ℓ-succinct M-SIS witness-compression variant (§4 "Reducing witnesses");
- a real post-quantum EUF-CMA signature in the compiler (we use a SHA-256 MAC);
- the LNP + LaBRADOR commit-and-prove zero-knowledge layer (Appendix D);
- a stateless PRF-derived `MemWitCreate`. -->
