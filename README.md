# Bordered Sparse Cholesky

Fast, backward-error-checked sparse symmetric positive-definite (SPD) linear solves under repeated nonnegative diagonal shifts at a fixed set of coordinates, using trailing-block Schur-complement refactoring on top of CHOLMOD.

- [Paper (PDF)](output/pdf/latex/sparse.pdf)
- [LaTeX source](manuscripts/latex/sparse.tex)
- [Markdown draft](manuscripts/sparse.md)

## Overview

Given a sparse SPD matrix $A$ and $k \leq 32$ fixed mutable diagonal coordinates $B$, permuting $B$ to the end of the elimination order isolates every diagonal update $(A + E\,\operatorname{diag}(s_t)\,E^\top)x_t = b_t$ inside the $k \times k$ trailing Schur complement $S = A_{BB} - A_{BI}A_{II}^{-1}A_{IB}$. The interior Cholesky factor $L_{II}$ and coupling block $L_{BI}$ remain invariant across all requests.

For each request, the native implementation (`native/sparse_border.h`) factors $S + \operatorname{diag}(s_t)$ in extended precision (`long double`), updates only the trailing block in CHOLMOD's simplicial $LDL^\top$ or supernodal $LL^\top$ storage, applies optional extended-precision iterative refinement when an unshifted pivot is small, and verifies the normwise backward error $\eta \leq 10^{-11}$ (falling back automatically to the checked adaptive solver on failure or for $k > 32$).

Across 19 public SuiteSparse SPD matrices (eight mutable sites, twelve chronological requests, 684 measured solves per method), bordered Cholesky passed 684/684 independent checks with maximum backward error $2.14 \times 10^{-15}$, achieving a median speedup of $2.15\times$ ($1.22\text{--}2.32\times$) over checked Woodbury ($2.00\times$ including shared AMD ordering) and $1.28\times$ ($1.19\text{--}1.39\times$) over adaptive recovery.

## Repository Layout

- `native/` — Native C/C++ implementation (`sparse_border.h`, `sparse.c`, `sparse_iterative.h`, `amd_policy.c`, `paramd.cpp`).
- `workhorses/` — Python `ctypes` interfaces (`sparse_border.py`, `sparse.py`, `sparse_iterative.py`).
- `tests/` — Regression test suite (`test_native.py`, `amd_bounds.c`).
- `benchmarks/` — Build helper (`build.py`), frozen matrix fetcher (`paper_data.py`), request generator/verifier (`followup_sparse.py`), and timing harness (`twice_sparse.py`).
- `manuscripts/` — LaTeX source (`latex/sparse.tex`), Markdown draft (`sparse.md`), protocol (`PROTOCOL.md`), archived run journal (`runs/sparse.jsonl`), and summary receipts (`summary.json`).
- `paper/`, `followup/`, `doubling/`, `twice/` — Archived manifests and historical benchmark journals.

## Reproduce

The archived benchmark suite ran on Python 3.12 on Windows with MSYS2 UCRT64 GCC building CHOLMOD/SuiteSparse (`sparse.dll`, `libamd.dll`). Set `WORKHORSES_UCRT64` if MSYS2 is installed outside its default path.

```sh
python -m pip install -r requirements.txt
python -B -m benchmarks.build packages sparse
python -B -m benchmarks.paper_data --restore --inputs-only sparse
python -B -m unittest discover -s tests -v
python -B -m benchmarks.twice_sparse --candidates adaptive bordered --output .research/replay/sparse.jsonl
```

For single-threaded timing comparisons, set BLAS, OpenMP, and MKL thread limits to `1` and use a fresh output path. Archived measurements and SHA-256 receipts are in [manuscripts/summary.json](manuscripts/summary.json) and [manuscripts/runs/sparse.jsonl](manuscripts/runs/sparse.jsonl).
