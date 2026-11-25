# Bordered Cholesky for repeated sparse diagonal shifts

## Abstract

Many applications solve a sequence of sparse systems whose coefficients change at a few known diagonal positions. We move those variables to the end of the elimination order and replace a small trailing factor for each request. On 19 SuiteSparse SPD matrices with eight changing sites and twelve chronological requests, median sequence throughput was 2.15× that of checked Woodbury and 1.28× that of the preceding adaptive solver. The 95% paired interval against Woodbury was 1.22–2.32×. Charging both methods for the shared AMD ordering gave 2.00×. All 684 measured solutions passed an independent backward-error test, and worst relative forward error was 0.53%. [Checked measurements](summary.json) · [Extension protocol](PROTOCOL.md)

## Problem and baseline

Let $A$ be a real symmetric positive-definite matrix and let the columns of $E$ select $k$ distinct coordinates. Request $t$ supplies an independent right-hand side $b_t$ and nonnegative shifts $s_t$. It asks for the solution of $(A+E\,\mathrm{diag}(s_t)E^\top)x_t=b_t$. The solver knows $A$ and the sites during setup. It sees the shifts and right-hand sides in chronological order.

The control factors $A$ in the supplied approximate minimum-degree order, computes $A^{-1}E$, and solves a small dense system for each Woodbury correction. Its acceptance test recomputes a residual. On failure the control factors the shifted matrix. The preceding adaptive method selects a simplicial or supernodal kernel and can update its existing factor after a rejected correction. Its factorization threshold was tuned on these matrices. CHOLMOD's factorization and update routines and its authors' work on dynamic supernodes form the numerical baseline. [Chen et al.](https://siva.science/publication/chen-2008-algorithm/) · [Davis and Hager](https://people.clas.ufl.edu/hager/files/cholmod.pdf) · [Sparse Matrix Collection](https://sparse.tamu.edu/)

## A trailing block for the changing variables

Preserve the supplied relative order among interior variables $I$ and append the mutable variables $B$ in caller order. Block elimination gives the Schur complement

$$
S=A_{BB}-A_{BI}A_{II}^{-1}A_{IB}.
$$

A request adds $\mathrm{diag}(s_t)$ only to $S$. The leading triangular factor and its coupling to the border are independent of $t$. The SPD assumption guarantees positive definiteness of $S+\mathrm{diag}(s_t)$ in exact arithmetic. The same block identity underlies static condensation in finite-element solvers. [MFEM's derivation](https://docs.mfem.org/html/classmfem_1_1StaticCondensation.html)

The implementation recovers $S$ from the unshifted factor's trailing block in CHOLMOD's simplicial $LDL^\top$ or supernodal $LL^\top$ storage. For each request it factors the shifted block in extended precision, writes its entries into the existing storage, and runs the triangular solves. Reconstructing each shift from the saved base avoids accumulating update errors. A small pivot relative to its original diagonal triggers one correction with an extended-precision residual. Borders larger than 32 or unsuccessful numerical checks use the checked adaptive solver. [Python interface](../workhorses/sparse_border.py) · [Native factor implementation](../native/sparse_border.h)

## Evaluation and accuracy

The frozen benchmark selects eight distinct sites on each of 19 public SPD patterns. Each matrix has twelve fixed systems with independently generated manufactured solution vectors. The benchmark forms each right-hand side by multiplying the shifted matrix by the manufactured vector. The timed solver receives the right-hand side alone with the requested shifts. An independent SciPy product checks the result and separately compares it to the manufactured solution. A warm-up and three measured sequences give 684 measured solutions per method. There are 19 independent matrix units. [Matrix manifest](../paper/inputs-sparse.json) · [Request generator and verifier](../benchmarks/followup_sparse.py)

For $M=A+E\,\mathrm{diag}(s_t)E^\top$, the acceptance statistic is

$$
\eta=\frac{\lVert M\hat x-b\rVert_\infty}
{\lVert M\rVert_\infty\lVert\hat x\rVert_\infty+\lVert b\rVert_\infty}.
$$

The required bound is $\eta\leq 10^{-11}$. Forward error is reported separately because conditioning can amplify a small backward error. Moving variables can increase fill, so we charge setup and all twelve solves. We take a median time within each matrix and then a median of the paired ratios. A second estimate adds the shared AMD ordering time to both methods. [Timing harness](../benchmarks/twice_sparse.py)

## Results and limits

The previous adaptive method passed 684/684 independent checks and replaced 108 fresh shifted factorizations with factor updates. Its median speedup over checked Woodbury was 1.50× [1.06–1.72]. Worst relative forward error was 0.48% for the adaptive method and 0.58% for the control. A zero-fill incomplete Cholesky method solved 216 requests iteratively and sent the remaining 468 through a checked direct solver. Its median speed ratio was 0.88×. [Earlier matrix-level and accuracy results](../twice/summary.json)

| Same-run measure | Checked Woodbury | Adaptive recovery | Trailing block |
| --- | ---: | ---: | ---: |
| Sequence speed relative to Woodbury | 1.00× | 1.44× | 2.15× |
| Speed including shared AMD | 1.00× | 1.40× | 2.00× |
| Largest independent backward error | 1.94 × 10⁻¹³ | 1.94 × 10⁻¹³ | 2.14 × 10⁻¹⁵ |
| Largest relative forward error | 0.58% | 0.48% | 0.53% |
| Verified measured requests | 684/684 | 684/684 | 684/684 |

The bordered solver used its trailing block on all measured requests. Eleven of the 19 matrices individually crossed 2×. Speed ratios ranged from 1.11× to 15.12×. The direct paired comparison with adaptive recovery was 1.28× [1.19–1.39]. Including shared AMD gave 2.00× [1.21–2.19] against Woodbury. The interval spans gains below twofold, and the largest forward error rose from the preceding adaptive solver's 0.48% to 0.53%. The matrix with the largest forward error was Oberwolfach/LF10000. [Per-matrix attempts](runs/sparse.jsonl) · [Aggregated results](summary.json)

The measured ratio combines the border ordering, removal of response-column setup and a native residual check. An ablation would be needed to apportion their costs. The study covers fixed sites and nonnegative shifts on SPD matrices. New patterns, changing sites and different sequence lengths would test how far the result transfers.
