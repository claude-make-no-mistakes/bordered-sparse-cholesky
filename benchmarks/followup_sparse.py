"""Measure fixed-pattern factoring and low-rank solves against fresh CHOLMOD."""

import argparse
from contextlib import ExitStack
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import triu

from benchmarks.common import BIN, ROOT, schedule
from benchmarks.fetch import sha256
from benchmarks.paper_common import Journal, PAPER, portable_file
from benchmarks.paper_data import CACHE, rank
from benchmarks.sparse import AMD, COLAMD, FactorStats, Native, PivotStats, load_matrix
from workhorses.sparse import DiagonalShiftFamily, FixedPatternCholesky, FixedPatternLU


FOLLOWUP = ROOT / "followup"
SEED = 1_623_795
REQUESTS = 12
LU_REQUESTS = 8
SITES = 8
REPEATS = 3
METHODS = {"numeric": ("fresh", "symbolic"), "diagonal": ("fresh", "symbolic", "low-rank")}
LU_METHODS = ("fresh", "symbolic")


def _requests(a, *, seed, scenario, sites):
    rng = np.random.default_rng(seed + (scenario == "diagonal"))
    diagonal = a.diagonal()[sites]
    prepared = []
    for _ in range(REQUESTS):
        scale = 1.0 if scenario == "diagonal" else float(rng.uniform(0.7, 1.4))
        shifts = rng.uniform(0.05, 1.5, len(sites)) * diagonal
        expected = rng.uniform(-1, 1, a.shape[0])
        rhs = scale * (a @ expected)
        rhs[sites] += shifts * expected[sites]
        prepared.append((scale, shifts, rhs, expected))
    return prepared


def _error(a, row_norm, base_norm, z, rhs, scale, shifts, sites):
    difference = scale * (a @ z) - rhs
    difference[sites] += shifts * z[sites]
    norm = max(scale * base_norm, float(np.max(scale * row_norm[sites] + shifts)))
    return float(np.max(abs(difference)) / (norm * np.max(abs(z)) + np.max(abs(rhs))))


def _outcome(seconds, answer, expected, backward, fallback=False):
    forward = float(np.max(abs(answer - expected)) / np.max(abs(expected)))
    return {"seconds": seconds, "verified": bool(np.isfinite(backward) and backward <= 1e-11),
            "backward_error": float(backward) if np.isfinite(backward) else None,
            "forward_error": forward if np.isfinite(forward) else None, "fallback": fallback}


def _recorded_error(start, error):
    return {"seconds": perf_counter() - start, "verified": False, "status": "failed", "error": str(error)}


def _attempt(native, method, a, upper, perm, prepared, sites, diag_positions, row_norm):
    base_norm = float(row_norm.max())
    with ExitStack() as owners:
        setup_start = perf_counter()
        if method == "symbolic":
            plan = owners.enter_context(FixedPatternCholesky(upper, permutation=perm))
        elif method in ("low-rank", "factor-update", "factor-mixed"):
            strategy = {"low-rank": "woodbury", "factor-update": "update", "factor-mixed": "mixed"}[method]
            plan = owners.enter_context(DiagonalShiftFamily(
                upper, sites, permutation=perm, strategy=strategy))
        setup = perf_counter() - setup_start
        result = []
        for scale, shifts, rhs, expected in prepared:
            start = perf_counter()
            used_fallback = False
            try:
                if method in ("low-rank", "factor-update", "factor-mixed"):
                    z, backward, used_fallback = plan.checked_solve(shifts, rhs)
                else:
                    values = scale * upper.data
                    values[diag_positions] += shifts
                    if method == "fresh":
                        z = np.empty_like(rhs)
                        stats = FactorStats()
                        status = native.cholesky(a.shape[0], upper.indptr, upper.indices, values, rhs, perm, z, ct.byref(stats))
                        if not status or not stats.permutation_match:
                            raise np.linalg.LinAlgError(f"fresh factorization failed (status {stats.status})")
                    else:
                        z = plan.refactor_values_and_solve(values, rhs)
                    backward = _error(a, row_norm, base_norm, z, rhs, scale, shifts, sites)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as failure:
                result.append(_recorded_error(start, failure))
                continue
            # The manufactured solution is checked independently of the timed guard.
            seconds = perf_counter() - start
            if method in ("factor-update", "factor-mixed"):
                backward = _error(a, row_norm, base_norm, z, rhs, scale, shifts, sites)
            result.append(_outcome(seconds, z, expected, backward, used_fallback))
        return {"setup_seconds": setup, "steps": result, "verified": all(r["verified"] for r in result),
                "status": "ok", "amortized_seconds": {str(k): setup + sum(r["seconds"] for r in result[:k])
                                                      for k in (1, 4, REQUESTS)}}


def _lu(native, a, seed):
    n = a.shape[0]
    perm = np.empty(n, dtype=np.int32)
    start = perf_counter()
    status = native.colamd(n, a.indptr, a.indices, perm, *COLAMD["colamd"])
    ordering_time = perf_counter() - start
    if status < 0 or not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError("COLAMD could not generate the shared permutation")
    columns = np.repeat(np.arange(n, dtype=np.int32), np.diff(a.indptr))
    rng = np.random.default_rng(seed)
    prepared = []
    magnitude = abs(a)
    for _ in range(LU_REQUESTS):
        left, right = rng.uniform(0.7, 1.3, (2, n))
        expected = rng.uniform(-1, 1, n)
        rhs = left * (a @ (right * expected))
        norm = float(np.max(left * (magnitude @ right)))
        prepared.append((left, right, rhs, expected, norm))

    rows = []
    for repeat, method in schedule(LU_METHODS, seed=seed, repeats=REPEATS + 1):
        try:
            setup = perf_counter()
            plan = FixedPatternLU(a, permutation=perm, reuse_symbolic=method == "symbolic")
            setup = perf_counter() - setup
            steps = []
            with plan:
                for left, right, rhs, expected, norm in prepared if repeat else prepared[:1]:
                    start = perf_counter()
                    try:
                        values = (a.data * left[a.indices]) * right[columns]
                        answer = plan.refactor_values_and_solve(values, rhs)
                        residual = float(np.max(abs(left * (a @ (right * answer)) - rhs)))
                        backward = residual / (norm * np.max(abs(answer)) + np.max(abs(rhs)))
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as failure:
                        steps.append(_recorded_error(start, failure))
                        continue
                    steps.append(_outcome(perf_counter() - start, answer, expected, backward))
            result = {"setup_seconds": setup, "steps": steps, "verified": all(r["verified"] for r in steps),
                      "status": "ok", "amortized_seconds": {str(k): setup + sum(r["seconds"] for r in steps[:k])
                                                            for k in (1, 4, LU_REQUESTS)}}
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as failure:
            result = {"status": "failed", "verified": False, "error": str(failure)}
        rows.append({"method": method, "scenario": "lu", "repeat": repeat - 1, **result})
    return ordering_time, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FOLLOWUP / "runs" / "sparse.jsonl")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    manifest = PAPER / "inputs-sparse.json"
    items = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
    protocol = {"study": "sparse-repeated-v1", "probe": args.probe, "seed": SEED, "requests": REQUESTS,
                "repeats": REPEATS, "sites": SITES, "lu_requests": LU_REQUESTS,
                "methods": {key: list(v) for key, v in {**METHODS, "lu": LU_METHODS}.items()},
                "input_manifest_sha256": sha256(manifest), "numpy": np.__version__, "scipy": scipy.__version__,
                "program_sha256": {f: sha256(BIN / f) for f in ("sparse.dll", "libcholmod.dll", "libumfpack.dll", "libopenblas.dll")}}
    journal = Journal(args.output, protocol)
    native = Native()
    if args.probe:
        items = [next(row for row in reversed(items) if row["kind"] == kind) for kind in ("spd", "lu")]
    for item in items:
        matrix = item["matrix"]
        if matrix in journal.completed:
            continue
        a = load_matrix(portable_file(CACHE, item), matrix.rsplit("/", 1)[-1])
        seed = SEED + int(rank("shift-sequence", matrix)[:11], 16)
        traits = {"matrix": matrix, "group": item["group"], "kind": item["kind"], "seed": seed,
                  "order": a.shape[0], "nnz": a.nnz}
        if item["kind"] == "lu":
            order_seconds, rows = _lu(native, a, seed)
            journal.add(matrix, attributes={**traits, "common_colamd_seconds": order_seconds}, attempts=rows)
            print(f"{matrix} verified {sum(r['verified'] for r in rows)}/{len(rows)} LU blocks", flush=True)
            continue
        upper = triu(a, format="csc")
        upper.sort_indices()
        if np.any(upper.indptr[1:] == upper.indptr[:-1]) or not np.array_equal(upper.indices[upper.indptr[1:] - 1], np.arange(a.shape[0])):
            raise ValueError(f"SPD matrix {matrix} is missing a stored diagonal entry")
        row_norm = np.asarray(abs(a).sum(axis=1)).ravel()
        perm = np.empty(a.shape[0], dtype=np.int32)
        stats = PivotStats()
        order_start = perf_counter()
        code = native.order(a.shape[0], upper.indptr, upper.indices, perm, *AMD["amd"], ct.byref(stats))
        order_seconds = perf_counter() - order_start
        if code < 0 or not np.array_equal(np.sort(perm), np.arange(a.shape[0])):
            raise ValueError(f"AMD could not order {matrix}")
        sites = np.sort(np.random.default_rng(seed).choice(a.shape[0], SITES, replace=False))
        rows = []
        for scenario, methods in METHODS.items():
            support = sites if scenario == "diagonal" else np.arange(a.shape[0], dtype=np.intp)
            diag_positions = upper.indptr[support + 1] - 1
            prepared = _requests(a, seed=seed, scenario=scenario, sites=support)
            for repeat, method in schedule(methods, seed=seed, repeats=REPEATS + 1):
                try:
                    record = _attempt(native, method, a, upper, perm, prepared if repeat else prepared[:1],
                                      support, diag_positions, row_norm)
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as failure:
                    record = {"status": "failed", "error": str(failure), "verified": False}
                rows.append({"method": method, "scenario": scenario, "repeat": repeat - 1, **record})
        journal.add(matrix, attributes={**traits, "sites": sites.tolist(), "common_amd_seconds": order_seconds}, attempts=rows)
        print(f"{matrix} verified {sum(r['verified'] for r in rows)}/{len(rows)} blocks", flush=True)


if __name__ == "__main__":
    main()
