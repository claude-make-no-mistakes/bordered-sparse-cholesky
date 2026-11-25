"""Replay AMD bucket policies and verify their downstream sparse solves."""

import argparse
import ctypes as ct
from hashlib import sha256 as digest
from pathlib import Path
import statistics
import tarfile
from time import perf_counter

import numpy as np
import scipy
from scipy.io import mmread
from scipy.sparse import csc_matrix, triu
from scipy.sparse.linalg import splu

from benchmarks.common import BIN, SEED, interval, metadata, schedule, write_result
from benchmarks.fetch import SPARSE, cached_asset


class PivotStats(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ("decisions", "candidates", "changed", "inspected", "incomplete")]


class FactorStats(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in ("analysis_seconds", "factor_seconds", "factor_nnz", "flops",
                                                "stored_entries", "memory_peak")]
    _fields_ += [("permutation_match", ct.c_int), ("status", ct.c_int)]


def _fields(stats):
    return {name: getattr(stats, name) for name, _ in stats._fields_}


AMD = {
    "amd": (10.0, 1, 0, 0, 0, 0),
    "amd-dense4": (4.0, 1, 0, 0, 0, 0),
    "amd-dense24": (24.0, 1, 0, 0, 0, 0),
    "hook-off": (10.0, 1, 1, 0, 0, 0),
    "exact-local-fill": (10.0, 1, 4, 64, 2_147_483_647, 0),
    "approx-local-fill": (10.0, 1, 4, 64, 192, 1),
    "bounded-64": (10.0, 1, 4, 64, 64, 0),
    "bounded-192": (10.0, 1, 4, 64, 192, 0),
}
COLAMD = {
    "colamd": (10.0, 10.0),
    "colamd-row4": (4.0, 10.0),
    "colamd-col4": (10.0, 4.0),
    "colamd-adaptive": None,
}


class Native:
    def __init__(self):
        self.library = ct.CDLL(str(BIN / "sparse.dll"))
        i, d = ct.c_int, ct.c_double
        ints = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
        values = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        self.order = self.library.wh_order
        self.order.argtypes = [i, ints, ints, ints, d, i, i, i, i, i, ct.POINTER(PivotStats)]
        self.colamd = self.library.wh_colamd
        self.colamd.argtypes = [i, ints, ints, ints, d, d]
        self.cholesky = self.library.wh_cholesky
        self.cholesky.argtypes = [i, ints, ints, values, values, ints, values, ct.POINTER(FactorStats)]
        self.metis = ct.CDLL(str(BIN / "libmetis.dll")).METIS_NodeND
        self.metis.argtypes = [ct.POINTER(i), ints, ints, ct.c_void_p, ct.c_void_p, ints, ints]
        self.paramd = None
        if (BIN / "paramd.dll").is_file():
            self.paramd = ct.CDLL(str(BIN / "paramd.dll")).wh_paramd
            self.paramd.argtypes = [i, ints, ints, ints, i]


def _matrix(name):
    short = name.split("/")[-1]
    path, received_hash = cached_asset(f"{short}.tar.gz")
    return load_matrix(path, short), received_hash


def load_matrix(path, short):
    with tarfile.open(path) as archive:
        member = next(item for item in archive.getmembers() if Path(item.name).name == f"{short}.mtx")
        with archive.extractfile(member) as stream:
            raw = mmread(stream, spmatrix=False)
    a = csc_matrix(raw, dtype=np.float64)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    if a.shape[0] != a.shape[1] or a.indptr.dtype != np.int32:
        raise ValueError(f"unsupported matrix shape or index width: {short}")
    return a


def _adjacency(matrix):
    graph = matrix.copy()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _density_knobs(matrix):
    """Discard unusually concentrated tails, preserving independent row/column knobs."""
    n = matrix.shape[0]
    row_degrees = np.bincount(matrix.indices, minlength=n)
    col_degrees = np.diff(matrix.indptr).astype(np.int64)

    def cutoff(degrees, mass):
        threshold = int(np.quantile(degrees, 0.98, method="higher"))
        tail = degrees > threshold
        if degrees.max() > 2 * max(16, threshold) and mass.sum() and mass[tail].sum() > 0.5 * mass.sum():
            return min(10.0, (max(16, threshold) + 0.5) / np.sqrt(n))
        return 10.0

    return cutoff(row_degrees, row_degrees * (row_degrees - 1) // 2), cutoff(col_degrees, col_degrees)


def _error(a, result, rhs, norm):
    delta = float(np.linalg.norm(a @ result - rhs, ord=np.inf))
    scale = norm * float(np.linalg.norm(result, ord=np.inf)) + float(np.linalg.norm(rhs, ord=np.inf))
    return delta / scale if scale else delta


def run_method(native, method, kind, a, upper, graph, rhs, norm):
    n = a.shape[0]
    ordering = np.empty(n, dtype=np.int32)
    pivots = PivotStats()
    start = perf_counter()
    knobs = None
    if method in AMD:
        code = native.order(n, upper.indptr, upper.indices, ordering, *AMD[method], ct.byref(pivots))
        success = code >= 0
    elif method == "metis":
        inverse = np.empty_like(ordering)
        code = native.metis(ct.byref(ct.c_int(n)), graph.indptr, graph.indices, None, None, ordering, inverse)
        success = code == 1
    elif method.startswith("paramd-"):
        code = native.paramd(n, graph.indptr, graph.indices, ordering, int(method.split("-")[1]))
        success = code == 0
    else:
        knobs = COLAMD[method] or _density_knobs(a)
        code = native.colamd(n, a.indptr, a.indices, ordering, *knobs)
        success = code >= 0
    ordering_time = perf_counter() - start
    row = {"ordering_seconds": ordering_time, "ordering_status": code, "pivots": _fields(pivots),
           "knobs": list(knobs) if knobs else None, "verified": False}
    if not success:
        return {**row, "status": "ordering_failed"}
    if not np.array_equal(np.sort(ordering), np.arange(n)):
        return {**row, "status": "invalid_permutation"}
    row["permutation_sha256"] = digest(ordering.tobytes()).hexdigest()
    if kind == "spd":
        result = np.empty_like(rhs)
        stats = FactorStats()
        ok = native.cholesky(n, upper.indptr, upper.indices, upper.data, rhs, ordering, result, ct.byref(stats))
        row.update(_fields(stats))
        row["factor_status"] = row.pop("status")
        if not ok:
            return {**row, "status": "cholesky_failed"}
    else:
        try:
            start = perf_counter()
            ordered = a[:, ordering].tocsc()
            ordered.sort_indices()
            row["analysis_seconds"] = perf_counter() - start
            start = perf_counter()
            factor = splu(ordered, permc_spec="NATURAL", diag_pivot_thresh=1.0, options={"Equil": False})
            row["factor_seconds"] = perf_counter() - start
        except (MemoryError, RuntimeError) as exc:
            return {**row, "status": type(exc).__name__, "error": str(exc)}
        result = np.empty_like(rhs)
        result[ordering] = factor.solve(rhs)
        parts = factor.L, factor.U
        row["factor_nnz"] = sum(item.nnz for item in parts)
        row["stored_factor_bytes"] = sum(item.data.nbytes + item.indptr.nbytes + item.indices.nbytes for item in parts)
    backward = _error(a, result, rhs, norm)
    verified = bool(np.isfinite(backward) and backward <= 1e-11)
    return {**row, "status": "verified" if verified else "backward_error",
            "backward_error": backward if np.isfinite(backward) else None, "verified": verified,
            "total_seconds": ordering_time + row["analysis_seconds"] + row["factor_seconds"]}


def summarize(rows):
    results = {}
    for method in sorted({r["method"] for r in rows}):
        candidates = [r for r in rows if r["method"] == method]
        kind = candidates[0]["kind"]
        baseline = "amd" if kind == "spd" else "colamd"
        by_matrix = {}
        for name, relabel in sorted({(r["matrix"], r["relabeling"]) for r in candidates}):
            controls = [r for r in rows if r["method"] == baseline and r["matrix"] == name and r["relabeling"] == relabel]
            tests = [r for r in candidates if r["matrix"] == name and r["relabeling"] == relabel]
            if not controls or not all(r["verified"] for r in controls + tests):
                continue
            result = {
                key: statistics.median(r[key] for r in controls) / statistics.median(r[key] for r in tests)
                for key in ("total_seconds", "factor_nnz", "ordering_seconds")
            }
            result["factor_pipeline"] = (
                statistics.median(r["analysis_seconds"] + r["factor_seconds"] for r in controls)
                / statistics.median(r["analysis_seconds"] + r["factor_seconds"] for r in tests)
            )
            baseline_order = statistics.median(r["ordering_seconds"] + r["analysis_seconds"] for r in controls)
            test_order = statistics.median(r["ordering_seconds"] + r["analysis_seconds"] for r in tests)
            per_factor_saving = statistics.median(r["factor_seconds"] for r in controls) - statistics.median(
                r["factor_seconds"] for r in tests)
            result["amortization_factorizations"] = (
                max(1, int(np.ceil((test_order - baseline_order) / per_factor_saving)))
                if per_factor_saving > 0 else None
            )
            by_matrix.setdefault(name, []).append(result)
        metrics = ("total_seconds", "factor_nnz", "ordering_seconds", "factor_pipeline")
        results[method] = {
            "attempts": len(candidates), "verified": sum(r["verified"] for r in candidates),
            "reference": baseline, "by_matrix_and_relabeling": by_matrix,
            **{key: interval(statistics.median(entry[key] for entry in group) for group in by_matrix.values())
               for key in metrics},
            "changed_pivots": sum(r["pivots"]["changed"] for r in candidates if r["repeat"] == 0),
        }
    return results


def study(args):
    native = Native()
    methods = {
        "spd": tuple(args.amd) + ("metis",),
        "lu": tuple(COLAMD),
    }
    if native.paramd:
        methods["spd"] += tuple(f"paramd-{t}" for t in args.parallel_threads)
    rows = []
    for kind, names in SPARSE[args.cohort].items():
        if args.kind != "all" and args.kind != kind:
            continue
        for index, name in enumerate(names):
            matrix, hash_ = _matrix(name)
            if kind == "spd" and (matrix != matrix.T).nnz:
                raise ValueError(f"SPD benchmark requires numerical symmetry: {name}")
            for relabel in range(args.relabelings + 1):
                rng = np.random.default_rng(SEED + 1000 * index + relabel)
                a = matrix if not relabel else matrix[(permutation := rng.permutation(matrix.shape[0])), :][:, permutation].tocsc()
                a.sort_indices()
                n = a.shape[0]
                norm = float(np.asarray(abs(a).sum(axis=1)).max())
                rhs = np.asarray(a @ rng.uniform(0.5, 1.5, n))
                upper, graph = (triu(a, format="csc"), _adjacency(a)) if kind == "spd" else (None, None)
                if upper is not None:
                    upper.sort_indices()
                for repeat, method in schedule(methods[kind], seed=SEED + index * 17 + relabel, repeats=args.repeats + 1):
                    result = run_method(native, method, kind, a, upper, graph, rhs, norm)
                    if repeat:
                        rows.append({"matrix": name, "kind": kind, "sha256": hash_, "n": n, "nnz": a.nnz,
                                     "relabeling": relabel, "repeat": repeat - 1, "method": method, **result})
                if kind == "spd" and {"amd", "hook-off"}.issubset(methods[kind]):
                    checks = {r["permutation_sha256"] for r in rows if r["matrix"] == name and r["relabeling"] == relabel
                              and r["method"] in ("amd", "hook-off") and "permutation_sha256" in r}
                    if len(checks) != 1:
                        raise RuntimeError(f"hook-off parity failed on {name}, relabeling {relabel}")
            print(f"{name} finished", flush=True)
    settings = vars(args).copy()
    settings.pop("output")
    write_result(args.output, {"metadata": {**metadata(), "scipy": scipy.__version__, "numpy": np.__version__, "seed": SEED},
                              "settings": {**settings, "amd_options": AMD, "colamd_options": COLAMD,
                                           "paramd_available": bool(native.paramd)},
                              "summary": summarize(rows), "attempts": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=SPARSE, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("all", "spd", "lu"), default="all")
    parser.add_argument("--amd", nargs="+", choices=AMD, default=list(AMD))
    parser.add_argument("--relabelings", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel-threads", type=int, nargs="*", default=[1, 4])
    args = parser.parse_args()
    if min(args.relabelings, args.repeats - 1) < 0 or any(t < 1 for t in args.parallel_threads):
        parser.error("invalid repetition or parallel thread count")
    study(args)


if __name__ == "__main__":
    main()
