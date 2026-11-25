"""Compare configured Cholesky kernels on the frozen diagonal-shift sequences."""

import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import triu

from benchmarks.common import BIN, ROOT, schedule
from benchmarks.fetch import sha256
from benchmarks import followup_sparse as sparse
from benchmarks.paper_common import Journal, PAPER, load_journal, portable_file
from benchmarks.paper_data import CACHE, rank
from workhorses.sparse import DiagonalShiftFamily, _CHOLESKY_MODES
from workhorses.sparse_iterative import IncompleteShiftFamily
from workhorses.sparse_border import BorderedShiftFamily


FAILURES = (RuntimeError, ValueError, np.linalg.LinAlgError)
REPEATS = 3
ITERATIVE = {"pcg-24": 24, "pcg-64": 64}


def _trial(name, a, upper, ordering, sites, requests, row_norm):
    start = perf_counter()
    used_base = False
    iterative = name in ITERATIVE
    try:
        if name == "bordered":
            plan = BorderedShiftFamily(upper, sites, permutation=ordering)
        elif iterative:
            plan = IncompleteShiftFamily(upper, sites, permutation=ordering, max_iterations=ITERATIVE[name])
        else:
            strategy = "recover" if name == "adaptive" else "woodbury"
            plan = DiagonalShiftFamily(upper, sites, permutation=ordering, strategy=strategy, _factorization=name)
    except FAILURES:
        if name == "auto":
            raise
        used_base = True
        plan = DiagonalShiftFamily(upper, sites, permutation=ordering)
    setup = perf_counter() - start
    with plan:
        metrics = None if iterative and not used_base else plan._plan._metrics()
        rows = []
        for scale, shift, rhs, truth in requests:
            start = perf_counter()
            try:
                answer, _, fallback = plan.checked_solve(shift, rhs)
            except FAILURES as error:
                rows.append(sparse._recorded_error(start, error))
                continue
            seconds = perf_counter() - start
            backward = sparse._error(a, row_norm, row_norm.max(), answer, rhs, scale, shift, sites)
            rows.append({**sparse._outcome(seconds, answer, truth, backward, fallback or used_base),
                         "iterations": getattr(plan, "last_iterations", None),
                         "updated_factor": bool(getattr(plan, "_updates_active", False))})
    return {"setup_seconds": setup, "setup_fallback": used_base, "factor": metrics,
            "verified": all(r["verified"] for r in rows), "steps": rows,
            "elapsed": {str(k): setup + sum(r["seconds"] for r in rows[:k]) for k in (1, 4, len(rows))}}


def run(path, *, probe, candidates, iterative_probe=False):
    manifest = PAPER / "inputs-sparse.json"
    matrices = [r for r in json.loads(manifest.read_text(encoding="utf-8"))["cases"] if r["kind"] == "spd"]
    frozen_path = ROOT / "doubling/runs/sparse-mixed.jsonl"
    _, frozen = load_journal(frozen_path)
    references = {u["unit"]: u["attributes"] for u in frozen}
    if iterative_probe:
        modes = ("auto", *ITERATIVE)
    else:
        modes = tuple(_CHOLESKY_MODES) if probe else ("auto", *candidates)
    repeats = 2 if probe or iterative_probe else REPEATS
    protocol = {"study": "configured-cholesky-v1", "probe": probe or iterative_probe, "methods": list(modes),
                             "repeats": repeats, "requests": sparse.REQUESTS, "sites": sparse.SITES, "seed": sparse.SEED,
                             "manifest_sha256": sha256(manifest), "reference_sha256": sha256(frozen_path),
                             "numpy": np.__version__, "scipy": scipy.__version__,
                             "code": {f: sha256(ROOT / "workhorses" / f) for f in ("sparse.py", "sparse_iterative.py")},
                             "programs": {f: sha256(BIN / f) for f in ("sparse.dll", "libcholmod.dll", "libopenblas.dll")}}
    if "bordered" in modes:
        protocol["study"] = "bordered-cholesky-v1"
        protocol["code"].update({p: sha256(ROOT / p) for p in (
            "workhorses/sparse_border.py", "native/sparse_border.h", "native/sparse.c", "benchmarks/twice_sparse.py")})
    journal = Journal(path, protocol)
    native = sparse.Native()
    for item in matrices:
        name = item["matrix"]
        if name in journal.completed:
            continue
        a = sparse.load_matrix(portable_file(CACHE, item), name.rsplit("/", 1)[-1])
        n = a.shape[0]
        upper = triu(a, format="csc")
        upper.sort_indices()
        row_norm = np.asarray(abs(a).sum(axis=1)).ravel()
        perm = np.empty(n, dtype=np.int32)
        start = perf_counter()
        ok = native.order(n, upper.indptr, upper.indices, perm, *sparse.AMD["amd"], ct.byref(sparse.PivotStats()))
        ordering = perf_counter() - start
        if ok < 0 or not np.array_equal(np.sort(perm), np.arange(n)):
            raise ValueError(f"cannot reproduce AMD for {name}")
        seed = sparse.SEED + int(rank("shift-sequence", name)[:11], 16)
        sites = np.sort(np.random.default_rng(seed).choice(n, sparse.SITES, replace=False))
        tags = {"seed": seed, "sites": sites.tolist(), "order": n, "nnz": a.nnz}
        if any(references[name][key] != value for key, value in tags.items()):
            raise ValueError(f"numeric input sequence changed for {name}")
        prepared = sparse._requests(a, seed=seed, scenario="diagonal", sites=sites)
        rows = []
        for repeat, mode in schedule(modes, seed=seed, repeats=repeats + 1):
            try:
                values = _trial(mode, a, upper, perm, sites, prepared, row_norm)
            except FAILURES as failure:
                values = {"verified": False, "error": str(failure)}
            rows.append({"method": mode, "repeat": repeat - 1, **values})
        journal.add(name, attributes={"matrix": name, "common_amd_seconds": ordering, **tags}, attempts=rows)
        print(f"{name} verified {sum(r['verified'] for r in rows)}/{len(rows)} sequences", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument("--probe", action="store_true")
    probes.add_argument("--iterative-probe", action="store_true")
    parser.add_argument("--candidates", nargs="+", choices=[k for k in (*_CHOLESKY_MODES, *ITERATIVE, "bordered") if k != "auto"],
                        default=["adaptive", "simplicial", "wide", "pcg-24"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default = ("sparse-iterative-probe.jsonl" if args.iterative_probe else
               "sparse-probe.jsonl" if args.probe else "sparse.jsonl")
    run(args.output or ROOT / "twice/runs" / default, probe=args.probe, candidates=args.candidates,
        iterative_probe=args.iterative_probe)
