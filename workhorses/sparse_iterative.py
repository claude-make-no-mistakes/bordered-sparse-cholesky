"""Checked diagonal-shift solves with bounded incomplete-Cholesky PCG."""

import ctypes as ct
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.sparse import tril

from workhorses.sparse import DiagonalShiftFamily, _DLL, _INTS, _VALUES, _canonical, _library, _ordering, _real_array, _rhs


@lru_cache(maxsize=4)
def _iterative_library(path):
    native = _library(path)
    native.wh_ic0.argtypes = (ct.c_int, _INTS, _INTS, _VALUES, ct.c_double, _VALUES)
    native.wh_pcg.argtypes = (ct.c_int, _INTS, _INTS, _VALUES, _INTS, _INTS, _VALUES,
                              ct.c_int, _INTS, _VALUES, _VALUES, ct.c_double, ct.c_int,
                              ct.c_double, _VALUES, _VALUES)
    return native


class IncompleteShiftFamily:
    """Try an IC(0) preconditioner and switch to checked direct solves on failure."""

    def __init__(self, upper, sites, *, permutation=None, max_iterations=48, library=_DLL):
        self._upper = upper = _canonical(upper).copy()
        self._order = n = upper.shape[0]
        sites = np.asarray(sites)
        if (sites.ndim != 1 or not len(sites) or sites.dtype.kind not in "iu"
                or np.any(sites >= n) or np.any(sites < 0) or len(np.unique(sites)) != len(sites)):
            raise ValueError("diagonal sites must be distinct valid positions")
        if not isinstance(max_iterations, int) or not 1 <= max_iterations <= 512:
            raise ValueError("iteration limit must be an integer from 1 to 512")
        if (not np.isfinite(upper.data).all()
                or np.any(upper.indices > np.repeat(np.arange(n, dtype=np.int32), np.diff(upper.indptr)))):
            raise ValueError("expected a finite upper-triangular CSC matrix")
        self._sites = np.array(sites, dtype=np.int32)
        self._permutation = None if permutation is None else _ordering(permutation, n)
        self._limit = max_iterations
        self._library_path = library
        self._native = _iterative_library(Path(library))
        self._direct = None
        self._closed = False
        self._lock = Lock()
        self.last_iterations = None
        self._ic = np.empty_like(upper.data)
        for bump in (0., .01):
            if self._native.wh_ic0(n, upper.indptr, upper.indices, upper.data, bump, self._ic):
                break
        else:
            self._use_direct()
            return
        self._matrix = (upper + tril(upper.T, k=-1)).tocsr()
        if self._matrix.indptr.dtype != np.int32 or self._matrix.indices.dtype != np.int32:
            self._use_direct()
            return
        self._row_norm = np.asarray(abs(self._matrix).sum(axis=1)).ravel()
        self._norm = float(self._row_norm.max())
        self._diagonals = upper.indptr[self._sites + 1] - 1
        self._roots = 1 / self._ic[self._diagonals]
        self._scratch = np.empty(4 * n)

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        with self._lock:
            if self._direct is not None:
                self._direct.close()
            self._closed = True

    def _use_direct(self):
        self._direct = DiagonalShiftFamily(self._upper, self._sites, permutation=self._permutation,
                                           library=self._library_path)
        self._ic = None
        self._matrix = None
        self._scratch = None

    def _solve_columns(self, shifts, rhs):
        matrix, triangle = self._matrix, self._upper
        norm = max(self._norm, float(np.max(self._row_norm[self._sites] + shifts)))
        result = np.empty(rhs.shape, dtype=np.float64, order="F")
        self._ic[self._diagonals] = 1 / np.hypot(self._roots, np.sqrt(shifts))
        iterations = []
        for index in range(rhs.shape[1]):
            count = self._native.wh_pcg(
                self._order, matrix.indptr, matrix.indices, matrix.data, triangle.indptr, triangle.indices,
                self._ic, len(shifts), self._sites, shifts, rhs[:, index], norm, self._limit,
                1e-13, result[:, index], self._scratch,
            )
            if count < 0:
                return None, None
            iterations.append(count)
        if not np.isfinite(result).all():
            return None, None
        discrepancy = matrix @ result - rhs
        discrepancy[self._sites] += shifts[:, None] * result[self._sites]
        denominator = norm * np.max(abs(result), axis=0) + np.max(abs(rhs), axis=0)
        if not np.isfinite(denominator).all():
            return None, None
        error = float(np.max(np.max(abs(discrepancy), axis=0) / np.where(denominator > 0, denominator, 1.)))
        if not np.isfinite(error) or error > 1e-11:
            return None, None
        self.last_iterations = max(iterations)
        return result, error

    def checked_solve(self, shifts, rhs):
        shifts = np.ascontiguousarray(_real_array(shifts))
        if shifts.shape != self._sites.shape or np.any(shifts < 0) or not np.isfinite(shifts).all():
            raise ValueError("provide one finite nonnegative shift per diagonal site")
        b = _rhs(rhs, self._order)
        with self._lock, np.errstate(over="ignore", invalid="ignore"):
            if self._closed:
                raise RuntimeError("the incomplete Cholesky plan is closed")
            self.last_iterations = None
            if self._direct is None:
                x, eta = self._solve_columns(shifts, np.asfortranarray(b.reshape(self._order, -1)))
                if x is not None:
                    return x.reshape(b.shape), eta, False
                self._use_direct()
            x, eta, _ = self._direct.checked_solve(shifts, b)
            return x, eta, True
