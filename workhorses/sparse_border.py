"""Checked repeated diagonal shifts using a trailing dense Schur complement."""

import ctypes as ct
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np

from workhorses.sparse import (
    DiagonalShiftFamily, FixedPatternCholesky, _DLL, _INTS, _VALUES, _canonical, _library,
    _ordering, _real_array, _rhs,
)


@lru_cache(maxsize=4)
def _border_library(path):
    native = _library(path)
    native.wh_border_prepare.argtypes = (ct.c_void_p, ct.c_int, _INTS, _VALUES)
    native.wh_border_solve.argtypes = (ct.c_void_p, _VALUES, _INTS, _VALUES, ct.c_size_t,
                                       _VALUES, _VALUES, ct.POINTER(ct.c_double))
    return native


def _amd(upper, library):
    routine = ct.CDLL(str(Path(library).with_name("libamd.dll"))).amd_order
    routine.argtypes = (ct.c_int, _INTS, _INTS, _INTS, ct.c_void_p, ct.c_void_p)
    order = np.empty(upper.shape[0], dtype=np.int32)
    if routine(len(order), upper.indptr, upper.indices, order, None, None) < 0:
        raise ValueError("AMD failed to order the upper triangle")
    return order


class BorderedShiftFamily:
    """Keep a fixed interior factor and replace its small trailing factor per solve.

    The supplied order sets the relative ordering of interior variables. The
    changing sites occupy the suffix in caller order. Larger borders and
    unsuccessful numerical checks use the checked adaptive direct solver.
    """

    def __init__(self, upper, sites, *, permutation=None, library=_DLL, _factorization="adaptive"):
        a = self._upper = _canonical(upper).copy()
        n = a.shape[0]
        sites = np.asarray(sites)
        if (sites.ndim != 1 or not sites.size or sites.dtype.kind not in "iu"
                or np.any(sites >= n) or np.any(sites < 0) or len(np.unique(sites)) != len(sites)):
            raise ValueError("diagonal sites must be distinct valid positions")
        self._sites = np.array(sites, dtype=np.int32)
        self._order = _amd(a, library) if permutation is None else _ordering(permutation, n)
        self._library_path = library
        self._lock = Lock()
        self._closed = False
        self._direct = self._plan = None
        if len(sites) <= 32:
            interior = np.ones(n, dtype=bool)
            interior[self._sites] = False
            order = np.concatenate((self._order[interior[self._order]], self._sites))
            self._native = _border_library(Path(library))
            self._plan = FixedPatternCholesky(a, permutation=order, library=library, _factorization=_factorization)
            try:
                self._plan.factorize(a)
                if self._native.wh_border_prepare(self._plan._handle, len(sites), self._sites, a.data):
                    return
            except np.linalg.LinAlgError:
                pass
            except BaseException:
                self._plan.close()
                raise
        self._fallback()

    def _fallback(self):
        if self._plan is not None:
            self._plan.close()
            self._plan = None
        self._direct = DiagonalShiftFamily(self._upper, self._sites, permutation=self._order,
                                            library=self._library_path, strategy="recover", _factorization="adaptive")
        self._plan = self._direct._plan

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        with self._lock:
            if self._plan is not None:
                self._plan.close()
            self._closed = True

    def checked_solve(self, shifts, rhs):
        shift = np.ascontiguousarray(_real_array(shifts))
        if shift.shape != self._sites.shape or np.any(shift < 0) or not np.isfinite(shift).all():
            raise ValueError("provide one finite nonnegative shift per diagonal site")
        b = _rhs(rhs, self._upper.shape[0])
        with self._lock:
            if self._closed:
                raise RuntimeError("the bordered factorization plan is closed")
            if self._direct is None and self._plan is not None:
                source = b.ravel(order="F")
                answer = np.empty(source.size)
                err = ct.c_double()
                if self._native.wh_border_solve(self._plan._handle, self._upper.data, self._sites, shift,
                                                b.size // len(b), source, answer, ct.byref(err)):
                    return answer.reshape(b.shape, order="F"), err.value, False
            if self._direct is None:
                self._fallback()
            x, err, _ = self._direct.checked_solve(shift, b)
            return x, err, True
