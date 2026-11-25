"""Fixed-pattern SPD factorization and repeated diagonal-shift solves."""

import ctypes as ct
from functools import lru_cache
import math
from pathlib import Path
from threading import Lock, RLock
from weakref import finalize

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse import csc_matrix, diags, issparse, tril


_DLL = Path(__file__).resolve().parents[1] / ".research" / "bin" / "sparse.dll"
_UMFPACK = _DLL.with_name("libumfpack.dll")
_INTS = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
_VALUES = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
_FLOATS = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_CHOLESKY_MODES = {"auto": (1, 0, 0), "simplicial": (0, 0, 0), "supernodal": (2, 0, 0),
                   "postorder": (1, 1, 0), "supernodal-postorder": (2, 1, 0), "wide": (2, 1, 1),
                   "adaptive": (1, 0, 2)}


@lru_cache(maxsize=4)
def _library(path):
    native = ct.CDLL(str(path))
    native.wh_chol_plan_create.restype = ct.c_void_p
    native.wh_chol_plan_create.argtypes = [ct.c_int, _INTS, _INTS, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int)]
    native.wh_chol_plan_free.restype = None
    native.wh_chol_plan_free.argtypes = [ct.c_void_p]
    native.wh_chol_plan_factor.argtypes = [ct.c_void_p, _VALUES, ct.POINTER(ct.c_int)]
    native.wh_chol_plan_solve.argtypes = [ct.c_void_p, ct.c_size_t, _VALUES, _VALUES]
    return native


@lru_cache(maxsize=4)
def _updates_library(path):
    native = _library(path)
    native.wh_chol_plan_permutation.restype = None
    native.wh_chol_plan_permutation.argtypes = [ct.c_void_p, _INTS]
    native.wh_chol_plan_modify.argtypes = [ct.c_void_p, ct.c_int, ct.c_size_t, _INTS, _INTS, _VALUES]
    return native


@lru_cache(maxsize=4)
def _single_library(path):
    native = _library(path)
    native.wh_chol_plan_create_f32.restype = ct.c_void_p
    native.wh_chol_plan_create_f32.argtypes = native.wh_chol_plan_create.argtypes
    native.wh_chol_plan_factor_f32.argtypes = [ct.c_void_p, _FLOATS, ct.POINTER(ct.c_int)]
    native.wh_chol_plan_solve_f32.argtypes = [ct.c_void_p, ct.c_size_t, _FLOATS, _FLOATS]
    return native


@lru_cache(maxsize=4)
def _recovery_library(path):
    native = _updates_library(path)
    native.wh_extended_residual.argtypes = [ct.c_int, _INTS, _INTS, _VALUES, ct.c_size_t, ct.c_int,
                                            _INTS, _VALUES, _VALUES, _VALUES, _VALUES]
    return native


@lru_cache(maxsize=4)
def _tuned_library(path):
    native = _library(path)
    native.wh_chol_plan_create_tuned.restype = ct.c_void_p
    native.wh_chol_plan_create_tuned.argtypes = [*native.wh_chol_plan_create.argtypes[:-1],
                                                 ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_int)]
    native.wh_chol_plan_metrics.restype = None
    native.wh_chol_plan_metrics.argtypes = [ct.c_void_p, _VALUES]
    return native


def _canonical(matrix):
    if (not issparse(matrix) or matrix.ndim != 2 or np.iscomplexobj(matrix)
            or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1):
        raise ValueError("expected a nonempty square real sparse matrix")
    result = csc_matrix(matrix, dtype=np.float64, copy=False)
    result.check_format(full_check=True)
    if not result.has_canonical_format:
        result = result.copy()
        result.sum_duplicates()
    if result.indptr.dtype != np.int32 or result.indices.dtype != np.int32:
        raise ValueError("native factorization plans require 32-bit CSC indices")
    return result


def _real_array(value):
    if np.iscomplexobj(value):
        raise ValueError("numeric inputs must be real")
    return np.asarray(value, dtype=np.float64)


def _rhs(value, order):
    value = _real_array(value)
    if value.ndim not in (1, 2) or value.shape[0] != order or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("right-hand sides must be finite vectors or columns of matrix order")
    return value


def _ordering(permutation, order):
    indices = np.asarray(permutation)
    if (indices.shape != (order,) or indices.dtype.kind not in "iu"
            or not np.array_equal(np.sort(indices), np.arange(order))):
        raise ValueError("permutation must contain each index exactly once")
    return np.array(indices, dtype=np.int32)


class FixedPatternCholesky:
    """Retain CHOLMOD symbolic state for numeric versions of an upper CSC pattern.

    The constructor copies the pattern. Explicit zeros are part of that
    pattern. Pass a permutation to fix the ordering or let CHOLMOD use AMD.
    """

    def __init__(self, upper, *, permutation=None, library=_DLL, _single=False, _factorization="auto"):
        if _factorization not in _CHOLESKY_MODES or (_single and _factorization != "auto"):
            raise ValueError("unknown or incompatible CHOLMOD factorization mode")
        matrix = _canonical(upper)
        self.order = matrix.shape[0]
        columns = np.repeat(np.arange(self.order, dtype=np.int32), np.diff(matrix.indptr))
        if np.any(matrix.indices > columns):
            raise ValueError("supply the upper triangle of the symmetric matrix")
        self._column, self._row = matrix.indptr.copy(), matrix.indices.copy()
        if permutation is not None:
            ordering = _ordering(permutation, self.order)
            permutation = ordering.ctypes.data_as(ct.POINTER(ct.c_int32))
        self._native = _single_library(Path(library)) if _single else _library(Path(library))
        self._dtype = np.float32 if _single else np.float64
        create = self._native.wh_chol_plan_create_f32 if _single else self._native.wh_chol_plan_create
        self._evaluate = self._native.wh_chol_plan_factor_f32 if _single else self._native.wh_chol_plan_factor
        self._execute = self._native.wh_chol_plan_solve_f32 if _single else self._native.wh_chol_plan_solve
        status = ct.c_int()
        arguments = (self.order, self._column, self._row, permutation)
        if _factorization == "auto":
            self._handle = create(*arguments, ct.byref(status))
        else:
            tuned = _tuned_library(Path(library))
            self._handle = tuned.wh_chol_plan_create_tuned(*arguments, *_CHOLESKY_MODES[_factorization], ct.byref(status))
        if not self._handle:
            raise RuntimeError(f"CHOLMOD could not prepare the pattern (status {status.value})")
        self._lock = Lock()
        self._release = finalize(self, self._native.wh_chol_plan_free, self._handle)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def close(self):
        with self._lock:
            self._release()
            self._handle = None

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("the factorization plan is closed")

    def _metrics(self):
        with self._lock:
            self._require_open()
            stats = np.empty(5)
            _tuned_library(Path(self._native._name)).wh_chol_plan_metrics(self._handle, stats)
            return dict(zip(("flops", "factor_nnz", "is_super", "supernodes", "stored_entries"), stats.tolist()))

    def _factor(self, values):
        self._require_open()
        with np.errstate(over="ignore"):
            values = np.ascontiguousarray(_real_array(values), dtype=self._dtype)
        if values.shape != self._row.shape or not np.isfinite(values).all():
            raise ValueError("numeric values must be finite and match the frozen CSC pattern")
        status = ct.c_int()
        if not self._evaluate(self._handle, values, ct.byref(status)):
            raise np.linalg.LinAlgError(f"CHOLMOD could not factor the numeric matrix (status {status.value})")

    def factorize(self, upper):
        matrix = _canonical(upper)
        if (matrix.shape[0] != self.order or not np.array_equal(matrix.indptr, self._column)
                or not np.array_equal(matrix.indices, self._row)):
            raise ValueError("the input matrix has a different CSC pattern")
        with self._lock:
            self._factor(matrix.data)

    def _solve(self, rhs):
        self._require_open()
        rhs = _rhs(rhs, self.order)
        with np.errstate(over="ignore"):
            source = rhs.ravel(order="F").astype(self._dtype, copy=False)
        if self._dtype is np.float32 and not np.isfinite(source).all():
            raise np.linalg.LinAlgError("right-hand side exceeds binary32 range")
        answer = np.empty_like(source)
        if not self._execute(self._handle, rhs.size // self.order, source, answer):
            raise np.linalg.LinAlgError("CHOLMOD could not solve the factored system")
        if not np.isfinite(answer).all():
            raise np.linalg.LinAlgError("CHOLMOD produced a non-finite solution")
        return answer.reshape(rhs.shape, order="F").astype(np.float64, copy=False)

    def solve(self, rhs):
        with self._lock:
            return self._solve(rhs)

    def refactor_values_and_solve(self, values, rhs):
        """Atomically refactor values in the original canonical CSC order and solve."""
        with self._lock:
            self._factor(values)
            return self._solve(rhs)


class DiagonalShiftFamily:
    """Solve A + E diag(shifts) E.T for fixed SPD A and selected diagonal sites.

    Shifts are finite and nonnegative. The default Woodbury method stores
    one dense column per site. The update strategy modifies the sparse
    factor between requests. The mixed strategy uses a binary32 factor
    and binary64 iterative refinement. The recover strategy switches to
    corrected factor updates if the Woodbury result fails its check.
    """

    def __init__(self, upper, sites, *, permutation=None, library=_DLL, strategy="woodbury", _factorization="auto"):
        if strategy not in ("woodbury", "update", "mixed", "recover"):
            raise ValueError("unknown diagonal shift strategy")
        upper = _canonical(upper).copy()
        n = upper.shape[0]
        sites = np.asarray(sites)
        if (sites.ndim != 1 or not len(sites) or sites.dtype.kind not in "iu"
                or np.any(sites >= n) or np.any(sites < 0) or len(np.unique(sites)) != len(sites)):
            raise ValueError("diagonal sites must be distinct valid positions")
        self._sites = sites.astype(np.intp, copy=True)
        self._upper = upper
        self._permutation = None if permutation is None else np.array(permutation)
        self._library_path = library
        self._matrix = upper + tril(upper.T, k=-1, format="csc")
        self._row_norm = np.asarray(abs(self._matrix).sum(axis=1)).ravel()
        self._norm = float(self._row_norm.max())
        self._recover_updates = strategy == "recover"
        self._updates_active = False
        self._shift_lock = RLock() if self._recover_updates else Lock() if strategy == "update" else None
        self._scale = None
        self._setup_fallback = False
        if strategy == "mixed":
            self._scale = self._try_single_factor(upper)
            self._setup_fallback = self._scale is None
            if self._scale is not None:
                return
        self._plan = FixedPatternCholesky(upper, permutation=self._permutation, library=library,
                                          _factorization=_factorization)
        try:
            self._plan.factorize(upper)
            if strategy == "update":
                self._start_updates()
                return
            basis = np.zeros((n, len(sites)), order="F")
            basis[self._sites, np.arange(len(sites))] = 1
            self._responses = self._plan.solve(basis)
        except (AttributeError, RuntimeError, ValueError, np.linalg.LinAlgError):
            self._plan.close()
            raise
        gram = self._responses[self._sites, :]
        self._gram = gram / 2 + gram.T / 2

    def _start_updates(self):
        self._plan._require_open()
        path = Path(self._library_path)
        self._modifier = _recovery_library(path) if self._recover_updates else _updates_library(path)
        n = self._plan.order
        order = np.empty(n, dtype=np.int32)
        self._modifier.wh_chol_plan_permutation(self._plan._handle, order)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(n, dtype=np.int32)
        self._permuted_sites = inverse[self._sites]
        self._diagonal = self._upper.indptr[self._sites + 1] - 1
        self._base_diagonal = self._upper.data[self._diagonal]
        self._column = np.arange(len(self._sites) + 1, dtype=np.int32)
        self._native_sites = self._sites.astype(np.int32)
        self._shifts = np.zeros(len(self._sites))
        self._updates_active = True

    def _refine_update(self, answer, rhs, shifts):
        a = self._matrix
        if a.indptr.dtype != np.int32 or a.indices.dtype != np.int32:
            return answer
        residual = np.empty(rhs.size)
        if not self._modifier.wh_extended_residual(
            self._plan.order, a.indptr, a.indices, a.data, rhs.size // self._plan.order,
            len(shifts), self._native_sites, np.ascontiguousarray(shifts),
            rhs.ravel(order="F"), answer.ravel(order="F"), residual,
        ) or not np.isfinite(residual).all():
            return answer
        return answer + self._plan.solve(residual.reshape(rhs.shape, order="F"))

    def _try_single_factor(self, upper):
        diagonal = upper.diagonal()
        if (diagonal <= 0).any() or not np.isfinite(diagonal).all():
            return None
        scale = 1 / np.sqrt(diagonal)
        scaled = upper.copy()
        scaled.data *= scale[scaled.indices]
        scaled.data *= np.repeat(scale, np.diff(scaled.indptr))
        plan = None
        try:
            plan = FixedPatternCholesky(scaled, permutation=self._permutation,
                                       library=self._library_path, _single=True)
            plan.factorize(scaled)
            basis = np.zeros((plan.order, len(self._sites)), order="F")
            basis[self._sites, np.arange(len(self._sites))] = scale[self._sites]
            self._responses = scale[:, None] * plan.solve(basis)
        except (AttributeError, RuntimeError, ValueError, np.linalg.LinAlgError):
            if plan is not None:
                plan.close()
            return None
        self._plan = plan
        gram = self._responses[self._sites, :]
        self._gram = gram / 2 + gram.T / 2
        return scale[:, None]

    def close(self):
        if self._shift_lock is None:
            self._plan.close()
        else:
            with self._shift_lock:
                self._plan.close()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def _updated_solve(self, shifts, rhs, error):
        with self._shift_lock:
            self._plan._require_open()
            if self._shifts is not None:
                delta = shifts - self._shifts
                # A large change can lose small entries of the factor through
                # cancellation, even when the global residual is small.
                if not np.any(abs(delta) * np.sqrt(np.finfo(float).eps) > self._base_diagonal):
                    try:
                        for increase, use in ((True, delta > 0), (False, delta < 0)):
                            count = int(use.sum())
                            if count and not self._modifier.wh_chol_plan_modify(
                                self._plan._handle, increase, count, self._column[:count + 1],
                                self._permuted_sites[use], np.sqrt(abs(delta[use])),
                            ):
                                raise np.linalg.LinAlgError("CHOLMOD could not modify the factor")
                        answer = self._plan.solve(rhs)
                        if self._recover_updates:
                            answer = self._refine_update(answer, rhs, shifts)
                        residual, _ = error(answer)
                        if np.isfinite(residual) and residual <= 1e-11:
                            self._shifts = shifts.copy()
                            return answer, residual, False
                    except np.linalg.LinAlgError:
                        pass
            self._shifts = None
            values = self._upper.data.copy()
            values[self._diagonal] += shifts
            answer = self._plan.refactor_values_and_solve(values, rhs)
            residual, _ = error(answer)
            if not np.isfinite(residual) or residual > 1e-11:
                raise np.linalg.LinAlgError(f"shifted solve failed the backward-error check ({residual})")
            self._shifts = shifts.copy()
            return answer, residual, True

    def checked_solve(self, shifts, rhs):
        """Return a verified solution, its normwise backward error, and fallback status."""
        if self._recover_updates:
            with self._shift_lock:
                return self._checked_solve(shifts, rhs)
        return self._checked_solve(shifts, rhs)

    def _checked_solve(self, shifts, rhs):
        shifts = _real_array(shifts)
        if shifts.shape != self._sites.shape or np.any(shifts < 0) or not np.isfinite(shifts).all():
            raise ValueError("provide one finite nonnegative shift per diagonal site")
        rhs = _rhs(rhs, self._plan.order)
        b = rhs.reshape(len(rhs), -1)

        def error(answer):
            answer = answer.reshape(len(rhs), -1)
            discrepancy = self._matrix @ answer - b
            discrepancy[self._sites] += shifts[:, None] * answer[self._sites]
            norm = max(self._norm, float(np.max(self._row_norm[self._sites] + shifts)))
            scale = norm * np.max(abs(answer), axis=0) + np.max(abs(b), axis=0)
            residual = np.max(abs(discrepancy), axis=0)
            if not np.isfinite(scale).all():
                return np.inf, discrepancy
            return float(np.max(residual / np.where(scale > 0, scale, 1))), discrepancy

        with np.errstate(over="ignore", invalid="ignore"):
            if self._updates_active:
                return self._updated_solve(shifts, rhs, error)
            root = np.sqrt(shifts)
            small = (root[:, None] * self._gram) * root[None, :]
            small.flat[::len(root) + 1] += 1
            try:
                if np.isfinite(small).all():
                    l = np.linalg.cholesky(small)

                    def corrected(right):
                        if self._scale is not None:
                            scaled_right = self._scale * right
                            if not np.isfinite(scaled_right).all():
                                raise np.linalg.LinAlgError("scaled right-hand side overflowed")
                            z = self._scale * self._plan.solve(scaled_right)
                        else:
                            z = self._plan.solve(right)
                        local = root[:, None] * z[self._sites]
                        y = solve_triangular(l, local, lower=True, check_finite=False)
                        updated = solve_triangular(l, y, lower=True, trans="T", check_finite=False)
                        return z - self._responses @ (root[:, None] * updated)

                    answer = corrected(b)
                    previous = np.inf
                    for step in range(4 if self._scale is not None else 1):
                        residual, discrepancy = error(answer)
                        if np.isfinite(residual) and residual <= 1e-11:
                            return answer.reshape(rhs.shape), residual, self._setup_fallback
                        if self._scale is None or step == 3 or residual >= previous:
                            break
                        previous = residual
                        answer += corrected(-discrepancy)
            except np.linalg.LinAlgError:
                pass
            if self._recover_updates:
                self._start_updates()
                self._responses = self._gram = None
                return self._updated_solve(shifts, rhs, error)
            diagonal = np.zeros(len(rhs))
            diagonal[self._sites] = shifts
            shifted = self._upper + diags(diagonal, format="csc")
            with FixedPatternCholesky(shifted, permutation=self._permutation, library=self._library_path) as plan:
                plan.factorize(shifted)
                answer = plan.solve(rhs)
            residual, _ = error(answer)
        if not np.isfinite(residual) or residual > 1e-11:
            raise np.linalg.LinAlgError(f"shifted solve failed the backward-error check ({residual})")
        return answer, residual, True


@lru_cache(maxsize=4)
def _umfpack(path):
    native = ct.CDLL(str(path))
    pointer = ct.POINTER(ct.c_void_p)
    native.umfpack_di_defaults.argtypes = [_VALUES]
    native.umfpack_di_defaults.restype = None
    native.umfpack_di_qsymbolic.argtypes = [ct.c_int, ct.c_int, _INTS, _INTS, _VALUES,
                                           _INTS, pointer, _VALUES, ct.c_void_p]
    native.umfpack_di_numeric.argtypes = [_INTS, _INTS, _VALUES, ct.c_void_p, pointer, _VALUES, ct.c_void_p]
    native.umfpack_di_solve.argtypes = [ct.c_int, _INTS, _INTS, _VALUES, _VALUES, _VALUES,
                                       ct.c_void_p, _VALUES, ct.c_void_p]
    for routine in (native.umfpack_di_free_symbolic, native.umfpack_di_free_numeric):
        routine.argtypes = [pointer]
        routine.restype = None
    return native


class FixedPatternLU:
    """Apply the same column ordering and UMFPACK symbolic plan to new CSC values."""

    def __init__(self, matrix, *, permutation, reuse_symbolic=True, library=_UMFPACK):
        matrix = _canonical(matrix)
        self._order = matrix.shape[0]
        self._permutation = _ordering(permutation, self._order)
        self._column, self._row = matrix.indptr.copy(), matrix.indices.copy()
        self._native = _umfpack(Path(library))
        self._control = np.empty(20, dtype=np.float64)  # UMFPACK_CONTROL for the real/int interface.
        self._native.umfpack_di_defaults(self._control)
        self._control[5] = 1  # UMFPACK_STRATEGY_UNSYMMETRIC
        self._control[13] = 1  # UMFPACK_FIXQ
        self._lock = Lock()
        self._closed = False
        self._symbolic = self._analyze(np.ascontiguousarray(matrix.data)) if reuse_symbolic else None
        self._release = (finalize(self, self._native.umfpack_di_free_symbolic, ct.byref(self._symbolic))
                         if self._symbolic is not None else None)

    def _analyze(self, values):
        if not np.isfinite(values).all():
            raise ValueError("symbolic setup requires a finite numeric matrix")
        symbolic = ct.c_void_p()
        status = self._native.umfpack_di_qsymbolic(self._order, self._order, self._column, self._row, values,
                                                  self._permutation, ct.byref(symbolic), self._control, None)
        if status != 0:
            self._native.umfpack_di_free_symbolic(ct.byref(symbolic))
            raise np.linalg.LinAlgError(f"UMFPACK symbolic analysis failed (status {status})")
        return symbolic

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def close(self):
        with self._lock:
            if self._release is not None:
                self._release()
            self._closed = True

    def refactor_values_and_solve(self, values, rhs, *, min_pivot_ratio=0.0):
        """Factor values and optionally reject a small UMFPACK U-diagonal ratio.

        The ratio is a conditioning diagnostic after UMFPACK's scaling. It
        measures the pivot spread and cannot certify forward accuracy.
        """
        if not math.isfinite(min_pivot_ratio) or not 0 <= min_pivot_ratio <= 1:
            raise ValueError("the minimum pivot ratio must be in [0, 1]")
        with self._lock:
            if self._closed:
                raise RuntimeError("the LU plan is closed")
            values, rhs = np.ascontiguousarray(_real_array(values)), np.ascontiguousarray(_rhs(rhs, self._order))
            if values.shape != self._row.shape or not np.isfinite(values).all() or rhs.ndim != 1:
                raise ValueError("LU requires finite values in the frozen CSC order and one RHS vector")
            symbolic = self._symbolic if self._symbolic is not None else self._analyze(values)
            numeric = ct.c_void_p()
            try:
                info = np.empty(90, dtype=np.float64) if min_pivot_ratio else None  # UMFPACK_INFO
                status = self._native.umfpack_di_numeric(self._column, self._row, values, symbolic,
                                                         ct.byref(numeric), self._control, info.ctypes.data if info is not None else None)
                if status != 0:
                    raise np.linalg.LinAlgError(f"UMFPACK numeric factorization failed (status {status})")
                if info is not None and not info[67] >= min_pivot_ratio:  # UMFPACK_RCOND
                    raise np.linalg.LinAlgError(f"UMFPACK pivot ratio {info[67]:.3g} is below {min_pivot_ratio:.3g}")
                answer = np.empty_like(rhs)
                status = self._native.umfpack_di_solve(0, self._column, self._row, values, answer, rhs,
                                                       numeric, self._control, None)
                if status != 0 or not np.isfinite(answer).all():
                    raise np.linalg.LinAlgError(f"UMFPACK solve failed (status {status})")
                return answer
            finally:
                self._native.umfpack_di_free_numeric(ct.byref(numeric))
                if self._symbolic is None:
                    self._native.umfpack_di_free_symbolic(ct.byref(symbolic))
