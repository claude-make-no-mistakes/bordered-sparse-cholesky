import ctypes
import json
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.sparse import csc_matrix, triu

from benchmarks.common import BIN
from benchmarks.sparse import FactorStats, Native, _adjacency, run_method
from workhorses.sparse import DiagonalShiftFamily, FixedPatternCholesky, FixedPatternLU
from workhorses.sparse_iterative import IncompleteShiftFamily
from workhorses.sparse_border import BorderedShiftFamily


def symbolic_fill(a, ordering):
    graph = [set(np.flatnonzero(row)) - {i} for i, row in enumerate(a)]
    nnz, operations = 0, 0
    for pivot in ordering:
        remaining = graph[pivot]
        length = len(remaining) + 1
        nnz += length
        operations += length ** 2
        for vertex in remaining:
            graph[vertex].update(remaining - {vertex})
            graph[vertex].discard(pivot)
        graph[pivot] = set()
    return nnz, operations


@unittest.skipUnless((BIN / "sparse.dll").is_file(), "build native sparse tools with python -m benchmarks.build sparse")
class SparseTests(unittest.TestCase):
    def test_bordered_tail_matches_manufactured_rhs_in_both_factor_layouts(self):
        # The disconnected last component creates structural zeros in the border.
        a = np.array([[5., .5, -.25, 0], [.5, 4., 2., 0], [-.25, 2., 6., 0], [0, 0, 0, 1.]])
        expected = np.array([[1., -2.], [-.5, 1.5], [.3, 2.], [0., 7.]])
        sites = [3, 0, 2]
        for mode in ("simplicial", "supernodal"):
            with BorderedShiftFamily(csc_matrix(np.triu(a)), sites, _factorization=mode) as plan:
                for shifts in ([0., 1e6, 4.], [3., .25, 0.], [0., 0., 0.]):
                    matrix = a.copy()
                    matrix[sites, sites] += shifts
                    # Multiplication by the explicitly shifted matrix generates both RHS columns.
                    result, error, fallback = plan.checked_solve(shifts, matrix @ expected)
                    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
                    self.assertFalse(fallback)
                    self.assertLessEqual(error, 1e-11)
                with self.assertRaises(ValueError):
                    plan.checked_solve([0., -1., 0.], expected)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                plan.checked_solve([0., 0., 0.], expected)

    def test_wide_border_uses_checked_direct_fallback(self):
        # For a diagonal matrix the solution is the RHS divided elementwise by its diagonal.
        with BorderedShiftFamily(csc_matrix(np.eye(33)), np.arange(33)) as plan:
            x, err, fallback = plan.checked_solve(np.arange(33), np.ones(33))
            np.testing.assert_allclose(x, 1 / np.arange(1, 34), atol=1e-14)
            self.assertLessEqual(err, 1e-11)
            self.assertTrue(fallback)

    def test_bordered_updates_across_different_site_counts_and_permutations(self):
        rng = np.random.default_rng(69421)
        for n, k in ((9, 1), (13, 8), (17, 17)):
            entries = rng.normal(size=(n, n)) * (rng.random((n, n)) < .20)
            a = entries @ entries.T + 2 * np.eye(n)
            sites = rng.choice(n, k, replace=False)
            order = rng.permutation(n)
            rhs = rng.normal(size=(n, 3))
            for mode in ("simplicial", "supernodal"):
                with BorderedShiftFamily(csc_matrix(np.triu(a)), sites, permutation=order,
                                         _factorization=mode) as plan:
                    for shift in (np.geomspace(1e-8, 1e6, k), np.zeros(k), np.linspace(10., 0., k)):
                        changed = a.copy()
                        changed[sites, sites] += shift
                        # Independent dense LAPACK solves use the original order and all three RHS columns.
                        expected = np.linalg.solve(changed, rhs)
                        answer, error, fallback = plan.checked_solve(shift, rhs)
                        np.testing.assert_allclose(answer, expected, atol=2e-13, rtol=2e-13)
                        self.assertLessEqual(error, 1e-11)
                        self.assertFalse(fallback)

    def test_sparse_recovery_reuses_a_factor_after_a_failed_woodbury_check(self):
        matrix = np.array([[4.2, .2, -.8], [.2, 3.1, .3], [-.8, .3, 5.4]])
        upper = csc_matrix(np.triu(matrix))
        expected = np.array([[1., 2.], [.5, -1.], [-3., .3]])
        with DiagonalShiftFamily(upper, [0, 2], permutation=[2, 0, 1],
                                 strategy="recover", _factorization="adaptive") as solver:
            # Force the quality gate on the model while the base factor
            # remains valid. Subsequent solves must use the same factor.
            solver._responses *= 32
            for shifts in ([2., 1.], [.5, 1.6]):
                rhs = (matrix + np.diag([shifts[0], 0, shifts[1]])) @ expected
                answer, error, fallback = solver.checked_solve(shifts, rhs)
                self.assertTrue(solver._updates_active)
                self.assertLessEqual(error, 1e-11)
                self.assertFalse(fallback)
                np.testing.assert_allclose(answer, expected, atol=1e-13, rtol=1e-13)

    def test_incomplete_solver_and_chronological_direct_fallback(self):
        matrix = np.array([[4., -1., 0., 0.], [-1., 5., -1., 0.], [0., -1., 6., 2.], [0., 0., 2., 7.]])
        upper = csc_matrix(np.triu(matrix))
        sites = [0, 3]
        shifts = np.array([5., 2.])
        expected = np.array([[1., -1.], [2., .3], [0., 1.5], [-3., 4.]])
        rhs = (matrix + np.diag([shifts[0], 0, 0, shifts[1]])) @ expected
        for budget in (1, 24):
            with IncompleteShiftFamily(upper, sites, max_iterations=budget) as solver:
                for _ in range(2):
                    actual, error, fallback = solver.checked_solve(shifts, rhs)
                    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
                    self.assertLessEqual(error, 1e-11)
                    self.assertEqual(fallback, budget == 1)
                    self.assertEqual(solver.last_iterations is None, budget == 1)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                solver.checked_solve(shifts, rhs)

    def test_every_truncated_interval_brackets_enumerated_missing_edges(self):
        subprocess.run([str(BIN / "amd-bounds-test.exe")], check=True, timeout=5)

    def test_fixed_pattern_recovers_after_numeric_failure_and_checks_shape(self):
        matrix = np.array([[4, 1, 0], [1, 5, -2], [0, -2, 6]], dtype=float)
        upper = csc_matrix(np.triu(matrix))
        expected = np.array([1., -3., 2.])
        with FixedPatternCholesky(upper, permutation=[2, 0, 1]) as plan:
            plan.factorize(upper)
            # The manufactured solution uses the explicitly given SPD matrix.
            np.testing.assert_allclose(plan.solve(matrix @ expected), expected, rtol=1e-14)
            with self.assertRaises(ValueError):
                plan.factorize(csc_matrix(np.eye(3)))
            with self.assertRaises(np.linalg.LinAlgError):
                plan.refactor_values_and_solve(-upper.data, matrix @ expected)
            with self.assertRaises(np.linalg.LinAlgError):
                plan.solve(matrix @ expected)
            # Rescaling by 1.5 preserves SPD and the stored pattern.
            solved = plan.refactor_values_and_solve(1.5 * upper.data, 1.5 * matrix @ expected)
            np.testing.assert_allclose(solved, expected, rtol=1e-14)
        with self.assertRaises(RuntimeError):
            plan.solve(matrix @ expected)

    def test_checked_diagonal_shifts_match_dense_reference_for_multiple_rhs(self):
        matrix = np.array([[4, 1, 0], [1, 5, -2], [0, -2, 6]], dtype=float)
        rhs = np.array([[1, 2], [0, 1], [5, 1]], dtype=float)
        with DiagonalShiftFamily(csc_matrix(np.triu(matrix)), [2, 0], permutation=[2, 0, 1]) as family:
            for shifts in ([0., 0.], [0.7, 3.], [1e-6, 1e4]):
                actual, error, fallback = family.checked_solve(shifts, rhs)
                expected = np.linalg.solve(matrix + np.diag([shifts[1], 0, shifts[0]]), rhs)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)
                self.assertLess(error, 1e-11)
                self.assertFalse(fallback)
            with self.assertRaises(ValueError):
                family.checked_solve([-1., 0.], rhs)

    def test_diagonal_shift_falls_back_when_unshifted_solution_overflows(self):
        with DiagonalShiftFamily(csc_matrix([[1e-200]]), [0]) as family:
            answer, error, fallback = family.checked_solve([1.0], [1e150])
        # The shifted scalar factor is 1 in FP64, and its finite solution is 1e150.
        np.testing.assert_allclose(answer, [1e150], rtol=1e-14)
        self.assertLess(error, 1e-11)
        self.assertTrue(fallback)

    def test_sparse_updates_handle_mixed_steps_and_refactor_large_changes(self):
        matrix = np.array([[5., -2., .25], [-2., 4., -1.], [.25, -1., 3.]])
        target = np.array([[1., -2.], [.5, 1.], [-1., 3.]])
        with DiagonalShiftFamily(csc_matrix(np.triu(matrix)), [2, 0], permutation=[2, 0, 1],
                                 strategy="update") as family:
            conditions = [([2., 4.], False), ([3., 1.], False), ([1e60, 2e10], True),
                          ([0., 0.], True), ([2., 1.], False)]
            for shifts, should_refactor in conditions:
                shifted = matrix + np.diag([shifts[1], 0., shifts[0]])
                answer, backward, fallback = family.checked_solve(shifts, shifted @ target)
                np.testing.assert_allclose(answer, target, rtol=1e-12)
                self.assertLess(backward, 1e-11)
                self.assertEqual(fallback, should_refactor)
        with self.assertRaises(RuntimeError):
            family.checked_solve([0., 0.], matrix @ target)

    def test_mixed_precision_refines_both_right_hand_sides_in_binary64(self):
        matrix = np.array([[5.1, -.8, .15], [-.8, 4.3, -.7], [.15, -.7, 2.9]])
        target = np.array([[1., -2.], [.5, 1.], [-1., 3.]])
        shifts = [7.1, .03]
        rhs = (matrix + np.diag([shifts[0], 0., shifts[1]])) @ target
        with DiagonalShiftFamily(csc_matrix(np.triu(matrix)), [0, 2], strategy="mixed") as family:
            answer, backward, fallback = family.checked_solve(shifts, rhs)
        np.testing.assert_allclose(answer, target, rtol=1e-11)
        self.assertLess(backward, 1e-11)
        self.assertFalse(fallback)

    def test_reused_unsymmetric_plan_checks_numeric_failure_and_solves_changed_values(self):
        matrix = np.array([[7, 0, 2], [1, 3, 2], [0, 1, 5]], dtype=float)
        given = csc_matrix(matrix)
        expected = np.array([1., 2., -3.])
        diagonal = np.array([0.8, 1.2, 1.1])
        for cached in (False, True):
            with FixedPatternLU(given, permutation=[2, 0, 1], reuse_symbolic=cached) as plan:
                with self.assertRaises(np.linalg.LinAlgError):
                    plan.refactor_values_and_solve(np.zeros_like(given.data), expected)
                changed = csc_matrix(diagonal[:, None] * matrix)
                # Positive row scaling keeps the actual system invertible and
                # produces a manufactured right-hand side for the same x.
                answer = plan.refactor_values_and_solve(changed.data, changed @ expected)
                np.testing.assert_allclose(answer, expected, rtol=1e-14)

    def test_lu_conditioning_guard_rejects_nearly_identical_rows(self):
        # The second row differs at only one entry, by about 1e-14.
        matrix = csc_matrix([[1.0, 1.0], [1.0, 1.0 + 1e-14]])
        with FixedPatternLU(matrix, permutation=[0, 1]) as plan:
            with self.assertRaisesRegex(np.linalg.LinAlgError, "pivot ratio"):
                plan.refactor_values_and_solve(matrix.data, [1.0, 2.0], min_pivot_ratio=1e-12)

    def test_exact_symbolic_fill_and_numerical_invariance_under_relabeling(self):
        native = Native()
        rng = np.random.default_rng(721)
        for n in (7, 15, 37):
            edges = np.triu(rng.random((n, n)) < 0.15, 1)
            pattern = edges | edges.T
            full = pattern.astype(float)
            np.fill_diagonal(full, full.sum(axis=0) + 1)
            a = csc_matrix(full)
            upper = triu(a, format="csc")
            graph = _adjacency(a)
            rhs = a @ rng.normal(size=n)
            norm = np.linalg.norm(full, ord=np.inf)
            for method in ("amd", "hook-off", "exact-local-fill", "bounded-64", "metis"):
                kwargs = (a, upper, graph, rhs, norm)
                result = run_method(native, method, "spd", *kwargs)
                # The matrix is strictly diagonally dominant and symmetric, hence SPD.
                self.assertTrue(result["verified"], (n, method, result))
                self.assertEqual(result["permutation_match"], 1)
                if method == "amd":
                    baseline = result["permutation_sha256"]
                if method == "hook-off":
                    self.assertEqual(result["permutation_sha256"], baseline)
            pattern_order = np.arange(n, dtype=np.int32)[::-1].copy()
            stats = FactorStats()
            vector = np.empty(n)
            self.assertEqual(native.cholesky(n, upper.indptr, upper.indices, upper.data, rhs, pattern_order,
                                             vector, ctypes.byref(stats)), 1)
            # Graph clique elimination counts the exact symbolic column lengths.
            self.assertEqual((stats.factor_nnz, stats.flops), symbolic_fill(full, pattern_order))




if __name__ == "__main__":
    unittest.main()
