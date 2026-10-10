/* row_scaling.c
 *
 * Row equilibration: builds the row-coordinate map
 *
 *   d[i]   = 1.0 / ||A[i, :]||_2
 *   D      = diag(d)
 *   G      = D A A^T D                 (m x m symmetric dense, host)
 *   (w, V) = eigh(G)
 *   W      = V diag(1 / sqrt(w)) V^T   (host product)
 *   M      = W * diag(d)               (column scaling, row map)
 *   A_new  = M @ A_old                 (dense output, host)
 *   b_new  = M @ b_old
 *   dual_start_new = M^{-T} @ dual_start_old
 *
 * Behaviour matches the Python reference `row_whitening.py`
 * `whiten_equalities`:
 *   - M = (D A A^T D)^(-1/2) D
 *   - G computed column-bucket outer product (correct formula)
 *   - A_new is full dense product (m * n scratch) then thresholded
 *     below 1e-12 * |max| to mimic `A_new.eliminate_zeros()`
 *   - M is stashed in a thread-local handle so the original-space dual
 *     can be recovered at the end of optimize() via
 *     y_orig = M^T y_new.
 *
 * The cuSOLVER eigendecomposition and the dense outer product limit this
 * to m <= row_scaling_max_rows (default 2000).  The C-side guard is in
 * solver.cu: row_scaling is attempted iff
 *   params->enable_row_scaling && num_constraints <= row_scaling_max_rows.
 *
 * CHANGELOG (vs the original buggy version):
 *   1. G outer product fix: removed the extraneous `* d[j]`
 *      (vi already includes d[i] and col_vals[q] already includes d[j],
 *      so `vi * col_vals[q]` alone equals (D A A^T D)[i,j]).
 *   2. A_new dense product is now followed by an `eliminate_zeros`
 *      step (Python: `A_new.eliminate_zeros()`) so the new CSR
 *      does not keep every roundoff-flushed entry.
 *   3. Return status enum values are aligned with `internal/row_scaling.h`
 *      (PDHCG_ROW_SCALING_APPLIED, _DISABLED, _RANK_DEFICIENT, _ERROR,
 *      _NOT_EQUALITY).
 *   4. Function signatures match `internal/row_scaling.h`.
 */

#include "row_scaling.h"
#include "internal_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define ROW_SCALING_REL_RANK_THRESHOLD 1e-12

const char *pdhcg_row_scaling_status_str(pdhcg_row_scaling_status_t s)
{
    switch (s) {
        case PDHCG_ROW_SCALING_APPLIED:       return "applied";
        case PDHCG_ROW_SCALING_DISABLED:      return "disabled";
        case PDHCG_ROW_SCALING_NOT_EQUALITY:  return "non-equality rows";
        case PDHCG_ROW_SCALING_RANK_DEFICIENT:return "rank deficient";
        case PDHCG_ROW_SCALING_ERROR:         return "error";
    }
    return "unknown";
}

#define CUDA_OK(call)                                                            \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            fprintf(stderr, "[row_scaling] CUDA error %s at %s:%d\n",             \
                    cudaGetErrorString(_e), __FILE__, __LINE__);                  \
            return PDHCG_ROW_SCALING_ERROR;                                      \
        }                                                                         \
    } while (0)

#define CUSOLVER_OK(call)                                                         \
    do {                                                                          \
        cusolverStatus_t _s = (call);                                             \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "[row_scaling] cuSOLVER error %d at %s:%d\n",         \
                    (int)_s, __FILE__, __LINE__);                                 \
            return PDHCG_ROW_SCALING_ERROR;                                      \
        }                                                                         \
    } while (0)

/* ---- thread-local pending M (survives rescale_info_free()) ----
 *
 * The map M is needed at the very end of optimize() to convert
 *   y_new = dual after solve
 * back to the user's original coordinate convention
 *   y_orig = M^T y_new.
 *
 * rescale_info_t is freed earlier in optimize() (before iteration
 * starts), so the map has to live in thread-local storage until
 * pdhcg_recover_dual_from_pending_row_scaling() runs.
 */
static _Thread_local double *g_pending_M = NULL;
static _Thread_local int      g_pending_M_size = 0;

void pdhcg_release_pending_row_scaling(void)
{
    if (g_pending_M) {
        free(g_pending_M);
        g_pending_M = NULL;
    }
    g_pending_M_size = 0;
}

void pdhcg_recover_dual_from_pending_row_scaling(double *dual_solution)
{
    if (!g_pending_M || g_pending_M_size <= 0 || !dual_solution) return;
    const int m = g_pending_M_size;
    /* y_orig = M^T y_new.  Allocate a small scratch for the result. */
    double *y_orig = (double *)malloc((size_t)m * sizeof(double));
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int j = 0; j < m; ++j) s += g_pending_M[j * m + i] * dual_solution[j];
        y_orig[i] = s;
    }
    memcpy(dual_solution, y_orig, (size_t)m * sizeof(double));
    free(y_orig);
}

void pdhcg_recover_dual_from_row_scaling(rescale_info_t *info, double *dual_solution)
{
    /* Legacy path: read M from the rescale_info (if it was stashed there)
     * or fall back to the thread-local handle.  The current C code does
     * not stash M on rescale_info, so this routes through the thread-local
     * handle. */
    (void)info;
    pdhcg_recover_dual_from_pending_row_scaling(dual_solution);
}

void pdhcg_row_scaling_info_clear(rescale_info_t *info)
{
    (void)info;
    pdhcg_release_pending_row_scaling();
}

static bool all_rows_are_equalities(const qp_problem_t *problem)
{
    if (!problem || !problem->constraint_lower_bound || !problem->constraint_upper_bound) {
        return false;
    }
    for (int i = 0; i < problem->num_constraints; ++i) {
        double l = problem->constraint_lower_bound[i];
        double u = problem->constraint_upper_bound[i];
        if (!isfinite(l) || !isfinite(u) || l != u) return false;
    }
    return true;
}

static int compute_row_scaling_and_transform(qp_problem_t *problem, double **out_M)
{
    const int m = problem->num_constraints;
    const int n = problem->num_variables;
    CsrComponent *A = problem->constraint_matrix;
    if (!A || !A->row_ptr || !A->col_ind || !A->val) {
        return PDHCG_ROW_SCALING_ERROR;
    }

    fprintf(stderr, "[row_scaling] start, m=%d, n=%d, nnz=%d\n",
            m, n, problem->constraint_matrix_num_nonzeros);

    /* ---- 1. row norms -> d (host) ---- */
    double *d = (double *)malloc(m * sizeof(double));
    for (int i = 0; i < m; ++i) {
        double s2 = 0.0;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k) {
            double v = A->val[k];
            s2 += v * v;
        }
        double r = sqrt(s2);
        if (!isfinite(r) || r <= 0.0) {
            free(d);
            return PDHCG_ROW_SCALING_ERROR;
        }
        d[i] = 1.0 / r;
    }

    /* ---- 2. G = D A A^T D (m x m dense, host, symmetric) ----
     *
     * Bug fix: the previous C implementation multiplied
     *     G[i,j] += vi * col_vals[q] * d[j]
     * which double-counted d[j] (col_vals already includes d[j]).  The
     * correct formula is
     *     G[i,j] += vi * col_vals[q]
     * because vi = A[i,k]*d[i] and col_vals[q] = A[j,k]*d[j], giving
     *     G[i,j] = sum_k A[i,k]*A[j,k]*d[i]*d[j] = (D A A^T D)[i,j].
     */
    double *G = (double *)calloc((size_t)m * m, sizeof(double));
    {
        int nnz_total = A->row_ptr[m];
        int *col_offsets = (int *)calloc(n + 1, sizeof(int));
        int *col_rows = (int *)malloc((size_t)nnz_total * sizeof(int));
        double *col_vals = (double *)malloc((size_t)nnz_total * sizeof(double));

        /* Count nnz per column. */
        for (int i = 0; i < m; ++i) {
            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k) {
                col_offsets[A->col_ind[k]]++;
            }
        }
        /* Prefix sum. */
        int total = 0;
        for (int k = 0; k < n; ++k) {
            int c = col_offsets[k];
            col_offsets[k] = total;
            total += c;
        }
        col_offsets[n] = total;
        /* Scatter (row, val*d[row]) into per-column buffers. */
        int *cursor = (int *)malloc(n * sizeof(int));
        memcpy(cursor, col_offsets, n * sizeof(int));
        for (int i = 0; i < m; ++i) {
            double di = d[i];
            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k) {
                int col = A->col_ind[k];
                int dst = cursor[col]++;
                col_rows[dst] = i;
                col_vals[dst] = A->val[k] * di;
            }
        }
        free(cursor);
        /* Outer product per column: G[i,j] += vi * vj (no extra d[j]). */
        for (int k = 0; k < n; ++k) {
            int start = col_offsets[k];
            int end = col_offsets[k + 1];
            for (int p = start; p < end; ++p) {
                int i = col_rows[p];
                double vi = col_vals[p];
                for (int q = start; q < end; ++q) {
                    int j = col_rows[q];
                    G[i * m + j] += vi * col_vals[q];
                }
            }
        }
        free(col_offsets);
        free(col_rows);
        free(col_vals);
    }

    /* ---- 3. eigendecomposition via cuSOLVER (GPU) ---- */
    cublasHandle_t cublas_h = NULL;
    cusolverDnHandle_t cusolver_h = NULL;
    cublasCreate(&cublas_h);
    cusolverDnCreate(&cusolver_h);

    double *d_G = NULL;
    double *d_w = NULL;
    double *d_work = NULL;
    int *d_Ipiv = NULL;

    CUDA_OK(cudaMalloc((void **)&d_G, (size_t)m * m * sizeof(double)));
    CUDA_OK(cudaMalloc((void **)&d_w, m * sizeof(double)));
    CUDA_OK(cudaMalloc((void **)&d_Ipiv, m * sizeof(int)));

    CUDA_OK(cudaMemcpy(d_G, G, (size_t)m * m * sizeof(double), cudaMemcpyHostToDevice));

    int lwork = 0;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    CUSOLVER_OK(cusolverDnDsyevd_bufferSize(cusolver_h, jobz, uplo, m, d_G, m, d_w, &lwork));
    CUDA_OK(cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double)));

    /* cuSOLVER overwrites the input matrix with eigenvectors on success. */
    CUSOLVER_OK(cusolverDnDsyevd(cusolver_h, jobz, uplo, m, d_G, m, d_w, d_work, lwork, d_Ipiv));

    /* ---- 4. rank check (host) ---- */
    double *w = (double *)malloc(m * sizeof(double));
    CUDA_OK(cudaMemcpy(w, d_w, m * sizeof(double), cudaMemcpyDeviceToHost));

    {
        double wmax = w[m - 1];
        if (!(wmax > 0.0)) {
            fprintf(stderr, "[row_scaling] all Gram eigenvalues <= 0 (rejected)\n");
            free(d); free(G); free(w);
            cudaFree(d_G); cudaFree(d_w); cudaFree(d_Ipiv); cudaFree(d_work);
            cublasDestroy(cublas_h); cusolverDnDestroy(cusolver_h);
            return PDHCG_ROW_SCALING_RANK_DEFICIENT;
        }
        int rank = 0;
        double wmin_pos = wmax;
        for (int k = 0; k < m; ++k) {
            double wk = w[k];
            if (wk < 0.0 && wk > -1e-9 * wmax) wk = 0.0;
            if (wk > 1e-12 * wmax) {
                rank++;
                if (wk < wmin_pos) wmin_pos = wk;
            }
        }
        if (rank < m) {
            fprintf(stderr, "[row_scaling] Gram rank=%d/%d, min_pos=%.3g (rejected: rank-deficient)\n",
                    rank, m, wmin_pos / wmax);
            free(d); free(G); free(w);
            cudaFree(d_G); cudaFree(d_w); cudaFree(d_Ipiv); cudaFree(d_work);
            cublasDestroy(cublas_h); cusolverDnDestroy(cusolver_h);
            return PDHCG_ROW_SCALING_RANK_DEFICIENT;
        }
        if (!(wmin_pos > ROW_SCALING_REL_RANK_THRESHOLD * wmax)) {
            fprintf(stderr, "[row_scaling] Gram min_pos/max = %.3g (rejected)\n",
                    wmin_pos / wmax);
            free(d); free(G); free(w);
            cudaFree(d_G); cudaFree(d_w); cudaFree(d_Ipiv); cudaFree(d_work);
            cublasDestroy(cublas_h); cusolverDnDestroy(cusolver_h);
            return PDHCG_ROW_SCALING_RANK_DEFICIENT;
        }
    }

    /* ---- 5. Build W = V diag(1/sqrt(w)) V^T (host, row-major) ---- */
    double *V_rm = (double *)malloc((size_t)m * m * sizeof(double));
    {
        double *V_cm = (double *)malloc((size_t)m * m * sizeof(double));
        CUDA_OK(cudaMemcpy(V_cm, d_G, (size_t)m * m * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                V_rm[i * m + j] = V_cm[j * m + i];
        free(V_cm);
    }
    for (int j = 0; j < m; ++j) {
        double s = 1.0 / sqrt(w[j]);
        for (int i = 0; i < m; ++i) V_rm[i * m + j] *= s;
    }
    double *W = (double *)malloc((size_t)m * m * sizeof(double));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int k = 0; k < m; ++k) s += V_rm[i * m + k] * V_rm[j * m + k];
            W[i * m + j] = s;
        }
    }
    free(V_rm);

    /* ---- 6. M = W * diag(d) ---- */
    double *M = (double *)malloc((size_t)m * m * sizeof(double));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            M[i * m + j] = W[i * m + j] * d[j];
    free(W);

    /* ---- 7. Rewrite A in place: A_new = M @ A_old (dense) ---- */
    {
        double *A_old_dense = (double *)calloc((size_t)m * n, sizeof(double));
        for (int i = 0; i < m; ++i)
            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; ++k)
                A_old_dense[i * n + A->col_ind[k]] = A->val[k];

        double *A_new_dense = (double *)calloc((size_t)m * n, sizeof(double));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < m; ++k)
                    s += M[i * m + k] * A_old_dense[k * n + j];
                A_new_dense[i * n + j] = s;
            }
        }

        /* eliminate_zeros equivalent: threshold dense output. */
        double out_max = 0.0;
        for (size_t k = 0; k < (size_t)m * n; ++k)
            if (fabs(A_new_dense[k]) > out_max) out_max = fabs(A_new_dense[k]);
        double out_tol = (out_max > 0.0) ? 1e-12 * out_max : 0.0;
        if (out_tol > 0.0) {
            for (size_t k = 0; k < (size_t)m * n; ++k)
                if (fabs(A_new_dense[k]) < out_tol) A_new_dense[k] = 0.0;
        }

        int *rp2 = (int *)malloc((m + 1) * sizeof(int));
        rp2[0] = 0;
        int nnz = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j)
                if (A_new_dense[i * n + j] != 0.0) ++nnz;
            rp2[i + 1] = nnz;
        }
        int *ci = (int *)malloc(nnz * sizeof(int));
        double *vv = (double *)malloc(nnz * sizeof(double));
        int write = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                double v = A_new_dense[i * n + j];
                if (v != 0.0) { ci[write] = j; vv[write] = v; ++write; }
            }

        free(A_old_dense);
        free(A_new_dense);
        free(A->row_ptr); free(A->col_ind); free(A->val);
        A->row_ptr = rp2;
        A->col_ind = ci;
        A->val = vv;
        problem->constraint_matrix_num_nonzeros = nnz;
        fprintf(stderr, "[row_scaling] A_new nnz=%d (density=%.4f), tol=%.3e\n",
                nnz, (double)nnz / ((double)m * (double)n), out_tol);
    }

    /* ---- 8. b_new = M @ b_old ---- */
    {
        double *b_new = (double *)malloc(m * sizeof(double));
        for (int i = 0; i < m; ++i) {
            double s = 0.0;
            for (int j = 0; j < m; ++j)
                s += M[i * m + j] * problem->constraint_lower_bound[j];
            b_new[i] = s;
        }
        memcpy(problem->constraint_lower_bound, b_new, m * sizeof(double));
        memcpy(problem->constraint_upper_bound, b_new, m * sizeof(double));
        free(b_new);
    }

    /* ---- 9. dual_start = M^{-T} @ dual_start_old (host LU-style) ---- */
    if (problem->dual_start) {
        /* M is symmetric, so M^{-T} = M^{-1}.  Use Gauss-Jordan. */
        double *aug = (double *)malloc((size_t)m * 2 * m * sizeof(double));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) aug[i * (2 * m) + j] = M[i * m + j];
            for (int j = 0; j < m; ++j) aug[i * (2 * m) + (m + j)] = (i == j) ? 1.0 : 0.0;
        }
        for (int k = 0; k < m; ++k) {
            double piv = aug[k * (2 * m) + k];
            if (fabs(piv) < 1e-30) {
                free(aug); free(M); free(d); free(G); free(w);
                cudaFree(d_G); cudaFree(d_w); cudaFree(d_Ipiv); cudaFree(d_work);
                cublasDestroy(cublas_h); cusolverDnDestroy(cusolver_h);
                return PDHCG_ROW_SCALING_ERROR;
            }
            for (int j = 0; j < 2 * m; ++j) aug[k * (2 * m) + j] /= piv;
            for (int i = 0; i < m; ++i) {
                if (i == k) continue;
                double factor = aug[i * (2 * m) + k];
                if (factor == 0.0) continue;
                for (int j = 0; j < 2 * m; ++j)
                    aug[i * (2 * m) + j] -= factor * aug[k * (2 * m) + j];
            }
        }
        double *y_new = (double *)malloc(m * sizeof(double));
        for (int i = 0; i < m; ++i) {
            double s = 0.0;
            for (int j = 0; j < m; ++j)
                s += aug[i * (2 * m) + (m + j)] * problem->dual_start[j];
            y_new[i] = s;
        }
        memcpy(problem->dual_start, y_new, m * sizeof(double));
        free(y_new);
        free(aug);
    }

    /* ---- 10. Save M for postsolve ---- */
    if (out_M) {
        *out_M = M;
    } else {
        free(M);
    }

    /* ---- 11. Cleanup ---- */
    free(d);
    free(G);
    free(w);
    cudaFree(d_G); cudaFree(d_w); cudaFree(d_Ipiv); cudaFree(d_work);
    cublasDestroy(cublas_h);
    cusolverDnDestroy(cusolver_h);
    return PDHCG_ROW_SCALING_APPLIED;
}

pdhcg_row_scaling_status_t pdhcg_apply_row_scaling(
    const pdhg_parameters_t *params,
    qp_problem_t *problem,
    rescale_info_t *rescale_info)
{
    /* Stash rescale_info in the thread-local handle; this is needed so
     * rescale_info_free() can find the row_scaling timer and not crash. */
    (void)rescale_info;
    if (!params || !problem) return PDHCG_ROW_SCALING_ERROR;
    if (!params->enable_row_scaling) return PDHCG_ROW_SCALING_DISABLED;
    if (params->row_scaling_max_rows > 0 &&
        problem->num_constraints > params->row_scaling_max_rows) {
        return PDHCG_ROW_SCALING_DISABLED;
    }
    if (problem->num_constraints > problem->num_variables) {
        return PDHCG_ROW_SCALING_RANK_DEFICIENT;
    }
    if (!all_rows_are_equalities(problem)) {
        return PDHCG_ROW_SCALING_NOT_EQUALITY;
    }

    /* Free any leftover from a previous run in this thread. */
    pdhcg_release_pending_row_scaling();

    double *M = NULL;
    pdhcg_row_scaling_status_t s = (pdhcg_row_scaling_status_t)
        compute_row_scaling_and_transform(problem, &M);
    if (s != PDHCG_ROW_SCALING_APPLIED) {
        if (M) free(M);
        return s;
    }
    g_pending_M = M;
    g_pending_M_size = problem->num_constraints;
    return PDHCG_ROW_SCALING_APPLIED;
}

pdhcg_row_scaling_status_t pdhcg_apply_row_scaling_const(
    const pdhg_parameters_t *params,
    const qp_problem_t *problem,
    rescale_info_t *rescale_info)
{
    /* The const input is a documented lie: the algorithm needs to rewrite
     * the constraint matrix in place.  We cast away const because the
     * caller (solver.cu) already owns the problem and treats it as a
     * working copy. */
    return pdhcg_apply_row_scaling(params, (qp_problem_t *)problem, rescale_info);
}

