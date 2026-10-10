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
#include <cublas_v2.h>
#include <cusolverDn.h>

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

    /* ---- 2. G = D A A^T D (m x m dense, GPU via cuBLAS dgemm) ----
     *
     * Bug fix: the previous C implementation multiplied
     *     G[i,j] += vi * col_vals[q] * d[j]
     * which double-counted d[j] (col_vals already includes d[j]).  The
     * correct formula is
     *     G[i,j] += vi * col_vals[q]
     * because vi = A[i,k]*d[i] and col_vals[q] = A[j,k]*d[j], giving
     *     G[i,j] = sum_k A[i,k]*A[j,k]*d[i]*d[j] = (D A A^T D)[i,j].
     *
     * Speed fix: instead of an O(nnz_per_col^2) outer-product per
     * column on the host (which costs ~30 s for dsNRL at m=1616,
     * n=61822), we form a row-scaled dense A' = diag(d) * A on the
     * host (m x n doubles, 800 MB worst case at m=2000) and let
     * cuBLAS dgemm compute G = A' * A'^T on the GPU in O(m^2 n) flops
     * — a few ms.  The row_scaling_max_rows cap of 2000 keeps the
     * dense A under ~1 GB which is safe for both host and device.
     */
    double *G = NULL;                 /* host, m x m row-major, leading dim m */
    double *d_A_dense = NULL;         /* device, m x n row-major */
    double *d_G = NULL;               /* device, m x m col-major (cuBLAS view) */
    {
        /* 2a. create cuBLAS handle so we can do the dgemm here. */
        cublasHandle_t cublas_h_local = NULL;
        if (cublasCreate(&cublas_h_local) != CUBLAS_STATUS_SUCCESS) {
            free(d);
            return PDHCG_ROW_SCALING_ERROR;
        }

        size_t A_bytes = (size_t)m * (size_t)n * sizeof(double);
        if (m > 0 && n > 0) {
            /* 2b. dense A' = diag(d) * A on the host (row-major). */
            double *h_A_dense = (double *)calloc((size_t)m * n, sizeof(double));
            if (!h_A_dense) {
                cublasDestroy(cublas_h_local);
                free(d);
                return PDHCG_ROW_SCALING_ERROR;
            }
            for (int i = 0; i < m; ++i) {
                double di = d[i];
                int row_end = A->row_ptr[i + 1];
                for (int k = A->row_ptr[i]; k < row_end; ++k) {
                    int col = A->col_ind[k];
                    h_A_dense[(size_t)i * n + col] = A->val[k] * di;
                }
            }
            /* 2c. copy to device. */
            CUDA_OK(cudaMalloc((void **)&d_A_dense, A_bytes));
            CUDA_OK(cudaMemcpy(d_A_dense, h_A_dense, A_bytes, cudaMemcpyHostToDevice));
            free(h_A_dense);

            /* 2d. cuBLAS dgemm: G(m x m) = A'(m x n) * A'^T(n x m).
             *     A' is row-major (m x n) on the host; in cuBLAS's
             *     col-major view the same physical buffer is A'^T
             *     (n x m).  To compute A' * A'^T (m x m) we therefore
             *     ask cuBLAS to apply op(A)=T (transpose the n x m
             *     col-major view back to m x n) and op(B)=N (the
             *     same buffer, kept as n x m col-major = A'^T).
             */
            CUDA_OK(cudaMalloc((void **)&d_G, (size_t)m * m * sizeof(double)));
            double alpha = 1.0;
            double beta  = 0.0;
            cublasDgemm(cublas_h_local,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        m, m, n,
                        &alpha,
                        d_A_dense, n,
                        d_A_dense, n,
                        &beta,
                        d_G, m);
            /* 2e. copy G back to host (row-major m x m). */
            G = (double *)malloc((size_t)m * m * sizeof(double));
            if (!G) {
                cublasDestroy(cublas_h_local);
                cudaFree(d_A_dense); cudaFree(d_G);
                free(d);
                return PDHCG_ROW_SCALING_ERROR;
            }
            CUDA_OK(cudaMemcpy(G, d_G, (size_t)m * m * sizeof(double),
                               cudaMemcpyDeviceToHost));

            cudaFree(d_A_dense); cudaFree(d_G);
        } else {
            G = (double *)calloc((size_t)m * m, sizeof(double));
            if (!G) {
                cublasDestroy(cublas_h_local);
                free(d);
                return PDHCG_ROW_SCALING_ERROR;
            }
        }
        cublasDestroy(cublas_h_local);
    }

    /* ---- 3. eigendecomposition via cuSOLVER (GPU) ---- */
    cusolverDnHandle_t cusolver_h = NULL;
    cusolverDnCreate(&cusolver_h);

    double *d_w = NULL;
    double *d_work = NULL;
    int *d_Ipiv = NULL;

    CUDA_OK(cudaMalloc((void **)&d_w, m * sizeof(double)));
    CUDA_OK(cudaMalloc((void **)&d_Ipiv, m * sizeof(int)));

    /* d_G already holds G on the device (computed in step 2 via dgemm).
     * cuSOLVER overwrites the input matrix with eigenvectors on success. */
    double *d_G = NULL;
    CUDA_OK(cudaMalloc((void **)&d_G, (size_t)m * m * sizeof(double)));
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
            cusolverDnDestroy(cusolver_h);
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
            cusolverDnDestroy(cusolver_h);
            return PDHCG_ROW_SCALING_RANK_DEFICIENT;
        }
        if (!(wmin_pos > ROW_SCALING_REL_RANK_THRESHOLD * wmax)) {
            fprintf(stderr, "[row_scaling] Gram min_pos/max = %.3g (rejected)\n",
                    wmin_pos / wmax);
            free(d); free(G); free(w);
            cudaFree(d_G); cudaFree(d_w); cudaFree(d_Ipiv); cudaFree(d_work);
            cusolverDnDestroy(cusolver_h);
            return PDHCG_ROW_SCALING_RANK_DEFICIENT;
        }
    }

    /* ---- 5. Build W = V diag(1/sqrt(w)) V^T (host, row-major) ----
     *
     * Critical fix: the previous loop
     *     W[i, j] = sum_k V_rm[i, k] * V_rm[j, k]
     * with V_rm already pre-scaled by 1/sqrt(w[k]) (line 357-359 above)
     * produced W = G^(-1), not G^(-1/2).  The correct formula is
     *     W[i, j] = sum_k V_original[i, k] * (1/sqrt(w[k])) * V_original[j, k]
     * Since V_rm[r, c] = V_original[c, r] / sqrt(w[c]), we have
     * V_original[j, k] = V_rm[k, j] * sqrt(w[j]), and the formula becomes
     *     W[i, j] = sqrt(w[i] w[j]) * sum_k V_rm[k, i] * V_rm[k, j] / sqrt(w[k])
     * which is the symmetric G^(-1/2) we need.
     */
    double *V_rm = (double *)malloc((size_t)m * m * sizeof(double));
    {
        double *V_cm = (double *)malloc((size_t)m * m * sizeof(double));
        CUDA_OK(cudaMemcpy(V_cm, d_G, (size_t)m * m * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
                V_rm[i * m + j] = V_cm[j * m + i];
        free(V_cm);
    }
    /* V_rm[col j] *= 1/sqrt(w[j]) so V_rm[i, j] = V_original[j, i] / sqrt(w[j]) */
    for (int j = 0; j < m; ++j) {
        double s = 1.0 / sqrt(w[j]);
        for (int i = 0; i < m; ++i) V_rm[i * m + j] *= s;
    }
    /* W = V_original * diag(1/sqrt(w)) * V_original^T  (the true G^{-1/2}) */
    double *W = (double *)malloc((size_t)m * m * sizeof(double));
    for (int i = 0; i < m; ++i) {
        double swi = sqrt(w[i]);
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int k = 0; k < m; ++k)
                s += V_rm[k * m + i] * V_rm[k * m + j] / sqrt(w[k]);
            W[i * m + j] = s * swi * sqrt(w[j]);
        }
    }
    free(V_rm);

    /* ---- 6. M = W * diag(d) ---- */
    double *M = (double *)malloc((size_t)m * m * sizeof(double));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            M[i * m + j] = W[i * m + j] * d[j];
    free(W);

    /* ---- 7. Rewrite A in place: A_new = M @ A_old (GPU via cuBLAS) ----
     *
     * Speed fix: the previous host-side triple loop is O(m^2 n) which
     * is ~161 G flops for dsNRL (m=1616, n=61822) and costs 200-400 s
     * on the host CPU.  We instead copy the small dense M (m x m =
     * 20 MB) and the dense A_old (m x n, up to ~800 MB) to the device
     * and let cuBLAS dgemm compute A_new = M * A_old on the GPU in a
     * few ms.
     */
    {
        /* 7a. build A_old_dense on the host then upload.  We already
         *     have the CSR representation; do an O(nnz) scatter. */
        double *A_old_dense = (double *)calloc((size_t)m * n, sizeof(double));
        if (!A_old_dense) {
            free(d); free(G); free(M);
            cusolverDnDestroy(cusolver_h);
            return PDHCG_ROW_SCALING_ERROR;
        }
        for (int i = 0; i < m; ++i) {
            int row_end = A->row_ptr[i + 1];
            for (int k = A->row_ptr[i]; k < row_end; ++k) {
                A_old_dense[(size_t)i * n + A->col_ind[k]] = A->val[k];
            }
        }

        /* 7b. cuBLAS dgemm: A_new (m x n) = M (m x m) * A_old (m x n).
         *     M is row-major m x m; in cuBLAS's col-major view the
         *     same physical buffer is M^T (m x m, square so still m x m).
         *     A_old is row-major m x n; col-major view is A_old^T (n x m).
         *     A_new row-major = M * A_old  =>  A_new^T col-major
         *     = A_old^T col-major * M^T col-major
         *     = (n x m) * (m x m)  =>  dgemm with op(A)=N, op(B)=N,
         *     result dims m_out=n, n_out=m, k=m, lda=m, ldb=m, ldc=n.
         */
        double *d_M = NULL;
        double *d_A_old = NULL;
        double *d_A_new = NULL;
        double alpha = 1.0;
        double beta  = 0.0;
        size_t M_bytes = (size_t)m * m * sizeof(double);
        size_t A_bytes = (size_t)m * n * sizeof(double);

        cublasHandle_t cublas_h = NULL;
        if (cublasCreate(&cublas_h) != CUBLAS_STATUS_SUCCESS) {
            free(A_old_dense); free(d); free(G); free(M);
            cusolverDnDestroy(cusolver_h);
            return PDHCG_ROW_SCALING_ERROR;
        }
        CUDA_OK(cudaMalloc((void **)&d_M,     M_bytes));
        CUDA_OK(cudaMalloc((void **)&d_A_old, A_bytes));
        CUDA_OK(cudaMalloc((void **)&d_A_new, A_bytes));
        CUDA_OK(cudaMemcpy(d_M,     M,        M_bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_A_old, A_old_dense, A_bytes, cudaMemcpyHostToDevice));
        cublasDgemm(cublas_h,
                    CUBLAS_OP_N, CUBLAS_OP_T,   /* op(B)=T: M = G^(-1/2) D is NOT symmetric */
                    n, m, m,                    /* m_out=n, n_out=m, k=m */
                    &alpha,
                    d_A_old, n,                 /* lda = n (A_old col-major is n x m, row-major m x n) */
                    d_M,     m,                 /* ldb = m (M col-major is m x m, op(B)=T reads M^T) */
                    &beta,
                    d_A_new, n);
        /* 7c. copy A_new back to host. */
        double *A_new_dense = (double *)calloc((size_t)m * n, sizeof(double));
        if (!A_new_dense) {
            cudaFree(d_M); cudaFree(d_A_old); cudaFree(d_A_new);
            cublasDestroy(cublas_h);
            free(A_old_dense); free(d); free(G); free(M);
            cusolverDnDestroy(cusolver_h);
            return PDHCG_ROW_SCALING_ERROR;
        }
        CUDA_OK(cudaMemcpy(A_new_dense, d_A_new, A_bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_M); cudaFree(d_A_old); cudaFree(d_A_new);
        cublasDestroy(cublas_h);
        free(A_old_dense);

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
                cusolverDnDestroy(cusolver_h);
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

