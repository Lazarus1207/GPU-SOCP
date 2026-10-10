/*
Copyright 2026 PDHCG-Benchmark

Licensed under the Apache License, Version 2.0.
*/

#ifndef PDHCG_ROW_SCALING_H
#define PDHCG_ROW_SCALING_H

#include "pdhcg_types.h"
#include "internal_types.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PDHCG_ROW_SCALING_DISABLED = 0,
    PDHCG_ROW_SCALING_APPLIED,
    PDHCG_ROW_SCALING_NOT_EQUALITY,
    PDHCG_ROW_SCALING_RANK_DEFICIENT,
    PDHCG_ROW_SCALING_ERROR
} pdhcg_row_scaling_status_t;

/* Apply M = (D A A^T D)^{-1/2} D in place when:
 *   - params->enable_row_scaling is true
 *   - num_constraints <= params->row_scaling_max_rows (default 2000)
 *   - all rows are finite equalities (constraint_lower == constraint_upper)
 *
 * The map M is stashed in a thread-local handle so it survives the
 * rescale_info_free() call earlier in optimize(). On graceful skip
 * (DISABLED / NOT_EQUALITY / RANK_DEFICIENT) the problem is untouched.
 */
pdhcg_row_scaling_status_t pdhcg_apply_row_scaling(
    const pdhg_parameters_t *params,
    qp_problem_t *problem,
    rescale_info_t *rescale_info);

/* Same as pdhcg_apply_row_scaling but accepts a const qp_problem_t *; the
 * problem is still mutated in place because the algorithm needs to rewrite
 * the constraint matrix. Use only when the caller can legitimately
 * discard the const qualifier (e.g. the qp_problem_t is owned locally). */
pdhcg_row_scaling_status_t pdhcg_apply_row_scaling_const(
    const pdhg_parameters_t *params,
    const qp_problem_t *problem,
    rescale_info_t *rescale_info);


/* Recover the original dual solution: y_orig = M^T y_new. */
void pdhcg_recover_dual_from_pending_row_scaling(double *dual_solution);

/* Free the thread-local row map. */
void pdhcg_release_pending_row_scaling(void);

/* Legacy wrappers. */
void pdhcg_recover_dual_from_row_scaling(rescale_info_t *info, double *dual_solution);
void pdhcg_row_scaling_info_clear(rescale_info_t *info);

const char *pdhcg_row_scaling_status_str(pdhcg_row_scaling_status_t s);

#ifdef __cplusplus
}
#endif

#endif
