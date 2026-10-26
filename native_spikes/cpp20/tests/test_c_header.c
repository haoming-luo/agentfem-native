// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "agentfem_native/p1_batch.h"

#include <stddef.h>

int main(void) {
  if (afn_p1_abi_version() != AFN_P1_ABI_VERSION) {
    return 2;
  }
  {
    const int64_t pointer[2] = {0, 0};
    const double diagonal[1] = {4.0};
    const double rhs[1] = {8.0};
    double pivot[1] = {0.0};
    double result[1] = {0.0};
    int64_t failed_row = -1;
    if (afn_ic0_factor(1, 0, pointer, NULL, NULL, diagonal, NULL, pivot,
                      &failed_row) != AFN_P1_SUCCESS) return 4;
    if (afn_ldlt_solve(1, 0, pointer, NULL, NULL, pivot, rhs, result) !=
        AFN_P1_SUCCESS || result[0] != 2.0) return 5;
  }
  const int status = afn_p1_diffusion_assemble(
      (size_t)0, (size_t)0, NULL, NULL, NULL, 0.0, NULL, NULL, NULL, NULL);
  if (status != AFN_P1_NULL_POINTER) {
    return 1;
  }
  return afn_csr_fill_from_contributions(
             (size_t)0, (size_t)0, NULL, NULL, NULL) == AFN_P1_SUCCESS
             ? 0
             : 3;
}
