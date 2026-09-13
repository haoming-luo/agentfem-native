// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "agentfem_native/p1_batch.h"

#include <stddef.h>

int main(void) {
  if (afn_p1_abi_version() != AFN_P1_ABI_VERSION) {
    return 2;
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
