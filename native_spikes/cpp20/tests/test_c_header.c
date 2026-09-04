// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "agentfem_native/p1_batch.h"

#include <stddef.h>

int main(void) {
  const int status = afn_p1_diffusion_assemble(
      (size_t)0, (size_t)0, NULL, NULL, NULL, 0.0, NULL, NULL, NULL, NULL);
  return status == AFN_P1_NULL_POINTER ? 0 : 1;
}
