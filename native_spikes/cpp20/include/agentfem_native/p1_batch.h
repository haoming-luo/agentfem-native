// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(AFN_BUILDING_LIBRARY)
#    define AFN_API __declspec(dllexport)
#  else
#    define AFN_API __declspec(dllimport)
#  endif
#else
#  define AFN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum AfnP1Status {
  AFN_P1_SUCCESS = 0,
  AFN_P1_NULL_POINTER = 1,
  AFN_P1_INVALID_CONDUCTIVITY = 2,
  AFN_P1_INVALID_CELL = 3,
  AFN_P1_NONFINITE_INPUT = 4,
} AfnP1Status;

AFN_API int afn_p1_diffusion_assemble(
    size_t node_count,
    size_t cell_count,
    const double* points_xy,
    const int64_t* cells,
    const double* conductivity_2x2,
    double source,
    int64_t* rows,
    int64_t* columns,
    double* data,
    double* load);

#ifdef __cplusplus
}
#endif
