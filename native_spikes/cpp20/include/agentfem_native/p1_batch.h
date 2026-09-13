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

#define AFN_P1_ABI_VERSION 0x00010004u

typedef enum AfnP1Status {
  AFN_P1_SUCCESS = 0,
  AFN_P1_NULL_POINTER = 1,
  AFN_P1_INVALID_CONDUCTIVITY = 2,
  AFN_P1_INVALID_CELL = 3,
  AFN_P1_NONFINITE_INPUT = 4,
  AFN_P1_INVALID_THREAD_COUNT = 5,
  AFN_P1_RESOURCE_FAILURE = 6,
} AfnP1Status;

AFN_API uint32_t afn_p1_abi_version(void);

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

AFN_API int afn_p1_diffusion_assemble_cells(
    size_t node_count,
    size_t cell_count,
    const double* points_xy,
    const int64_t* cells,
    const double* conductivity_cells_2x2,
    double source,
    int64_t* rows,
    int64_t* columns,
    double* data,
    double* load);

AFN_API int afn_t3_elasticity_assemble_cells(
    size_t node_count,
    size_t cell_count,
    const double* points_xy,
    const int64_t* cells,
    const double* constitutive_cells_3x3,
    const double* body_force_xy,
    double thickness,
    int64_t* rows,
    int64_t* columns,
    double* data,
    double* load);

AFN_API int afn_t3_elasticity_assemble_cells_parallel(
    size_t node_count,
    size_t cell_count,
    size_t thread_count,
    const double* points_xy,
    const int64_t* cells,
    const double* constitutive_cells_3x3,
    const double* body_force_xy,
    double thickness,
    int64_t* rows,
    int64_t* columns,
    double* data,
    double* load);

AFN_API int afn_t4_elasticity_assemble_cells(
    size_t node_count,
    size_t cell_count,
    const double* points_xyz,
    const int64_t* cells,
    const double* constitutive_cells_6x6,
    const double* body_force_xyz,
    int64_t* rows,
    int64_t* columns,
    double* data,
    double* load);

AFN_API int afn_t4_elasticity_assemble_cells_parallel(
    size_t node_count,
    size_t cell_count,
    size_t thread_count,
    const double* points_xyz,
    const int64_t* cells,
    const double* constitutive_cells_6x6,
    const double* body_force_xyz,
    int64_t* rows,
    int64_t* columns,
    double* data,
    double* load);

AFN_API int afn_csr_spmv(
    size_t row_count,
    size_t column_count,
    size_t nonzero_count,
    const int64_t* indptr,
    const int64_t* indices,
    const double* data,
    const double* vector,
    double* result);

AFN_API int afn_csr_fill_from_contributions(
    size_t contribution_count,
    size_t nonzero_count,
    const int64_t* coo_to_csr,
    const double* contributions,
    double* data);

#ifdef __cplusplus
}
#endif
