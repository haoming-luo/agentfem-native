// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "agentfem_native/p1_batch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace {

bool finite_tensor(const double* tensor) {
  return std::isfinite(tensor[0]) && std::isfinite(tensor[1]) &&
         std::isfinite(tensor[2]) && std::isfinite(tensor[3]);
}

bool positive_definite_tensor(const double* tensor) {
  if (!finite_tensor(tensor)) {
    return false;
  }
  const double scale = std::max(
      {1.0, std::abs(tensor[0]) + std::abs(tensor[1]),
       std::abs(tensor[2]) + std::abs(tensor[3])});
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * scale;
  if (std::abs(tensor[1] - tensor[2]) > tolerance) {
    return false;
  }
  const double determinant = tensor[0] * tensor[3] - tensor[1] * tensor[2];
  return tensor[0] > tolerance && determinant > tolerance * scale;
}

}  // namespace

extern "C" AFN_API int afn_p1_diffusion_assemble(
    const std::size_t node_count,
    const std::size_t cell_count,
    const double* points_xy,
    const std::int64_t* cells,
    const double* conductivity_2x2,
    const double source,
    std::int64_t* rows,
    std::int64_t* columns,
    double* data,
    double* load) {
  if (node_count == 0 || cell_count == 0 || points_xy == nullptr ||
      cells == nullptr || conductivity_2x2 == nullptr || rows == nullptr ||
      columns == nullptr || data == nullptr || load == nullptr) {
    return AFN_P1_NULL_POINTER;
  }
  if (!positive_definite_tensor(conductivity_2x2)) {
    return AFN_P1_INVALID_CONDUCTIVITY;
  }
  if (!std::isfinite(source)) {
    return AFN_P1_NONFINITE_INPUT;
  }
  std::fill(load, load + node_count, 0.0);

  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    const std::int64_t* cell = cells + 3 * cell_index;
    for (std::size_t local = 0; local < 3; ++local) {
      if (cell[local] < 0 || static_cast<std::size_t>(cell[local]) >= node_count) {
        return AFN_P1_INVALID_CELL;
      }
    }
    if (cell[0] == cell[1] || cell[1] == cell[2] || cell[2] == cell[0]) {
      return AFN_P1_INVALID_CELL;
    }

    const double x0 = points_xy[2 * cell[0]];
    const double y0 = points_xy[2 * cell[0] + 1];
    const double x1 = points_xy[2 * cell[1]];
    const double y1 = points_xy[2 * cell[1] + 1];
    const double x2 = points_xy[2 * cell[2]];
    const double y2 = points_xy[2 * cell[2] + 1];
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) ||
        !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
      return AFN_P1_NONFINITE_INPUT;
    }
    const double ax = x1 - x0;
    const double ay = y1 - y0;
    const double bx = x2 - x0;
    const double by = y2 - y0;
    const double determinant = ax * by - bx * ay;
    const double edge_scale =
        std::max(std::hypot(ax, ay), std::hypot(bx, by));
    const double threshold = 32.0 * std::numeric_limits<double>::epsilon() *
                             edge_scale * edge_scale;
    if (std::abs(determinant) <= threshold) {
      return AFN_P1_INVALID_CELL;
    }

    const double inverse_determinant = 1.0 / determinant;
    const std::array<std::array<double, 2>, 3> gradients{{
        {{(ay - by) * inverse_determinant,
          (bx - ax) * inverse_determinant}},
        {{by * inverse_determinant, -bx * inverse_determinant}},
        {{-ay * inverse_determinant, ax * inverse_determinant}},
    }};
    const double area = 0.5 * std::abs(determinant);
    const std::size_t entry_offset = 9 * cell_index;
    for (std::size_t local_row = 0; local_row < 3; ++local_row) {
      load[cell[local_row]] += source * area / 3.0;
      for (std::size_t local_column = 0; local_column < 3; ++local_column) {
        const std::size_t entry = entry_offset + 3 * local_row + local_column;
        rows[entry] = cell[local_row];
        columns[entry] = cell[local_column];
        const auto& left = gradients[local_row];
        const auto& right = gradients[local_column];
        const double applied_x = conductivity_2x2[0] * right[0] +
                                 conductivity_2x2[1] * right[1];
        const double applied_y = conductivity_2x2[2] * right[0] +
                                 conductivity_2x2[3] * right[1];
        data[entry] = area * (left[0] * applied_x + left[1] * applied_y);
      }
    }
  }
  return AFN_P1_SUCCESS;
}
