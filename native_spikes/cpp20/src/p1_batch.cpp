// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "agentfem_native/p1_batch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

bool positive_definite_constitutive(const double* matrix) {
  for (std::size_t index = 0; index < 9; ++index) {
    if (!std::isfinite(matrix[index])) {
      return false;
    }
  }
  const double scale = std::max(
      {1.0, std::abs(matrix[0]) + std::abs(matrix[1]) + std::abs(matrix[2]),
       std::abs(matrix[3]) + std::abs(matrix[4]) + std::abs(matrix[5]),
       std::abs(matrix[6]) + std::abs(matrix[7]) + std::abs(matrix[8])});
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * scale;
  if (std::abs(matrix[1] - matrix[3]) > tolerance ||
      std::abs(matrix[2] - matrix[6]) > tolerance ||
      std::abs(matrix[5] - matrix[7]) > tolerance) {
    return false;
  }
  const double leading_two = matrix[0] * matrix[4] - matrix[1] * matrix[3];
  const double determinant =
      matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
      matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
      matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  return matrix[0] > tolerance && leading_two > tolerance * scale &&
         determinant > tolerance * scale * scale;
}

std::array<double, 3> strain_column(
    const std::array<std::array<double, 2>, 3>& gradients,
    const std::size_t local_dof) {
  const std::size_t node = local_dof / 2;
  if (local_dof % 2 == 0) {
    return {{gradients[node][0], 0.0, gradients[node][1]}};
  }
  return {{0.0, gradients[node][1], gradients[node][0]}};
}

int assemble_impl(
    const std::size_t node_count,
    const std::size_t cell_count,
    const double* points_xy,
    const std::int64_t* cells,
    const double* conductivity,
    const bool conductivity_per_cell,
    const double source,
    std::int64_t* rows,
    std::int64_t* columns,
    double* data,
    double* load) {
  if (node_count == 0 || cell_count == 0 || points_xy == nullptr ||
      cells == nullptr || conductivity == nullptr || rows == nullptr ||
      columns == nullptr || data == nullptr || load == nullptr) {
    return AFN_P1_NULL_POINTER;
  }
  if (node_count > std::numeric_limits<std::size_t>::max() / 2 ||
      cell_count > std::numeric_limits<std::size_t>::max() / 9) {
    return AFN_P1_INVALID_CELL;
  }
  if (!conductivity_per_cell && !positive_definite_tensor(conductivity)) {
    return AFN_P1_INVALID_CONDUCTIVITY;
  }
  if (!std::isfinite(source)) {
    return AFN_P1_NONFINITE_INPUT;
  }
  std::fill(load, load + node_count, 0.0);

  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    const double* conductivity_2x2 =
        conductivity + (conductivity_per_cell ? 4 * cell_index : 0);
    if (conductivity_per_cell &&
        !positive_definite_tensor(conductivity_2x2)) {
      return AFN_P1_INVALID_CONDUCTIVITY;
    }
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

}  // namespace

extern "C" AFN_API std::uint32_t afn_p1_abi_version() {
  return AFN_P1_ABI_VERSION;
}

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
  return assemble_impl(node_count, cell_count, points_xy, cells,
                       conductivity_2x2, false, source, rows, columns, data,
                       load);
}

extern "C" AFN_API int afn_p1_diffusion_assemble_cells(
    const std::size_t node_count,
    const std::size_t cell_count,
    const double* points_xy,
    const std::int64_t* cells,
    const double* conductivity_cells_2x2,
    const double source,
    std::int64_t* rows,
    std::int64_t* columns,
    double* data,
    double* load) {
  return assemble_impl(node_count, cell_count, points_xy, cells,
                       conductivity_cells_2x2, true, source, rows, columns,
                       data, load);
}

extern "C" AFN_API int afn_t3_elasticity_assemble_cells(
    const std::size_t node_count,
    const std::size_t cell_count,
    const double* points_xy,
    const std::int64_t* cells,
    const double* constitutive_cells_3x3,
    const double* body_force_xy,
    const double thickness,
    std::int64_t* rows,
    std::int64_t* columns,
    double* data,
    double* load) {
  if (node_count == 0 || cell_count == 0 || points_xy == nullptr ||
      cells == nullptr || constitutive_cells_3x3 == nullptr ||
      body_force_xy == nullptr || rows == nullptr || columns == nullptr ||
      data == nullptr || load == nullptr) {
    return AFN_P1_NULL_POINTER;
  }
  if (node_count > std::numeric_limits<std::size_t>::max() / 2 ||
      cell_count > std::numeric_limits<std::size_t>::max() / 36) {
    return AFN_P1_INVALID_CELL;
  }
  if (!std::isfinite(thickness) || thickness <= 0.0 ||
      !std::isfinite(body_force_xy[0]) || !std::isfinite(body_force_xy[1])) {
    return AFN_P1_NONFINITE_INPUT;
  }
  std::fill(load, load + 2 * node_count, 0.0);
  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    const double* constitutive = constitutive_cells_3x3 + 9 * cell_index;
    if (!positive_definite_constitutive(constitutive)) {
      return AFN_P1_INVALID_CONDUCTIVITY;
    }
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
    const double scale = thickness * 0.5 * std::abs(determinant);
    for (std::size_t local_node = 0; local_node < 3; ++local_node) {
      for (std::size_t component = 0; component < 2; ++component) {
        const std::size_t local_row = 2 * local_node + component;
        const std::int64_t global_row = 2 * cell[local_node] +
                                        static_cast<std::int64_t>(component);
        load[global_row] += scale * body_force_xy[component] / 3.0;
        const auto left = strain_column(gradients, local_row);
        for (std::size_t local_column = 0; local_column < 6; ++local_column) {
          const std::size_t column_node = local_column / 2;
          const std::int64_t global_column =
              2 * cell[column_node] + static_cast<std::int64_t>(local_column % 2);
          const std::size_t entry = 36 * cell_index + 6 * local_row + local_column;
          rows[entry] = global_row;
          columns[entry] = global_column;
          const auto right = strain_column(gradients, local_column);
          double value = 0.0;
          for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
              value += left[i] * constitutive[3 * i + j] * right[j];
            }
          }
          data[entry] = scale * value;
        }
      }
    }
  }
  return AFN_P1_SUCCESS;
}
