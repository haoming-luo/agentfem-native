// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "agentfem_native/p1_batch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

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

template <std::size_t Size>
bool positive_definite_symmetric(const double* matrix) {
  double scale = 1.0;
  for (std::size_t row = 0; row < Size; ++row) {
    double row_scale = 0.0;
    for (std::size_t column = 0; column < Size; ++column) {
      const double value = matrix[Size * row + column];
      if (!std::isfinite(value)) {
        return false;
      }
      row_scale += std::abs(value);
    }
    scale = std::max(scale, row_scale);
  }
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * scale;
  std::array<double, Size * Size> lower{};
  for (std::size_t row = 0; row < Size; ++row) {
    for (std::size_t column = 0; column < row; ++column) {
      if (std::abs(matrix[Size * row + column] -
                   matrix[Size * column + row]) > tolerance) {
        return false;
      }
    }
    for (std::size_t column = 0; column <= row; ++column) {
      double value = matrix[Size * row + column];
      for (std::size_t inner = 0; inner < column; ++inner) {
        value -= lower[Size * row + inner] *
                 lower[Size * column + inner];
      }
      if (row == column) {
        if (!std::isfinite(value) || value <= tolerance) {
          return false;
        }
        lower[Size * row + column] = std::sqrt(value);
      } else {
        lower[Size * row + column] =
            value / lower[Size * column + column];
      }
    }
  }
  return true;
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

std::array<double, 6> solid_strain_column(
    const std::array<std::array<double, 3>, 4>& gradients,
    const std::size_t local_dof) {
  const std::size_t node = local_dof / 3;
  const std::size_t component = local_dof % 3;
  const double gx = gradients[node][0];
  const double gy = gradients[node][1];
  const double gz = gradients[node][2];
  if (component == 0) {
    return {{gx, 0.0, 0.0, gy, 0.0, gz}};
  }
  if (component == 1) {
    return {{0.0, gy, 0.0, gx, gz, 0.0}};
  }
  return {{0.0, 0.0, gz, 0.0, gy, gx}};
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

// 每个工作线程只写独占的单元 COO 区间；节点载荷必须先写入线程私有缓冲区，
// 再按线程编号归并。这个顺序是跨平台可重复性的组成部分，不能改成原子累加。
template <typename AssembleRange>
int assemble_parallel_ranges(
    const std::size_t cell_count,
    const std::size_t thread_count,
    const std::size_t load_size,
    double* load,
    AssembleRange assemble_range) {
  const std::size_t worker_count = std::min(cell_count, thread_count);
  if (load_size > std::numeric_limits<std::size_t>::max() / worker_count) {
    return AFN_P1_RESOURCE_FAILURE;
  }

  try {
    std::vector<double> private_load(worker_count * load_size, 0.0);
    std::vector<int> statuses(worker_count, AFN_P1_SUCCESS);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    const std::size_t base_count = cell_count / worker_count;
    const std::size_t remainder = cell_count % worker_count;
    std::size_t begin = 0;
    try {
      for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t count = base_count + (worker < remainder ? 1 : 0);
        const std::size_t range_begin = begin;
        workers.emplace_back([&, worker, range_begin, count]() {
          try {
            statuses[worker] = assemble_range(
                range_begin, count, private_load.data() + worker * load_size);
          } catch (...) {
            statuses[worker] = AFN_P1_RESOURCE_FAILURE;
          }
        });
        begin += count;
      }
    } catch (...) {
      for (auto& worker : workers) {
        if (worker.joinable()) {
          worker.join();
        }
      }
      return AFN_P1_RESOURCE_FAILURE;
    }

    for (auto& worker : workers) {
      worker.join();
    }
    for (const int status : statuses) {
      if (status != AFN_P1_SUCCESS) {
        return status;
      }
    }

    std::fill(load, load + load_size, 0.0);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      const double* source = private_load.data() + worker * load_size;
      for (std::size_t index = 0; index < load_size; ++index) {
        load[index] += source[index];
      }
    }
    return AFN_P1_SUCCESS;
  } catch (...) {
    return AFN_P1_RESOURCE_FAILURE;
  }
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
      cell_count > std::numeric_limits<std::size_t>::max() / 36 ||
      node_count - 1 >
          static_cast<std::size_t>(
              std::numeric_limits<std::int64_t>::max()) /
              2) {
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

extern "C" AFN_API int afn_t3_elasticity_assemble_cells_parallel(
    const std::size_t node_count,
    const std::size_t cell_count,
    const std::size_t thread_count,
    const double* points_xy,
    const std::int64_t* cells,
    const double* constitutive_cells_3x3,
    const double* body_force_xy,
    const double thickness,
    std::int64_t* rows,
    std::int64_t* columns,
    double* data,
    double* load) {
  if (thread_count == 0) {
    return AFN_P1_INVALID_THREAD_COUNT;
  }
  if (thread_count == 1 || cell_count <= 1) {
    return afn_t3_elasticity_assemble_cells(
        node_count, cell_count, points_xy, cells, constitutive_cells_3x3,
        body_force_xy, thickness, rows, columns, data, load);
  }
  if (node_count == 0 || points_xy == nullptr || cells == nullptr ||
      constitutive_cells_3x3 == nullptr || body_force_xy == nullptr ||
      rows == nullptr || columns == nullptr || data == nullptr ||
      load == nullptr) {
    return AFN_P1_NULL_POINTER;
  }
  if (node_count > std::numeric_limits<std::size_t>::max() / 2 ||
      cell_count > std::numeric_limits<std::size_t>::max() / 36) {
    return AFN_P1_INVALID_CELL;
  }

  return assemble_parallel_ranges(
      cell_count, thread_count, 2 * node_count, load,
      [&](const std::size_t begin, const std::size_t count,
          double* private_load) {
        return afn_t3_elasticity_assemble_cells(
            node_count, count, points_xy, cells + 3 * begin,
            constitutive_cells_3x3 + 9 * begin, body_force_xy, thickness,
            rows + 36 * begin, columns + 36 * begin, data + 36 * begin,
            private_load);
      });
}

extern "C" AFN_API int afn_t4_elasticity_assemble_cells(
    const std::size_t node_count,
    const std::size_t cell_count,
    const double* points_xyz,
    const std::int64_t* cells,
    const double* constitutive_cells_6x6,
    const double* body_force_xyz,
    std::int64_t* rows,
    std::int64_t* columns,
    double* data,
    double* load) {
  if (node_count == 0 || cell_count == 0 || points_xyz == nullptr ||
      cells == nullptr || constitutive_cells_6x6 == nullptr ||
      body_force_xyz == nullptr || rows == nullptr || columns == nullptr ||
      data == nullptr || load == nullptr) {
    return AFN_P1_NULL_POINTER;
  }
  if (node_count > std::numeric_limits<std::size_t>::max() / 3 ||
      cell_count > std::numeric_limits<std::size_t>::max() / 144 ||
      node_count - 1 >
          static_cast<std::size_t>(
              std::numeric_limits<std::int64_t>::max()) /
              3) {
    return AFN_P1_INVALID_CELL;
  }
  for (std::size_t component = 0; component < 3; ++component) {
    if (!std::isfinite(body_force_xyz[component])) {
      return AFN_P1_NONFINITE_INPUT;
    }
  }
  std::fill(load, load + 3 * node_count, 0.0);
  for (std::size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
    const double* constitutive = constitutive_cells_6x6 + 36 * cell_index;
    if (!positive_definite_symmetric<6>(constitutive)) {
      return AFN_P1_INVALID_CONDUCTIVITY;
    }
    const std::int64_t* cell = cells + 4 * cell_index;
    for (std::size_t local = 0; local < 4; ++local) {
      if (cell[local] < 0 ||
          static_cast<std::size_t>(cell[local]) >= node_count) {
        return AFN_P1_INVALID_CELL;
      }
      for (std::size_t prior = 0; prior < local; ++prior) {
        if (cell[local] == cell[prior]) {
          return AFN_P1_INVALID_CELL;
        }
      }
    }
    std::array<std::array<double, 3>, 4> point{};
    for (std::size_t local = 0; local < 4; ++local) {
      for (std::size_t component = 0; component < 3; ++component) {
        point[local][component] =
            points_xyz[3 * cell[local] + component];
        if (!std::isfinite(point[local][component])) {
          return AFN_P1_NONFINITE_INPUT;
        }
      }
    }
    const double a00 = point[1][0] - point[0][0];
    const double a10 = point[1][1] - point[0][1];
    const double a20 = point[1][2] - point[0][2];
    const double a01 = point[2][0] - point[0][0];
    const double a11 = point[2][1] - point[0][1];
    const double a21 = point[2][2] - point[0][2];
    const double a02 = point[3][0] - point[0][0];
    const double a12 = point[3][1] - point[0][1];
    const double a22 = point[3][2] - point[0][2];
    const double determinant =
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);
    const double edge_scale = std::max(
        {std::hypot(a00, a10, a20), std::hypot(a01, a11, a21),
         std::hypot(a02, a12, a22)});
    const double threshold = 64.0 * std::numeric_limits<double>::epsilon() *
                             edge_scale * edge_scale * edge_scale;
    if (std::abs(determinant) <= threshold) {
      return AFN_P1_INVALID_CELL;
    }
    const double inverse_determinant = 1.0 / determinant;
    const std::array<std::array<double, 3>, 3> inverse{{
        {{(a11 * a22 - a12 * a21) * inverse_determinant,
          (a02 * a21 - a01 * a22) * inverse_determinant,
          (a01 * a12 - a02 * a11) * inverse_determinant}},
        {{(a12 * a20 - a10 * a22) * inverse_determinant,
          (a00 * a22 - a02 * a20) * inverse_determinant,
          (a02 * a10 - a00 * a12) * inverse_determinant}},
        {{(a10 * a21 - a11 * a20) * inverse_determinant,
          (a01 * a20 - a00 * a21) * inverse_determinant,
          (a00 * a11 - a01 * a10) * inverse_determinant}},
    }};
    const std::array<std::array<double, 3>, 4> reference{{
        {{-1.0, -1.0, -1.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
    }};
    std::array<std::array<double, 3>, 4> gradients{};
    for (std::size_t node = 0; node < 4; ++node) {
      for (std::size_t physical = 0; physical < 3; ++physical) {
        for (std::size_t reference_axis = 0; reference_axis < 3;
             ++reference_axis) {
          gradients[node][physical] +=
              reference[node][reference_axis] *
              inverse[reference_axis][physical];
        }
      }
    }
    const double volume = std::abs(determinant) / 6.0;
    for (std::size_t local_node = 0; local_node < 4; ++local_node) {
      for (std::size_t component = 0; component < 3; ++component) {
        const std::size_t local_row = 3 * local_node + component;
        const std::int64_t global_row =
            3 * cell[local_node] + static_cast<std::int64_t>(component);
        load[global_row] += volume * body_force_xyz[component] / 4.0;
        const auto left = solid_strain_column(gradients, local_row);
        for (std::size_t local_column = 0; local_column < 12;
             ++local_column) {
          const std::size_t column_node = local_column / 3;
          const std::int64_t global_column =
              3 * cell[column_node] +
              static_cast<std::int64_t>(local_column % 3);
          const std::size_t entry =
              144 * cell_index + 12 * local_row + local_column;
          rows[entry] = global_row;
          columns[entry] = global_column;
          const auto right = solid_strain_column(gradients, local_column);
          double value = 0.0;
          for (std::size_t i = 0; i < 6; ++i) {
            for (std::size_t j = 0; j < 6; ++j) {
              value += left[i] * constitutive[6 * i + j] * right[j];
            }
          }
          data[entry] = volume * value;
        }
      }
    }
  }
  return AFN_P1_SUCCESS;
}

extern "C" AFN_API int afn_t4_elasticity_assemble_cells_parallel(
    const std::size_t node_count,
    const std::size_t cell_count,
    const std::size_t thread_count,
    const double* points_xyz,
    const std::int64_t* cells,
    const double* constitutive_cells_6x6,
    const double* body_force_xyz,
    std::int64_t* rows,
    std::int64_t* columns,
    double* data,
    double* load) {
  if (thread_count == 0) {
    return AFN_P1_INVALID_THREAD_COUNT;
  }
  if (thread_count == 1 || cell_count <= 1) {
    return afn_t4_elasticity_assemble_cells(
        node_count, cell_count, points_xyz, cells, constitutive_cells_6x6,
        body_force_xyz, rows, columns, data, load);
  }
  if (node_count == 0 || points_xyz == nullptr || cells == nullptr ||
      constitutive_cells_6x6 == nullptr || body_force_xyz == nullptr ||
      rows == nullptr || columns == nullptr || data == nullptr ||
      load == nullptr) {
    return AFN_P1_NULL_POINTER;
  }
  if (node_count > std::numeric_limits<std::size_t>::max() / 3 ||
      cell_count > std::numeric_limits<std::size_t>::max() / 144) {
    return AFN_P1_INVALID_CELL;
  }

  return assemble_parallel_ranges(
      cell_count, thread_count, 3 * node_count, load,
      [&](const std::size_t begin, const std::size_t count,
          double* private_load) {
        return afn_t4_elasticity_assemble_cells(
            node_count, count, points_xyz, cells + 4 * begin,
            constitutive_cells_6x6 + 36 * begin, body_force_xyz,
            rows + 144 * begin, columns + 144 * begin, data + 144 * begin,
            private_load);
      });
}

extern "C" AFN_API int afn_csr_spmv(
    const std::size_t row_count,
    const std::size_t column_count,
    const std::size_t nonzero_count,
    const std::int64_t* indptr,
    const std::int64_t* indices,
    const double* data,
    const double* vector,
    double* result) {
  if (row_count == 0 || column_count == 0 || indptr == nullptr ||
      indices == nullptr || data == nullptr || vector == nullptr ||
      result == nullptr) {
    return AFN_P1_NULL_POINTER;
  }
  if (row_count > static_cast<std::size_t>(
                      std::numeric_limits<std::int64_t>::max()) ||
      column_count > static_cast<std::size_t>(
                         std::numeric_limits<std::int64_t>::max()) ||
      nonzero_count > static_cast<std::size_t>(
                          std::numeric_limits<std::int64_t>::max()) ||
      indptr[0] != 0 ||
      indptr[row_count] != static_cast<std::int64_t>(nonzero_count)) {
    return AFN_P1_INVALID_CELL;
  }
  for (std::size_t column = 0; column < column_count; ++column) {
    if (!std::isfinite(vector[column])) {
      return AFN_P1_NONFINITE_INPUT;
    }
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    const std::int64_t start = indptr[row];
    const std::int64_t stop = indptr[row + 1];
    if (start < 0 || stop < start ||
        stop > static_cast<std::int64_t>(nonzero_count)) {
      return AFN_P1_INVALID_CELL;
    }
    std::int64_t prior = -1;
    double value = 0.0;
    for (std::int64_t entry = start; entry < stop; ++entry) {
      const std::int64_t column = indices[entry];
      if (column <= prior || column < 0 ||
          column >= static_cast<std::int64_t>(column_count)) {
        return AFN_P1_INVALID_CELL;
      }
      if (!std::isfinite(data[entry])) {
        return AFN_P1_NONFINITE_INPUT;
      }
      value += data[entry] * vector[column];
      prior = column;
    }
    if (!std::isfinite(value)) {
      return AFN_P1_NONFINITE_INPUT;
    }
    result[row] = value;
  }
  return AFN_P1_SUCCESS;
}
