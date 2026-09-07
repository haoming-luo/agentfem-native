// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "agentfem_native/p1_batch.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

bool close(const double left, const double right, const double tolerance = 1e-15) {
  return std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
  static_assert(AFN_P1_ABI_VERSION == 0x00010001u);
  assert(afn_p1_abi_version() == AFN_P1_ABI_VERSION);
  const std::array<double, 8> points{{0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0}};
  const std::array<std::int64_t, 6> cells{{0, 1, 2, 0, 2, 3}};
  const std::array<double, 4> conductivity{{1.0, 0.0, 0.0, 1.0}};
  std::array<std::int64_t, 18> rows{};
  std::array<std::int64_t, 18> columns{};
  std::array<double, 18> data{};
  std::array<double, 4> load{};
  const int status = afn_p1_diffusion_assemble(
      4, 2, points.data(), cells.data(), conductivity.data(), 2.0,
      rows.data(), columns.data(), data.data(), load.data());
  assert(status == AFN_P1_SUCCESS);
  assert(rows[0] == 0 && rows[8] == 2 && rows[9] == 0 && rows[17] == 3);
  assert(columns[0] == 0 && columns[8] == 2 && columns[9] == 0 &&
         columns[17] == 3);
  const std::array<double, 9> first_expected{{
      0.5, -0.5, 0.0, -0.5, 1.0, -0.5, 0.0, -0.5, 0.5}};
  for (std::size_t index = 0; index < first_expected.size(); ++index) {
    assert(close(data[index], first_expected[index]));
  }
  assert(close(load[0], 2.0 / 3.0));
  assert(close(load[1], 1.0 / 3.0));
  assert(close(load[2], 2.0 / 3.0));
  assert(close(load[3], 1.0 / 3.0));

  const std::array<double, 8> cell_conductivity{{
      1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0}};
  assert(afn_p1_diffusion_assemble_cells(
             4, 2, points.data(), cells.data(), cell_conductivity.data(), 2.0,
             rows.data(), columns.data(), data.data(), load.data()) ==
         AFN_P1_SUCCESS);
  assert(close(data[0], first_expected[0]));
  assert(close(data[9], 2.0 * first_expected[0]));

  const std::array<std::int64_t, 3> reversed_cell{{0, 2, 1}};
  std::array<std::int64_t, 9> reversed_rows{};
  std::array<std::int64_t, 9> reversed_columns{};
  std::array<double, 9> reversed_data{};
  std::array<double, 4> reversed_load{};
  assert(afn_p1_diffusion_assemble(
             4, 1, points.data(), reversed_cell.data(), conductivity.data(),
             0.0, reversed_rows.data(), reversed_columns.data(),
             reversed_data.data(), reversed_load.data()) == AFN_P1_SUCCESS);
  const std::array<std::size_t, 3> permutation{{0, 2, 1}};
  for (std::size_t local_row = 0; local_row < 3; ++local_row) {
    for (std::size_t local_column = 0; local_column < 3; ++local_column) {
      const std::size_t entry = 3 * local_row + local_column;
      const std::size_t expected_entry =
          3 * permutation[local_row] + permutation[local_column];
      assert(close(reversed_data[entry], first_expected[expected_entry]));
    }
  }

  const std::array<double, 4> invalid_conductivity{{0.0, 0.0, 0.0, 0.0}};
  assert(afn_p1_diffusion_assemble(
             4, 2, points.data(), cells.data(), invalid_conductivity.data(),
             0.0, rows.data(), columns.data(), data.data(), load.data()) ==
         AFN_P1_INVALID_CONDUCTIVITY);
  const std::array<std::int64_t, 3> invalid_cell{{0, 1, 1}};
  assert(afn_p1_diffusion_assemble(
             4, 1, points.data(), invalid_cell.data(), conductivity.data(),
             0.0, rows.data(), columns.data(), data.data(), load.data()) ==
         AFN_P1_INVALID_CELL);
  std::array<double, 8> nonfinite_points = points;
  nonfinite_points[0] = std::numeric_limits<double>::quiet_NaN();
  assert(afn_p1_diffusion_assemble(
             4, 2, nonfinite_points.data(), cells.data(), conductivity.data(),
             0.0, rows.data(), columns.data(), data.data(), load.data()) ==
         AFN_P1_NONFINITE_INPUT);

  const std::array<double, 18> constitutive{{
      106.66666666666667, 26.666666666666668, 0.0,
      26.666666666666668, 106.66666666666667, 0.0,
      0.0, 0.0, 40.0,
      106.66666666666667, 26.666666666666668, 0.0,
      26.666666666666668, 106.66666666666667, 0.0,
      0.0, 0.0, 40.0}};
  const std::array<double, 2> body{{2.0, -1.0}};
  std::array<std::int64_t, 72> elasticity_rows{};
  std::array<std::int64_t, 72> elasticity_columns{};
  std::array<double, 72> elasticity_data{};
  std::array<double, 8> elasticity_load{};
  assert(afn_t3_elasticity_assemble_cells(
             4, 2, points.data(), cells.data(), constitutive.data(), body.data(),
             0.5, elasticity_rows.data(), elasticity_columns.data(),
             elasticity_data.data(), elasticity_load.data()) == AFN_P1_SUCCESS);
  assert(elasticity_rows[0] == 0 && elasticity_columns[35] == 5);
  assert(close(elasticity_load[0], 1.0 / 3.0));
  assert(close(elasticity_load[1], -1.0 / 6.0));
  return 0;
}
