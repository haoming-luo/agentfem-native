// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#![deny(warnings)]

use std::ffi::{c_double, c_int};
use std::slice;

// This comparison spike intentionally implements only the ABI 1.0 diffusion
// subset. It must not advertise the production C++ ABI 1.1 T3 entry point.
pub const AFN_P1_ABI_VERSION: u32 = 0x0001_0000;
pub const AFN_P1_SUCCESS: c_int = 0;
pub const AFN_P1_NULL_POINTER: c_int = 1;
pub const AFN_P1_INVALID_CONDUCTIVITY: c_int = 2;
pub const AFN_P1_INVALID_CELL: c_int = 3;
pub const AFN_P1_NONFINITE_INPUT: c_int = 4;

fn positive_definite_tensor(tensor: &[f64]) -> bool {
    let scale = 1.0_f64
        .max(tensor[0].abs() + tensor[1].abs())
        .max(tensor[2].abs() + tensor[3].abs());
    let tolerance = 64.0 * f64::EPSILON * scale;
    let determinant = tensor[0] * tensor[3] - tensor[1] * tensor[2];
    tensor.iter().all(|value| value.is_finite())
        && (tensor[1] - tensor[2]).abs() <= tolerance
        && tensor[0] > tolerance
        && determinant > tolerance * scale
}

#[allow(clippy::too_many_arguments)]
unsafe fn assemble_impl(
    node_count: usize,
    cell_count: usize,
    points_xy: *const c_double,
    cells: *const i64,
    conductivity: *const c_double,
    conductivity_per_cell: bool,
    source: c_double,
    rows: *mut i64,
    columns: *mut i64,
    data: *mut c_double,
    load: *mut c_double,
) -> c_int {
    if node_count == 0
        || cell_count == 0
        || points_xy.is_null()
        || cells.is_null()
        || conductivity.is_null()
        || rows.is_null()
        || columns.is_null()
        || data.is_null()
        || load.is_null()
    {
        return AFN_P1_NULL_POINTER;
    }
    let Some(point_value_count) = node_count.checked_mul(2) else {
        return AFN_P1_INVALID_CELL;
    };
    let Some(cell_value_count) = cell_count.checked_mul(3) else {
        return AFN_P1_INVALID_CELL;
    };
    let Some(entry_count) = cell_count.checked_mul(9) else {
        return AFN_P1_INVALID_CELL;
    };
    let tensor_count = if conductivity_per_cell {
        let Some(count) = cell_count.checked_mul(4) else {
            return AFN_P1_INVALID_CELL;
        };
        count
    } else {
        4
    };

    // Safety: the public ABI requires valid, aligned, non-overlapping buffers
    // with the lengths implied by the counts. Nulls and arithmetic overflow are
    // checked above before any slice is formed.
    let points = unsafe { slice::from_raw_parts(points_xy, point_value_count) };
    let cell_nodes = unsafe { slice::from_raw_parts(cells, cell_value_count) };
    let tensors = unsafe { slice::from_raw_parts(conductivity, tensor_count) };
    let row_output = unsafe { slice::from_raw_parts_mut(rows, entry_count) };
    let column_output = unsafe { slice::from_raw_parts_mut(columns, entry_count) };
    let data_output = unsafe { slice::from_raw_parts_mut(data, entry_count) };
    let load_output = unsafe { slice::from_raw_parts_mut(load, node_count) };

    if !source.is_finite() {
        return AFN_P1_NONFINITE_INPUT;
    }
    if !conductivity_per_cell && !positive_definite_tensor(tensors) {
        return AFN_P1_INVALID_CONDUCTIVITY;
    }
    load_output.fill(0.0);

    for cell_index in 0..cell_count {
        let cell = &cell_nodes[3 * cell_index..3 * cell_index + 3];
        let tensor_offset = if conductivity_per_cell {
            4 * cell_index
        } else {
            0
        };
        let tensor = &tensors[tensor_offset..tensor_offset + 4];
        if conductivity_per_cell && !positive_definite_tensor(tensor) {
            return AFN_P1_INVALID_CONDUCTIVITY;
        }
        if cell
            .iter()
            .any(|node| *node < 0 || *node as usize >= node_count)
            || cell[0] == cell[1]
            || cell[1] == cell[2]
            || cell[2] == cell[0]
        {
            return AFN_P1_INVALID_CELL;
        }
        let node0 = cell[0] as usize;
        let node1 = cell[1] as usize;
        let node2 = cell[2] as usize;
        let x0 = points[2 * node0];
        let y0 = points[2 * node0 + 1];
        let x1 = points[2 * node1];
        let y1 = points[2 * node1 + 1];
        let x2 = points[2 * node2];
        let y2 = points[2 * node2 + 1];
        if [x0, y0, x1, y1, x2, y2]
            .iter()
            .any(|value| !value.is_finite())
        {
            return AFN_P1_NONFINITE_INPUT;
        }

        let ax = x1 - x0;
        let ay = y1 - y0;
        let bx = x2 - x0;
        let by = y2 - y0;
        let determinant = ax * by - bx * ay;
        let edge_scale = ax.hypot(ay).max(bx.hypot(by));
        let threshold = 32.0 * f64::EPSILON * edge_scale * edge_scale;
        if determinant.abs() <= threshold {
            return AFN_P1_INVALID_CELL;
        }

        let inverse_determinant = 1.0 / determinant;
        let gradients = [
            [
                (ay - by) * inverse_determinant,
                (bx - ax) * inverse_determinant,
            ],
            [by * inverse_determinant, -bx * inverse_determinant],
            [-ay * inverse_determinant, ax * inverse_determinant],
        ];
        let area = 0.5 * determinant.abs();
        for local_row in 0..3 {
            let global_row = cell[local_row] as usize;
            load_output[global_row] += source * area / 3.0;
            for local_column in 0..3 {
                let entry = 9 * cell_index + 3 * local_row + local_column;
                row_output[entry] = cell[local_row];
                column_output[entry] = cell[local_column];
                let left = gradients[local_row];
                let right = gradients[local_column];
                let applied_x = tensor[0] * right[0] + tensor[1] * right[1];
                let applied_y = tensor[2] * right[0] + tensor[3] * right[1];
                data_output[entry] = area * (left[0] * applied_x + left[1] * applied_y);
            }
        }
    }
    AFN_P1_SUCCESS
}

#[no_mangle]
pub extern "C" fn afn_p1_abi_version() -> u32 {
    AFN_P1_ABI_VERSION
}

/// Assemble one global static conductivity tensor.
///
/// # Safety
///
/// All pointers must be valid, naturally aligned, non-overlapping buffers of
/// the lengths implied by `node_count` and `cell_count`.
#[no_mangle]
pub unsafe extern "C" fn afn_p1_diffusion_assemble(
    node_count: usize,
    cell_count: usize,
    points_xy: *const c_double,
    cells: *const i64,
    conductivity_2x2: *const c_double,
    source: c_double,
    rows: *mut i64,
    columns: *mut i64,
    data: *mut c_double,
    load: *mut c_double,
) -> c_int {
    unsafe {
        assemble_impl(
            node_count,
            cell_count,
            points_xy,
            cells,
            conductivity_2x2,
            false,
            source,
            rows,
            columns,
            data,
            load,
        )
    }
}

/// Assemble one static conductivity tensor per cell.
///
/// # Safety
///
/// All pointers must be valid, naturally aligned, non-overlapping buffers of
/// the lengths implied by `node_count` and `cell_count`.
#[no_mangle]
pub unsafe extern "C" fn afn_p1_diffusion_assemble_cells(
    node_count: usize,
    cell_count: usize,
    points_xy: *const c_double,
    cells: *const i64,
    conductivity_cells_2x2: *const c_double,
    source: c_double,
    rows: *mut i64,
    columns: *mut i64,
    data: *mut c_double,
    load: *mut c_double,
) -> c_int {
    unsafe {
        assemble_impl(
            node_count,
            cell_count,
            points_xy,
            cells,
            conductivity_cells_2x2,
            true,
            source,
            rows,
            columns,
            data,
            load,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_triangle_contract_and_failures() {
        let points = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let cells = [0_i64, 1, 2, 0, 2, 3];
        let tensor = [1.0, 0.0, 0.0, 1.0];
        let mut rows = [0_i64; 18];
        let mut columns = [0_i64; 18];
        let mut data = [0.0; 18];
        let mut load = [0.0; 4];
        let status = unsafe {
            afn_p1_diffusion_assemble(
                4,
                2,
                points.as_ptr(),
                cells.as_ptr(),
                tensor.as_ptr(),
                2.0,
                rows.as_mut_ptr(),
                columns.as_mut_ptr(),
                data.as_mut_ptr(),
                load.as_mut_ptr(),
            )
        };
        assert_eq!(status, AFN_P1_SUCCESS);
        assert_eq!(afn_p1_abi_version(), AFN_P1_ABI_VERSION);
        assert_eq!(&rows[0..3], &[0, 0, 0]);
        assert!((data[0] - 0.5).abs() <= 1.0e-15);
        assert!((load.iter().sum::<f64>() - 2.0).abs() <= 1.0e-15);

        let invalid_tensor = [0.0; 4];
        let invalid_status = unsafe {
            afn_p1_diffusion_assemble(
                4,
                2,
                points.as_ptr(),
                cells.as_ptr(),
                invalid_tensor.as_ptr(),
                0.0,
                rows.as_mut_ptr(),
                columns.as_mut_ptr(),
                data.as_mut_ptr(),
                load.as_mut_ptr(),
            )
        };
        assert_eq!(invalid_status, AFN_P1_INVALID_CONDUCTIVITY);
    }
}
