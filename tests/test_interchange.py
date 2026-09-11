# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentfem_native.interchange import (
    NativeInterchangeBundle,
    read_native_bundle,
    write_native_bundle,
)
from agentfem_native.mesh import unit_square_triangles
from agentfem_native.volume_mesh import unit_cube_tetrahedra


class NativeInterchangeTests(unittest.TestCase):
    def test_empty_bundle_defaults_are_independent(self) -> None:
        first = NativeInterchangeBundle(unit_square_triangles(1))
        second = NativeInterchangeBundle(unit_square_triangles(1))
        self.assertEqual(first.point_fields, {})
        self.assertIsNot(first.point_fields, second.point_fields)

    def test_triangle_bundle_round_trip_is_deterministic(self) -> None:
        mesh = unit_square_triangles(2)
        bundle = NativeInterchangeBundle(
            mesh,
            {"temperature": mesh.points[:, 0] + 2.0 * mesh.points[:, 1]},
            {"indicator": np.arange(mesh.cell_count, dtype=np.float64)},
            {"units": {"length": "m"}, "revision": 3},
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.afn.json"
            second = Path(directory) / "second.afn.json"
            write_native_bundle(first, bundle)
            recovered = read_native_bundle(first)
            write_native_bundle(second, recovered)
            self.assertEqual(first.read_bytes(), second.read_bytes())
        np.testing.assert_array_equal(recovered.mesh.points, mesh.points)
        np.testing.assert_array_equal(
            recovered.point_fields["temperature"], bundle.point_fields["temperature"]
        )

    def test_tetrahedral_vector_and_tensor_fields_round_trip(self) -> None:
        mesh = unit_cube_tetrahedra(1)
        displacement = np.column_stack(
            (mesh.points[:, 0], -mesh.points[:, 1], mesh.points[:, 2])
        )
        stress = np.tile(np.arange(6, dtype=np.float64), (mesh.cell_count, 1))
        bundle = NativeInterchangeBundle(
            mesh, {"displacement": displacement}, {"stress": stress}, {}
        )
        with tempfile.TemporaryDirectory() as directory:
            recovered = read_native_bundle(
                write_native_bundle(Path(directory) / "solid.afn.json", bundle)
            )
        np.testing.assert_array_equal(recovered.mesh.cells, mesh.cells)
        np.testing.assert_array_equal(recovered.cell_fields["stress"], stress)

    def test_invalid_field_and_format_fail_explicitly(self) -> None:
        mesh = unit_square_triangles(1)
        with self.assertRaisesRegex(ValueError, "finite"):
            NativeInterchangeBundle(
                mesh, {"bad": np.full(mesh.node_count, np.nan)}, {}, {}
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"format":"unknown"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                read_native_bundle(path)


if __name__ == "__main__":
    unittest.main()
