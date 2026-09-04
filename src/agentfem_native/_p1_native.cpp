// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "agentfem_native/p1_batch.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

class BufferView {
 public:
  BufferView() = default;
  BufferView(const BufferView&) = delete;
  BufferView& operator=(const BufferView&) = delete;
  ~BufferView() {
    if (acquired_) {
      PyBuffer_Release(&view_);
    }
  }

  bool acquire(PyObject* object, const Py_ssize_t expected_bytes,
               const bool writable, const bool floating,
               const char* argument_name) {
    const int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT |
                      (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(object, &view_, flags) < 0) {
      return false;
    }
    acquired_ = true;
    const char* format = view_.format;
    if (format != nullptr && (format[0] == '@' || format[0] == '=')) {
      ++format;
    }
    const bool valid_format =
        format != nullptr && format[0] != '\0' && format[1] == '\0' &&
        (floating ? format[0] == 'd' : (format[0] == 'q' || format[0] == 'l'));
    if ((expected_bytes >= 0 && view_.len != expected_bytes) ||
        view_.itemsize != 8 || !valid_format) {
      if (expected_bytes >= 0) {
        PyErr_Format(
            PyExc_ValueError,
            "%s must be a native-endian, C-contiguous %s buffer of %zd bytes.",
            argument_name, floating ? "float64" : "int64", expected_bytes);
      } else {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a native-endian, C-contiguous %s buffer.",
                     argument_name, floating ? "float64" : "int64");
      }
      return false;
    }
    return true;
  }

  Py_ssize_t size_bytes() const { return view_.len; }

  template <typename Value>
  Value* data() const {
    return static_cast<Value*>(view_.buf);
  }

 private:
  Py_buffer view_{};
  bool acquired_{false};
};

bool checked_bytes(const Py_ssize_t count, const Py_ssize_t multiplier,
                   Py_ssize_t* result) {
  if (count < 0 || multiplier < 0 ||
      (multiplier != 0 &&
       count > std::numeric_limits<Py_ssize_t>::max() / multiplier)) {
    PyErr_SetString(PyExc_OverflowError, "Native buffer size is out of range.");
    return false;
  }
  *result = count * multiplier;
  return true;
}

PyObject* abi_version(PyObject*, PyObject*) {
  return PyLong_FromUnsignedLong(afn_p1_abi_version());
}

PyObject* assemble_into(PyObject*, PyObject* arguments) {
  Py_ssize_t node_count = 0;
  Py_ssize_t cell_count = 0;
  PyObject* points_object = nullptr;
  PyObject* cells_object = nullptr;
  PyObject* conductivity_object = nullptr;
  double source = 0.0;
  PyObject* rows_object = nullptr;
  PyObject* columns_object = nullptr;
  PyObject* data_object = nullptr;
  PyObject* load_object = nullptr;
  if (!PyArg_ParseTuple(arguments, "nnOOOdOOOO", &node_count, &cell_count,
                        &points_object, &cells_object, &conductivity_object,
                        &source, &rows_object, &columns_object, &data_object,
                        &load_object)) {
    return nullptr;
  }

  Py_ssize_t points_bytes = 0;
  Py_ssize_t cells_bytes = 0;
  Py_ssize_t entries_bytes = 0;
  Py_ssize_t load_bytes = 0;
  if (!checked_bytes(node_count, 2 * 8, &points_bytes) ||
      !checked_bytes(cell_count, 3 * 8, &cells_bytes) ||
      !checked_bytes(cell_count, 9 * 8, &entries_bytes) ||
      !checked_bytes(node_count, 8, &load_bytes)) {
    return nullptr;
  }

  BufferView points;
  BufferView cells;
  BufferView conductivity;
  BufferView rows;
  BufferView columns;
  BufferView data;
  BufferView load;
  if (!points.acquire(points_object, points_bytes, false, true, "points") ||
      !cells.acquire(cells_object, cells_bytes, false, false, "cells") ||
      !rows.acquire(rows_object, entries_bytes, true, false, "rows") ||
      !columns.acquire(columns_object, entries_bytes, true, false, "columns") ||
      !data.acquire(data_object, entries_bytes, true, true, "data") ||
      !load.acquire(load_object, load_bytes, true, true, "load")) {
    return nullptr;
  }

  const Py_ssize_t global_conductivity_bytes = 4 * 8;
  Py_ssize_t cell_conductivity_bytes = 0;
  if (!checked_bytes(cell_count, 4 * 8, &cell_conductivity_bytes)) {
    return nullptr;
  }
  if (!conductivity.acquire(conductivity_object, -1, false, true,
                            "conductivity")) {
    return nullptr;
  }
  const bool per_cell = conductivity.size_bytes() == cell_conductivity_bytes;
  if (conductivity.size_bytes() != global_conductivity_bytes && !per_cell) {
    PyErr_Format(
        PyExc_ValueError,
        "conductivity must contain either 4 or %zd native float64 values.",
        4 * cell_count);
    return nullptr;
  }

  int status = AFN_P1_NULL_POINTER;
  Py_BEGIN_ALLOW_THREADS
  if (per_cell) {
    status = afn_p1_diffusion_assemble_cells(
        static_cast<std::size_t>(node_count),
        static_cast<std::size_t>(cell_count), points.data<double>(),
        cells.data<std::int64_t>(), conductivity.data<double>(), source,
        rows.data<std::int64_t>(), columns.data<std::int64_t>(),
        data.data<double>(), load.data<double>());
  } else {
    status = afn_p1_diffusion_assemble(
        static_cast<std::size_t>(node_count),
        static_cast<std::size_t>(cell_count), points.data<double>(),
        cells.data<std::int64_t>(), conductivity.data<double>(), source,
        rows.data<std::int64_t>(), columns.data<std::int64_t>(),
        data.data<double>(), load.data<double>());
  }
  Py_END_ALLOW_THREADS
  return PyLong_FromLong(status);
}

PyMethodDef methods[] = {
    {"abi_version", abi_version, METH_NOARGS,
     "Return the encoded AgentFEM Native P1 C ABI version."},
    {"assemble_into", assemble_into, METH_VARARGS,
     "Fill owned contiguous output buffers using the C++20 P1 kernel."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_p1_native",
    "Owned AgentFEM Native C++20 P1 accelerator.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__p1_native() { return PyModule_Create(&module); }
