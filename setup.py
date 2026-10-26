# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""构建自主 C++20 加速器，不依赖第三方绑定层。"""

from __future__ import annotations

import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    # 中文源码与诊断不能受 Windows 当前 ANSI 代码页影响。
    compile_arguments = ["/std:c++20", "/O2", "/W4", "/WX", "/EHsc", "/utf-8"]
    link_arguments: list[str] = []
else:
    compile_arguments = [
        "-std=c++20",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-fvisibility=hidden",
        "-pthread",
    ]
    link_arguments = ["-pthread"]

setup(
    ext_modules=[
        Extension(
            "agentfem_native._p1_native",
            sources=[
                "src/agentfem_native/_p1_native.cpp",
                "native_spikes/cpp20/src/p1_batch.cpp",
            ],
            include_dirs=["native_spikes/cpp20/include"],
            define_macros=[
                ("Py_LIMITED_API", "0x030B0000"),
                ("AFN_BUILDING_LIBRARY", "1"),
            ],
            extra_compile_args=compile_arguments,
            extra_link_args=link_arguments,
            language="c++",
            py_limited_api=True,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
