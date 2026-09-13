# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
cmake_minimum_required(VERSION 3.20)
project(agentfem_native_cpp20_spike VERSION 0.1.0 LANGUAGES C CXX)
option(AFN_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
find_package(Threads REQUIRED)

add_library(agentfem_native_p1 SHARED src/p1_batch.cpp)
target_compile_features(agentfem_native_p1 PUBLIC cxx_std_20)
target_compile_definitions(agentfem_native_p1 PRIVATE AFN_BUILDING_LIBRARY)
target_link_libraries(agentfem_native_p1 PRIVATE Threads::Threads)
target_include_directories(
  agentfem_native_p1
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
set_target_properties(
  agentfem_native_p1
  PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
)

if(MSVC)
  target_compile_options(agentfem_native_p1 PRIVATE /W4 /WX)
else()
  target_compile_options(agentfem_native_p1 PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

enable_testing()
add_executable(agentfem_native_p1_test tests/test_p1_batch.cpp)
target_compile_features(agentfem_native_p1_test PRIVATE cxx_std_20)
target_link_libraries(agentfem_native_p1_test PRIVATE agentfem_native_p1)

if(AFN_ENABLE_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "AFN_ENABLE_SANITIZERS is not configured for MSVC")
  endif()
  foreach(target agentfem_native_p1 agentfem_native_p1_test)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

add_test(NAME p1_batch_contract COMMAND agentfem_native_p1_test)

add_executable(agentfem_native_c_header_test tests/test_c_header.c)
target_compile_features(agentfem_native_c_header_test PRIVATE c_std_11)
target_link_libraries(agentfem_native_c_header_test PRIVATE agentfem_native_p1)
if(MSVC)
  target_compile_options(agentfem_native_c_header_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    agentfem_native_c_header_test PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
endif()
if(AFN_ENABLE_SANITIZERS)
  target_compile_options(
    agentfem_native_c_header_test
    PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
  )
  target_link_options(
    agentfem_native_c_header_test PRIVATE -fsanitize=address,undefined
  )
endif()
add_test(NAME c_header_contract COMMAND agentfem_native_c_header_test)
