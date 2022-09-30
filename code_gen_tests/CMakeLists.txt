#
# Copyright (c) 2022-present, IO Visor Project
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


enable_testing()

add_executable(
  code_gen_tests
  bpf_assembler.cc
  code_gen_tests.cc
)

target_include_directories("code_gen_tests" PRIVATE
  "${CMAKE_SOURCE_DIR}/vm"
  "${CMAKE_BINARY_DIR}/vm"
  "${CMAKE_SOURCE_DIR}/vm/inc"
  "${CMAKE_BINARY_DIR}/vm/inc"
)

target_link_libraries(
  code_gen_tests
  GTest::gtest_main
  ubpf
  ubpf_settings
)

include(GoogleTest)
gtest_discover_tests(code_gen_tests
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )