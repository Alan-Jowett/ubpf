#
# Copyright (c) 2022-present, IO Visor Project
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

# MIPS64r6 cross-compilation toolchain using clang + musl (static).
# Produces fully static R6 binaries that run on QEMU without needing
# an R6-compatible sysroot (Ubuntu only ships pre-R6 MIPS64 libraries).

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mips64)
set(CMAKE_SYSTEM_VERSION 1)

# Use clang as a cross-compiler — it natively supports any target triple.
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Target MIPS64r6 little-endian with musl libc
set(CMAKE_C_FLAGS_INIT "--target=mips64el-linux-musl -march=mips64r6 -static")
set(CMAKE_CXX_FLAGS_INIT "--target=mips64el-linux-musl -march=mips64r6 -static -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "--target=mips64el-linux-musl -march=mips64r6 -static -fuse-ld=lld -stdlib=libc++ -lc++abi")

# Use LLVM tools for cross-compilation
set(CMAKE_AR llvm-ar)
set(CMAKE_RANLIB llvm-ranlib)
set(CMAKE_STRIP llvm-strip)

# Sysroot will be set by the CI workflow after building musl
# set(CMAKE_SYSROOT "/path/to/musl-mips64r6el-sysroot")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
