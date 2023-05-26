# uBPF

Userspace eBPF VM

[![Main](https://github.com/iovisor/ubpf/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/iovisor/ubpf/actions/workflows/main.yml)
[![Coverage Status](https://coveralls.io/repos/iovisor/ubpf/badge.svg?branch=main&service=github)](https://coveralls.io/github/iovisor/ubpf?branch=master)

## About

This project aims to create an Apache-licensed library for executing eBPF programs. The primary implementation of eBPF lives in the Linux kernel, but due to its GPL license it can't be used in many projects.

[Linux documentation for the eBPF instruction set](https://www.kernel.org/doc/Documentation/networking/filter.txt)

[Instruction set reference](https://github.com/iovisor/bpf-docs/blob/master/eBPF.md)

[API Documentation](https://iovisor.github.io/ubpf)

This project includes an eBPF assembler, disassembler, interpreter (for all platforms),
and JIT compiler (for x86-64 and Arm64 targets).

## Checking Out

Before following any of the instructions below for [building](#building-with-cmake),
[testing](#running-the-tests), [contributing](#contributing), etc, please be
sure to properly check out the source code which requires properly initializing submodules:

```
git submodule update --init --recursive
```

## Preparing system for build

In order to prepare your system to successfully generate the build system using CMake, follow the platform-specific instructions below.

### Windows

Building, compiling and testing on Windows requires an installation of Visual Studio (*not* VS Code -- the MSVC compiler is necessary!).

> Note: There are free-to-use versions of Visual Studio for individual developers. These versions are known as the [community version](https://visualstudio.microsoft.com/vs/community/).

You *can* build, compile and test uBPF using [VS Code but Visual Studio is still required](https://code.visualstudio.com/docs/cpp/config-msvc).

The other requirement is that you have [`nuget.exe`](https://learn.microsoft.com/en-us/nuget/install-nuget-client-tools) in your `PATH`. You can determine if your host meets this criteria by testing whether

```console
> nuget.exe
```

produces output about how to execute the program. With `nuget.exe` installed, the `cmake` configuration system will download all the required developer libraries as it configures the build system.

### macOS
First, make sure that you have the XCode Command Line Tools installed:

```console
$ xcode-select --install
```

Installing the XCode Command Linux Tools will install Apple's version of the Clang compiler and other developer-support tools.

uBpf requires that your host have several support libraries installed. The easiest way to configure your host to meet these requirements,

1. Install [homebrew](https://brew.sh/)
2. Install [boost](https://www.boost.org/):
```console
$ brew install boost
```
3. Install [LLVM](https://llvm.org/) (and related tools):
```console
$ brew install llvm cmake
$ brew install clang-format
```

Installing LLVM from Homebrew is optional for developing and using uBPF on macOS. It is required if you plan on compiling/creating eBPF programs by compiling LLVM and storing them in ELF files. If you *do* install LLVM from Homebrew, add `-DUBPF_ALTERNATE_LLVM_PATH=/opt/homebrew/opt/llvm/bin` to the `cmake` configuration command:

```console
cmake -S . -B build -DUBPF_ENABLE_TESTS=true -DUBPF_ALTERNATE_LLVM_PATH=/opt/homebrew/opt/llvm/bin
```

### Linux

```bash
./scripts/build-libbpf.sh
```

## Building with CMake
Note: This works on Windows, Linux, and MacOS, provided the prerequisites are installed.
For a more detailed list of instructions, including list of dependencies,
see [CI/CD steps](.github/workflows/main.yml).
```
cmake -S . -B build -DUBPF_ENABLE_TESTS=true
cmake --build build --config Debug
```
## Running the tests

### Linux and MacOS native
```
cmake --build build --target test --
```

### Linux aarch64 cross-compile
Note: This requires qemu and the aarch64 toolchain.

To install the required tools (assuming Debian derived distro). For a more
detailed list of instructions, including list of dependencies, see
[CI/CD steps](.github/workflows/main.yml).
```
sudo apt install -y \
    g++-aarch64-linux-gnu \
    gcc-aarch64-linux-gnu \
    qemu-user
```

Building for aarch64.
```
# Build bpf_conformance natively as a workaround to missing boost libraries on aarch64.
cmake -S external/bpf_conformance -B build_bpf_conformance
cmake --build build_bpf_conformance
# Build ubpf for aarch64
cmake -S . -B build -DUBPF_ENABLE_TESTS=true -DUBPF_SKIP_EXTERNAL=true \
    -DBPF_CONFORMANCE_RUNNER="$(pwd)/build_bpf_conformance/bin/bpf_conformance_runner" \
    -DCMAKE_TOOLCHAIN_FILE=cmake/arm64.cmake
cmake --build build
cmake --build build --target test --
```

### Windows
```
ctest --test-dir build
```

## Contributing

## Running the tests (Linux)
To run the tests, you first need to build the vm code then use nosetests to execute the tests. Note: The tests have some dependencies that need to be present. See the [.travis.yml](https://github.com/iovisor/ubpf/blob/main/.travis.yml) for details.

### Preparing Code Contributions

We aim to maintain code coverage with every code change. The CI/CD pipeline will verify this invariant as part of the contribution process. However, you can calculate code coverage locally by

```console
coveralls --gcov-options '\-lp' -i $PWD/vm/ubpf_vm.c -i $PWD/vm/ubpf_jit_x86_64.c -i $PWD/vm/ubpf_loader.c
```

We also aim to maintain a consistent code format. The pre-commit git hooks configured for the uBPF repository will guarantee that code changes match the format we expect. In order for those hooks to work effectively, you must have `clang-format` installed and available on your system.

## Compiling C to eBPF

You'll need [Clang 11](https://github.com/llvm/llvm-project/releases/tag/llvmorg-11.1.0).

    clang -g -O2 -target bpf -c prog.c -o prog.o

You can then pass the contents of `prog.o` to `ubpf_test`.

## License

Copyright 2015, Big Switch Networks, Inc. Licensed under the Apache License, Version 2.0
<LICENSE.txt or http://www.apache.org/licenses/LICENSE-2.0>.
