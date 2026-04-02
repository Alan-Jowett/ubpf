#!/bin/bash
# Copyright (c) Microsoft Corporation
# SPDX-License-Identifier: Apache-2.0

# Wrapper script for running ubpf_plugin with JIT mode via QEMU
qemu-mips64el -cpu I6500 ../bin/ubpf_plugin "$@" --jit
