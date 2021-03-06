/*
 * Copyright 2015 Big Switch Networks, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef UBPF_H
#define UBPF_H

#include <stdint.h>
#include <stddef.h>

struct ubpf_vm;
typedef uint64_t (*ubpf_jit_fn)(void *mem, size_t mem_len);

struct ubpf_map_def {
    unsigned int type;
    unsigned int key_size;
    unsigned int value_size;
    unsigned int max_entries;
    unsigned int map_flags;
    unsigned int inner_map_idx;
    unsigned int numa_node;
};

typedef uint64_t (*ubpf_map_create_fn)(void *context, const struct ubpf_map_def * map_def);
typedef uint64_t (*ubpf_map_resolver_fn)(void *context, uint64_t fd);
typedef uint64_t (*ubpf_helper_resolver_fn)(void * context, uint32_t helper);

struct ubpf_vm *ubpf_create(void);
void ubpf_destroy(struct ubpf_vm *vm);

/*
 * Enable / disable bounds_check
 *
 * Bounds check is enabled by default, but it may be too restrictive
 * Pass true to enable, false to disable
 * Returns previous state
 */
bool toggle_bounds_check(struct ubpf_vm *vm, bool enable);

/*
 * Register an external function
 *
 * The immediate field of a CALL instruction is an index into an array of
 * functions registered by the user. This API associates a function with
 * an index.
 *
 * 'name' should be a string with a lifetime longer than the VM.
 *
 * Returns 0 on success, -1 on error.
 */
int ubpf_register(struct ubpf_vm *vm, unsigned int idx, const char *name, void *fn);

/*
 * Register a function used to create a map and return a fd.
 *
 * When loading an ELF file, the ubpf_map_create_fn is invoked for each map.
 *
 * Returns 0 on sucess, -1 on error.
 */
int ubpf_register_map_create(struct ubpf_vm *vm, void *context, ubpf_map_create_fn create_fn);

/*
 * Register a function used to resolve helper function addresses.
 *
 * When processing a EBPF_OP_CALL instruction, lookup the helper 
 * address.
 *
 * Returns 0 on sucess, -1 on error.
 */
int ubpf_register_helper_resolver(struct ubpf_vm *vm, void *context, ubpf_helper_resolver_fn resolver_fn);

/*
 * Register a function used to resolve map FD to address.
 *
 * When processing a lddw instruction with the map_fd bit set, invoke the
 * call back to resolve map fd to address.
 *
 * Returns 0 on sucess, -1 on error.
 */
int ubpf_register_map_resolver(struct ubpf_vm *vm, void *context, ubpf_map_resolver_fn resolver_fn);

/*
 * Load code into a VM
 *
 * This must be done before calling ubpf_exec or ubpf_compile and after
 * registering all functions.
 *
 * 'code' should point to eBPF bytecodes and 'code_len' should be the size in
 * bytes of that buffer.
 *
 * Returns 0 on success, -1 on error. In case of error a pointer to the error
 * message will be stored in 'errmsg' and should be freed by the caller.
 */
int ubpf_load(struct ubpf_vm *vm, const void *code, uint32_t code_len, char **errmsg);

/*
 * Load code from an ELF file
 *
 * This must be done before calling ubpf_exec or ubpf_compile and after
 * registering all functions.
 *
 * 'elf' should point to a copy of an ELF file in memory and 'elf_len' should
 * be the size in bytes of that buffer.
 *
 * The ELF file must be 64-bit little-endian with a single text section
 * containing the eBPF bytecodes. This is compatible with the output of
 * Clang.
 *
 * Returns 0 on success, -1 on error. In case of error a pointer to the error
 * message will be stored in 'errmsg' and should be freed by the caller.
 */
int ubpf_load_elf(struct ubpf_vm *vm, const void *elf, size_t elf_len, char **errmsg);

int ubpf_load_elf_by_name(struct ubpf_vm *vm, const void *elf, size_t elf_len, const char * name, char **errmsg);


uint64_t ubpf_exec(const struct ubpf_vm *vm, void *mem, size_t mem_len);

ubpf_jit_fn ubpf_compile(struct ubpf_vm *vm, char **errmsg);

/* 
 * Translate the eBPF byte code to x64 machine code and store in buffer.
 * 
 * This must be called after registering all functions.
 * 
 * Returns 0 on success, -1 on error. In case of error a pointer to the error
 * message will be stored in 'errmsg' and should be freed by the caller.
 */
int ubpf_translate(struct ubpf_vm *vm, uint8_t * buffer, size_t * size, char **errmsg);

#endif
