// Copyright (c) uBPF contributors
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

#include "asm_unmarshal.hpp"
#include "crab_verifier.hpp"
#include "platform.hpp"

extern "C"
{
#define ebpf_inst ebpf_inst_ubpf
#include "ebpf.h"
#include "ubpf.h"
#undef ebpf_inst
}

#include "test_helpers.h"
#include <cassert>

typedef struct _ubpf_context
{
    uint64_t data;
    uint64_t data_end;
    uint64_t stack_start;
    uint64_t stack_end;
} ubpf_context_t;

ebpf_context_descriptor_t g_ebpf_context_descriptor_ubpf = {
    .size = sizeof(ubpf_context_t),
    .data = 0,
    .end = 8,
    .meta = -1,
};


EbpfProgramType g_ubpf_program_type = {
    .name = "ubpf",
    .context_descriptor = &g_ebpf_context_descriptor_ubpf,
    .platform_specific_data = 0,
    .section_prefixes = {},
    .is_privileged = false,
};

EbpfProgramType ubpf_get_program_type(const std::string& section, const std::string& path)
{
    UNREFERENCED_PARAMETER(section);
    UNREFERENCED_PARAMETER(path);
    return g_ubpf_program_type;
}

EbpfMapType ubpf_get_map_type(uint32_t platform_specific_type)
{
    UNREFERENCED_PARAMETER(platform_specific_type);
    return {};
}

EbpfHelperPrototype ubpf_get_helper_prototype(int32_t n)
{
    UNREFERENCED_PARAMETER(n);
    return {};
}

bool ubpf_is_helper_usable(int32_t n)
{
    UNREFERENCED_PARAMETER(n);
    return false;
}


ebpf_platform_t g_ebpf_platform_ubpf_fuzzer = {
    .get_program_type = ubpf_get_program_type,
    .get_helper_prototype = ubpf_get_helper_prototype,
    .is_helper_usable = ubpf_is_helper_usable,
    .map_record_size = 0,
    .parse_maps_section = nullptr,
    .get_map_descriptor = nullptr,
    .get_map_type = ubpf_get_map_type,
    .resolve_inner_map_references = nullptr,
    .supported_conformance_groups = bpf_conformance_groups_t::default_groups,
};


uint64_t test_helpers_dispatcher(uint64_t p0, uint64_t p1,uint64_t p2,uint64_t p3, uint64_t p4, unsigned int idx, void* cookie) {
    UNREFERENCED_PARAMETER(cookie);
    return helper_functions[idx](p0, p1, p2, p3, p4);
}

bool test_helpers_validator(unsigned int idx, const struct ubpf_vm *vm) {
    UNREFERENCED_PARAMETER(vm);
    return helper_functions.contains(idx);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size);

int null_printf(FILE* stream, const char* format, ...)
{
    if (!stream) {
        return 0;
    }
    if (!format) {
        return 0;
    }
    return 0;
}

bool verify_bpf_byte_code(const std::vector<uint8_t>& program_code)
try
{
    std::ostringstream error;
    auto instruction_array = reinterpret_cast<const ebpf_inst*>(program_code.data());
    size_t instruction_count = program_code.size() / sizeof(ebpf_inst);
    const ebpf_platform_t* platform = &g_ebpf_platform_ubpf_fuzzer;
    std::vector<ebpf_inst> instructions{instruction_array, instruction_array + instruction_count};
    program_info info{
        .platform = platform,
        .type = g_ubpf_program_type,
    };
    std::string section;
    std::string file;
    raw_program raw_prog{file, section, 0, {}, instructions, info};

    std::variant<InstructionSeq, std::string> prog_or_error = unmarshal(raw_prog);
    if (!std::holds_alternative<InstructionSeq>(prog_or_error)) {
        //std::cout << "Failed to unmarshal program : " << std::get<std::string>(prog_or_error) << std::endl;
        return false;
    }
    InstructionSeq& prog = std::get<InstructionSeq>(prog_or_error);

    // First try optimized for the success case.
    ebpf_verifier_options_t options = ebpf_verifier_default_options;
    ebpf_verifier_stats_t stats;
    options.check_termination = true;
    options.store_pre_invariants = true;
    options.simplify = false;

    std::ostringstream error_stream;

    return ebpf_verify_program(error_stream, prog, raw_prog.info, &options, &stats);
}
catch (const std::exception& ex)
{
    return false;
}

typedef std::unique_ptr<ubpf_vm, decltype(&ubpf_destroy)> ubpf_vm_ptr;

/**
 * @brief Create a ubpf vm object and load the program code into it.
 *
 * @param[in] program_code The program code to load into the VM.
 * @return A unique pointer to the ubpf_vm object or nullptr if the VM could not be created.
 */
ubpf_vm_ptr create_ubpf_vm(const std::vector<uint8_t>& program_code)
{
    // Automatically free the VM when it goes out of scope.
    std::unique_ptr<ubpf_vm, decltype(&ubpf_destroy)> vm(ubpf_create(), ubpf_destroy);

    if (vm == nullptr) {
        // Failed to create the VM.
        // This is not interesting, as the fuzzer input is invalid.
        // Do not add it to the corpus.
        return {nullptr, nullptr};
    }

    ubpf_toggle_undefined_behavior_check(vm.get(), true);

    char* error_message = nullptr;

    ubpf_set_error_print(vm.get(), null_printf);

    if (ubpf_load(vm.get(), program_code.data(), program_code.size(), &error_message) != 0) {
        // The program failed to load, due to a validation error.
        // This is not interesting, as the fuzzer input is invalid.
        // Do not add it to the corpus.
        free(error_message);
        return {nullptr, nullptr};
    }

    ubpf_toggle_bounds_check(vm.get(), true);

    if (ubpf_register_external_dispatcher(vm.get(), test_helpers_dispatcher, test_helpers_validator) != 0) {
        // Failed to register the external dispatcher.
        // This is not interesting, as the fuzzer input is invalid.
        // Do not add it to the corpus.
        return {nullptr, nullptr};
    }

    if (ubpf_set_instruction_limit(vm.get(), 10000, nullptr) != 0) {
        // Failed to set the instruction limit.
        // This is not interesting, as the fuzzer input is invalid.
        // Do not add it to the corpus.
        return {nullptr, nullptr};
    }

    return vm;
}


bool ubpf_is_packet(ubpf_context_t* context, uint64_t register_value)
{
    return register_value >= context->data && register_value < context->data_end;
}

bool ubpf_is_context(ubpf_context_t* context, uint64_t register_value)
{
    return register_value >= reinterpret_cast<uint64_t>(context) && register_value < reinterpret_cast<uint64_t>(context) + sizeof(ubpf_context_t);
}

bool ubpf_is_stack(ubpf_context_t* context, uint64_t register_value)
{
    return register_value >= context->stack_start && register_value < context->stack_end;
}

void
ubpf_debug_function(
    void* context, int program_counter, const uint64_t registers[16], const uint8_t* stack_start, size_t stack_length, uint64_t register_mask)
{
    ubpf_context_t* ubpf_context = reinterpret_cast<ubpf_context_t*>(context);
    UNREFERENCED_PARAMETER(stack_start);
    UNREFERENCED_PARAMETER(stack_length);

    std::string label = std::to_string(program_counter) + ":-1";

    if (program_counter == 0) {
        return;
    }

    // Build set of string constraints from the register values.
    std::set<std::string> constraints;
    for (int i = 0; i < 10; i++) {
        if ((register_mask & (1 << i)) == 0) {
            continue;
        }
        uint64_t reg = registers[i];
        std::string register_name = "r" + std::to_string(i);

        if (ubpf_is_packet(ubpf_context, reg)) {
            constraints.insert(register_name + ".type=packet");
            constraints.insert(register_name + ".packet_offset=" + std::to_string(reg - ubpf_context->data));
            constraints.insert(register_name +".packet_size=" + std::to_string(ubpf_context->data_end - ubpf_context->data));
        }
        else if (ubpf_is_context(ubpf_context, reg)) {
            constraints.insert(register_name + ".type=ctx");
            constraints.insert(register_name + ".ctx_offset=" + std::to_string(reg - reinterpret_cast<uint64_t>(ubpf_context)));
        } else if (ubpf_is_stack(ubpf_context, reg)) {
            constraints.insert(register_name + ".type=stack");
            constraints.insert(register_name + ".stack_offset=" + std::to_string(reg - ubpf_context->stack_start));
        }
        else {
            constraints.insert("r" + std::to_string(i) + ".uvalue=" + std::to_string(registers[i]));
            constraints.insert("r" + std::to_string(i) + ".svalue=" + std::to_string(static_cast<int64_t>(registers[i])));
        }
    }


    // Call ebpf_check_constraints_at_label with the set of string constraints at this label.

    std::ostringstream os;

    if (!ebpf_check_constraints_at_label(os, label, constraints)) {
        std::cerr << "Label: " << label << std::endl;
        std::cerr << os.str() << std::endl;
        throw std::runtime_error("ebpf_check_constraints_at_label failed");
    }
}


ubpf_context_t ubpf_context_from(std::vector<uint8_t>& memory, std::vector<uint8_t>& ubpf_stack)
{
    ubpf_context_t context;
    context.data = reinterpret_cast<uint64_t>(memory.data());
    context.data_end = context.data + memory.size();
    context.stack_start = reinterpret_cast<uint64_t>(ubpf_stack.data());
    context.stack_end = context.stack_start + ubpf_stack.size();
    return context;
}

/**
 * @brief Invoke the ubpf interpreter with the given program code and input memory.
 *
 * @param[in] program_code The program code to execute.
 * @param[in,out] memory The input memory to use when executing the program. May be modified by the program.
 * @param[in,out] ubpf_stack The stack to use when executing the program. May be modified by the program.
 * @param[out] interpreter_result The result of the program execution.
 * @return true if the program executed successfully.
 * @return false if the program failed to execute.
 */
bool call_ubpf_interpreter(const std::vector<uint8_t>& program_code, std::vector<uint8_t>& memory, std::vector<uint8_t>& ubpf_stack, uint64_t& interpreter_result)
{
    auto vm = create_ubpf_vm(program_code);

    ubpf_context_t context = ubpf_context_from(memory, ubpf_stack);

    if (vm == nullptr) {
        // VM creation failed.
        return false;
    }

    ubpf_register_debug_fn(vm.get(), &context, ubpf_debug_function);

    // Execute the program using the input memory.
    if (ubpf_exec_ex(vm.get(), &context, 0, &interpreter_result, ubpf_stack.data(), ubpf_stack.size()) != 0) {
        // VM execution failed.
        return false;
    }

    // VM execution succeeded.
    return true;
}

/**
 * @brief Execute the given program code using the ubpf JIT.
 *
 * @param[in] program_code The program code to execute.
 * @param[in,out] memory The input memory to use when executing the program. May be modified by the program.
 * @param[in,out] ubpf_stack The stack to use when executing the program. May be modified by the program.
 * @param[out] interpreter_result The result of the program execution.
 * @return true if the program executed successfully.
 * @return false if the program failed to execute.
 */
bool call_ubpf_jit(const std::vector<uint8_t>& program_code, std::vector<uint8_t>& memory, std::vector<uint8_t>& ubpf_stack, uint64_t& jit_result)
{
    auto vm = create_ubpf_vm(program_code);

    ubpf_context_t context = ubpf_context_from(memory, ubpf_stack);

    char* error_message = nullptr;

    if (vm == nullptr) {
        // VM creation failed.
        return false;
    }

    auto fn = ubpf_compile_ex(vm.get(), &error_message, JitMode::ExtendedJitMode);

    if (fn == nullptr) {
        free(error_message);

        // Compilation failed.
        return false;
    }

    jit_result = fn(&context, 0, ubpf_stack.data(), ubpf_stack.size());

    // Compilation succeeded.
    return true;
}

/**
 * @brief Copy the program and memory from the input buffer into separate buffers.
 *
 * @param[in] data The input buffer from the fuzzer.
 * @param[in] size The size of the input buffer.
 * @param[out] program The program code extracted from the input buffer.
 * @param[out] memory The input memory extracted from the input buffer.
 * @return true if the input buffer was successfully split.
 * @return false if the input buffer is malformed.
 */
bool split_input(const uint8_t* data, std::size_t size, std::vector<uint8_t>& program, std::vector<uint8_t>& memory)
{
    if (size < 4)
        return false;

    uint32_t program_length = *reinterpret_cast<const uint32_t*>(data);
    uint32_t memory_length = size - 4 - program_length;
    const uint8_t* program_start = data + 4;
    const uint8_t* memory_start = data + 4 + program_length;

    if (program_length > size) {
        // The program length is larger than the input size.
        // This is not interesting, as the fuzzer input is invalid.
        return false;
    }

    if (program_length == 0) {
        // The program length is zero.
        // This is not interesting, as the fuzzer input is invalid.
        return false;
    }

    if (program_length + 4u > size) {
        // The program length is larger than the input size.
        // This is not interesting, as the fuzzer input is invalid.
        return false;
    }

    if ((program_length % sizeof(ebpf_inst)) != 0) {
        // The program length needs to be a multiple of sizeof(ebpf_inst_t).
        // This is not interesting, as the fuzzer input is invalid.
        return false;
    }

    // Copy any input memory into a writable buffer.
    if (memory_length > 0) {
        memory.resize(memory_length);
        std::memcpy(memory.data(), memory_start, memory_length);
    }

    program.resize(program_length);
    std::memcpy(program.data(), program_start, program_length);

    return true;
}

/**
 * @brief Accept an input buffer and size.
 *
 * @param[in] data Pointer to the input buffer.
 * @param[in] size Size of the input buffer.
 * @return -1 if the input is invalid
 * @return 0 if the input is valid and processed.
 */
int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size)
{
    // Assume the fuzzer input is as follows:
    // 32-bit program length
    // program byte
    // test data

    std::vector<uint8_t> program;
    std::vector<uint8_t> memory;
    std::vector<uint8_t> ubpf_stack(3*4096);

    if (!split_input(data, size, program, memory)) {
        // The input is invalid. Not interesting.
        return -1;
    }

    if (!verify_bpf_byte_code(program)) {
        // The program failed verification.
        return 0;
    }

    uint64_t interpreter_result = 0;
    uint64_t jit_result = 0;

    if (!call_ubpf_interpreter(program, memory, ubpf_stack, interpreter_result)) {
        // Failed to load or execute the program in the interpreter.
        // This is not interesting, as the fuzzer input is invalid.
        return 0;
    }

    if (!split_input(data, size, program, memory)) {
        // The input was successfully split, but failed to split again.
        // This should not happen.
        assert(!"split_input failed");
    }

    if (!call_ubpf_jit(program, memory, ubpf_stack, jit_result)) {
        // Failed to load or execute the program in the JIT.
        // This is not interesting, as the fuzzer input is invalid.
        return 0;
    }

    // If interpreter_result is not equal to jit_result, raise a fatal signal
    if (interpreter_result != jit_result) {
        printf("%lx ubpf_stack\n", reinterpret_cast<uintptr_t>(ubpf_stack.data()) + ubpf_stack.size());
        printf("interpreter_result: %lx\n", interpreter_result);
        printf("jit_result: %lx\n", jit_result);
        throw std::runtime_error("interpreter_result != jit_result");
    }

    // Program executed successfully.
    // Add it to the corpus as it may be interesting.
    return 0;
}
