// Copyright (c) 2022-present, IO Visor Project
// SPDX-License-Identifier: MIT

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "bpf_assembler.h"
#include "test_helpers.h"
extern "C"
{
#include "ubpf.h"
}


std::tuple<std::string, std::string, std::vector<ebpf_inst>>
parse_test_file(const std::filesystem::path &data_file)
{
    enum class _state
    {
        state_ignore,
        state_assembly,
        state_raw,
        state_result,
        state_memory,
    } state = _state::state_ignore;

    std::stringstream data_out;
    std::ifstream data_in(data_file);

    std::string result;
    std::string mem;
    std::string line;
    while (std::getline(data_in, line))
    {
        if (line.find("--") != std::string::npos)
        {
            if (line.find("asm") != std::string::npos)
            {
                state = _state::state_assembly;
                continue;
            }
            else if (line.find("result") != std::string::npos)
            {
                state = _state::state_result;
                continue;
            }
            else if (line.find("mem") != std::string::npos)
            {
                state = _state::state_memory;
                continue;
            }
            else if (line.find("raw") != std::string::npos)
            {
                state = _state::state_ignore;
                continue;
            }
            else if (line.find("result") != std::string::npos)
            {
                state = _state::state_result;
                continue;
            }
            else if (line.find("no register offset") != std::string::npos)
            {
                state = _state::state_ignore;
                continue;
            }
            else if (line.find(" c") != std::string::npos)
            {
                state = _state::state_ignore;
                continue;
            }
            else
            {
                std::cout << "Unknown directive " << line << std::endl;
                state = _state::state_ignore;
                continue;
            }
        }
        if (line.empty())
        {
            continue;
        }

        switch (state)
        {
        case _state::state_assembly:
            if (line.find("#") != std::string::npos)
            {
                line = line.substr(0, line.find("#"));
            }
            data_out << line << std::endl;
            break;
        case _state::state_result:
            result = line;
            break;
        case _state::state_memory:
            mem += std::string(" ") + line;
            break;
        default:
            continue;
        }
    }

    if (result.find("0x") != std::string::npos)
    {
        result = result.substr(result.find("0x") + 2);
    }
    data_out.seekg(0);
    auto instructions = bpf_assembler(data_out);
    return {mem, result, instructions};
}

ubpf_vm *
prepare_ubpf_vm(const std::vector<ebpf_inst> instructions)
{
    ubpf_vm *vm = ubpf_create();
    if (vm == nullptr)
        throw std::runtime_error("Failed to create VM");

    char *error = nullptr;
    for (auto &[key, value] : helper_functions)
    {
        if (ubpf_register(vm, key, "unnamed", reinterpret_cast<void*>(value)) != 0)
            throw std::runtime_error("Failed to register helper function");
    }

    if (ubpf_set_unwind_function_index(vm, 5) != 0)
        throw std::runtime_error("Failed to set unwind function index");

    if (ubpf_load(vm, instructions.data(), static_cast<uint32_t>(instructions.size() * sizeof(ebpf_inst)), &error) != 0)
        throw std::runtime_error("Failed to load program: " + std::string(error));

    return vm;
}

void run_ubpf_jit_test(const std::filesystem::path &data_file)
{
    if (data_file.filename().string().find("err") != std::string::npos)
    {
        GTEST_SKIP();
        return;
    }
    auto [mem, result, instructions] = parse_test_file(data_file);
    if (result.empty() || instructions.empty())
    {
        GTEST_SKIP();
        return;
    }

    char *error = nullptr;
    ubpf_vm *vm = prepare_ubpf_vm(instructions);
    if (vm == nullptr)
        throw std::runtime_error("Failed to create VM");

    ubpf_jit_fn jit = ubpf_compile(vm, &error);
    if (jit == nullptr)
        throw std::runtime_error("Failed to compile program: " + std::string(error));

    std::vector<uint8_t> input_buffer;

    if (!mem.empty())
    {
        std::stringstream ss(mem);
        uint32_t value;
        while (ss >> std::hex >> value)
        {
            input_buffer.push_back(static_cast<uint8_t>(value));
        }
    }

    uint64_t expected_result = std::stoull(result, nullptr, 16);

    uint64_t actual_result = jit(input_buffer.data(), input_buffer.size());

    if (actual_result != expected_result) {
        std::cout << "Expected: " << expected_result << " Actual: " << actual_result << std::endl;
        throw std::runtime_error("Result mismatch");
    }

    ubpf_destroy(vm);
}

void run_ubpf_interpret_test(const std::filesystem::path &data_file)
{
    if (data_file.filename().string().find("err") != std::string::npos)
    {
        GTEST_SKIP();
        return;
    }
    auto [mem, result, instructions] = parse_test_file(data_file);
    if (result.empty() || instructions.empty())
    {
        GTEST_SKIP();
        return;
    }

    ubpf_vm *vm = prepare_ubpf_vm(instructions);
    if (vm == nullptr)
        throw std::runtime_error("Failed to create VM");

    std::vector<uint8_t> input_buffer;

    if (!mem.empty())
    {
        std::stringstream ss(mem);
        uint32_t value;
        while (ss >> std::hex >> value)
        {
            input_buffer.push_back(static_cast<uint8_t>(value));
        }
    }

    uint64_t expected_result = std::stoull(result, nullptr, 16);

    uint64_t actual_result;
    if (ubpf_exec(vm, input_buffer.data(), input_buffer.size(), &actual_result) != 0)
        throw std::runtime_error("Failed to execute program");

    if (actual_result != expected_result) {
        std::cout << "Expected: " << expected_result << " Actual: " << actual_result << std::endl;
        throw std::runtime_error("Result mismatch");
    }

    ubpf_destroy(vm);
}

class ubpf_test : public ::testing::TestWithParam<std::filesystem::path>
{
};

TEST_P(ubpf_test, jit)
{
    run_ubpf_jit_test(GetParam());
}

TEST_P(ubpf_test, interpret)
{
    run_ubpf_interpret_test(GetParam());
}

std::vector<std::filesystem::path> get_test_files()
{
    std::vector<std::filesystem::path> result;
    for (auto &p : std::filesystem::directory_iterator("tests/"))
    {
        if (p.path().extension() == ".data")
        {
            result.push_back(p.path());
        }
    }
    return result;
}

INSTANTIATE_TEST_SUITE_P(ubpf_tests, ubpf_test, ::testing::ValuesIn(get_test_files()));
