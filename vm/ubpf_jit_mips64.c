// Copyright (c) 2015 Big Switch Networks, Inc
// SPDX-License-Identifier: Apache-2.0

/*
 * Copyright 2015 Big Switch Networks, Inc
 * Copyright 2017 Google Inc.
 * Copyright 2022 Linaro Limited
 * Copyright 2025 MIPS64r6 JIT backend
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
 *
 * References:
 * [MIPS64r6]: MIPS Architecture for Programmers Volume II-A: The MIPS64 Instruction Set Manual, Rev 6
 */

#include <stdint.h>
#define _GNU_SOURCE
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <assert.h>
#include "ubpf_int.h"
#include "ubpf_jit_support.h"

#if defined(__mips64) || defined(__mips__)

#if !defined(_countof)
#define _countof(array) (sizeof(array) / sizeof(array[0]))
#endif

// All MIPS64 general-purpose registers (n64 ABI).
enum MipsRegister
{
    REG_ZERO = 0,   // Hardwired zero
    REG_AT = 1,     // Assembler temporary
    REG_V0 = 2,     // Return value 0
    REG_V1 = 3,     // Return value 1
    REG_A0 = 4,     // Argument 0
    REG_A1 = 5,     // Argument 1
    REG_A2 = 6,     // Argument 2
    REG_A3 = 7,     // Argument 3
    REG_A4 = 8,     // Argument 4 (n64 ABI)
    REG_A5 = 9,     // Argument 5
    REG_A6 = 10,    // Argument 6
    REG_A7 = 11,    // Argument 7
    REG_T4 = 12,    // Temporary
    REG_T5 = 13,    // Temporary
    REG_T6 = 14,    // Temporary
    REG_T7 = 15,    // Temporary
    REG_S0 = 16,    // Callee-saved
    REG_S1 = 17,    // Callee-saved
    REG_S2 = 18,    // Callee-saved
    REG_S3 = 19,    // Callee-saved
    REG_S4 = 20,    // Callee-saved
    REG_S5 = 21,    // Callee-saved
    REG_S6 = 22,    // Callee-saved
    REG_S7 = 23,    // Callee-saved
    REG_T8 = 24,    // Temporary
    REG_T9 = 25,    // Temporary
    REG_K0 = 26,    // Kernel reserved
    REG_K1 = 27,    // Kernel reserved
    REG_GP = 28,    // Global pointer
    REG_SP = 29,    // Stack pointer
    REG_FP = 30,    // Frame pointer (s8)
    REG_RA = 31,    // Return address
};

// Scratch/temp register assignments (Spec §2.2).
static enum MipsRegister temp_register = REG_T4;           // Large immediate materialization, constant blinding
static enum MipsRegister temp_div_register = REG_T5;       // Division scratch, atomic scratch
static enum MipsRegister offset_register = REG_T6;         // Address computation, atomic address
static enum MipsRegister temp_extra_register = REG_T7;     // Additional scratch
static enum MipsRegister helper_table_register = REG_S5;   // Helper table base
static enum MipsRegister context_register = REG_S6;        // Context/cookie pointer

// Number of eBPF registers
#define REGISTER_MAP_SIZE 11

// Register map (Spec §2.1):
//   BPF        MIPS64      Usage
//   r0         v0          Return value
//   r1 - r5    a0 - a4     Function parameters, caller-saved
//   r6 - r9    s0 - s3     Callee-saved registers
//   r10        s4          Frame pointer (read-only in BPF)
//              t4          Temp - large immediate materialization
//              t5          Temp - division/atomic scratch
//              t6          Temp - address computation
//              t7          Temp - additional scratch
//              s5          Helper table base pointer
//              s6          Context/cookie pointer
static enum MipsRegister register_map[REGISTER_MAP_SIZE] = {
    REG_V0,  // BPF R0 = return value
    REG_A0,  // BPF R1 = param 1 (context pointer)
    REG_A1,  // BPF R2 = param 2
    REG_A2,  // BPF R3 = param 3
    REG_A3,  // BPF R4 = param 4
    REG_A4,  // BPF R5 = param 5
    REG_S0,  // BPF R6 = callee-saved
    REG_S1,  // BPF R7 = callee-saved
    REG_S2,  // BPF R8 = callee-saved
    REG_S3,  // BPF R9 = callee-saved
    REG_S4,  // BPF R10 = frame pointer
};

/* Return the MIPS64 register for the given eBPF register */
static enum MipsRegister
map_register(int r)
{
    assert(r < REGISTER_MAP_SIZE);
    return register_map[r % REGISTER_MAP_SIZE];
}

/* Some forward declarations. */
static void
emit_mips64_immediate(struct jit_state* state, enum MipsRegister rd, int64_t imm);
static void
divmod(struct jit_state* state, uint8_t opcode, enum MipsRegister dst, enum MipsRegister src, int16_t offset);

static uint32_t inline
align_to(uint32_t amount, uint64_t boundary)
{
    return (amount + (boundary - 1)) & ~(boundary - 1);
}

static void
emit_bytes(struct jit_state* state, void* data, uint32_t len)
{
    if (!(len <= state->size && state->offset <= state->size - len)) {
        state->jit_status = NotEnoughSpace;
        return;
    }

    if ((state->offset + len) > state->size) {
        state->offset = state->size;
        return;
    }
    memcpy(state->buf + state->offset, data, len);
    state->offset += len;
}

static void
emit_instruction(struct jit_state* state, uint32_t instr)
{
    emit_bytes(state, &instr, 4);
}

/* ========================================================================
 * MIPS64r6 instruction encoding constants and format helpers
 * ======================================================================== */

// R-type: [opcode(6) | rs(5) | rt(5) | rd(5) | sa(5) | funct(6)]
static inline uint32_t
mips_r_type(uint32_t opcode, uint32_t rs, uint32_t rt, uint32_t rd, uint32_t sa, uint32_t funct)
{
    return (opcode << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) |
           ((rd & 0x1F) << 11) | ((sa & 0x1F) << 6) | (funct & 0x3F);
}

// I-type: [opcode(6) | rs(5) | rt(5) | immediate(16)]
static inline uint32_t
mips_i_type(uint32_t opcode, uint32_t rs, uint32_t rt, uint16_t imm)
{
    return (opcode << 26) | ((rs & 0x1F) << 21) | ((rt & 0x1F) << 16) | (imm & 0xFFFF);
}

// Compact branch with 26-bit offset: [opcode(6) | offset(26)]
static inline uint32_t
mips_j26_type(uint32_t opcode, uint32_t offset26)
{
    return (opcode << 26) | (offset26 & 0x03FFFFFF);
}

// Compact branch with 21-bit offset: [opcode(6) | rs(5) | offset(21)]
static inline uint32_t
mips_cmpbr21_type(uint32_t opcode, uint32_t rs, uint32_t offset21)
{
    return (opcode << 26) | ((rs & 0x1F) << 21) | (offset21 & 0x001FFFFF);
}

// Major opcodes
#define MIPS_OP_SPECIAL    0x00
#define MIPS_OP_SPECIAL3   0x1F
#define MIPS_OP_DADDIU     0x19
#define MIPS_OP_LUI        0x0F
#define MIPS_OP_ORI        0x0D
#define MIPS_OP_ANDI       0x0C
#define MIPS_OP_XORI       0x0E
#define MIPS_OP_LB         0x20
#define MIPS_OP_LBU        0x24
#define MIPS_OP_LH         0x21
#define MIPS_OP_LHU        0x25
#define MIPS_OP_LW         0x23
#define MIPS_OP_LWU        0x27
#define MIPS_OP_LD         0x37
#define MIPS_OP_SB         0x28
#define MIPS_OP_SH         0x29
#define MIPS_OP_SW         0x2B
#define MIPS_OP_SD         0x3F

// R6 compact branches
#define MIPS_OP_BC         0x32   // Unconditional compact branch (26-bit offset)
#define MIPS_OP_BALC       0x3A   // Branch and link compact (26-bit offset)
#define MIPS_OP_POP06      0x08   // BEQC (rs < rt), BOVC (rs >= rt)
#define MIPS_OP_POP16      0x18   // BNEC (rs < rt), BNVC (rs >= rt)
#define MIPS_OP_POP26      0x16   // BGEC (rs < rt), BLTC (rs >= rt) [signed]
#define MIPS_OP_POP27      0x17   // BGEUC (rs < rt), BLTUC (rs >= rt) [unsigned]
#define MIPS_OP_BEQZC      0x36   // Branch if equal zero compact (21-bit offset)
#define MIPS_OP_BNEZC      0x3E   // Branch if not equal zero compact (21-bit offset)

// SPECIAL function codes
#define MIPS_FN_SLL        0x00
#define MIPS_FN_SRL        0x02
#define MIPS_FN_SRA        0x03
#define MIPS_FN_SLLV       0x04
#define MIPS_FN_SRLV       0x06
#define MIPS_FN_SRAV       0x07
#define MIPS_FN_JALR       0x09
#define MIPS_FN_DSLLV      0x14
#define MIPS_FN_DSRLV      0x16
#define MIPS_FN_DSRAV      0x17
#define MIPS_FN_ADDU       0x21
#define MIPS_FN_SUBU       0x23
#define MIPS_FN_AND        0x24
#define MIPS_FN_OR         0x25
#define MIPS_FN_XOR        0x26
#define MIPS_FN_NOR        0x27
#define MIPS_FN_SLT        0x2A
#define MIPS_FN_SLTU       0x2B
#define MIPS_FN_DADDU      0x2D
#define MIPS_FN_DSUBU      0x2F
#define MIPS_FN_DSLL       0x38
#define MIPS_FN_DSRL       0x3A
#define MIPS_FN_DSRA       0x3B
#define MIPS_FN_DSLL32     0x3C
#define MIPS_FN_DSRL32     0x3E
#define MIPS_FN_DSRA32     0x3F

// R6 multiply/divide function codes (all SPECIAL, sa field distinguishes)
#define MIPS_FN_MUL_R6     0x18   // sa=2 for MUL, sa=3 for MUH
#define MIPS_FN_DMUL_R6    0x1C   // sa=2 for DMUL, sa=3 for DMUH
#define MIPS_FN_DIV_R6     0x1A   // sa=2 for DIV, sa=3 for MOD
#define MIPS_FN_DIVU_R6    0x1B   // sa=2 for DIVU, sa=3 for MODU
#define MIPS_FN_DDIV_R6    0x1E   // sa=2 for DDIV, sa=3 for DMOD
#define MIPS_FN_DDIVU_R6   0x1F   // sa=2 for DDIVU, sa=3 for DMODU
// sa values for multiply/divide distinction
#define MIPS_MUL_SA        0x02   // MUL/DIV/DIVU etc.
#define MIPS_MOD_SA        0x03   // MUH/MOD/MODU etc.

// SPECIAL3 function codes
#define MIPS_FN_BSHFL      0x20   // Contains SEB (sa=0x10), SEH (sa=0x18), WSBH (sa=0x02)
#define MIPS_FN_DBSHFL     0x24   // Contains DSBH (sa=0x02), DSHD (sa=0x05)
// sa values for BSHFL/DBSHFL
#define MIPS_BSHFL_WSBH    0x02
#define MIPS_BSHFL_SEB     0x10
#define MIPS_BSHFL_SEH     0x18
#define MIPS_DBSHFL_DSBH   0x02
#define MIPS_DBSHFL_DSHD   0x05

// ROTR: SRL with rs=1
#define MIPS_ROTR_RS       0x01

// R6 LL/SC (SPECIAL3 opcode, with 9-bit offset)
// LL:  [011111 | base(5) | rt(5) | offset(9) | 0 | 110110]
// SC:  [011111 | base(5) | rt(5) | offset(9) | 0 | 100110]
// LLD: [011111 | base(5) | rt(5) | offset(9) | 0 | 110111]
// SCD: [011111 | base(5) | rt(5) | offset(9) | 0 | 100111]
#define MIPS_R6_LL_FN      0x36
#define MIPS_R6_SC_FN      0x26
#define MIPS_R6_LLD_FN     0x37
#define MIPS_R6_SCD_FN     0x27

/* ========================================================================
 * Specific instruction emit functions
 * ======================================================================== */

// Spec §3.1: ALU64 register operations
static void
emit_daddu(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_DADDU));
}

static void
emit_dsubu(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_DSUBU));
}

static void
emit_or(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_OR));
}

static void
emit_and(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_AND));
}

static void
emit_xor(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_XOR));
}

static void
emit_addu(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_ADDU));
}

static void
emit_slt(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_SLT));
}

static void
emit_sltu(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_SLTU));
}

// I-type ALU
static void
emit_daddiu(struct jit_state* state, enum MipsRegister rt, enum MipsRegister rs, int16_t imm)
{
    emit_instruction(state, mips_i_type(MIPS_OP_DADDIU, rs, rt, (uint16_t)imm));
}

static void
emit_ori(struct jit_state* state, enum MipsRegister rt, enum MipsRegister rs, uint16_t imm)
{
    emit_instruction(state, mips_i_type(MIPS_OP_ORI, rs, rt, imm));
}

static void
emit_andi(struct jit_state* state, enum MipsRegister rt, enum MipsRegister rs, uint16_t imm)
{
    emit_instruction(state, mips_i_type(MIPS_OP_ANDI, rs, rt, imm));
}

static void
emit_xori(struct jit_state* state, enum MipsRegister rt, enum MipsRegister rs, uint16_t imm)
{
    emit_instruction(state, mips_i_type(MIPS_OP_XORI, rs, rt, imm));
}

static void
emit_lui(struct jit_state* state, enum MipsRegister rt, uint16_t imm)
{
    emit_instruction(state, mips_i_type(MIPS_OP_LUI, REG_ZERO, rt, imm));
}

// Shift instructions (fixed amount) - Spec §3.1
static void
emit_dsll(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_DSLL));
}

static void
emit_dsrl(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_DSRL));
}

static void
emit_dsra(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_DSRA));
}

static void
emit_dsll32(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_DSLL32));
}

static void
emit_dsrl32(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_DSRL32));
}

static void
emit_dsra32(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_DSRA32));
}

// 32-bit shifts
static void
emit_sll(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_SLL));
}

static void
emit_srl(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_SRL));
}

static void
emit_sra(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, 0, rt, rd, sa, MIPS_FN_SRA));
}

// Variable shifts
static void
emit_dsllv(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, enum MipsRegister rs)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_DSLLV));
}

static void
emit_dsrlv(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, enum MipsRegister rs)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_DSRLV));
}

static void
emit_dsrav(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, enum MipsRegister rs)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_DSRAV));
}

static void
emit_sllv(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, enum MipsRegister rs)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_SLLV));
}

static void
emit_srlv(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, enum MipsRegister rs)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_SRLV));
}

static void
emit_srav(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, enum MipsRegister rs)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, 0, MIPS_FN_SRAV));
}

// R6 multiply/divide
static void
emit_dmul(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, MIPS_MUL_SA, MIPS_FN_DMUL_R6));
}

static void
emit_ddiv(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, MIPS_MUL_SA, MIPS_FN_DDIV_R6));
}

static void
emit_dmod(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, MIPS_MOD_SA, MIPS_FN_DDIV_R6));
}

static void
emit_ddivu(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, MIPS_MUL_SA, MIPS_FN_DDIVU_R6));
}

static void
emit_dmodu(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rs, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, rs, rt, rd, MIPS_MOD_SA, MIPS_FN_DDIVU_R6));
}

// Load/Store
static void
emit_lb(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_LB, base, rt, (uint16_t)offset));
}

static void
emit_lbu(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_LBU, base, rt, (uint16_t)offset));
}

static void
emit_lh(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_LH, base, rt, (uint16_t)offset));
}

static void
emit_lhu(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_LHU, base, rt, (uint16_t)offset));
}

static void
emit_lw(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_LW, base, rt, (uint16_t)offset));
}

static void
emit_lwu(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_LWU, base, rt, (uint16_t)offset));
}

static void
emit_ld(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_LD, base, rt, (uint16_t)offset));
}

static void
emit_sb(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_SB, base, rt, (uint16_t)offset));
}

static void
emit_sh(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_SH, base, rt, (uint16_t)offset));
}

static void
emit_sw(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_SW, base, rt, (uint16_t)offset));
}

static void
emit_sd(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_SD, base, rt, (uint16_t)offset));
}

// R6 compact indirect jumps (no delay slot) — Spec §8.4
// JIC rt, offset: PC = GPR[rt] + sign_extend(offset). No delay slot.
// Encoding: POP66 (0x36) with rs=0.
static void
emit_jic(struct jit_state* state, enum MipsRegister rt, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_BEQZC, REG_ZERO, rt, (uint16_t)offset));
}

// JIALC rt, offset: $ra = PC + 4, PC = GPR[rt] + sign_extend(offset). No delay slot.
// Encoding: POP76 (0x3E) with rs=0.
static void
emit_jialc(struct jit_state* state, enum MipsRegister rt, int16_t offset)
{
    emit_instruction(state, mips_i_type(MIPS_OP_BNEZC, REG_ZERO, rt, (uint16_t)offset));
}

// Special instructions (SPECIAL3 opcode)

// SEB: Sign-extend byte (Spec §3.4)
static void
emit_seb(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL3, 0, rt, rd, MIPS_BSHFL_SEB, MIPS_FN_BSHFL));
}

// SEH: Sign-extend halfword (Spec §3.4)
static void
emit_seh(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL3, 0, rt, rd, MIPS_BSHFL_SEH, MIPS_FN_BSHFL));
}

// WSBH: Word Swap Bytes within Halfwords (Spec §3.5)
static void
emit_wsbh(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL3, 0, rt, rd, MIPS_BSHFL_WSBH, MIPS_FN_BSHFL));
}

// DSBH: Doubleword Swap Bytes within Halfwords
static void
emit_dsbh(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL3, 0, rt, rd, MIPS_DBSHFL_DSBH, MIPS_FN_DBSHFL));
}

// DSHD: Doubleword Swap Halfwords within Doublewords
static void
emit_dshd(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt)
{
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL3, 0, rt, rd, MIPS_DBSHFL_DSHD, MIPS_FN_DBSHFL));
}

// ROTR: Rotate Right (encoded as SRL with rs=1)
static void
emit_rotr(struct jit_state* state, enum MipsRegister rd, enum MipsRegister rt, uint8_t sa)
{
    assert(sa < 32);
    emit_instruction(state, mips_r_type(MIPS_OP_SPECIAL, MIPS_ROTR_RS, rt, rd, sa, MIPS_FN_SRL));
}

// R6 LL/SC (atomic) — 9-bit signed offset
// Format: [SPECIAL3(6) | base(5) | rt(5) | offset(9) | 0(1) | funct(6)]
static inline uint32_t
mips_r6_llsc(uint32_t base, uint32_t rt, int16_t offset9, uint32_t funct)
{
    return (MIPS_OP_SPECIAL3 << 26) | ((base & 0x1F) << 21) | ((rt & 0x1F) << 16) |
           (((uint32_t)(offset9 & 0x1FF)) << 7) | (funct & 0x3F);
}

static void
emit_ll(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    assert(offset >= -256 && offset <= 255);
    emit_instruction(state, mips_r6_llsc(base, rt, offset, MIPS_R6_LL_FN));
}

static void
emit_sc(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    assert(offset >= -256 && offset <= 255);
    emit_instruction(state, mips_r6_llsc(base, rt, offset, MIPS_R6_SC_FN));
}

static void
emit_lld(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    assert(offset >= -256 && offset <= 255);
    emit_instruction(state, mips_r6_llsc(base, rt, offset, MIPS_R6_LLD_FN));
}

static void
emit_scd(struct jit_state* state, enum MipsRegister rt, enum MipsRegister base, int16_t offset)
{
    assert(offset >= -256 && offset <= 255);
    emit_instruction(state, mips_r6_llsc(base, rt, offset, MIPS_R6_SCD_FN));
}

/* ========================================================================
 * 64-bit immediate materialization (Spec §3.9)
 * ======================================================================== */

// Materialize a 64-bit immediate into a register.
// Uses the shortest instruction sequence possible.
static void
emit_mips64_immediate(struct jit_state* state, enum MipsRegister rd, int64_t imm)
{
    uint64_t uimm = (uint64_t)imm;

    if (uimm == 0) {
        // OR rd, $zero, $zero (1 instruction)
        emit_or(state, rd, REG_ZERO, REG_ZERO);
    } else if (imm >= -32768 && imm <= 32767) {
        // DADDIU rd, $zero, imm (1 instruction, sign-extended)
        emit_daddiu(state, rd, REG_ZERO, (int16_t)imm);
    } else if (uimm <= 0xFFFF) {
        // ORI rd, $zero, imm (1 instruction, zero-extended)
        emit_ori(state, rd, REG_ZERO, (uint16_t)uimm);
    } else if ((int64_t)(int32_t)(uimm & 0xFFFFFFFF) == imm) {
        // 32-bit value: LUI + ORI (2 instructions)
        uint16_t upper = (uint16_t)((uimm >> 16) & 0xFFFF);
        uint16_t lower = (uint16_t)(uimm & 0xFFFF);
        emit_lui(state, rd, upper);
        if (lower != 0) {
            emit_ori(state, rd, rd, lower);
        }
    } else {
        // Full 64-bit: LUI + ORI + DSLL + ORI + DSLL + ORI (up to 6 instructions)
        uint16_t bits_63_48 = (uint16_t)((uimm >> 48) & 0xFFFF);
        uint16_t bits_47_32 = (uint16_t)((uimm >> 32) & 0xFFFF);
        uint16_t bits_31_16 = (uint16_t)((uimm >> 16) & 0xFFFF);
        uint16_t bits_15_0  = (uint16_t)(uimm & 0xFFFF);

        emit_lui(state, rd, bits_63_48);
        if (bits_47_32 != 0) {
            emit_ori(state, rd, rd, bits_47_32);
        }
        emit_dsll(state, rd, rd, 16);
        if (bits_31_16 != 0) {
            emit_ori(state, rd, rd, bits_31_16);
        }
        emit_dsll(state, rd, rd, 16);
        if (bits_15_0 != 0) {
            emit_ori(state, rd, rd, bits_15_0);
        }
    }
}

/* ========================================================================
 * Constant blinding (Spec §5.1)
 * ======================================================================== */

// EMIT_MIPS64_IMMEDIATE: Emit an immediate with optional constant blinding.
// If blinding is enabled, materializes (V^R) into dest, R into temp, then XORs.
#define EMIT_MIPS64_IMMEDIATE(vm, state, dest, imm) \
    do { \
        if ((vm)->constant_blinding_enabled) { \
            uint64_t _blind_key = ubpf_generate_blinding_constant(); \
            uint64_t _blind_val = (uint64_t)(imm) ^ _blind_key; \
            emit_mips64_immediate((state), (dest), (int64_t)_blind_val); \
            emit_mips64_immediate((state), temp_register, (int64_t)_blind_key); \
            emit_xor((state), (dest), (dest), temp_register); \
        } else { \
            emit_mips64_immediate((state), (dest), (int64_t)(imm)); \
        } \
    } while (0)

/* ========================================================================
 * Zero-extension helper (Spec §3.2)
 * ======================================================================== */

// Zero-extend a 32-bit value in a register to 64 bits.
// Spec §3.2: DSLL32 + DSRL32 clears upper 32 bits.
static void
emit_zero_extend_32(struct jit_state* state, enum MipsRegister rd)
{
    emit_dsll32(state, rd, rd, 0);
    emit_dsrl32(state, rd, rd, 0);
}

/* ========================================================================
 * Utility functions
 * ======================================================================== */

static bool
is_imm_op(const struct ebpf_inst* inst)
{
    int class = inst->opcode & EBPF_CLS_MASK;
    bool is_imm = (inst->opcode & EBPF_SRC_REG) == EBPF_SRC_IMM;
    bool is_endian = (inst->opcode & EBPF_ALU_OP_MASK) == 0xd0;
    bool is_neg = (inst->opcode & EBPF_ALU_OP_MASK) == 0x80;
    bool is_call = inst->opcode == EBPF_OP_CALL;
    bool is_exit = inst->opcode == EBPF_OP_EXIT;
    bool is_ja = inst->opcode == EBPF_OP_JA || inst->opcode == EBPF_OP_JA32;
    bool is_alu = (class == EBPF_CLS_ALU || class == EBPF_CLS_ALU64) && !is_endian && !is_neg;
    bool is_jmp = (class == EBPF_CLS_JMP && !is_ja && !is_call && !is_exit);
    bool is_jmp32 = (class == EBPF_CLS_JMP32 && inst->opcode != EBPF_OP_JA32);
    bool is_store = class == EBPF_CLS_ST;
    return (is_imm && (is_alu || is_jmp || is_jmp32)) || is_store;
}

static bool
is_simple_imm(const struct ebpf_inst* inst)
{
    return inst->imm >= -32768 && inst->imm <= 32767;
}

static uint8_t
to_reg_op(uint8_t opcode)
{
    return opcode | EBPF_SRC_REG;
}

/* ========================================================================
 * Local branch patching helpers
 *
 * These are used within functions (divmod, atomics, etc.) where branch
 * targets are local to the generated sequence — NOT for cross-BPF-instruction
 * jumps, which use the patchable_relative system.
 * ======================================================================== */

// Patch a 16-bit branch offset (BEQC/BNEC/BGEC/etc.)
// MIPS R6 compact branches: target = PC + 4 + sign_extend(offset << 2)
// So offset = (target - (branch_loc + 4)) / 4
static void
patch_branch_offset16(struct jit_state* state, uint32_t branch_loc)
{
    int32_t rel = ((int32_t)(state->offset - (branch_loc + 4))) / 4;
    uint32_t* instr = (uint32_t*)(state->buf + branch_loc);
    *instr = (*instr & 0xFFFF0000u) | ((uint32_t)rel & 0xFFFFu);
}

// Patch a 21-bit branch offset (BEQZC/BNEZC)
static void
patch_branch_offset21(struct jit_state* state, uint32_t branch_loc)
{
    int32_t rel = ((int32_t)(state->offset - (branch_loc + 4))) / 4;
    uint32_t* instr = (uint32_t*)(state->buf + branch_loc);
    *instr = (*instr & 0xFFE00000u) | ((uint32_t)rel & 0x001FFFFFu);
}

// Patch a 26-bit branch offset (BC/BALC)
static void
patch_branch_offset26(struct jit_state* state, uint32_t branch_loc)
{
    int32_t rel = ((int32_t)(state->offset - (branch_loc + 4))) / 4;
    uint32_t* instr = (uint32_t*)(state->buf + branch_loc);
    *instr = (*instr & 0xFC000000u) | ((uint32_t)rel & 0x03FFFFFFu);
}

// Emit BNEC with placeholder for local patching (returns branch location).
// Handles R6 register ordering constraint: BNEC requires rs < rt.
// Since != is symmetric, we can swap operands freely.
static uint32_t
emit_bnec_local(struct jit_state* state, enum MipsRegister rs, enum MipsRegister rt)
{
    if (rs > rt) {
        enum MipsRegister tmp = rs; rs = rt; rt = tmp;
    }
    uint32_t loc = state->offset;
    emit_instruction(state, mips_i_type(MIPS_OP_POP16, rs, rt, 0));
    return loc;
}

// Emit BEQZC with placeholder for local patching
static uint32_t
emit_beqzc_local(struct jit_state* state, enum MipsRegister rs)
{
    uint32_t loc = state->offset;
    emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, rs, 0));
    return loc;
}

// Emit BNEZC with placeholder for local patching
static uint32_t
emit_bnezc_local(struct jit_state* state, enum MipsRegister rs)
{
    uint32_t loc = state->offset;
    emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, rs, 0));
    return loc;
}

// Emit BC (unconditional) with placeholder for local patching
static uint32_t
emit_bc_local(struct jit_state* state)
{
    uint32_t loc = state->offset;
    emit_instruction(state, mips_j26_type(MIPS_OP_BC, 0));
    return loc;
}

/* ========================================================================
 * Division and modulo helper (Spec §3.3)
 * ======================================================================== */

static void
divmod(struct jit_state* state, uint8_t opcode, enum MipsRegister dst, enum MipsRegister src, int16_t offset)
{
    bool is_64 = (opcode & EBPF_CLS_MASK) == EBPF_CLS_ALU64;
    bool is_div = (opcode & EBPF_ALU_OP_MASK) == EBPF_ALU_OP_DIV;
    bool is_signed = (offset == 1);

    // For 32-bit operations: sign-extend operands into scratch registers
    // to avoid UNPREDICTABLE behavior with MIPS64 32-bit instructions,
    // and to make overflow guard comparisons correct (F-R008/F-R009).
    // We use 64-bit instructions + truncation instead of 32-bit instructions.
    enum MipsRegister op_dst = dst;
    enum MipsRegister op_src = src;
    if (!is_64) {
        // Sign-extend 32-bit values to 64-bit for correct operation
        emit_sll(state, temp_register, dst, 0);    // sign-extend dst → $t4
        emit_sll(state, temp_div_register, src, 0); // sign-extend src → $t5
        op_dst = temp_register;
        op_src = temp_div_register;
    }

    if (is_signed) {
        // Spec §3.3: Signed division/modulo with overflow guards

        // SMOD guard: any value % -1 == 0
        if (!is_div) {
            // Check if src == -1 (use sign-extended value for 32-bit)
            emit_daddiu(state, temp_extra_register, REG_ZERO, -1);
            uint32_t not_neg1 = emit_bnec_local(state, op_src, temp_extra_register);

            // src == -1: result is 0
            emit_or(state, dst, REG_ZERO, REG_ZERO);
            uint32_t done_mod = emit_bc_local(state);

            patch_branch_offset16(state, not_neg1);

            // Division-by-zero check
            uint32_t nonzero = emit_bnezc_local(state, op_src);
            // div-by-zero: dst unchanged for mod
            uint32_t done_dbz = emit_bc_local(state);
            patch_branch_offset21(state, nonzero);

            // Normal modulo — always use 64-bit to avoid UNPREDICTABLE
            emit_dmod(state, dst, op_dst, op_src);

            patch_branch_offset26(state, done_mod);
            patch_branch_offset26(state, done_dbz);
        } else {
            // SDIV: INT_MIN / -1 overflow guard + div-by-zero

            // Division-by-zero check first
            uint32_t nonzero = emit_bnezc_local(state, op_src);
            emit_or(state, dst, REG_ZERO, REG_ZERO);
            uint32_t done_dbz = emit_bc_local(state);
            patch_branch_offset21(state, nonzero);

            // Check src == -1
            emit_daddiu(state, temp_extra_register, REG_ZERO, -1);
            uint32_t not_neg1 = emit_bnec_local(state, op_src, temp_extra_register);

            // src == -1: check if dst == INT_MIN
            if (is_64) {
                // Materialize INT64_MIN
                emit_lui(state, temp_extra_register, 0x8000);
                emit_dsll32(state, temp_extra_register, temp_extra_register, 0);
            } else {
                // Materialize INT32_MIN (sign-extended to 64-bit for comparison)
                emit_lui(state, temp_extra_register, 0x8000);
                emit_sll(state, temp_extra_register, temp_extra_register, 0);
            }
            uint32_t not_intmin = emit_bnec_local(state, op_dst, temp_extra_register);
            // dst == INT_MIN && src == -1: result is INT_MIN
            emit_or(state, dst, temp_extra_register, REG_ZERO);
            uint32_t done_ovf = emit_bc_local(state);

            patch_branch_offset16(state, not_neg1);
            patch_branch_offset16(state, not_intmin);

            // Normal division — always use 64-bit
            emit_ddiv(state, dst, op_dst, op_src);

            patch_branch_offset26(state, done_dbz);
            patch_branch_offset26(state, done_ovf);
        }
    } else {
        // Spec §3.3: Unsigned division/modulo

        // For 32-bit unsigned, zero-extend (not sign-extend) the operands
        if (!is_64) {
            emit_zero_extend_32(state, temp_register);   // re-zero-extend dst copy
            emit_zero_extend_32(state, temp_div_register); // re-zero-extend src copy
        }

        // Division-by-zero check
        uint32_t nonzero = emit_bnezc_local(state, op_src);

        if (is_div) {
            // div-by-zero: dst = 0
            emit_or(state, dst, REG_ZERO, REG_ZERO);
        }
        // mod-by-zero: dst unchanged (just skip)
        uint32_t done = emit_bc_local(state);

        patch_branch_offset21(state, nonzero);

        // Normal operation — always use 64-bit to avoid UNPREDICTABLE
        if (is_div) {
            if (is_64) {
                emit_ddivu(state, dst, dst, src);
            } else {
                emit_ddivu(state, dst, op_dst, op_src);
            }
        } else {
            if (is_64) {
                emit_dmodu(state, dst, dst, src);
            } else {
                emit_dmodu(state, dst, op_dst, op_src);
            }
        }

        patch_branch_offset26(state, done);
    }
}

/* ========================================================================
 * Stack frame layout (Spec §4.1)
 * ======================================================================== */

// Callee-saved registers to preserve: $s0-$s6, $fp, $ra (9 registers)
#define MIPS64_CALLEE_SAVED_COUNT 9
#define MIPS64_CALLEE_SAVED_SIZE  (MIPS64_CALLEE_SAVED_COUNT * 8)  // 72 bytes

// Additional slot for saving $ra during helper calls (separate from local-call $ra slot)
#define MIPS64_HELPER_RA_SAVE_SIZE 8

// Compute total frame size (16-byte aligned)
static uint32_t
compute_frame_size(uint32_t bpf_stack_size)
{
    uint32_t total = MIPS64_CALLEE_SAVED_SIZE + MIPS64_HELPER_RA_SAVE_SIZE + bpf_stack_size;
    return align_to(total, 16);
}

// Offsets within the frame (from $sp after prologue)
// Layout (high to low address):
//   frame_size - 8:   $ra
//   frame_size - 16:  $fp
//   frame_size - 24:  $s0 (BPF R6)
//   frame_size - 32:  $s1 (BPF R7)
//   frame_size - 40:  $s2 (BPF R8)
//   frame_size - 48:  $s3 (BPF R9)
//   frame_size - 56:  $s4 (BPF R10)
//   frame_size - 64:  $s5 (helper table base)
//   frame_size - 72:  $s6 (context register)
//   frame_size - 80:  helper $ra save slot
//   [bpf_stack_size bytes]: BPF stack (grows downward)
//   0: $sp

#define RA_OFFSET(frame_size)           ((int16_t)((frame_size) - 8))
#define FP_OFFSET(frame_size)           ((int16_t)((frame_size) - 16))
#define S0_OFFSET(frame_size)           ((int16_t)((frame_size) - 24))
#define S1_OFFSET(frame_size)           ((int16_t)((frame_size) - 32))
#define S2_OFFSET(frame_size)           ((int16_t)((frame_size) - 40))
#define S3_OFFSET(frame_size)           ((int16_t)((frame_size) - 48))
#define S4_OFFSET(frame_size)           ((int16_t)((frame_size) - 56))
#define S5_OFFSET(frame_size)           ((int16_t)((frame_size) - 64))
#define S6_OFFSET(frame_size)           ((int16_t)((frame_size) - 72))
#define HELPER_RA_OFFSET(frame_size)    ((int16_t)((frame_size) - 80))
#define BPF_STACK_OFFSET                (MIPS64_CALLEE_SAVED_SIZE + MIPS64_HELPER_RA_SAVE_SIZE)

/* ========================================================================
 * Prologue (Spec §4.1)
 * ======================================================================== */

static void
emit_jit_prologue(struct jit_state* state, uint32_t bpf_stack_size)
{
    uint32_t frame_size = compute_frame_size(bpf_stack_size);
    state->stack_size = frame_size;

    // Spec §4.1: Allocate stack frame
    // Handle large frame sizes that don't fit in 16-bit DADDIU immediate
    if (frame_size <= 32767) {
        emit_daddiu(state, REG_SP, REG_SP, -(int16_t)frame_size);
    } else {
        emit_mips64_immediate(state, temp_register, (int64_t)frame_size);
        emit_dsubu(state, REG_SP, REG_SP, temp_register);
    }

    // Save return address and frame pointer
    emit_sd(state, REG_RA, REG_SP, RA_OFFSET(frame_size));
    emit_sd(state, REG_FP, REG_SP, FP_OFFSET(frame_size));
    emit_or(state, REG_FP, REG_SP, REG_ZERO);  // $fp = $sp

    // Save callee-saved registers (BPF R6-R10, helper table base, context)
    emit_sd(state, REG_S0, REG_SP, S0_OFFSET(frame_size));
    emit_sd(state, REG_S1, REG_SP, S1_OFFSET(frame_size));
    emit_sd(state, REG_S2, REG_SP, S2_OFFSET(frame_size));
    emit_sd(state, REG_S3, REG_SP, S3_OFFSET(frame_size));
    emit_sd(state, REG_S4, REG_SP, S4_OFFSET(frame_size));
    emit_sd(state, REG_S5, REG_SP, S5_OFFSET(frame_size));
    emit_sd(state, REG_S6, REG_SP, S6_OFFSET(frame_size));

    // Spec §4.1: Load helper table base into $s5 via BALC to get PC.
    // This will be resolved by resolve_leas() after code generation.
    {
        DECLARE_PATCHABLE_SPECIAL_TARGET(load_helper_tgt, LoadHelperTable)
        note_lea(state, load_helper_tgt);
        // BALC .Lpc — puts PC+4 into $ra
        emit_instruction(state, mips_j26_type(MIPS_OP_BALC, 0));
        // DADDIU $s5, $ra, <data_offset> — placeholder offset, patched by resolve_leas
        emit_daddiu(state, helper_table_register, REG_RA, 0);
    }

    // Setup BPF frame pointer (R10 = $s4 = top of BPF stack area)
    // BPF stack sits between $sp and $sp + (frame_size - metadata_size).
    // R10 points to the TOP so BPF accesses [R10 - offset] stay in-frame.
    if (state->jit_mode == BasicJitMode) {
        int32_t bpf_top = (int32_t)(frame_size - BPF_STACK_OFFSET);
        if (bpf_top >= -32768 && bpf_top <= 32767) {
            emit_daddiu(state, REG_S4, REG_SP, (int16_t)bpf_top);
        } else {
            emit_mips64_immediate(state, REG_S4, bpf_top);
            emit_daddu(state, REG_S4, REG_SP, REG_S4);
        }
    } else {
        // ExtendedJitMode: BPF stack is caller-provided ($a2=start, $a3=len)
        // $s4 = $a2 + $a3
        emit_daddu(state, REG_S4, REG_A2, REG_A3);
    }

    // Save context pointer for helper calls
    // Spec §4.1: $s6 preserves context across BPF execution
    emit_or(state, context_register, REG_A0, REG_ZERO);

    // Record entry location
    state->entry_loc = state->offset;
}

/* ========================================================================
 * Epilogue (Spec §4.1)
 * ======================================================================== */

static void
emit_jit_epilogue(struct jit_state* state, uint32_t bpf_stack_size)
{
    uint32_t frame_size = compute_frame_size(bpf_stack_size);

    // Record exit location
    state->exit_loc = state->offset;

    // Restore $sp from $fp in case we are returning from inside a local call
    // where $sp was adjusted. This is the same pattern as ARM64's epilogue.
    emit_or(state, REG_SP, REG_FP, REG_ZERO);  // $sp = $fp

    // Restore callee-saved registers (reverse order)
    emit_ld(state, REG_S6, REG_SP, S6_OFFSET(frame_size));
    emit_ld(state, REG_S5, REG_SP, S5_OFFSET(frame_size));
    emit_ld(state, REG_S4, REG_SP, S4_OFFSET(frame_size));
    emit_ld(state, REG_S3, REG_SP, S3_OFFSET(frame_size));
    emit_ld(state, REG_S2, REG_SP, S2_OFFSET(frame_size));
    emit_ld(state, REG_S1, REG_SP, S1_OFFSET(frame_size));
    emit_ld(state, REG_S0, REG_SP, S0_OFFSET(frame_size));

    // Restore frame pointer and return address
    emit_ld(state, REG_FP, REG_SP, FP_OFFSET(frame_size));
    emit_ld(state, REG_RA, REG_SP, RA_OFFSET(frame_size));

    // Deallocate stack frame
    if (frame_size <= 32767) {
        emit_daddiu(state, REG_SP, REG_SP, (int16_t)frame_size);
    } else {
        emit_mips64_immediate(state, temp_register, (int64_t)frame_size);
        emit_daddu(state, REG_SP, REG_SP, temp_register);
    }

    // Return — result already in $v0 (BPF R0)
    // Use R6 compact JIC (no delay slot) — Spec §8.4
    emit_jic(state, REG_RA, 0);
}

/* ========================================================================
 * Helper function dispatch (Spec §3.12, §6)
 * ======================================================================== */

// Emit a static helper call by index.
// BPF R1-R5 ($a0-$a4) are already in place — zero-cost mapping.
static void
emit_helper_call(struct jit_state* state, uint32_t helper_index)
{
    uint32_t frame_size = state->stack_size;

    // Save $ra (clobbered by JALR) — Spec §3.12
    emit_sd(state, REG_RA, REG_SP, HELPER_RA_OFFSET(frame_size));

    // 6th parameter: context cookie — Spec §6.3
    emit_or(state, REG_A5, context_register, REG_ZERO);

    // Load function pointer from helper table via $s5
    int32_t table_offset = (int32_t)(helper_index * 8);
    if (table_offset >= -32768 && table_offset <= 32767) {
        emit_daddiu(state, temp_register, helper_table_register, (int16_t)table_offset);
    } else {
        emit_mips64_immediate(state, temp_register, table_offset);
        emit_daddu(state, temp_register, helper_table_register, temp_register);
    }
    emit_ld(state, temp_register, temp_register, 0);

    // Call helper — R6 compact JIALC (no delay slot)
    emit_jialc(state, temp_register, 0);

    // Restore $ra
    emit_ld(state, REG_RA, REG_SP, HELPER_RA_OFFSET(frame_size));
}

// Emit a dispatcher call — Spec §3.12 (dynamic dispatcher)
static void
emit_dispatcher_call(struct jit_state* state, uint32_t helper_index)
{
    uint32_t frame_size = state->stack_size;

    // Save $ra
    emit_sd(state, REG_RA, REG_SP, HELPER_RA_OFFSET(frame_size));

    // Load dispatcher function pointer from data section via $s5
    // The dispatcher pointer is stored at dispatcher_loc offset
    DECLARE_PATCHABLE_SPECIAL_TARGET(disp_tgt, ExternalDispatcher)
    note_load(state, disp_tgt);
    emit_ld(state, temp_register, helper_table_register, 0); // placeholder, patched

    // $a0-$a4 = BPF R1-R5 (already mapped)
    // 6th param: helper index
    if (helper_index <= 0xFFFF) {
        emit_ori(state, REG_A5, REG_ZERO, (uint16_t)helper_index);
    } else {
        emit_mips64_immediate(state, REG_A5, (int64_t)helper_index);
    }
    // 7th param: context cookie
    emit_or(state, REG_A6, context_register, REG_ZERO);

    // Call dispatcher — R6 compact JIALC (no delay slot)
    emit_jialc(state, temp_register, 0);

    // Restore $ra
    emit_ld(state, REG_RA, REG_SP, HELPER_RA_OFFSET(frame_size));
}

/* ========================================================================
 * Local function calls (Spec §7)
 * ======================================================================== */

// Local-call $ra save slot: stored at 0($sp) of the callee's sub-frame
#define LOCAL_RA_SLOT_OFFSET 0

static void
emit_local_call(struct jit_state* state, struct ubpf_vm* vm, uint32_t target_pc)
{
    // Save $ra to native stack (current frame's local-call $ra slot)
    emit_sd(state, REG_RA, REG_SP, LOCAL_RA_SLOT_OFFSET);

    // Save callee-saved BPF registers R6-R9 to BPF stack via $s4 (BPF R10)
    emit_sd(state, REG_S0, REG_S4, -8);    // BPF R6
    emit_sd(state, REG_S1, REG_S4, -16);   // BPF R7
    emit_sd(state, REG_S2, REG_S4, -24);   // BPF R8
    emit_sd(state, REG_S3, REG_S4, -32);   // BPF R9

    // Adjust BPF frame pointer
    uint16_t stack_usage = ubpf_stack_usage_for_local_func(vm, target_pc);
    emit_mips64_immediate(state, temp_register, -(int64_t)stack_usage);
    emit_daddu(state, REG_S4, REG_S4, temp_register);

    // Allocate native stack space for callee's $ra save slot (16-byte aligned)
    emit_daddiu(state, REG_SP, REG_SP, -16);

    // Branch-and-link to local function
    {
        DECLARE_PATCHABLE_REGULAR_EBPF_TARGET(call_tgt, target_pc)
        emit_patchable_relative(
            state->local_calls, state->offset, call_tgt, state->num_local_calls++);
        emit_instruction(state, mips_j26_type(MIPS_OP_BALC, 0)); // placeholder
    }

    // After return: deallocate callee's native frame
    emit_daddiu(state, REG_SP, REG_SP, 16);
}

/* ========================================================================
 * Data section emission — helper table and dispatcher pointer
 * Emitted after code, referenced via $s5 (loaded in prologue)
 * ======================================================================== */

static uint32_t
emit_dispatched_external_helper_address(struct jit_state* state, uint64_t dispatcher_addr)
{
    uint32_t address_loc = state->offset;
    emit_bytes(state, &dispatcher_addr, sizeof(uint64_t));
    return address_loc;
}

static uint32_t
emit_helper_table(struct jit_state* state, struct ubpf_vm* vm)
{
    uint32_t table_loc = state->offset;
    for (int i = 0; i < MAX_EXT_FUNCS; i++) {
        emit_bytes(state, &vm->ext_funcs[i], sizeof(uint64_t));
    }
    return table_loc;
}

/* ========================================================================
 * translate() — main BPF instruction translation (Spec §3)
 * ======================================================================== */

static int
translate(struct ubpf_vm* vm, struct jit_state* state, char** errmsg)
{
    emit_jit_prologue(state, UBPF_EBPF_STACK_SIZE);

    for (int i = 0; i < vm->num_insts; i++) {
        if (state->jit_status != NoError) {
            break;
        }

        struct ebpf_inst inst = ubpf_fetch_instruction(vm, i);

        // Handle fallthrough into a local function boundary (same pattern as ARM64)
        uint32_t fallthrough_jump_source = 0;
        bool fallthrough_jump_present = false;
        if (i != 0 && vm->int_funcs[i]) {
            struct ebpf_inst prev_inst = ubpf_fetch_instruction(vm, i - 1);
            if (ubpf_instruction_has_fallthrough(prev_inst)) {
                DECLARE_PATCHABLE_REGULAR_EBPF_TARGET(default_tgt, 0)
                fallthrough_jump_source = state->offset;
                emit_patchable_relative(
                    state->jumps, state->offset, default_tgt, state->num_jumps++);
                emit_instruction(state, mips_j26_type(MIPS_OP_BC, 0));
                fallthrough_jump_present = true;
            }
        }

        // Emit local function prolog if this is a function entry
        if (i == 0 || vm->int_funcs[i]) {
            size_t prolog_start = state->offset;
            emit_mips64_immediate(state, temp_register, ubpf_stack_usage_for_local_func(vm, i));
            emit_daddiu(state, REG_SP, REG_SP, -16);
            emit_sd(state, temp_register, REG_SP, 0);
            emit_sd(state, temp_register, REG_SP, 8);
            if (state->bpf_function_prolog_size == 0) {
                state->bpf_function_prolog_size = state->offset - prolog_start;
            } else {
                assert(state->bpf_function_prolog_size == state->offset - prolog_start);
            }
        }

        if (fallthrough_jump_present) {
            DECLARE_PATCHABLE_REGULAR_JIT_TARGET(fallthrough_tgt, state->offset)
            modify_patchable_relatives_target(
                state->jumps, state->num_jumps, fallthrough_jump_source, fallthrough_tgt);
        }

        state->pc_locs[i] = state->offset;

        enum MipsRegister dst = map_register(inst.dst);
        enum MipsRegister src = map_register(inst.src);
        uint8_t opcode = inst.opcode;

        // Compute target PC for jumps
        int64_t target_pc_64;
        if (inst.opcode == EBPF_OP_JA32) {
            target_pc_64 = (int64_t)i + (int64_t)inst.imm + 1;
        } else {
            target_pc_64 = (int64_t)i + (int64_t)inst.offset + 1;
        }
        uint32_t target_pc = (uint32_t)target_pc_64;

        DECLARE_PATCHABLE_REGULAR_EBPF_TARGET(tgt, target_pc)

        // Convert immediate ops to register ops after materializing the immediate.
        // Exception: MOV_IMM/MOV64_IMM handled directly in switch.
        // MUL/DIV/MOD have no direct-immediate switch cases, so always convert them.
        // ORI/ANDI/XORI zero-extend their 16-bit immediate, so negative values
        // or values > 65535 must be materialized into a register first.
        bool has_no_imm_case = (opcode & EBPF_ALU_OP_MASK) == EBPF_ALU_OP_MUL ||
                               (opcode & EBPF_ALU_OP_MASK) == EBPF_ALU_OP_DIV ||
                               (opcode & EBPF_ALU_OP_MASK) == EBPF_ALU_OP_MOD;
        bool logical_needs_materialize = false;
        if (is_imm_op(&inst) && ((opcode & EBPF_ALU_OP_MASK) == EBPF_ALU_OP_OR ||
                                  (opcode & EBPF_ALU_OP_MASK) == EBPF_ALU_OP_AND ||
                                  (opcode & EBPF_ALU_OP_MASK) == EBPF_ALU_OP_XOR)) {
            // ORI/ANDI/XORI use 16-bit UNSIGNED immediate (0–65535)
            if (inst.imm < 0 || inst.imm > 65535) {
                logical_needs_materialize = true;
            }
        }
        if (is_imm_op(&inst) &&
            opcode != EBPF_OP_MOV_IMM &&
            opcode != EBPF_OP_MOV64_IMM &&
            (has_no_imm_case || logical_needs_materialize ||
             !is_simple_imm(&inst) || vm->constant_blinding_enabled)) {
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            src = temp_register;
            opcode = to_reg_op(opcode);
        }

        switch (opcode) {

        /* ====== ALU64 register operations (Spec §3.1) ====== */

        case EBPF_OP_ADD64_REG:
            emit_daddu(state, dst, dst, src);
            break;
        case EBPF_OP_ADD64_IMM:
            emit_daddiu(state, dst, dst, (int16_t)inst.imm);
            break;
        case EBPF_OP_SUB64_REG:
            emit_dsubu(state, dst, dst, src);
            break;
        case EBPF_OP_SUB64_IMM:
            emit_daddiu(state, dst, dst, (int16_t)(-inst.imm));
            break;
        case EBPF_OP_MUL64_REG:
            emit_dmul(state, dst, dst, src);
            break;
        case EBPF_OP_DIV64_REG:
        case EBPF_OP_MOD64_REG:
            divmod(state, opcode, dst, src, inst.offset);
            break;
        case EBPF_OP_OR64_REG:
            emit_or(state, dst, dst, src);
            break;
        case EBPF_OP_OR64_IMM:
            emit_ori(state, dst, dst, (uint16_t)(inst.imm & 0xFFFF));
            break;
        case EBPF_OP_AND64_REG:
            emit_and(state, dst, dst, src);
            break;
        case EBPF_OP_AND64_IMM:
            emit_andi(state, dst, dst, (uint16_t)(inst.imm & 0xFFFF));
            break;
        case EBPF_OP_XOR64_REG:
            emit_xor(state, dst, dst, src);
            break;
        case EBPF_OP_XOR64_IMM:
            emit_xori(state, dst, dst, (uint16_t)(inst.imm & 0xFFFF));
            break;
        case EBPF_OP_LSH64_REG:
            emit_dsllv(state, dst, dst, src);
            break;
        case EBPF_OP_LSH64_IMM:
            if (inst.imm < 32) {
                emit_dsll(state, dst, dst, (uint8_t)inst.imm);
            } else {
                emit_dsll32(state, dst, dst, (uint8_t)(inst.imm - 32));
            }
            break;
        case EBPF_OP_RSH64_REG:
            emit_dsrlv(state, dst, dst, src);
            break;
        case EBPF_OP_RSH64_IMM:
            if (inst.imm < 32) {
                emit_dsrl(state, dst, dst, (uint8_t)inst.imm);
            } else {
                emit_dsrl32(state, dst, dst, (uint8_t)(inst.imm - 32));
            }
            break;
        case EBPF_OP_ARSH64_REG:
            emit_dsrav(state, dst, dst, src);
            break;
        case EBPF_OP_ARSH64_IMM:
            if (inst.imm < 32) {
                emit_dsra(state, dst, dst, (uint8_t)inst.imm);
            } else {
                emit_dsra32(state, dst, dst, (uint8_t)(inst.imm - 32));
            }
            break;
        case EBPF_OP_NEG64:
            // Spec §3.1: DSUBU dst, $zero, dst
            emit_dsubu(state, dst, REG_ZERO, dst);
            break;
        case EBPF_OP_MOV64_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, dst, (int64_t)inst.imm);
            break;
        case EBPF_OP_MOV64_REG:
            if (inst.offset == 8) {
                // Spec §3.4: MOVSX 8-bit — SEB
                emit_seb(state, dst, src);
            } else if (inst.offset == 16) {
                // Spec §3.4: MOVSX 16-bit — SEH
                emit_seh(state, dst, src);
            } else if (inst.offset == 32) {
                // Spec §3.4: MOVSX 32-bit — SLL dst, src, 0
                emit_sll(state, dst, src, 0);
            } else {
                // Normal move: OR dst, src, $zero
                emit_or(state, dst, src, REG_ZERO);
            }
            break;

        /* ====== ALU32 register operations (Spec §3.2) ====== */
        // ALU32 ops: use 64-bit operations + zero-extension

        case EBPF_OP_ADD_REG:
            emit_daddu(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_ADD_IMM:
            emit_daddiu(state, dst, dst, (int16_t)inst.imm);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_SUB_REG:
            emit_dsubu(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_SUB_IMM:
            emit_daddiu(state, dst, dst, (int16_t)(-inst.imm));
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_MUL_REG:
            emit_dmul(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_DIV_REG:
        case EBPF_OP_MOD_REG:
            divmod(state, opcode, dst, src, inst.offset);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_OR_REG:
            emit_or(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_OR_IMM:
            emit_ori(state, dst, dst, (uint16_t)(inst.imm & 0xFFFF));
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_AND_REG:
            emit_and(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_AND_IMM:
            emit_andi(state, dst, dst, (uint16_t)(inst.imm & 0xFFFF));
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_XOR_REG:
            emit_xor(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_XOR_IMM:
            emit_xori(state, dst, dst, (uint16_t)(inst.imm & 0xFFFF));
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_LSH_REG:
            emit_sllv(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_LSH_IMM:
            emit_sll(state, dst, dst, (uint8_t)inst.imm);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_RSH_REG:
            emit_srlv(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_RSH_IMM:
            emit_srl(state, dst, dst, (uint8_t)inst.imm);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_ARSH_REG:
            emit_srav(state, dst, dst, src);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_ARSH_IMM:
            emit_sra(state, dst, dst, (uint8_t)inst.imm);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_NEG:
            emit_dsubu(state, dst, REG_ZERO, dst);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_MOV_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, dst, (int64_t)inst.imm);
            emit_zero_extend_32(state, dst);
            break;
        case EBPF_OP_MOV_REG:
            if (inst.offset == 8) {
                emit_seb(state, dst, src);
            } else if (inst.offset == 16) {
                emit_seh(state, dst, src);
            } else {
                emit_or(state, dst, src, REG_ZERO);
            }
            emit_zero_extend_32(state, dst);
            break;

        /* ====== Byte swap operations (Spec §3.5) ====== */

        case EBPF_OP_LE:
            // Little-endian target: LE is no-op, just truncate
            if (inst.imm == 16) {
                emit_andi(state, dst, dst, 0xFFFF);
            } else if (inst.imm == 32) {
                emit_zero_extend_32(state, dst);
            }
            // LE64: no-op
            break;

        case EBPF_OP_BE:
            // Big-endian swap on little-endian target
            if (inst.imm == 16) {
                // Spec §3.5: WSBH + ANDI
                emit_wsbh(state, dst, dst);
                emit_andi(state, dst, dst, 0xFFFF);
            } else if (inst.imm == 32) {
                // Spec §3.5: WSBH + ROTR 16 + zero-ext
                emit_wsbh(state, dst, dst);
                emit_rotr(state, dst, dst, 16);
                emit_zero_extend_32(state, dst);
            } else if (inst.imm == 64) {
                // Spec §3.5: DSBH + DSHD
                emit_dsbh(state, dst, dst);
                emit_dshd(state, dst, dst);
            }
            break;

        case EBPF_OP_BSWAP:
            // Unconditional byte swap (RFC 9669)
            if (inst.imm == 16) {
                emit_wsbh(state, dst, dst);
                emit_andi(state, dst, dst, 0xFFFF);
            } else if (inst.imm == 32) {
                emit_wsbh(state, dst, dst);
                emit_rotr(state, dst, dst, 16);
                emit_zero_extend_32(state, dst);
            } else if (inst.imm == 64) {
                emit_dsbh(state, dst, dst);
                emit_dshd(state, dst, dst);
            }
            break;

        /* ====== Memory loads (Spec §3.6) ====== */

        case EBPF_OP_LDXB:
            emit_lbu(state, dst, src, inst.offset);
            break;
        case EBPF_OP_LDXH:
            emit_lhu(state, dst, src, inst.offset);
            break;
        case EBPF_OP_LDXW:
            emit_lwu(state, dst, src, inst.offset);
            break;
        case EBPF_OP_LDXDW:
            emit_ld(state, dst, src, inst.offset);
            break;

        /* ====== Sign-extending loads (Spec §3.7) ====== */

        case EBPF_OP_LDXBSX:
            emit_lb(state, dst, src, inst.offset);
            break;
        case EBPF_OP_LDXHSX:
            emit_lh(state, dst, src, inst.offset);
            break;
        case EBPF_OP_LDXWSX:
            emit_lw(state, dst, src, inst.offset);
            break;

        /* ====== Memory stores (Spec §3.8) ====== */

        case EBPF_OP_STXB:
            emit_sb(state, src, dst, inst.offset);
            break;
        case EBPF_OP_STXH:
            emit_sh(state, src, dst, inst.offset);
            break;
        case EBPF_OP_STXW:
            emit_sw(state, src, dst, inst.offset);
            break;
        case EBPF_OP_STXDW:
            emit_sd(state, src, dst, inst.offset);
            break;

        /* Store immediate (Spec §3.8): materialize imm in $t4, then store */
        /* Use EMIT_MIPS64_IMMEDIATE for constant blinding support (Spec §5.1) */
        case EBPF_OP_STB:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_sb(state, temp_register, dst, inst.offset);
            break;
        case EBPF_OP_STH:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_sh(state, temp_register, dst, inst.offset);
            break;
        case EBPF_OP_STW:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_sw(state, temp_register, dst, inst.offset);
            break;
        case EBPF_OP_STDW:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_sd(state, temp_register, dst, inst.offset);
            break;

        /* ====== 64-bit immediate load (Spec §3.9) ====== */

        case EBPF_OP_LDDW: {
            struct ebpf_inst next_inst = ubpf_fetch_instruction(vm, ++i);
            uint64_t imm64 = (uint64_t)(uint32_t)inst.imm | ((uint64_t)(uint32_t)next_inst.imm << 32);
            EMIT_MIPS64_IMMEDIATE(vm, state, dst, (int64_t)imm64);
            break;
        }

        /* ====== Jump instructions (Spec §3.10) ====== */

        case EBPF_OP_JA:
        case EBPF_OP_JA32: {
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_j26_type(MIPS_OP_BC, 0));
            break;
        }

        case EBPF_OP_JEQ_REG: {
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            // BEQC: symmetric, swap handled internally
            enum MipsRegister r1 = dst, r2 = src;
            if (r1 > r2) { enum MipsRegister tmp = r1; r1 = r2; r2 = tmp; }
            emit_instruction(state, mips_i_type(MIPS_OP_POP06, r1, r2, 0));
            break;
        }
        case EBPF_OP_JEQ_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            {
                enum MipsRegister r1 = dst, r2 = temp_register;
                if (r1 > r2) { enum MipsRegister tmp = r1; r1 = r2; r2 = tmp; }
                emit_instruction(state, mips_i_type(MIPS_OP_POP06, r1, r2, 0));
            }
            break;

        case EBPF_OP_JNE_REG: {
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            enum MipsRegister r1 = dst, r2 = src;
            if (r1 > r2) { enum MipsRegister tmp = r1; r1 = r2; r2 = tmp; }
            emit_instruction(state, mips_i_type(MIPS_OP_POP16, r1, r2, 0));
            break;
        }
        case EBPF_OP_JNE_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            {
                enum MipsRegister r1 = dst, r2 = temp_register;
                if (r1 > r2) { enum MipsRegister tmp = r1; r1 = r2; r2 = tmp; }
                emit_instruction(state, mips_i_type(MIPS_OP_POP16, r1, r2, 0));
            }
            break;

        // For ordered comparisons: use SLT/SLTU + BNEZC/BEQZC.
        // This avoids R6 compact-branch register ordering constraints.
        case EBPF_OP_JGT_REG:
            // dst > src (unsigned) → SLTU $t4, src, dst; BNEZC $t4, target
            emit_sltu(state, temp_register, src, dst);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JGT_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_sltu(state, temp_register, temp_register, dst);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        case EBPF_OP_JGE_REG:
            // dst >= src (unsigned) → SLTU $t4, dst, src; BEQZC $t4, target
            emit_sltu(state, temp_register, dst, src);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;
        case EBPF_OP_JGE_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_sltu(state, temp_register, dst, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;

        case EBPF_OP_JLT_REG:
            // dst < src (unsigned) → SLTU $t4, dst, src; BNEZC $t4, target
            emit_sltu(state, temp_register, dst, src);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JLT_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_sltu(state, temp_register, dst, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        case EBPF_OP_JLE_REG:
            // dst <= src (unsigned) → SLTU $t4, src, dst; BEQZC $t4, target
            emit_sltu(state, temp_register, src, dst);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;
        case EBPF_OP_JLE_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_sltu(state, temp_register, temp_register, dst);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;

        case EBPF_OP_JSGT_REG:
            emit_slt(state, temp_register, src, dst);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JSGT_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_slt(state, temp_register, temp_register, dst);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        case EBPF_OP_JSGE_REG:
            emit_slt(state, temp_register, dst, src);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;
        case EBPF_OP_JSGE_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_slt(state, temp_register, dst, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;

        case EBPF_OP_JSLT_REG:
            emit_slt(state, temp_register, dst, src);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JSLT_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_slt(state, temp_register, dst, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        case EBPF_OP_JSLE_REG:
            emit_slt(state, temp_register, src, dst);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;
        case EBPF_OP_JSLE_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_slt(state, temp_register, temp_register, dst);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;

        case EBPF_OP_JSET_REG:
            // Spec §3.10: AND $t4, dst, src; BNEZC $t4, target
            emit_and(state, temp_register, dst, src);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JSET_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_and(state, temp_register, dst, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        /* ====== JMP32 — 32-bit comparisons (Spec §3.10) ====== */
        // Canonicalize to 32-bit via SLL $t4, dst, 0; SLL $t5, src, 0

        case EBPF_OP_JEQ32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            {
                enum MipsRegister r1 = temp_register, r2 = temp_div_register;
                if (r1 > r2) { enum MipsRegister tmp = r1; r1 = r2; r2 = tmp; }
                emit_instruction(state, mips_i_type(MIPS_OP_POP06, r1, r2, 0));
            }
            break;
        case EBPF_OP_JEQ32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            {
                enum MipsRegister r1 = temp_register, r2 = temp_div_register;
                if (r1 > r2) { enum MipsRegister tmp = r1; r1 = r2; r2 = tmp; }
                emit_instruction(state, mips_i_type(MIPS_OP_POP06, r1, r2, 0));
            }
            break;

        case EBPF_OP_JNE32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            {
                enum MipsRegister r1 = temp_register, r2 = temp_div_register;
                if (r1 > r2) { enum MipsRegister tmp = r1; r1 = r2; r2 = tmp; }
                emit_instruction(state, mips_i_type(MIPS_OP_POP16, r1, r2, 0));
            }
            break;
        case EBPF_OP_JNE32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            {
                enum MipsRegister r1 = temp_register, r2 = temp_div_register;
                if (r1 > r2) { enum MipsRegister tmp = r1; r1 = r2; r2 = tmp; }
                emit_instruction(state, mips_i_type(MIPS_OP_POP16, r1, r2, 0));
            }
            break;

        case EBPF_OP_JGT32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_sltu(state, temp_register, temp_div_register, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JGT32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_sltu(state, temp_register, temp_div_register, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        case EBPF_OP_JGE32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_sltu(state, temp_register, temp_register, temp_div_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;
        case EBPF_OP_JGE32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_sltu(state, temp_register, temp_register, temp_div_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;

        case EBPF_OP_JLT32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_sltu(state, temp_register, temp_register, temp_div_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JLT32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_sltu(state, temp_register, temp_register, temp_div_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        case EBPF_OP_JLE32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_sltu(state, temp_register, temp_div_register, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;
        case EBPF_OP_JLE32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_sltu(state, temp_register, temp_div_register, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;

        case EBPF_OP_JSGT32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_slt(state, temp_register, temp_div_register, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JSGT32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_slt(state, temp_register, temp_div_register, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        case EBPF_OP_JSGE32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_slt(state, temp_register, temp_register, temp_div_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;
        case EBPF_OP_JSGE32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_slt(state, temp_register, temp_register, temp_div_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;

        case EBPF_OP_JSLT32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_slt(state, temp_register, temp_register, temp_div_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JSLT32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_slt(state, temp_register, temp_register, temp_div_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        case EBPF_OP_JSLE32_REG:
            emit_sll(state, temp_register, dst, 0);
            emit_sll(state, temp_div_register, src, 0);
            emit_slt(state, temp_register, temp_div_register, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;
        case EBPF_OP_JSLE32_IMM:
            emit_sll(state, temp_register, dst, 0);
            emit_mips64_immediate(state, temp_div_register, (int64_t)(int32_t)inst.imm);
            emit_slt(state, temp_register, temp_div_register, temp_register);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BEQZC, temp_register, 0));
            break;

        case EBPF_OP_JSET32_REG:
            emit_and(state, temp_register, dst, src);
            emit_sll(state, temp_register, temp_register, 0);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;
        case EBPF_OP_JSET32_IMM:
            EMIT_MIPS64_IMMEDIATE(vm, state, temp_register, (int64_t)inst.imm);
            emit_and(state, temp_register, dst, temp_register);
            emit_sll(state, temp_register, temp_register, 0);
            emit_patchable_relative(state->jumps, state->offset, tgt, state->num_jumps++);
            emit_instruction(state, mips_cmpbr21_type(MIPS_OP_BNEZC, temp_register, 0));
            break;

        /* ====== CALL instructions (Spec §3.12) ====== */

        case EBPF_OP_CALL:
            if (inst.src == 0) {
                // External helper call
                if (vm->dispatcher != NULL) {
                    emit_dispatcher_call(state, inst.imm);
                } else {
                    emit_helper_call(state, inst.imm);
                }
            } else if (inst.src == 1) {
                // Local function call
                emit_local_call(state, vm, inst.imm + i + 1);
            } else {
                *errmsg = ubpf_error("unsupported call type: src=%d", inst.src);
                return -1;
            }
            break;

        /* ====== EXIT instruction (Spec §3.13) ====== */

        case EBPF_OP_EXIT:
            // Branch to epilogue. The epilogue restores $sp from $fp (undoing
            // any local-call stack pushes), restores callee-saved registers,
            // and returns to the native caller via JIC $ra.
            {
                DECLARE_PATCHABLE_SPECIAL_TARGET(exit_tgt, Exit)
                emit_patchable_relative(state->jumps, state->offset, exit_tgt, state->num_jumps++);
                emit_instruction(state, mips_j26_type(MIPS_OP_BC, 0));
            }
            break;

        /* ====== Atomic operations (Spec §3.11) ====== */

        case EBPF_OP_ATOMIC_STORE:
        case EBPF_OP_ATOMIC32_STORE: {
            bool is_64bit = (opcode == EBPF_OP_ATOMIC_STORE);
            uint8_t atomic_op = inst.imm;
            bool fetch = (atomic_op & EBPF_ATOMIC_OP_FETCH) != 0;

            // Compute address: $t6 = dst + offset
            emit_daddiu(state, offset_register, dst, inst.offset);

            if (atomic_op == EBPF_ATOMIC_OP_CMPXCHG) {
                // CMPXCHG: compare with BPF R0 ($v0)
                uint32_t retry = state->offset;
                if (is_64bit) {
                    emit_lld(state, temp_register, offset_register, 0);
                } else {
                    emit_ll(state, temp_register, offset_register, 0);
                }
                // If loaded != R0, don't store — branch to fail
                uint32_t fail = emit_bnec_local(state, temp_register, REG_V0);
                // Match: store new value
                emit_or(state, temp_div_register, src, REG_ZERO);
                if (is_64bit) {
                    emit_scd(state, temp_div_register, offset_register, 0);
                } else {
                    emit_sc(state, temp_div_register, offset_register, 0);
                }
                // SC sets temp_div_register to 0 on failure, retry
                uint32_t retry_br = emit_beqzc_local(state, temp_div_register);
                // Patch retry branch to go back to retry label
                int32_t retry_rel = ((int32_t)(retry - state->offset)) / 4;
                uint32_t* retry_instr = (uint32_t*)(state->buf + retry_br);
                *retry_instr = (*retry_instr & 0xFFE00000u) | ((uint32_t)retry_rel & 0x001FFFFFu);

                patch_branch_offset16(state, fail);
                // R0 = old value (always)
                emit_or(state, REG_V0, temp_register, REG_ZERO);
            } else if (atomic_op == EBPF_ATOMIC_OP_XCHG) {
                // XCHG: exchange
                uint32_t retry = state->offset;
                if (is_64bit) {
                    emit_lld(state, temp_register, offset_register, 0);
                } else {
                    emit_ll(state, temp_register, offset_register, 0);
                }
                emit_or(state, temp_div_register, src, REG_ZERO);
                if (is_64bit) {
                    emit_scd(state, temp_div_register, offset_register, 0);
                } else {
                    emit_sc(state, temp_div_register, offset_register, 0);
                }
                uint32_t retry_br = emit_beqzc_local(state, temp_div_register);
                int32_t retry_rel = ((int32_t)(retry - state->offset)) / 4;
                uint32_t* retry_instr = (uint32_t*)(state->buf + retry_br);
                *retry_instr = (*retry_instr & 0xFFE00000u) | ((uint32_t)retry_rel & 0x001FFFFFu);
                // Return old value in src
                emit_or(state, src, temp_register, REG_ZERO);
            } else {
                // ADD, OR, AND, XOR (with optional FETCH)
                uint8_t base_op = atomic_op & ~EBPF_ATOMIC_OP_FETCH;
                uint32_t retry = state->offset;
                if (is_64bit) {
                    emit_lld(state, temp_register, offset_register, 0);
                } else {
                    emit_ll(state, temp_register, offset_register, 0);
                }
                // Compute new value in $t5
                switch (base_op) {
                case EBPF_ALU_OP_ADD:
                    if (is_64bit) emit_daddu(state, temp_div_register, temp_register, src);
                    else emit_addu(state, temp_div_register, temp_register, src);
                    break;
                case EBPF_ALU_OP_OR:
                    emit_or(state, temp_div_register, temp_register, src);
                    break;
                case EBPF_ALU_OP_AND:
                    emit_and(state, temp_div_register, temp_register, src);
                    break;
                case EBPF_ALU_OP_XOR:
                    emit_xor(state, temp_div_register, temp_register, src);
                    break;
                default:
                    state->jit_status = UnexpectedInstruction;
                    break;
                }
                if (is_64bit) {
                    emit_scd(state, temp_div_register, offset_register, 0);
                } else {
                    emit_sc(state, temp_div_register, offset_register, 0);
                }
                uint32_t retry_br = emit_beqzc_local(state, temp_div_register);
                int32_t retry_rel = ((int32_t)(retry - state->offset)) / 4;
                uint32_t* retry_instr = (uint32_t*)(state->buf + retry_br);
                *retry_instr = (*retry_instr & 0xFFE00000u) | ((uint32_t)retry_rel & 0x001FFFFFu);

                if (fetch) {
                    // Return old value in src register
                    emit_or(state, src, temp_register, REG_ZERO);
                }
            }
            break;
        }

        default:
            *errmsg = ubpf_error("Unknown instruction at PC %d: opcode %02x", i, inst.opcode);
            return -1;
        }

        // Check for encoding errors after each instruction
        if (state->jit_status != NoError) {
            switch (state->jit_status) {
            case TooManyJumps:
                *errmsg = ubpf_error("Too many jumps");
                break;
            case TooManyLoads:
                *errmsg = ubpf_error("Too many loads");
                break;
            case TooManyLeas:
                *errmsg = ubpf_error("Too many LEAs");
                break;
            case TooManyLocalCalls:
                *errmsg = ubpf_error("Too many local calls");
                break;
            case NotEnoughSpace:
                *errmsg = ubpf_error("Target buffer too small");
                break;
            case UnexpectedInstruction:
                *errmsg = ubpf_error("Unexpected instruction at PC %d", i);
                break;
            default:
                *errmsg = ubpf_error("Unknown JIT error at PC %d", i);
                break;
            }
            return -1;
        }
    }

    emit_jit_epilogue(state, UBPF_EBPF_STACK_SIZE);

    // Emit data section after code: dispatcher pointer + helper function table.
    // $s5 (loaded in prologue via BALC) points to helper_table_loc.
    state->dispatcher_loc = emit_dispatched_external_helper_address(state, (uint64_t)vm->dispatcher);
    state->helper_table_loc = emit_helper_table(state, vm);

    return 0;
}

/* ========================================================================
 * Resolve functions — patch forward references after code generation
 * ======================================================================== */

// Resolve branch/jump forward references (Spec §9)
static bool
resolve_jumps(struct jit_state* state)
{
    for (int i = 0; i < state->num_jumps; i++) {
        struct patchable_relative jump = state->jumps[i];
        uint32_t target_loc;

        if (jump.target.is_special) {
            if (jump.target.target.special == Exit) {
                target_loc = state->exit_loc;
            } else {
                return false;
            }
        } else {
            uint32_t ebpf_pc = jump.target.target.regular.ebpf_target_pc;
            if (jump.target.target.regular.jit_target_pc != 0) {
                target_loc = jump.target.target.regular.jit_target_pc;
            } else {
                target_loc = state->pc_locs[ebpf_pc];
            }
        }

        // R6 compact branch offset: target = PC + 4 + sign_extend(offset << 2)
        int32_t rel = ((int32_t)(target_loc - (jump.offset_loc + 4))) / 4;
        uint32_t* instr = (uint32_t*)(state->buf + jump.offset_loc);
        uint32_t opcode_field = (*instr >> 26) & 0x3F;

        if (opcode_field == MIPS_OP_BC || opcode_field == MIPS_OP_BALC) {
            // 26-bit offset
            *instr = (*instr & 0xFC000000u) | ((uint32_t)rel & 0x03FFFFFFu);
        } else if (opcode_field == MIPS_OP_BNEZC || opcode_field == MIPS_OP_BEQZC) {
            // 21-bit offset
            *instr = (*instr & 0xFFE00000u) | ((uint32_t)rel & 0x001FFFFFu);
        } else {
            // 16-bit offset (BEQC, BNEC, etc.)
            *instr = (*instr & 0xFFFF0000u) | ((uint32_t)rel & 0xFFFFu);
        }
    }
    return true;
}

// Resolve load references (helper table, dispatcher pointer)
static bool
resolve_loads(struct jit_state* state)
{
    for (int i = 0; i < state->num_loads; i++) {
        struct patchable_relative load = state->loads[i];
        if (load.target.is_special && load.target.target.special == ExternalDispatcher) {
            // Patch the LD instruction at load.offset_loc.
            // The dispatcher pointer is at state->dispatcher_loc in the buffer.
            // $s5 points to state->helper_table_loc.
            // offset = dispatcher_loc - helper_table_loc
            int32_t disp_offset = (int32_t)(state->dispatcher_loc - state->helper_table_loc);
            if (disp_offset >= -32768 && disp_offset <= 32767) {
                uint32_t* ld_instr = (uint32_t*)(state->buf + load.offset_loc);
                *ld_instr = (*ld_instr & 0xFFFF0000u) | ((uint32_t)(int16_t)disp_offset & 0xFFFFu);
            } else {
                return false;
            }
        }
    }
    return true;
}

// Resolve LEA references (helper table base, data section addresses)
static bool
resolve_leas(struct jit_state* state)
{
    for (int i = 0; i < state->num_leas; i++) {
        struct patchable_relative lea = state->leas[i];
        if (lea.target.is_special && lea.target.target.special == LoadHelperTable) {
            // The BALC + DADDIU sequence: patch the DADDIU offset
            // BALC is at lea.offset_loc, DADDIU is at lea.offset_loc + 4
            // $s5 should point to helper_table_loc (set by translate)
            uint32_t balc_loc = lea.offset_loc;
            uint32_t daddiu_loc = balc_loc + 4;

            // After BALC, $ra = balc_loc + 4. We need $s5 = helper_table_loc.
            int32_t data_offset = (int32_t)(state->helper_table_loc - (balc_loc + 4));

            if (data_offset >= -32768 && data_offset <= 32767) {
                uint32_t* daddiu_instr = (uint32_t*)(state->buf + daddiu_loc);
                *daddiu_instr = (*daddiu_instr & 0xFFFF0000u) | ((uint32_t)(int16_t)data_offset & 0xFFFFu);
            } else {
                return false;
            }
        }
    }
    return true;
}

// Resolve local function call targets
static bool
resolve_local_calls(struct jit_state* state)
{
    for (int i = 0; i < state->num_local_calls; i++) {
        struct patchable_relative call = state->local_calls[i];
        assert(!call.target.is_special);

        int32_t target_loc = state->pc_locs[call.target.target.regular.ebpf_target_pc];
        int32_t rel = target_loc - call.offset_loc;
        rel -= state->bpf_function_prolog_size;

        // BALC: 26-bit offset
        int32_t rel_instr = rel / 4;
        uint32_t* instr = (uint32_t*)(state->buf + call.offset_loc);
        *instr = (*instr & 0xFC000000u) | ((uint32_t)rel_instr & 0x03FFFFFFu);
    }
    return true;
}

/* ========================================================================
 * Entry points (Spec §1)
 * ======================================================================== */

bool
ubpf_jit_update_dispatcher_mips64(
    struct ubpf_vm* vm, external_function_dispatcher_t new_dispatcher, uint8_t* buffer, size_t size, uint32_t offset)
{
    UNUSED_PARAMETER(vm);
    uint64_t jit_upper_bound = (uint64_t)buffer + size;
    void* dispatcher_address = (void*)((uint64_t)buffer + offset);
    if ((uint64_t)dispatcher_address + sizeof(void*) < jit_upper_bound) {
        memcpy(dispatcher_address, &new_dispatcher, sizeof(void*));
        return true;
    }
    return false;
}

bool
ubpf_jit_update_helper_mips64(
    struct ubpf_vm* vm,
    extended_external_helper_t new_helper,
    unsigned int idx,
    uint8_t* buffer,
    size_t size,
    uint32_t offset)
{
    UNUSED_PARAMETER(vm);
    uint64_t jit_upper_bound = (uint64_t)buffer + size;
    void* helper_address = (void*)((uint64_t)buffer + offset + (8 * idx));
    if ((uint64_t)helper_address + sizeof(void*) < jit_upper_bound) {
        memcpy(helper_address, &new_helper, sizeof(void*));
        return true;
    }
    return false;
}

struct ubpf_jit_result
ubpf_translate_mips64(struct ubpf_vm* vm, uint8_t* buffer, size_t* size, enum JitMode jit_mode)
{
    struct jit_state state;
    struct ubpf_jit_result compile_result;

    if (initialize_jit_state_result(&state, &compile_result, buffer, *size, jit_mode, &compile_result.errmsg) < 0) {
        goto out;
    }

    if (translate(vm, &state, &compile_result.errmsg) < 0) {
        goto out;
    }

    if (!resolve_jumps(&state) || !resolve_loads(&state) || !resolve_leas(&state) || !resolve_local_calls(&state)) {
        compile_result.errmsg = ubpf_error("Could not patch the relative addresses in the JIT'd code.");
        goto out;
    }

    compile_result.compile_result = UBPF_JIT_COMPILE_SUCCESS;
    *size = state.offset;
    compile_result.external_dispatcher_offset = state.dispatcher_loc;
    compile_result.external_helper_offset = state.helper_table_loc;

out:
    release_jit_state_result(&state, &compile_result);
    return compile_result;
}

#endif /* defined(__mips64) || defined(__mips__) */
