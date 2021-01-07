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

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <inttypes.h>
#include "ubpf_int.h"
#include <elf.h>

#define MAX_SECTIONS 32

#ifndef EM_BPF
#define EM_BPF 247
#endif

struct bounds {
    const void *base;
    uint64_t size;
};

struct section {
    const Elf64_Shdr *shdr;
    const void *data;
    uint64_t size;
};

static const void *
bounds_check(struct bounds *bounds, uint64_t offset, uint64_t size)
{
    if (offset + size > bounds->size || offset + size < offset) {
        return NULL;
    }
    return bounds->base + offset;
}

int 
ubpf_load_elf(struct ubpf_vm *vm, const void *elf, size_t elf_size, char **errmsg)
{
    return ubpf_load_elf_by_name(vm, elf, elf_size, NULL, errmsg);
}

static 
const char *
ubpf_lookup_string(struct section * string_table, uint64_t offset)
{
    const char *strings = string_table->data;
    if (offset > string_table->size) {
        return NULL;
    }
    else
    {
        return strings + offset;
    }
}

static
int ubpf_find_section(
    struct section * sections, 
    size_t section_count, 
    Elf64_Word section_type, 
    Elf64_Word section_flags,
    int str_shndx,
    const char * section_name)
{
    int i;
    for (i = 0; i < section_count; i++) {
        const Elf64_Shdr *shdr = sections[i].shdr;
        if (shdr->sh_type == section_type &&
                shdr->sh_flags == section_flags) {
            const char * name;
            name = ubpf_lookup_string(&sections[str_shndx], shdr->sh_name);
            if (!section_name || (name && strcmp(section_name, name) == 0)) {
                return i;
            }
        }
    }
    return 0;
}

int
ubpf_load_elf_by_name(struct ubpf_vm *vm, const void *elf, size_t elf_size, const char * name, char **errmsg)
{
    struct bounds b = { .base=elf, .size=elf_size };
    // text_copy is an array of eBPF instructions
    struct ebpf_inst *text_copy = NULL;
    int i;

    const Elf64_Ehdr *ehdr = bounds_check(&b, 0, sizeof(*ehdr));
    if (!ehdr) {
        *errmsg = ubpf_error("not enough data for ELF header");
        goto error;
    }

    if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG)) {
        *errmsg = ubpf_error("wrong magic");
        goto error;
    }

    if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
        *errmsg = ubpf_error("wrong class");
        goto error;
    }

    if (ehdr->e_ident[EI_DATA] != ELFDATA2LSB) {
        *errmsg = ubpf_error("wrong byte order");
        goto error;
    }

    if (ehdr->e_ident[EI_VERSION] != 1) {
        *errmsg = ubpf_error("wrong version");
        goto error;
    }

    if (ehdr->e_ident[EI_OSABI] != ELFOSABI_NONE) {
        *errmsg = ubpf_error("wrong OS ABI");
        goto error;
    }

    if (ehdr->e_type != ET_REL) {
        *errmsg = ubpf_error("wrong type, expected relocatable");
        goto error;
    }

    if (ehdr->e_machine != EM_NONE && ehdr->e_machine != EM_BPF) {
        *errmsg = ubpf_error("wrong machine, expected none or BPF, got %d",
                             ehdr->e_machine);
        goto error;
    }

    if (ehdr->e_shnum > MAX_SECTIONS) {
        *errmsg = ubpf_error("too many sections");
        goto error;
    }

    /* Parse section headers into an array */
    struct section sections[MAX_SECTIONS];
    for (i = 0; i < ehdr->e_shnum; i++) {
        const Elf64_Shdr *shdr = bounds_check(&b, ehdr->e_shoff + i*ehdr->e_shentsize, sizeof(*shdr));
        if (!shdr) {
            *errmsg = ubpf_error("bad section header offset or size");
            goto error;
        }

        const void *data = bounds_check(&b, shdr->sh_offset, shdr->sh_size);
        if (!data) {
            *errmsg = ubpf_error("bad section offset or size");
            goto error;
        }

        sections[i].shdr = shdr;
        sections[i].data = data;
        sections[i].size = shdr->sh_size;
    }

    /* Find string table index */
    int str_shndx = ubpf_find_section(sections, ehdr->e_shnum, SHT_STRTAB, 0, 0, NULL);
    if (!str_shndx) {
        *errmsg = ubpf_error("string table section not found");
        goto error;
    }

    int text_shndx = ubpf_find_section(sections, ehdr->e_shnum, SHT_PROGBITS, SHF_ALLOC|SHF_EXECINSTR, str_shndx, name);
    if (!text_shndx) {
        *errmsg = ubpf_error("text section not found");
        goto error;
    }

    // Look for any maps
    int maps_shndx = ubpf_find_section(sections, ehdr->e_shnum, SHT_PROGBITS, SHF_ALLOC|SHF_WRITE, str_shndx, "maps");
    if (maps_shndx) {
        struct section *maps = &sections[maps_shndx];
        int i;
        const struct ubpf_map_def * map_defs = maps->data;

        if (maps->size > (UBPF_MAX_MAPS * sizeof(struct ubpf_map_def))) {
            *errmsg = ubpf_error("too many maps");
            goto error;
        }

        for (i = 0; i < maps->size / sizeof(struct ubpf_map_def); i ++)
        {
            vm->maps_fd[i] = vm->map_create(vm->map_create_context, &map_defs[i]);
            if (vm->maps_fd[i] == -1) {
                *errmsg = ubpf_error("map creation failed");
                goto error;
            }
        }
    }

    struct section *text = &sections[text_shndx];

    /* May need to modify text for relocations, so make a copy */
    text_copy = malloc(text->size);
    if (!text_copy) {
        *errmsg = ubpf_error("failed to allocate memory");
        goto error;
    }
    memcpy(text_copy, text->data, text->size);

    /* Process each relocation section */
    for (i = 0; i < ehdr->e_shnum; i++) {
        struct section *rel = &sections[i];
        if (rel->shdr->sh_type != SHT_REL) {
            continue;
        } else if (rel->shdr->sh_info != text_shndx) {
            continue;
        }

        const Elf64_Rel *rs = rel->data;

        if (rel->shdr->sh_link >= ehdr->e_shnum) {
            *errmsg = ubpf_error("bad symbol table section index");
            goto error;
        }

        struct section *symtab = &sections[rel->shdr->sh_link];
        const Elf64_Sym *syms = symtab->data;
        uint32_t num_syms = symtab->size/sizeof(syms[0]);

        if (symtab->shdr->sh_link >= ehdr->e_shnum) {
            *errmsg = ubpf_error("bad string table section index");
            goto error;
        }

        int j;
        for (j = 0; j < rel->size/sizeof(Elf64_Rel); j++) {
            const Elf64_Rel *r = &rs[j];

            unsigned int r_indx = r->r_offset / sizeof(struct ebpf_inst);

            uint32_t sym_idx = ELF64_R_SYM(r->r_info);
            if (sym_idx >= num_syms) {
                *errmsg = ubpf_error("bad symbol index");
                goto error;
            }

            const Elf64_Sym *sym = &syms[sym_idx];

            switch (ELF64_R_TYPE(r->r_info)) {
            // Map relocation
            case 1: {
                if (!maps_shndx || sym->st_value >= sections[maps_shndx].size) {
                    *errmsg = ubpf_error("bad map index");
                    goto error;
                }
                // Set the destination register to 1 to flag this as a LDDW MAP_FD instruction.
                // Set the immediate value to the fd returned from map_create
                text_copy[r_indx].dst = 1;
                text_copy[r_indx].imm = vm->maps_fd[sym->st_value / sizeof(struct ubpf_map_def)];
                break;
            }
            // Function relocation
            case 2: {
                const char * sym_name = ubpf_lookup_string(&sections[str_shndx], sym->st_name);

                if (!sym_name) {
                    *errmsg = ubpf_error("bad symbol name");
                    goto error;
                }

                if (r->r_offset + 8 > text->size) {
                    *errmsg = ubpf_error("bad relocation offset");
                    goto error;
                }

                unsigned int imm = ubpf_lookup_registered_function(vm, sym_name);
                if (imm == -1) {
                    *errmsg = ubpf_error("function '%s' not found", sym_name);
                    goto error;
                }
                text_copy[r_indx].imm = imm;
                break;
            }
            default:
                *errmsg = ubpf_error("bad relocation type %u", ELF64_R_TYPE(r->r_info));
                goto error;
            }
        }
    }

    int rv = ubpf_load(vm, text_copy, sections[text_shndx].size, errmsg);
    free(text_copy);
    return rv;

error:
    free(text_copy);
    return -1;
}
