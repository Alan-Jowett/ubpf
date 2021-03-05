#pragma once

#if defined(NTDDI_VERSION)
#define uint8_t unsigned char
#define uint16_t unsigned short
#define uint32_t unsigned long
#define uint64_t unsigned long long
#define int8_t signed char
#define int16_t signed short
#define int32_t signed long
#define int64_t signed long long
#define bool unsigned char
#define true 1
#define false 1
#define UINT64_MAX 0xFFFFFFFFFFFFFFFF
#define UINT32_MAX 0xFFFFFFFF
#define INT32_MIN 0x80000000
#define INT32_MAX 0x7FFFFFFF
#else
#include <stdint.h>
#include <stdbool.h>
#endif

#pragma warning(disable:4214)

int vasprintf(char** target, const char* format, va_list argptr);

#define htobe16(X) swap16(X)
#define htobe32(X) swap32(X)
#define htobe64(X) swap64(X)

#define htole16(X) (X)
#define htole32(X) (X)
#define htole64(X) (X)

inline uint16_t swap16(uint16_t value)
{
    return value << 8 | value >> 8;
}
inline uint32_t swap32(uint32_t value)
{
    return swap16(value >> 16) | ((uint32_t)swap16(value & ((1 << 16) - 1))) << 16;
}

inline uint64_t swap64(uint64_t value)
{
    return swap32(value >> 32) | ((uint64_t)swap32(value & ((1ull << 32ull) - 1))) << 32;
}

