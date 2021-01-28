#pragma once

inline int vasprintf(char** target, const char* format, va_list argptr)
{
    int length = _vscprintf(format, argptr);
    *target = malloc(length);
    return vsprintf_s(*target, length, format, argptr);
}