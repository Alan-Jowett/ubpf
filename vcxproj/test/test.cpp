// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
extern "C"
{
#include "../../vm/inc/ubpf.h"
}
int main()
{
    unsigned short byte_code[] = { 0x00b7, 0x0000, 0x0000, 0x0000, 0x0095, 0x0000, 0x0000, 0x0000 };
    unsigned char machine_code[1024] = { 0 };
    size_t machine_code_size = 0;
    char* errmsg;

    auto vm = ubpf_create();
    if (ubpf_load(vm, byte_code, sizeof(byte_code), &errmsg))
    {
        printf("ubpf_load failed\n");
        return -1;
    }

    if (ubpf_translate(vm, machine_code, &machine_code_size, &errmsg))
    {
        printf("ubpf_load failed\n");
        return -1;
    }

    printf("machine_code_size=%d\n", machine_code_size);

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
