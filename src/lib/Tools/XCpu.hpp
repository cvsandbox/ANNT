/*
    ANNT - Artificial Neural Networks C++ library

    Copyright (C) 2018, cvsandbox, cvsandbox@gmail.com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

#pragma once
#ifndef ANNT_XCPU_HPP
#define ANNT_XCPU_HPP

#include <stdint.h>

namespace ANNT {

// Set of functions providing some CPU related information
class XCpu
{
private:
    XCpu( );

public:
    enum FeatureRegisters
    {
        Reg_EAX = 0,
        Reg_EBX = 1,
        Reg_ECX = 2,
        Reg_EDX = 3
    };

    // Some of the CPUID flags to check for for available instruction sets
    enum EbxFlags
    {
        Flag_AVX2   = 1 << 5,
    };

    enum EcxFlags
    {
        Flag_SSE3   = 1,
        Flag_SSSE3  = 1 << 9,
        Flag_SSE4_1 = 1 << 19,
        Flag_SSE4_2 = 1 << 20,
        Flag_AVX    = 1 << 28,
    };

    enum EdxFlags
    {
        Flag_MMX    = 1 << 24,
        Flag_SSE    = 1 << 25,
        Flag_SSE2   = 1 << 26,
    };

public:
    // Provide CPU ID - 4 32-bit registers describing CPU features
    static void CpuId( uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx );

    // Check if the particular feature is support by the CPU
    static bool IsFeatureSupported( uint32_t reg, uint32_t flag );

    // Get number of CPU cores provided by the system
    static uint32_t CoresCount( );
};

} // namespace ANNT

#endif // ANNT_XCPU_HPP
