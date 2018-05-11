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

#include <thread>

#ifdef _MSC_VER
    #include <intrin.h>
#elif __GNUC__
    #include <cpuid.h>
#endif

#include "XCpu.hpp"

namespace ANNT {

// Provide CPU ID - 4 32-bit registers describing CPU features
void XCpu::CpuId( uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx )
{
#ifdef _MSC_VER
    int cpuInfo[4];

    __cpuid( cpuInfo, 1 );

    eax = static_cast<uint32_t>( cpuInfo[0] );
    ebx = static_cast<uint32_t>( cpuInfo[1] );
    ecx = static_cast<uint32_t>( cpuInfo[2] );
    edx = static_cast<uint32_t>( cpuInfo[3] );
#elif __GNUC__
    __cpuid( 1, eax, ebx, ecx, edx );
#endif
}

// Check if the particular feature is support by the CPU
bool XCpu::IsFeatureSupported( uint32_t reg, uint32_t flag )
{
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    bool     ret = false;

    CpuId( eax, ebx, ecx, edx );

    switch ( reg )
    {
    case Reg_EAX:
        ret = ( ( eax & flag ) == flag );
        break;
    case Reg_EBX:
        ret = ( ( ebx & flag ) == flag );
        break;
    case Reg_ECX:
        ret = ( ( ecx & flag ) == flag );
        break;
    case Reg_EDX:
        ret = ( ( edx & flag ) == flag );
        break;
    }

    return ret;
}

// Get number of CPU cores provided by the system
uint32_t XCpu::CoresCount( )
{
    return std::thread::hardware_concurrency( );
}

} // namespace ANNT
