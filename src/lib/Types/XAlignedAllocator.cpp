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

#include "XAlignedAllocator.hpp"

#ifdef _WIN32
    #include <malloc.h>
#endif

#ifdef __MINGW32__
    #include <mm_malloc.h>
#endif

namespace ANNT {

// Allocate aligned memory
void* AlignedAlloc( std::size_t align, std::size_t size )
{
#if defined(_MSC_VER)
    return ::_aligned_malloc( size, align );
#elif defined(__MINGW32__)
    return ::_mm_malloc( size, align );
#else  // posix assumed
    void* p;

    if ( ::posix_memalign( &p, align, size ) != 0 )
    {
        p = 0;
    }

    return p;
#endif
}

// Free aligned memory
void AlignedFree( void* ptr )
{
#if defined(_MSC_VER)
    ::_aligned_free( ptr );
#elif defined(__MINGW32__)
    ::_mm_free( ptr );
#else
    ::free( ptr );
#endif
}

} // namespace ANNT
