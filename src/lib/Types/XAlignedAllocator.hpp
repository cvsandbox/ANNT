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
#ifndef ANNT_XALLIGNED_ALLOCATOR_HPP
#define ANNT_XALLIGNED_ALLOCATOR_HPP

#include <cstdlib>

#ifdef _WIN32
    #include <malloc.h>
#endif

#ifdef __MINGW32__
    #include <mm_malloc.h>
#endif

namespace ANNT {

// Allocate aligned memory
static void* AlignedAlloc( std::size_t align, std::size_t size )
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
static void AlignedFree( void* ptr )
{
#if defined(_MSC_VER)
    ::_aligned_free( ptr );
#elif defined(__MINGW32__)
    ::_mm_free( ptr );
#else
    ::free( ptr );
#endif
}
    
// Aligned allocator for standard containers
template <typename T, std::size_t Alignment>
class XAlignedAllocator
{
public:
    // Typedefs
    typedef T               value_type;
    typedef T*              pointer;
    typedef const T*        const_pointer;
    typedef T&              reference;
    typedef const T&        const_reference;
    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;

public:
    // Convert an allocator<T> to allocator<U>
    template <typename U>
    struct rebind
    {
        typedef XAlignedAllocator<U, Alignment> other;
    };

public:
    XAlignedAllocator( ) { }

    template <typename U>
    XAlignedAllocator( const XAlignedAllocator<U, Alignment> & ) { }

    // Address
    inline pointer address( reference value ) const
    {
        return std::addressof( value );
    }
    inline const_pointer address( const_reference value ) const
    {
        return std::addressof(value);
    }

    // Memory allocation
    inline pointer allocate( size_type size, const void* = nullptr )
    {
        void* p = AlignedAlloc( Alignment, sizeof( T ) * size );

        if ( ( p == nullptr ) && ( size > 0 ) )
        {
            throw std::bad_alloc( );
        }

        return static_cast<pointer>( p );
    }
    inline void deallocate( pointer ptr, size_type )
    {
        AlignedFree( ptr );
    }

    // Size
    inline size_type max_size( ) const
    {
        return ~static_cast<std::size_t>( 0 ) / sizeof( T );
    }

    // Construction/destruction
    inline void construct( pointer ptr, const_reference value )
    {
        ::new ( ptr ) value_type( value );
    }
    inline void destroy( pointer ptr )
    {
        if ( ptr )
        {
            ptr->~value_type( );
        }
    }

    inline bool operator==( const XAlignedAllocator& ) { return true; }
    inline bool operator!=( const XAlignedAllocator& rhs ) { return !operator==( rhs ); }
};

} // namespace ANNT

#endif // ANNT_XALLIGNED_ALLOCATOR_HPP
