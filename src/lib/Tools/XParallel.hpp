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
#ifndef ANNT_XPARALLEL_HPP
#define ANNT_XPARALLEL_HPP

#include "../Config.hpp"

namespace ANNT {

// Provides functions to use for paralleling for-loops
class XParallel
{
private:
    XParallel( );

public:
    // Runs the specified lambda in a parallel for loop
    template <typename Func> static inline void For( size_t size, Func func )
    {
        #ifdef ANNT_USE_OMP
        #pragma omp parallel for
        #endif
        for ( int i = 0; i < static_cast<int>( size ); i++ )
        {
            func( static_cast<size_t>( i ) );
        }
    }

    // Conditionally runs the specified lambda in a parallel for loop
    template <typename Func> static inline void For( size_t size, bool parallel, Func func )
    {
        #ifdef ANNT_USE_OMP
        if ( parallel )
        {
            #pragma omp parallel for
            for ( int i = 0; i < static_cast< int >( size ); i++ )
            {
                func( static_cast< size_t >( i ) );
            }
        }
        else
        #else
        ANNT_UNREFERENCED_PARAMETER( parallel );
        #endif
        {
            for ( size_t i = 0; i < size; i++ )
            {
                func( i );
            }
        }
    }
};

} // namespace ANNT

#endif // ANNT_XPARALLEL_HPP
