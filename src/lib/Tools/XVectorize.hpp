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
#ifndef ANNT_XVECTORIZE_HPP
#define ANNT_XVECTORIZE_HPP

#include "IVectorTools.hpp"

namespace ANNT {

class XVectorize
{
private:
    XVectorize( );

public:
    // Add two vectors: dst[i] += src[i]
    template <typename T> static inline void Add( const T* src, T* dst, size_t size )
    {
        mVectorTools->Add( src, dst, size );
    }

    // Element wise multiplication of two vectors (Hadamard product): dst[i] *= src[i]
    template <typename T> static inline void Mul( const T* src, T* dst, size_t size )
    {
        mVectorTools->Mul( src, dst, size );
    }

    // Dot product of two vectors: sum( vec1[i] * vec2[i] )
    template <typename T> static inline T Dot( const T* vec1, const T* vec2, size_t size )
    {
        return mVectorTools->Dot( vec1, vec2, size );
    }

    // Calculates maximum of the vector's elements and the specified value: dst[i] = max( src[i], alpha )
    template <typename T> static inline void Max( const T* src, T alpha, T* dst, size_t size )
    {
        mVectorTools->Max( src, alpha, dst, size );
    }

private:

    static IVectorTools* mVectorTools;
};

} // namespace ANNT

#endif // ANNT_XVECTORIZE_HPP
