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

#include "XVectorTools.hpp"

namespace ANNT {

// Helper class providing the actual implementation
class VectorToolsImpl
{
public:
    // Add two vectors: dst[i] += src[i]
    template <typename T> static inline void Add( const T* src, T* dst, size_t size )
    {
        for ( size_t i = 0; i < size; i++ )
        {
            dst[i] += src[i];
        }
    }

    // Multiply two vectors: dst[i] *= src[i]
    template <typename T> static inline void Mul( const T* src, T* dst, size_t size )
    {
        for ( size_t i = 0; i < size; i++ )
        {
            dst[i] *= src[i];
        }
    }

    // Dot product: sum( vec1[i] * vec2[i] )
    template <typename T> static inline T Dot( const T* vec1, const T* vec2, size_t size )
    {
        T dotProduct = T( 0 );

        for ( size_t i = 0; i < size; i++ )
        {
            dotProduct += vec1[i] * vec2[i];
        }

        return dotProduct;
    }

    // Calculates maximum of the vector's elements and the specified value: dst[i] = max( src[i], alpha )
    template <typename T> static inline void Max( const T* src, T alpha, T* dst, size_t size )
    {
        for ( size_t i = 0; i < size; i++ )
        {
            dst[i] = ( src[i] > alpha ) ? src[i] : alpha;
        }
    }
};

/* ============================================================================= */

// Check if the implementation of vector tools is available on the current system
bool XVectorTools::IsAvailable( ) const
{
    return true;
}

// Add two vectors: dst[i] += src[i]
void XVectorTools::Add( const float* src, float* dst, size_t size ) const
{
    VectorToolsImpl::Add( src, dst, size );
}
void XVectorTools::Add( const double* src, double* dst, size_t size ) const
{
    VectorToolsImpl::Add( src, dst, size );
};

// Element wise multiplication of two vectors (Hadamard product): dst[i] *= src[i]
void XVectorTools::Mul( const float*  src, float*  dst, size_t size ) const
{
    VectorToolsImpl::Mul( src, dst, size );
}
void XVectorTools::Mul( const double* src, double* dst, size_t size ) const
{
    VectorToolsImpl::Mul( src, dst, size );
}

// Dot product of two vectors: sum( vec1[i] * vec2[i] )
float XVectorTools::Dot( const float* vec1, const float* vec2, size_t size ) const
{
    return VectorToolsImpl::Dot( vec1, vec2, size );
}
double XVectorTools::Dot( const double* vec1, const double* vec2, size_t size ) const
{
    return VectorToolsImpl::Dot( vec1, vec2, size );
}

// Calculates maximum of the vector's elements and the specified value: dst[i] = max( src[i], alpha )
void XVectorTools::Max( const float* src, float alpha, float* dst, size_t size ) const
{
    VectorToolsImpl::Max( src, alpha, dst, size );
}
void XVectorTools::Max( const double* src, double alpha, double* dst, size_t size ) const
{
    VectorToolsImpl::Max( src, alpha, dst, size );
}

} // namespace ANNT
