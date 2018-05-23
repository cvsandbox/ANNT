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
#ifndef ANNT_IVECTOR_TOOLS_HPP
#define ANNT_IVECTOR_TOOLS_HPP

#include <cstddef>

namespace ANNT {

// Interface for some common operations performed on vectors
class IVectorTools
{
public:
    virtual ~IVectorTools( ) { }

    // Check if the implementation of vector tools is available on the current system
    virtual bool IsAvailable( ) const = 0;

    // Add two vectors: dst[i] += src[i]
    virtual void Add( const float*  src, float*  dst, size_t size ) const = 0;
    virtual void Add( const double* src, double* dst, size_t size ) const = 0;

    // Element wise multiplication of two vectors (Hadamard product): dst[i] *= src[i]
    virtual void Mul( const float*  src, float*  dst, size_t size ) const = 0;
    virtual void Mul( const double* src, double* dst, size_t size ) const = 0;

    // Dot product of two vectors: sum( vec1[i] * vec2[i] )
    virtual float  Dot( const float*  vec1, const float*  vec2, size_t size ) const = 0;
    virtual double Dot( const double* vec1, const double* vec2, size_t size ) const = 0;

    // Calculates maximum of the vector's elements and the specified value: dst[i] = max( src[i], alpha )
    virtual void Max( const float*  src, float  alpha, float*  dst, size_t size ) const = 0;
    virtual void Max( const double* src, double alpha, double* dst, size_t size ) const = 0;
};

} // namespace ANNT

#endif // ANNT_IVECTOR_TOOLS_HPP
