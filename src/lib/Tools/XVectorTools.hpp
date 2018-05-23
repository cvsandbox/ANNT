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
#ifndef ANNT_XVECTOR_TOOLS_HPP
#define ANNT_XVECTOR_TOOLS_HPP

#include "IVectorTools.hpp"

namespace ANNT {

// Default implementation of common vector routines without using
// any extended CPU instruction sets
class XVectorTools : public IVectorTools
{
public:
    // Check if the implementation of vector tools is available on the current system
    bool IsAvailable( ) const override;

    // Add two vectors: dst[i] += src[i]
    void Add( const float*  src, float*  dst, size_t size ) const override;
    void Add( const double* src, double* dst, size_t size ) const override;

    // Element wise multiplication of two vectors (Hadamard product): dst[i] *= src[i]
    void Mul( const float*  src, float*  dst, size_t size ) const override;
    void Mul( const double* src, double* dst, size_t size ) const override;

    // Dot product of two vectors: sum( vec1[i] * vec2[i] )
    float  Dot( const float*  vec1, const float*  vec2, size_t size ) const override;
    double Dot( const double* vec1, const double* vec2, size_t size ) const override;

    // Calculates maximum of the vector's elements and the specified value: dst[i] = max( src[i], alpha )
    void Max( const float*  src, float  alpha, float*  dst, size_t size ) const override;
    void Max( const double* src, double alpha, double* dst, size_t size ) const override;
};

} // namespace ANNT

#endif // ANNT_XVECTOR_TOOLS_HPP
