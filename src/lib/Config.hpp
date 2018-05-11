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
#ifndef ANNT_CONFIG_HPP
#define ANNT_CONFIG_HPP

// Enable support of SSE/SSE2 instructions set
#define ANNT_USE_SSE

// Enable support of AVX instructions set
#define ANNT_USE_AVX

// Enable Open MP usage for loops parallelization
#define ANNT_USE_OMP

// Use double or single precision floating numbers for neural networks' weights, parameters, gradients, etc.
// #define ANNT_USE_DOUBLE

#endif // ANNT_CONFIG_HPP
