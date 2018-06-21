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
#ifndef ANNT_TYPES_HPP
#define ANNT_TYPES_HPP

#include <limits>
#include <vector>
#include <cmath>

#include "../Config.hpp"
#include "XAlignedAllocator.hpp"

namespace ANNT {

// Numeric type used for neural network's data/callculations
// (weights, biases, errors, gradients, parameters, etc.)
#ifdef ANNT_USE_DOUBLE
typedef double float_t;
#else
typedef float  float_t;
#endif

// Vector type to use for network's input/output/error/gradient flow.
// 32 bytes aligned to enable SIMD operations on those.
typedef std::vector<float_t, XAlignedAllocator<float_t, 32>> fvector_t;

// Vector type with unsigned integers (size_t) as elements
typedef std::vector<size_t> uvector_t;

// Border handling modes for convolution and pooling
enum class BorderMode
{
    Valid,  // Output is smaller than input, since convolution is only computed
            // where input and filter fully overlap.

    Same    // Output is of the same size as input. To get this input is padded.
};

// Modes of selecting training samples into batches while running training epch.
enum class EpochSelectionMode
{
    Sequential,     // Samples are not shuffled and are chosen sequentially one after another in the provided order.

    RandomPick,     // Samples are not shuffled (order is kept), but individual items are chosed randomly into batches.

    Shuffle,        // Training samples are shuffled at the start of each epoch. Then chosen sequentially into batches.
};

// A value to represent missing connection (between inputs/outputs, neurons, layers, etc)
static const size_t ANNT_NOT_CONNECTED = std::numeric_limits<size_t>::max( );

// Macro to suppress warnings caused by unreferenced parameter
#define ANNT_UNREFERENCED_PARAMETER(param) (void)param

} // namespace ANNT

#endif // ANNT_TYPES_HPP
