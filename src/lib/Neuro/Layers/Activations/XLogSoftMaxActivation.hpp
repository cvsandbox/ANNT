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
#ifndef ANNT_XLOG_SOFT_MAX_ACTIVATION_HPP
#define ANNT_XLOG_SOFT_MAX_ACTIVATION_HPP

#include "IActivationLayer.hpp"

namespace ANNT { namespace Neuro {

// Implementation of Log-SoftMax activation function - to be used with XNegativeLogLikelihoodCost,
// which expects log-probabilities as input.
// http://www.mlpack.org/docs/mlpack-git/doxygen/classmlpack_1_1ann_1_1LogSoftMax.html
//
// Using it since SoftMax+CrossEntropy can lead to NaN in gradient for unbounded activations:
// https://github.com/Theano/Theano/issues/3162
//
class XLogSoftMaxActivation : public IActivationLayer
{
public:

    void ForwardActivate( const float_t* input, float_t* output, size_t len ) override
    {
        float_t sum = 0;
        float_t max = input[0];

        for ( size_t i = 1; i < len; i++ )
        {
            if ( input[i] > max ) max = input[i];
        }

        for ( size_t i = 0; i < len; i++ )
        {
            output[i] = std::exp( input[i] - max );
            sum      += output[i];
        }

        sum = std::log( sum );

        for ( size_t i = 0; i < len; i++ )
        {
            output[i] = input[i] - max - sum;
        }
    }

    void BackwardActivate( const float_t* /* input */, const float_t* output,
                           const float_t* delta, float_t* prevDelta, size_t len ) override
    {
        for ( size_t i = 0; i < len; i++ )
        {
            prevDelta[i] = std::exp( output[i] ) +  delta[i];
        }
    }
};

} } // namespace ANNT::Neuro

#endif // ANNT_XLOG_SOFT_MAX_ACTIVATION_HPP
