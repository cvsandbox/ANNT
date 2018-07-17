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
#ifndef ANNT_XSIGMOID_ACTIVATION_HPP
#define ANNT_XSIGMOID_ACTIVATION_HPP

#include "IActivationLayer.hpp"

namespace ANNT { namespace Neuro {

// Implementation of Sigmoid activation function, f(x) = 1 / ( 1 + exp(-x) )
class XSigmoidActivation : public IActivationLayer
{
public:

    void ForwardActivate( const float_t* input, float_t* output, size_t len ) override
    {
        for ( size_t i = 0; i < len; i++ )
        {
            output[i] = float_t( 1 ) / ( float_t( 1 ) + std::exp( -input[i] ) );
        }
    }

    void BackwardActivate( const float_t* /* input */, const float_t* output,
                           const float_t* delta, float_t* prevDelta, size_t len ) override
    {
        for ( size_t i = 0; i < len; i++ )
        {
            // derivative(Sigmoid) = y * ( 1 - y )
            prevDelta[i] = delta[i] * output[i] * ( float_t( 1 ) - output[i] );
        }
    }
};

} } // namespace ANNT::Neuro

#endif // ANNT_XSIGMOID_ACTIVATION_HPP
