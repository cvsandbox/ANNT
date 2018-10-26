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
#ifndef ANNT_XSOFT_MAX_ACTIVATION_HPP
#define ANNT_XSOFT_MAX_ACTIVATION_HPP

#include "IActivationLayer.hpp"

namespace ANNT { namespace Neuro {

// Implementation of SoftMax activation function
class XSoftMaxActivation : public IActivationLayer
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

        for ( size_t i = 0; i < len; i++ )
        {
            output[i] /= sum;
        }
    }

    void BackwardActivate( const float_t* /* input */, const float_t* output,
                           const float_t* delta, float_t* prevDelta, size_t len ) override
    {
        for ( size_t i = 0; i < len; i++ )
        {
            float_t sum = 0;

            for ( size_t j = 0; j < len; j++ )
            {
                //sum += delta[j] * der[j];
                sum += delta[j] * ( ( j == i ) ? output[i] * ( float_t( 1 ) - output[j] ) : -output[i] * output[j] );
            }

            prevDelta[i] = sum;
        }
    }
};

} } // namespace ANNT::Neuro

#endif // ANNT_XSOFT_MAX_ACTIVATION_HPP
