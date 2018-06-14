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
#ifndef ANNT_XRELU_ACTIVATION_HPP
#define ANNT_XRELU_ACTIVATION_HPP

#include "IActivationLayer.hpp"
#include "../../../Tools/XVectorize.hpp"

namespace ANNT { namespace Neuro {

// Implementation of ReLU activation function, f(x) = max(0, x)
class XReLuActivation : public IActivationLayer
{
public:

    void ForwardActivate( const fvector_t& input, fvector_t& output ) override
    {
        XVectorize::Max( input.data( ), float_t( 0 ), output.data( ), input.size( ) );
    }

    void BackwardActivate( const fvector_t& input, const fvector_t& output,
                           const fvector_t& delta, fvector_t& prevDelta ) override
    {
        for ( size_t i = 0, len = input.size( ); i < len; i++ )
        {
            // derivative(ReLU) = 1, y > 0
            //                  = 0, otherwise
            prevDelta[i] = ( output[i] > float_t( 0 ) ) ? delta[i] : float_t( 0 );
        }
    }
};

} } // namespace ANNT::Neuro

#endif // ANNT_XRELU_ACTIVATION_HPP
