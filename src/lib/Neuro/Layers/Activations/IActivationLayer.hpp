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
#ifndef ANNT_IACTIVATION_LAYER_HPP
#define ANNT_IACTIVATION_LAYER_HPP

#include "../ILayer.hpp"

namespace ANNT { namespace Neuro {

class IActivationLayer : public ILayer
{
public:
    IActivationLayer( ) : ILayer( 0, 0 )
    {
    }

    // None of the activation functions have weights/biases to train
    bool Trainable( ) const override
    {
        return false;
    }

    // Calls ForwardActivate() for individual input/output vectors passed by reference
    void ForwardCompute( const std::vector<vector_t*>& inputs,
                         std::vector<vector_t*>& outputs ) override
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            ForwardActivate( *( inputs[i] ), *( outputs[i] ) );
        }
    }

    // Calls BackwardActivate() for individual input/output/delta vectors passed by reference
    void BackwardCompute( const std::vector<vector_t*>& inputs,
                          const std::vector<vector_t*>& outputs,
                          const std::vector<vector_t*>& deltas,
                          std::vector<vector_t*>& prevDeltas,
                          vector_t& /* gradWeights */,
                          vector_t& /* gradBiases  */ ) override
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            BackwardActivate( *( inputs[i] ), *( outputs[i] ), *( deltas[i] ), *( prevDeltas[i] ) );
        }
    }

    // Applies activation function to the input vector
    virtual void ForwardActivate( const vector_t& input, vector_t& output ) = 0;

    // Propagates error back to previous layer by multiplying delta with activation function's derivative
    virtual void BackwardActivate( const vector_t& input, const vector_t& output,
                                   const vector_t& delta, vector_t& prevDelta ) = 0;
};

} } // namespace ANNT::Neuro

#endif // ANNT_IACTIVATION_LAYER_HPP
