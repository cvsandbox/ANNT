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
#include "../../../Tools/XParallel.hpp"

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
    void ForwardCompute( const std::vector<fvector_t*>& inputs,
                         std::vector<fvector_t*>& outputs,
                         const XNetworkContext& ctx ) override
    {
        XParallel::For( inputs.size( ), ctx.IsTraining( ), [&]( size_t i )
        {
            ForwardActivate( inputs[i]->data( ), outputs[i]->data( ), inputs[i]->size( ) );
        } );
    }

    // Calls BackwardActivate() for individual input/output/delta vectors passed by reference
    void BackwardCompute( const std::vector<fvector_t*>& inputs,
                          const std::vector<fvector_t*>& outputs,
                          const std::vector<fvector_t*>& deltas,
                          std::vector<fvector_t*>& prevDeltas,
                          fvector_t& /* gradWeights */,
                          const XNetworkContext& ctx ) override
    {
        XParallel::For( inputs.size( ), ctx.IsTraining( ), [&]( size_t i )
        {
            BackwardActivate( inputs[i]->data( ), outputs[i]->data( ), deltas[i]->data( ), prevDeltas[i]->data( ), inputs[i]->size( ) );
        } );
    }

    // Applies activation function to the input vector
    virtual void ForwardActivate( const float_t* input, float_t* output, size_t len ) = 0;

    // Propagates error back to previous layer by multiplying delta with activation function's derivative
    virtual void BackwardActivate( const float_t* input, const float_t* output,
                                   const float_t* delta, float_t* prevDelta, size_t len ) = 0;
};

} } // namespace ANNT::Neuro

#endif // ANNT_IACTIVATION_LAYER_HPP
