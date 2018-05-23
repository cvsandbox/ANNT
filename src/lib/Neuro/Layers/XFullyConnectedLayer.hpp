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
#ifndef ANNT_XFULLY_CONNECTED_LAYER_HPP
#define ANNT_XFULLY_CONNECTED_LAYER_HPP

#include "ITrainableLayer.hpp"

namespace ANNT { namespace Neuro {

class XFullyConnectedLayer : public ITrainableLayer
{
private:
    vector_t mWeights;
    vector_t mBiases;

public:
    XFullyConnectedLayer( size_t inputsCount, size_t outputsCount );

    // Reports number of weight coefficients the layer has
    size_t WeightsCount( ) const override
    {
        return mWeights.size( );
    }

    // Reports number of bias coefficients the layer has
    virtual size_t BiasesCount( ) const override
    {
        return mBiases.size( );
    }

    // Randomizes layer's weights, clears biases
    void Randomize( ) override;

    // Calculates outputs for the given inputs
    void ForwardCompute( const std::vector<vector_t*>& inputs,
                         std::vector<vector_t*>& outputs ) override;

    // Propagates error to the previous layer and calculates weights/biases gradients
    void BackwardCompute( const std::vector<vector_t*>& inputs,
                          const std::vector<vector_t*>& outputs,
                          const std::vector<vector_t*>& deltas,
                          std::vector<vector_t*>& prevDeltas,
                          vector_t& gradWeights,
                          vector_t& gradBiases ) override;

    // Applies updates to the layer's weights and biases
    void UpdateWeights( const vector_t& weightsUpdate,
                        const vector_t& biasesUpdate ) override;
};

} } // namespace ANNT::Neuro

#endif // ANNT_XFULLY_CONNECTED_LAYER_HPP
