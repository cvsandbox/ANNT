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

// Implementation of fully connected layer - each neuron is connected to each input
class XFullyConnectedLayer : public ITrainableLayer
{
private:
    // Weights and biases are all kept together
    fvector_t mAllWeights;

    // And here are their pointers
    float_t*  mWeights;
    float_t*  mBiases;

public:
    XFullyConnectedLayer( size_t inputsCount, size_t outputsCount );

    // Reports number of weight coefficients the layer has
    size_t WeightsCount( ) const override
    {
        return mAllWeights.size( );
    }

    // Get/set layer's weights
    fvector_t Weights( ) const override
    {
        return mAllWeights;
    }
    void SetWeights( const fvector_t& weights ) override
    {
        mAllWeights = weights;
    }

    // Randomizes layer's weights, clears biases
    void Randomize( ) override;

    // Calculates outputs for the given inputs
    void ForwardCompute( const std::vector<fvector_t*>& inputs,
                         std::vector<fvector_t*>& outputs,
                         const XNetworkContext& ctx ) override;

    // Propagates error to the previous layer and calculates weights/biases gradients
    void BackwardCompute( const std::vector<fvector_t*>& inputs,
                          const std::vector<fvector_t*>& outputs,
                          const std::vector<fvector_t*>& deltas,
                          std::vector<fvector_t*>& prevDeltas,
                          fvector_t& gradWeights,
                          const XNetworkContext& ctx ) override;

    // Applies updates to the layer's weights and biases
    void UpdateWeights( const fvector_t& updates ) override;

    // Saves layer's learnt parameters/weights
    bool SaveLearnedParams( FILE* file ) const override;
    // Loads layer's learnt parameters
    bool LoadLearnedParams( FILE* file ) override;
};

} } // namespace ANNT::Neuro

#endif // ANNT_XFULLY_CONNECTED_LAYER_HPP
