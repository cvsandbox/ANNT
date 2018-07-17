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
#ifndef ANNT_XRECURRENT_LAYER_HPP
#define ANNT_XRECURRENT_LAYER_HPP

#include "ITrainableLayer.hpp"
#include "Activations/XTanhActivation.hpp"

namespace ANNT { namespace Neuro {

// Implementation of simple recurent layer, which perform calculations as:
//
//  Internal activation:  A(t) = U * X(t) + W * H(t-1) + B
//  State:                H(t) = tanh(A(t))
//  Output:               O(t) = V * H(t) + C
//
// See: http://www.deeplearningbook.org/contents/rnn.html
//
class XRecurrentLayer : public ITrainableLayer
{
private:
    XTanhActivation    mTanh;

    // All weights and biases are stored together as two vectors.
    fvector_t   mWeights;
    fvector_t   mBiases;

    // And here are the pointers to specific weights/biases
    float_t*    mWeightsU;
    float_t*    mWeightsW;
    float_t*    mWeightsV;

    float_t*    mBiasesB;
    float_t*    mBiasesC;
    
    // --------------------------------------------------------------------------------------
    enum
    {
        BUFFER_INDEX_STATE          = 0,            // per batch
        BUFFER_INDEX_STATE_GRAD     = 1,            // per batch
        BUFFER_INDEX_STATE_PREV     = 2, // H(t-1)  // per sample
        BUFFER_INDEX_STATE_CURRENT  = 3, // H(t)    // per sample
    };

public:

    XRecurrentLayer( size_t inputsCount, size_t outputsCount );

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

    // Get/set layer's weights
    fvector_t Weights( ) const override
    {
        return mWeights;
    }
    void SetWeights( const fvector_t& weights ) override
    {
        mWeights = weights;
    }

    // Get/set layer's biases
    fvector_t Biases( ) const override
    {
        return mBiases;
    }
    void SetBiases( const fvector_t& biases ) override
    {
        mBiases = biases;
    }

    // Tells that we may need some extra memory for internal state/calculations
    uvector_t WorkingMemSize( bool /* trainingMode */ ) const override
    {
        uvector_t workingMemSize = uvector_t( 4, mOutputsCount * sizeof( float_t ) );

        return workingMemSize;
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
                          fvector_t& gradBiases,
                          const XNetworkContext& ctx ) override;

    // Applies updates to the layer's weights and biases
    void UpdateWeights( const fvector_t& weightsUpdate,
                        const fvector_t& biasesUpdate ) override;

    // Saves layer's learnt parameters/weights
    bool SaveLearnedParams( FILE* file ) const override;
    // Loads layer's learnt parameters
    bool LoadLearnedParams( FILE* file ) override;
};

} } // ANNT::Neuro

#endif // ANNT_XRECURRENT_LAYER_HPP
