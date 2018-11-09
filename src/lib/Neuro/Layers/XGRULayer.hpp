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
#ifndef ANNT_XGRU_LAYER_HPP
#define ANNT_XGRU_LAYER_HPP

#include "ITrainableLayer.hpp"
#include "Activations/XSigmoidActivation.hpp"
#include "Activations/XTanhActivation.hpp"

namespace ANNT { namespace Neuro {

// Implementation of Gated Recurrent Unit (GRU) layer
//
// http://colah.github.io/posts/2015-08-Understanding-LSTMs/
// https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
//
// Some info on GRU backpropagation. NOTE: be careful with that. First link does not
// provide all equations. While the second has some obvious errors.
// Some understanding of backpropagation and derivatives math should help though.
// https://medium.com/swlh/only-numpy-deriving-forward-feed-and-back-propagation-in-gated-recurrent-neural-networks-gru-8b6810f91bad
// https://cran.r-project.org/web/packages/rnn/vignettes/GRU_units.html
//
// Update gate:
//   Z(t) = sigmoid( Wz [X(t), H(t-1)] + Bz )
// Reset gate:
//   R(t) = sigmoid( Wr [X(t), H(t-1)] + Br )
// Current memory content:
//   H'(t) = tanh( W [X(t), R(t)*H(t-1)] + B )
// Output/History:
//   H(t) = (1 - Z(t)) * H(t-1) + Z(t) * H'(t)
//
class XGRULayer : public ITrainableLayer
{
private:
    XSigmoidActivation mSigmoid;
    XTanhActivation    mTanh;

    // Weights and biases are all kept together
    fvector_t mAllWeights;

    // And here are the pointers to specific weights/biases, which are used to calculate different
    // vectors from current layer's input, X(t), and its previous output/history, H(t-1):

    // 1) to calculate "update gate" vector, Z(t);
    float_t* mWeightsX2Z;
    float_t* mWeightsH2Z;
    float_t* mBiasesZ;
    // 2) to calculate "reset gate" vector, R(t);
    float_t* mWeightsX2R;
    float_t* mWeightsH2R;
    float_t* mBiasesR;
    // 3) to calculate "current memory content", H'(t);
    float_t* mWeightsX2H;
    float_t* mWeightsHR2H;
    float_t* mBiasesH;

    // --------------------------------------------------------------------------------------
    enum
    {
        // per batch
        BUFFER_INDEX_HISTORY            = 0,
        BUFFER_INDEX_HISTORY_GRAD       = 1,
        BUFFER_INDEX_DELTA              = 2,    // sum of the incoming gradient (from the next layer)
                                                // and history gradient

        // per sample
        BUFFER_INDEX_HISTORY_PREV       = 3,    // H(t-1)
        BUFFER_INDEX_UPDATE_GATE        = 4,    // Z(t)
        BUFFER_INDEX_RESET_GATE         = 5,    // R(t)
        BUFFER_INDEX_HISTORY_PREV_RESET = 6,    // H(t-1) * R(t)
        BUFFER_INDEX_HISTORY_HAT        = 7,    // H'(t)

        BUFFER_INDEX_UPDATE_GATE_DELTA  = 8,
        BUFFER_INDEX_RESET_GATE_DELTA   = 9,
        BUFFER_INDEX_HISTORY_HAT_DELTA  = 10,
    };

public:
    XGRULayer( size_t inputsCount, size_t outputsCount );

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

    // Tells that we may need some extra memory for internal state/calculations
    uvector_t WorkingMemSize( bool /* trainingMode */ ) const override
    {
        uvector_t workingMemSize = uvector_t( 11, mOutputsCount * sizeof( float_t ) );

        return workingMemSize;
    }

    // Randomizes layer's weights, clears biases (forget gate biases are set to 1 though)
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

} } // ANNT::Neuro

#endif // ANNT_XGRU_LAYER_HPP
