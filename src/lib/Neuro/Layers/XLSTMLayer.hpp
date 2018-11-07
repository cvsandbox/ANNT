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
#ifndef ANNT_XLSTM_LAYER_HPP
#define ANNT_XLSTM_LAYER_HPP

#include "ITrainableLayer.hpp"
#include "Activations/XSigmoidActivation.hpp"
#include "Activations/XTanhActivation.hpp"

namespace ANNT { namespace Neuro {

// Implementation of Long Short-Term Memory (LSTM) recurrent layer
//
// http://colah.github.io/posts/2015-08-Understanding-LSTMs/
// https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
//
// Forget gate:
//   F(t) = sigmoid( Wf [X(t), H(t-1)] + Bf )
// Input gate:
//   I(t) = sigmoid( Wi [X(t), H(t-1)] + Bi )
// Candidate state:
//   Z(t) = tanh( Wz [X(t), H(t-1)] + Bz )
// State:
//   C(t) = F(t) * C(t-1) + I(t) * Z(t)
// Output gate:
//   O(t) = sigmoid( Wo [X(t), H(t-1)] + Bo )
// Output/History:
//   H(t) = O(t) * tanh( C(t) )
//
class XLSTMLayer : public ITrainableLayer
{
private:
    XSigmoidActivation mSigmoid;
    XTanhActivation    mTanh;

    // Weights and biases are all kept together
    fvector_t mAllWeights;

    // And here are the pointers to specific weights/biases, which are used to calculate different
    // vectors from current layer's input, X(t), and its previous output/history, H(t-1):

    // 1) to calculate "forget gate" vector, F(t);
    float_t* mWeightsX2F;
    float_t* mWeightsH2F;
    float_t* mBiasesF;
    // 2) to calculate "input gate" vector, I(t);
    float_t* mWeightsX2I;
    float_t* mWeightsH2I;
    float_t* mBiasesI;
    // 3) to calculate "candidate state" vector, Z(t) (C with tilde);
    float_t* mWeightsX2Z;
    float_t* mWeightsH2Z;
    float_t* mBiasesZ;
    // 4) to calculate "output gate" vector, O(t)
    float_t* mWeightsX2O;
    float_t* mWeightsH2O;
    float_t* mBiasesO;

    // --------------------------------------------------------------------------------------
    enum
    {
        // per batch
        BUFFER_INDEX_STATE           = 0,
        BUFFER_INDEX_STATE_GRAD      = 1,
        BUFFER_INDEX_HISTORY         = 2,
        BUFFER_INDEX_HISTORY_GRAD    = 3,
        BUFFER_INDEX_DELTA           = 4,   // sum of the incoming gradient (from the next layer)
                                            // and history gradient
        BUFFER_INDEX_STATE_DELTA     = 5,   // BUFFER_INDEX_DELTA passed backward through output gate 

        // per sample
        BUFFER_INDEX_STATE_PREV      = 6,   // C(t-1)
        BUFFER_INDEX_STATE_NEXT      = 7,   // C(t)
        BUFFER_INDEX_HISTORY_PREV    = 8,   // H(t-1)
        BUFFER_INDEX_FORGET_GATE     = 9,   // F(t)
        BUFFER_INDEX_INPUT_GATE      = 10,  // I(t)
        BUFFER_INDEX_OUTPUT_GATE     = 11,  // O(t)
        BUFFER_INDEX_CANDIDATE_STATE = 12,  // Z(t)
        BUFFER_INDEX_STATE_NEXT_TANH = 13,  // tanh(C(t))

        BUFFER_INDEX_CANDIDATE_STATE_DELTA = 14,
        BUFFER_INDEX_INPUT_GATE_DELTA      = 15,
        BUFFER_INDEX_FORGET_GATE_DELTA     = 16,
        BUFFER_INDEX_OUTPUT_GATE_DELTA     = 17,
    };

public:

    XLSTMLayer( size_t inputsCount, size_t outputsCount );

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
        uvector_t workingMemSize = uvector_t( 18, mOutputsCount * sizeof( float_t ) );

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

#endif // ANNT_XLSTM_LAYER_HPP
