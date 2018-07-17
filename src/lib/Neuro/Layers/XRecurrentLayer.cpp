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

#include "XRecurrentLayer.hpp"
#include "../../Tools/XParallel.hpp"
#include "../../Tools/XVectorize.hpp"
#include <cstring>

using namespace std;

namespace ANNT { namespace Neuro {

XRecurrentLayer::XRecurrentLayer( size_t inputsCount, size_t outputsCount ) :
    ITrainableLayer( inputsCount, outputsCount ),
    mTanh( ),
    mWeights( ( inputsCount + outputsCount + outputsCount ) * outputsCount ),
    mBiases( outputsCount * 2 )
{
    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    // set up weights pointers
    mWeightsU = mWeights.data( );
    mWeightsW = mWeightsU + weightsCountInputs;
    mWeightsV = mWeightsW + weightsCountHistory;

    // set up biases pointers
    mBiasesB = mBiases.data( );
    mBiasesC = mBiasesB + mOutputsCount;

    Randomize( );
}

// Randomizes layer's weights, clears biases
void XRecurrentLayer::Randomize( )
{
    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    float halfRangeX = sqrt( 3.0f / mInputsCount );
    float halfRangeH = sqrt( 3.0f / mOutputsCount );

    for ( size_t i = 0; i < weightsCountInputs; i++ )
    {
        mWeightsU[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeX ) - halfRangeX;
    }

    for ( size_t i = 0; i < weightsCountHistory; i++ )
    {
        mWeightsW[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
        mWeightsV[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
    }

    for ( auto& b : mBiases )
    {
        b = 0;
    }
}

// Calculates outputs for the given inputs
void XRecurrentLayer::ForwardCompute( const vector<fvector_t*>& inputs,
                                      vector<fvector_t*>& outputs,
                                      const XNetworkContext& ctx )
{
    size_t sequenceLen = ctx.TrainingSequenceLength( );
    size_t batchSize   = inputs.size( ) / sequenceLen;

    XParallel::For( batchSize, ctx.IsTraining( ), [&]( size_t batchIndex )
    {
        float_t* state = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE, batchIndex ) );

        for ( size_t sequenceIndex = 0; sequenceIndex < sequenceLen; sequenceIndex++ )
        {
            size_t    sampleIndex  = batchIndex * sequenceLen + sequenceIndex;
            float_t*  input        = inputs [sampleIndex]->data( );
            float_t*  output       = outputs[sampleIndex]->data( );

            float_t*  statePrev    = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_PREV, sampleIndex ) );    // H(t-1)
            float_t*  stateCurrent = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_CURRENT, sampleIndex ) ); // H(t)

            // remember previous state for this particular sample
            memcpy( statePrev, state, mOutputsCount * sizeof( float_t ) );

            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                state[outputIndex] =
                    // X(t) * U
                    XVectorize::Dot( input, &( mWeightsU[outputIndex * mInputsCount] ), mInputsCount ) +
                    // H(t-1) * W
                    XVectorize::Dot( statePrev, &( mWeightsW[outputIndex * mOutputsCount] ), mOutputsCount ) +
                    // B
                    mBiasesB[outputIndex];
            }

            // apply tanh() to get the final H(t)
            mTanh.ForwardActivate( state, state, mOutputsCount );

            // remember current state for this sample, to use on backward step
            memcpy( stateCurrent, state, mOutputsCount * sizeof( float_t ) );

            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                output[outputIndex] =
                    // V * H(t)
                    XVectorize::Dot( state, &( mWeightsV[outputIndex * mOutputsCount] ), mOutputsCount ) +
                    // C
                    mBiasesC[outputIndex];
            }
        }
    } );
}

// Propagates error to the previous layer and calculates weights/biases gradients
void XRecurrentLayer::BackwardCompute( const vector<fvector_t*>& inputs,
                                       const vector<fvector_t*>& /* outputs */,
                                       const vector<fvector_t*>& deltas,
                                       vector<fvector_t*>& prevDeltas,
                                       fvector_t& gradWeights,
                                       fvector_t& gradBiases,
                                       const XNetworkContext& ctx )
{
    size_t sequenceLen = ctx.TrainingSequenceLength( );
    size_t batchSize   = inputs.size( ) / sequenceLen;

    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    // set up weights gradient pointers
    float_t* gradWeightsU = gradWeights.data( );
    float_t* gradWeightsW = gradWeightsU + weightsCountInputs;
    float_t* gradWeightsV = gradWeightsW + weightsCountHistory;

    // set up biases gradient pointers
    float_t* gradBiasesB = gradBiases.data( );
    float_t* gradBiasesC = gradBiasesB + mOutputsCount;

    // temporary buffer to calculate state delta for current sample
    fvector_t  stateDeltaCurrnet( mOutputsCount );

    for ( size_t batchIndex = 0; batchIndex < batchSize; batchIndex++ )
    {
        // accumulated state delta
        float_t* stateGrad = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_GRAD, batchIndex ) );

        for ( int sequenceIndex = (int) sequenceLen - 1; sequenceIndex >= 0; sequenceIndex-- )
        {
            size_t  sampleIndex         = batchIndex * sequenceLen + sequenceIndex;

            const fvector_t& input      = *( inputs[sampleIndex] );
            const fvector_t& delta      = *( deltas[sampleIndex] );
            fvector_t&       prevDelta  = *( prevDeltas[sampleIndex] );

            float_t*  statePrev         = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_PREV, sampleIndex ) );    // H(t-1)
            float_t*  stateCurrent      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_CURRENT, sampleIndex ) ); // H(t)

            // initial calculation of state delta for the current sample
            for ( size_t outputIndex2 = 0; outputIndex2 < mOutputsCount; outputIndex2++ )
            {
                size_t  weightIndex = outputIndex2;
                float_t sum         = 0;

                for ( size_t otputIndex = 0; otputIndex < mOutputsCount; otputIndex++, weightIndex += mOutputsCount )
                {
                    sum += delta[otputIndex] * mWeightsV[weightIndex];
                }

                stateDeltaCurrnet[outputIndex2] = sum;
            }

            // add state gradient from the future
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                stateDeltaCurrnet[outputIndex] += stateGrad[outputIndex];
            }

            // backward pass through Tanh activation to get final state delta for current sample
            mTanh.BackwardActivate( stateCurrent, stateCurrent, stateDeltaCurrnet.data( ), stateDeltaCurrnet.data( ), mOutputsCount );

            // input deltas for the previous layer
            for ( size_t inputIndex = 0; inputIndex < mInputsCount; inputIndex++ )
            {
                size_t  weightIndex = inputIndex;
                float_t sum = 0;

                for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++, weightIndex += mInputsCount )
                {
                    sum += stateDeltaCurrnet[outputIndex] * mWeightsU[weightIndex];
                }

                prevDelta[inputIndex] = sum;
            }

            // state gradients for the previous sample in the time series
            for ( size_t outputIndex2 = 0; outputIndex2 < mOutputsCount; outputIndex2++ )
            {
                size_t  weightIndex = outputIndex2;
                float_t sum = 0;

                for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++, weightIndex += mOutputsCount )
                {
                    sum += stateDeltaCurrnet[outputIndex] * mWeightsW[weightIndex];
                }

                stateGrad[outputIndex2] = sum;
            }

            // calculate weights/biases updates

            // dU
            for ( size_t outputIndex = 0, weightIndexStart = 0; outputIndex < mOutputsCount; outputIndex++, weightIndexStart += mInputsCount )
            {
                for ( size_t inputIndex = 0, weightIndex = weightIndexStart; inputIndex < mInputsCount; inputIndex++, weightIndex++ )
                {
                    gradWeightsU[weightIndex] += stateDeltaCurrnet[outputIndex] * input[inputIndex];
                }
            }

            // dW
            if ( sequenceIndex != 0 )
            {
                for ( size_t outputIndex = 0, weightIndexStart = 0; outputIndex < mOutputsCount; outputIndex++, weightIndexStart += mOutputsCount )
                {
                    for ( size_t outputIndex2 = 0, weightIndex = weightIndexStart; outputIndex < mOutputsCount; outputIndex++, weightIndex++ )
                    {
                        gradWeightsW[weightIndex] += stateDeltaCurrnet[outputIndex] * statePrev[outputIndex2];
                    }
                }
            }

            // dV
            for ( size_t outputIndex = 0, weightIndexStart = 0; outputIndex < mOutputsCount; outputIndex++, weightIndexStart += mOutputsCount )
            {
                for ( size_t outputIndex2 = 0, weightIndex = weightIndexStart; outputIndex < mOutputsCount; outputIndex++, weightIndex++ )
                {
                    gradWeightsV[weightIndex] += delta[outputIndex] * stateCurrent[outputIndex2];
                }
            }

            // dB
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                gradBiasesB[outputIndex] += stateDeltaCurrnet[outputIndex];
            }

            // dC
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                gradBiasesC[outputIndex] += delta[outputIndex];
            }
        }
    }
}

// Applies updates to the layer's weights and biases
void XRecurrentLayer::UpdateWeights( const fvector_t& weightsUpdate,
                                     const fvector_t& biasesUpdate )
{
    for ( size_t i = 0, n = mWeights.size( ); i < n; i++ )
    {
        mWeights[i] += weightsUpdate[i];
    }
    for ( size_t i = 0, n = mBiases.size( ); i < n; i++ )
    {
        mBiases[i] += biasesUpdate[i];
    }
}

// Saves layer's learnt parameters/weights
bool XRecurrentLayer::SaveLearnedParams( FILE* file ) const
{
    vector<const fvector_t*> params( { &mWeights, &mBiases } );

    return SaveLearnedParamsHelper( file, LayerID::RecurrentBasic, params );
}

// Loads layer's learnt parameters
bool XRecurrentLayer::LoadLearnedParams( FILE* file )
{
    vector<fvector_t*> params( { &mWeights, &mBiases } );

    return LoadLearnedParamsHelper( file, LayerID::RecurrentBasic, params );
}

} } // ANNT::Neuro
