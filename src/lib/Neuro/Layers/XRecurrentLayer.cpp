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
    mAllWeights( ( inputsCount + outputsCount ) * outputsCount  + outputsCount )
{
    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    // set up weights pointers
    mWeightsU = mAllWeights.data( );
    mWeightsW = mWeightsU + weightsCountInputs;

    // set up biases pointers
    mBiasesB  = mWeightsW + weightsCountHistory;

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
    }

    for ( size_t i = 0; i < mOutputsCount; i++ )
    {
        mBiasesB[i] = 0;
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

            // copy state to output
            memcpy( output, state, mOutputsCount * sizeof( float_t ) );

            // remember current state for this sample, to use on backward step
            memcpy( stateCurrent, state, mOutputsCount * sizeof( float_t ) );
        }
    } );
}

// Propagates error to the previous layer and calculates weights/biases gradients
void XRecurrentLayer::BackwardCompute( const vector<fvector_t*>& inputs,
                                       const vector<fvector_t*>& /* outputs */,
                                       const vector<fvector_t*>& deltas,
                                       vector<fvector_t*>& prevDeltas,
                                       fvector_t& gradWeights,
                                       const XNetworkContext& ctx )
{
    size_t sequenceLen   = ctx.TrainingSequenceLength( );
    size_t batchSize     = inputs.size( ) / sequenceLen;

    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    // set up weights gradient pointers
    float_t* gradWeightsU = gradWeights.data( );
    float_t* gradWeightsW = gradWeightsU + weightsCountInputs;
    
    // set up biases gradient pointers
    float_t* gradBiasesB = gradWeightsW + weightsCountHistory;
    
    XParallel::For( batchSize, [&]( size_t batchIndex )
    {
        // accumulated state delta
        float_t* stateGrad = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_GRAD, batchIndex ) );

        for ( int sequenceIndex = (int) sequenceLen - 1; sequenceIndex >= 0; sequenceIndex-- )
        {
            size_t  sampleIndex         = batchIndex * sequenceLen + sequenceIndex;

            const fvector_t& delta      = *( deltas[sampleIndex] );
            fvector_t&       prevDelta  = *( prevDeltas[sampleIndex] );

            float_t*  stateCurrent      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_CURRENT, sampleIndex ) ); // H(t)
            float_t*  stateDeltaCurrent = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_DELTA_CURRENT, sampleIndex ) );

            // add state gradient from the future
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                stateDeltaCurrent[outputIndex] = delta[outputIndex] + stateGrad[outputIndex];
            }

            // backward pass through Tanh activation to get final state delta for current sample
            mTanh.BackwardActivate( stateCurrent, stateCurrent, stateDeltaCurrent, stateDeltaCurrent, mOutputsCount );

            // input deltas for the previous layer
            for ( size_t inputIndex = 0; inputIndex < mInputsCount; inputIndex++ )
            {
                size_t  weightIndex = inputIndex;
                float_t sum = 0;

                for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++, weightIndex += mInputsCount )
                {
                    sum += stateDeltaCurrent[outputIndex] * mWeightsU[weightIndex];
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
                    sum += stateDeltaCurrent[outputIndex] * mWeightsW[weightIndex];
                }

                stateGrad[outputIndex2] = sum;
            }
        }
    } );

    XParallel::For( mOutputsCount, [&]( size_t outputIndex )
    {
        size_t weightIndexStartI = outputIndex * mInputsCount;
        size_t weightIndexStartH = outputIndex * mOutputsCount;

        for ( size_t batchIndex = 0; batchIndex < batchSize; batchIndex++ )
        {
            for ( int sequenceIndex = (int) sequenceLen - 1; sequenceIndex >= 0; sequenceIndex-- )
            {
                size_t  sampleIndex         = batchIndex * sequenceLen + sequenceIndex;
                const fvector_t& input      = *( inputs[sampleIndex] );
                float_t*  statePrev         = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_PREV, sampleIndex ) );    // H(t-1)
                float_t*  stateDeltaCurrnet = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_DELTA_CURRENT, sampleIndex ) );

                // accumulate weights' gradients
                // dU
                for ( size_t inputIndex = 0, weightIndex = weightIndexStartI; inputIndex < mInputsCount; inputIndex++, weightIndex++ )
                {
                    gradWeightsU[weightIndex] += stateDeltaCurrnet[outputIndex] * input[inputIndex];
                }

                // dW
                if ( sequenceIndex != 0 )
                {
                    for ( size_t historyIndex = 0, weightIndex = weightIndexStartH; historyIndex < mOutputsCount; historyIndex++, weightIndex++ )
                    {
                        gradWeightsW[weightIndex] += stateDeltaCurrnet[outputIndex] * statePrev[historyIndex];
                    }
                }

                // accumulate biases' gradients
                // dB
                gradBiasesB[outputIndex] += stateDeltaCurrnet[outputIndex];
            }
        }
    } );
}

// Applies updates to the layer's weights and biases
void XRecurrentLayer::UpdateWeights( const fvector_t& updates )
{
    for ( size_t i = 0, n = mAllWeights.size( ); i < n; i++ )
    {
        mAllWeights[i] += updates[i];
    }
}

// Saves layer's learnt parameters/weights
bool XRecurrentLayer::SaveLearnedParams( FILE* file ) const
{
    vector<const fvector_t*> params( { &mAllWeights } );

    return SaveLearnedParamsHelper( file, LayerID::RecurrentBasic, params );
}

// Loads layer's learnt parameters
bool XRecurrentLayer::LoadLearnedParams( FILE* file )
{
    vector<fvector_t*> params( { &mAllWeights } );

    return LoadLearnedParamsHelper( file, LayerID::RecurrentBasic, params );
}

} } // ANNT::Neuro
