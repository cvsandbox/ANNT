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

#include "XGRULayer.hpp"
#include "../../Tools/XParallel.hpp"
#include "../../Tools/XVectorize.hpp"
#include <cstring>

using namespace std;

namespace ANNT { namespace Neuro {

XGRULayer::XGRULayer( size_t inputsCount, size_t outputsCount ) :
    ITrainableLayer( inputsCount, outputsCount ),
    mSigmoid( ), mTanh( ),
    mAllWeights( ( inputsCount * outputsCount + outputsCount * outputsCount ) * 3 + outputsCount * 3 )
{
    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    // set up weights pointers
    mWeightsX2Z  = mAllWeights.data( );
    mWeightsH2Z  = mWeightsX2Z + weightsCountInputs;

    mWeightsX2R  = mWeightsH2Z + weightsCountHistory;
    mWeightsH2R  = mWeightsX2R + weightsCountInputs;

    mWeightsX2H  = mWeightsH2R + weightsCountHistory;
    mWeightsHR2H = mWeightsX2H + weightsCountInputs;

    // set up biases pointers
    mBiasesZ = mWeightsHR2H + weightsCountHistory;
    mBiasesR = mBiasesZ + mOutputsCount;
    mBiasesH = mBiasesR + mOutputsCount;

    Randomize( );
}

// Randomizes layer's weights, clears biases
void XGRULayer::Randomize( )
{
    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    float halfRangeX = sqrt( 3.0f / mInputsCount );
    float halfRangeH = sqrt( 3.0f / mOutputsCount );

    for ( size_t i = 0; i < weightsCountInputs; i++ )
    {
        mWeightsX2Z[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeX ) - halfRangeX;
        mWeightsX2R[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeX ) - halfRangeX;
        mWeightsX2H[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeX ) - halfRangeX;
    }

    for ( size_t i = 0; i < weightsCountHistory; i++ )
    {
        mWeightsH2Z[i]  = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
        mWeightsH2R[i]  = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
        mWeightsHR2H[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
    }

    // See "Model Parameters" explaining why biases for Reset Gate are set to -1.0
    // https://danijar.com/tips-for-training-recurrent-neural-networks/
    for ( size_t i = 0; i < mOutputsCount; i++ )
    {
        mBiasesZ[i] =  0.0f;
        mBiasesR[i] = -1.0f;
        mBiasesH[i] =  0.0f;
    }
}

// Calculates outputs for the given inputs
void XGRULayer::ForwardCompute( const vector<fvector_t*>& inputs, vector<fvector_t*>& outputs, const XNetworkContext& ctx )
{
    size_t sequenceLen = ctx.TrainingSequenceLength( );
    size_t batchSize   = inputs.size( ) / sequenceLen;

    XParallel::For( batchSize, ctx.IsTraining( ), [&]( size_t batchIndex )
    {
        float_t* history = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY, batchIndex ) );

        for ( size_t sequenceIndex = 0; sequenceIndex < sequenceLen; sequenceIndex++ )
        {
            size_t   sampleIndex      = batchIndex * sequenceLen + sequenceIndex;
            float_t* input            = inputs [sampleIndex]->data( );
            float_t* output           = outputs[sampleIndex]->data( );   // H(t)

            float_t* historyPrev      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_PREV, sampleIndex ) );        // H(t-1)
            float_t* updateGate       = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_UPDATE_GATE, sampleIndex ) );         // Z(t)
            float_t* resetGate        = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_RESET_GATE, sampleIndex ) );          // R(t)
            float_t* historyPrevReset = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_PREV_RESET, sampleIndex ) );  // H(t-1) * R(t)
            float_t* historyHat       = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_HAT, sampleIndex ) );         // H'(t)

            // remember previous history for this particular sample
            memcpy( historyPrev, history, mOutputsCount * sizeof( float_t ) );

            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                // Wz [X(t), H(t-1)] + Bz
                updateGate[outputIndex] = XVectorize::Dot( input, &( mWeightsX2Z[outputIndex * mInputsCount] ), mInputsCount ) +
                                          XVectorize::Dot( historyPrev, &( mWeightsH2Z[outputIndex * mOutputsCount] ), mOutputsCount ) +
                                          mBiasesZ[outputIndex];
                // Wr [X(t), H(t-1)] + Br
                resetGate[outputIndex] =  XVectorize::Dot( input, &( mWeightsX2R[outputIndex * mInputsCount] ), mInputsCount ) +
                                          XVectorize::Dot( historyPrev, &( mWeightsH2R[outputIndex * mOutputsCount] ), mOutputsCount ) +
                                          mBiasesR[outputIndex];
                // W [X(t)] + B
                historyHat[outputIndex] = XVectorize::Dot( input, &( mWeightsX2H[outputIndex * mInputsCount] ), mInputsCount ) +
                                          mBiasesH[outputIndex];
            }

            // apply activations
            mSigmoid.ForwardActivate( updateGate, updateGate, mOutputsCount );
            mSigmoid.ForwardActivate( resetGate, resetGate, mOutputsCount );

            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                // previous history multiplied by reset gate
                historyPrevReset[outputIndex] = historyPrev[outputIndex] * resetGate[outputIndex];

                // first part of the ouput: (1 - Z(t)) * H(t-1)
                output[outputIndex] = historyPrev[outputIndex] * ( float_t( 1 ) - updateGate[outputIndex] );
            }

            // complete current memory content by adding reseted previous history ...
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                historyHat[outputIndex] += XVectorize::Dot( historyPrevReset, &( mWeightsHR2H[outputIndex * mOutputsCount] ), mOutputsCount );
            }
            // ... and passing through tanh() activation
            mTanh.ForwardActivate( historyHat, historyHat, mOutputsCount );

            // get the final output and put into history as well
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                // second part of the ouput: Z(t) * H'(t)
                output[outputIndex] += historyHat[outputIndex] * updateGate[outputIndex];
                history[outputIndex] = output[outputIndex];
            }
        }
    } );
}

// Propagates error to the previous layer and calculates weights/biases gradients
void XGRULayer::BackwardCompute( const vector<fvector_t*>& inputs,
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
    float_t* gradWeightsX2Z  = gradWeights.data( );
    float_t* gradWeightsH2Z  = gradWeightsX2Z + weightsCountInputs;

    float_t* gradWeightsX2R  = gradWeightsH2Z + weightsCountHistory;
    float_t* gradWeightsH2R  = gradWeightsX2R + weightsCountInputs;

    float_t* gradWeightsX2H  = gradWeightsH2R + weightsCountHistory;
    float_t* gradWeightsHR2H = gradWeightsX2H + weightsCountInputs;

    // set up biases gradient pointers
    float_t* gradBiasesZ = gradWeightsHR2H + weightsCountHistory;
    float_t* gradBiasesR = gradBiasesZ + mOutputsCount;
    float_t* gradBiasesH = gradBiasesR + mOutputsCount;

    XParallel::For( batchSize, [&]( size_t batchIndex )
    {
        float_t* historyGrad = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_GRAD, batchIndex ) );
        float_t* delta       = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_DELTA, batchIndex ) );

        for ( int sequenceIndex = (int) sequenceLen - 1; sequenceIndex >= 0; sequenceIndex-- )
        {
            size_t   sampleIndex = batchIndex * sequenceLen + sequenceIndex;
            float_t* prevDelta   = prevDeltas[sampleIndex]->data( );

            float_t* historyPrev = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_PREV, sampleIndex ) );        // H(t-1)
            float_t* updateGate  = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_UPDATE_GATE, sampleIndex ) );         // Z(t)
            float_t* resetGate   = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_RESET_GATE, sampleIndex ) );          // R(t)
            float_t* historyHat  = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_HAT, sampleIndex ) );         // H'(t)

            float_t* dUpdateGate = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_UPDATE_GATE_DELTA, sampleIndex ) );
            float_t* dResetGate  = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_RESET_GATE_DELTA, sampleIndex ) );
            float_t* dHistoryHat = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_HAT_DELTA, sampleIndex ) );

            // add history gradient from the future
            memcpy( delta, deltas[sampleIndex]->data( ), sizeof( float_t ) * mOutputsCount );
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                delta[outputIndex] += historyGrad[outputIndex];
            }

            // dE/dWz
            // dH(t)/dZ(t) = H'(t) - H(t-1)
            // delta * ( H'(t) - H(t-1) )
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                dUpdateGate[outputIndex] = delta[outputIndex] * ( historyHat[outputIndex] - historyPrev[outputIndex] );
            }
            mSigmoid.BackwardActivate( updateGate, updateGate, dUpdateGate, dUpdateGate, mOutputsCount );

            // dE/dWh
            // dH(t)/dH'(t) = Z(t)
            // delta * Z(t)
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                dHistoryHat[outputIndex] = delta[outputIndex] * updateGate[outputIndex];
            }
            mTanh.BackwardActivate( historyHat, historyHat, dHistoryHat, dHistoryHat, mOutputsCount );

            // progress with reset gate gradient
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                float_t weightedGradHistoryHat = float_t( 0 );

                for ( size_t outputIndex2 = 0, weightIndex = outputIndex; outputIndex2 < mOutputsCount; outputIndex2++, weightIndex += mOutputsCount )
                {
                    weightedGradHistoryHat += dHistoryHat[outputIndex2] * mWeightsHR2H[weightIndex];
                }

                // multiply with previous history to find reset gradient then
                dResetGate[outputIndex]  = weightedGradHistoryHat * historyPrev[outputIndex];
                // multiply with reset gate value to direct error gradient to previous history gradient
                historyGrad[outputIndex] = weightedGradHistoryHat * resetGate[outputIndex];
            }
            mSigmoid.BackwardActivate( resetGate, resetGate, dResetGate, dResetGate, mOutputsCount );
            
            // add error gradient from output to previous history gradient
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                historyGrad[outputIndex] += ( ( 1 - updateGate[outputIndex] ) * delta[outputIndex] );
            }

            // input deltas for the previous layer
            for ( size_t inputIndex = 0; inputIndex < mInputsCount; inputIndex++ )
            {
                size_t  weightIndex = inputIndex;
                float_t sum         = 0;

                for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++, weightIndex += mInputsCount )
                {
                    sum += dUpdateGate[outputIndex] * mWeightsX2Z[weightIndex];
                    sum += dResetGate[outputIndex]  * mWeightsX2R[weightIndex];
                    sum += dHistoryHat[outputIndex] * mWeightsX2H[weightIndex];
                }

                prevDelta[inputIndex] = sum;
            }

            // add more to history gradient for the previous sequence of this layer
            for ( size_t outputIndex2 = 0; outputIndex2 < mOutputsCount; outputIndex2++ )
            {
                size_t  weightIndex = outputIndex2;
                float_t sum         = 0;

                for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++, weightIndex += mOutputsCount )
                {
                    sum += dUpdateGate[outputIndex]  * mWeightsH2Z[weightIndex];
                    sum += dResetGate[outputIndex]   * mWeightsH2R[weightIndex];
                }

                historyGrad[outputIndex2] += sum;
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
                size_t   sampleIndex      = batchIndex * sequenceLen + sequenceIndex;
                const float_t* input      = inputs[sampleIndex]->data( );

                float_t* historyPrev      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_PREV, sampleIndex ) );        // H(t-1)
                float_t* historyPrevReset = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_PREV_RESET, sampleIndex ) );  // H(t-1) * R(t)

                float_t* dUpdateGate      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_UPDATE_GATE_DELTA, sampleIndex ) );
                float_t* dResetGate       = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_RESET_GATE_DELTA, sampleIndex ) );
                float_t* dHistoryHat      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_HAT_DELTA, sampleIndex ) );

                float_t  dUpdateGateVal   = dUpdateGate[outputIndex];
                float_t  dResetGateVal    = dResetGate[outputIndex];
                float_t  dHistoryHatVal   = dHistoryHat[outputIndex];

                // accumulate gradients for inputs' weights
                for ( size_t inputIndex = 0, weightIndex = weightIndexStartI; inputIndex < mInputsCount; inputIndex++, weightIndex++ )
                {
                    gradWeightsX2Z[weightIndex] += dUpdateGateVal * input[inputIndex];
                    gradWeightsX2R[weightIndex] += dResetGateVal  * input[inputIndex];
                    gradWeightsX2H[weightIndex] += dHistoryHatVal * input[inputIndex];
                }

                // accumulate gradients for history weights
                if ( sequenceIndex != 0 )
                {
                    for ( size_t historyIndex = 0, weightIndex = weightIndexStartH; historyIndex < mOutputsCount; historyIndex++, weightIndex++ )
                    {
                        gradWeightsH2Z[weightIndex]  += dUpdateGateVal * historyPrev[historyIndex];
                        gradWeightsH2R[weightIndex]  += dResetGateVal  * historyPrev[historyIndex];
                        gradWeightsHR2H[weightIndex] += dHistoryHatVal * historyPrevReset[historyIndex];
                    }
                }

                // accumulate gradients for biases
                gradBiasesZ[outputIndex] += dUpdateGateVal;
                gradBiasesR[outputIndex] += dResetGateVal;
                gradBiasesH[outputIndex] += dHistoryHatVal;
            }
        }
    } );
}

// Applies updates to the layer's weights and biases
void XGRULayer::UpdateWeights( const fvector_t& updates )
{
    for ( size_t i = 0, n = mAllWeights.size( ); i < n; i++ )
    {
        mAllWeights[i] += updates[i];
    }
}

// Saves layer's learnt parameters/weights
bool XGRULayer::SaveLearnedParams( FILE* file ) const
{
    vector<const fvector_t*> params( { &mAllWeights } );

    return SaveLearnedParamsHelper( file, LayerID::RecurrentGRU, params );
}

// Loads layer's learnt parameters
bool XGRULayer::LoadLearnedParams( FILE* file )
{
    vector<fvector_t*> params( { &mAllWeights } );

    return LoadLearnedParamsHelper( file, LayerID::RecurrentGRU, params );
}

} } // ANNT::Neuro
