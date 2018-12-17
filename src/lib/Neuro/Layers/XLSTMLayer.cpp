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

#include "XLSTMLayer.hpp"
#include "../../Tools/XParallel.hpp"
#include "../../Tools/XVectorize.hpp"
#include <cstring>

using namespace std;

namespace ANNT { namespace Neuro {

XLSTMLayer::XLSTMLayer( size_t inputsCount, size_t outputsCount ) :
    ITrainableLayer( inputsCount, outputsCount ),
    mSigmoid( ), mTanh( ),
    mAllWeights( ( inputsCount * outputsCount + outputsCount * outputsCount ) * 4 + outputsCount * 4 )
{
    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    // set up weights pointers
    mWeightsX2F = mAllWeights.data( );
    mWeightsH2F = mWeightsX2F + weightsCountInputs;

    mWeightsX2I = mWeightsH2F + weightsCountHistory;
    mWeightsH2I = mWeightsX2I + weightsCountInputs;

    mWeightsX2Z = mWeightsH2I + weightsCountHistory;
    mWeightsH2Z = mWeightsX2Z + weightsCountInputs;

    mWeightsX2O = mWeightsH2Z + weightsCountHistory;
    mWeightsH2O = mWeightsX2O + weightsCountInputs;

    // set up biases pointers
    mBiasesF = mWeightsH2O + weightsCountHistory;
    mBiasesI = mBiasesF + mOutputsCount;
    mBiasesZ = mBiasesI + mOutputsCount;
    mBiasesO = mBiasesZ + mOutputsCount;

    Randomize( );
}

// Randomizes layer's weights, clears biases
void XLSTMLayer::Randomize( )
{
    size_t weightsCountInputs  = mInputsCount  * mOutputsCount;
    size_t weightsCountHistory = mOutputsCount * mOutputsCount;

    float halfRangeX = sqrt( 3.0f / mInputsCount );
    float halfRangeH = sqrt( 3.0f / mOutputsCount );

    for ( size_t i = 0; i < weightsCountInputs; i++ )
    {
        mWeightsX2F[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeX ) - halfRangeX;
        mWeightsX2I[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeX ) - halfRangeX;
        mWeightsX2Z[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeX ) - halfRangeX;
        mWeightsX2O[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeX ) - halfRangeX;
    }

    for ( size_t i = 0; i < weightsCountHistory; i++ )
    {
        mWeightsH2F[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
        mWeightsH2I[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
        mWeightsH2Z[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
        mWeightsH2O[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRangeH ) - halfRangeH;
    }

    // See "Model Parameters" explaining why biases for Forget Gate are set to 1.0
    // https://danijar.com/tips-for-training-recurrent-neural-networks/
    for ( size_t i = 0; i < mOutputsCount; i++ )
    {
        mBiasesF[i] = 1.0f;
        mBiasesI[i] = 0.0f;
        mBiasesZ[i] = 0.0f;
        mBiasesO[i] = 0.0f;
    }
}

// Calculates outputs for the given inputs
void XLSTMLayer::ForwardCompute( const vector<fvector_t*>& inputs, vector<fvector_t*>& outputs, const XNetworkContext& ctx )
{
    size_t sequenceLen = ctx.TrainingSequenceLength( );
    size_t batchSize   = inputs.size( ) / sequenceLen;

    XParallel::For( batchSize, ctx.IsTraining( ), [&]( size_t batchIndex )
    {
        float_t* state   = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE,   batchIndex ) );
        float_t* history = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY, batchIndex ) );

        for ( size_t sequenceIndex = 0; sequenceIndex < sequenceLen; sequenceIndex++ )
        {
            size_t   sampleIndex    = batchIndex * sequenceLen + sequenceIndex;
            float_t* input          = inputs [sampleIndex]->data( );
            float_t* output         = outputs[sampleIndex]->data( );   // H(t)
            float_t* statePrev      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_PREV, sampleIndex ) );      // C(t-1)
            float_t* stateNext      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_NEXT, sampleIndex ) );      // C(t)
            float_t* historyPrev    = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_PREV, sampleIndex ) );    // H(t-1)
            float_t* forgetGate     = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_FORGET_GATE, sampleIndex ) );     // F(t)
            float_t* inputGate      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_INPUT_GATE, sampleIndex ) );      // I(t)
            float_t* candidateState = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_CANDIDATE_STATE, sampleIndex ) ); // Z(t)
            float_t* outputGate     = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_OUTPUT_GATE, sampleIndex ) );     // O(t)
            float_t* stateNextTanh  = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_NEXT_TANH, sampleIndex ) ); // tanh(C(t))

            // remember previous state/history for this particular sample
            memcpy( statePrev, state, mOutputsCount * sizeof( float_t ) );
            memcpy( historyPrev, history, mOutputsCount * sizeof( float_t ) );

            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                // Wf [X(t), H(t-1)] + Bf
                forgetGate[outputIndex]     = XVectorize::Dot( input, &( mWeightsX2F[outputIndex * mInputsCount] ), mInputsCount ) +
                                              XVectorize::Dot( historyPrev, &( mWeightsH2F[outputIndex * mOutputsCount] ), mOutputsCount ) +
                                              mBiasesF[outputIndex];
                // Wi [X(t), H(t-1)] + Bi
                inputGate[outputIndex]      = XVectorize::Dot( input, &( mWeightsX2I[outputIndex * mInputsCount] ), mInputsCount ) +
                                              XVectorize::Dot( historyPrev, &( mWeightsH2I[outputIndex * mOutputsCount] ), mOutputsCount ) +
                                              mBiasesI[outputIndex];
                // Wz [X(t), H(t-1)] + Bz
                candidateState[outputIndex] = XVectorize::Dot( input, &( mWeightsX2Z[outputIndex * mInputsCount] ), mInputsCount ) +
                                              XVectorize::Dot( historyPrev, &( mWeightsH2Z[outputIndex * mOutputsCount] ), mOutputsCount ) +
                                              mBiasesZ[outputIndex];
                // Wo [X(t), H(t-1)] + Bo
                outputGate[outputIndex]     = XVectorize::Dot( input, &( mWeightsX2O[outputIndex * mInputsCount] ), mInputsCount ) +
                                              XVectorize::Dot( historyPrev, &( mWeightsH2O[outputIndex * mOutputsCount] ), mOutputsCount ) +
                                              mBiasesO[outputIndex];
            }

            // apply activations
            mSigmoid.ForwardActivate( forgetGate, forgetGate, mOutputsCount );
            mSigmoid.ForwardActivate( inputGate, inputGate, mOutputsCount );
            mSigmoid.ForwardActivate( outputGate, outputGate, mOutputsCount );
            mTanh.ForwardActivate( candidateState, candidateState, mOutputsCount );

            // get the new state: C(t) = F(t) * C(t-1) + I(t) * Z(t)
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                state[outputIndex]     =
                stateNext[outputIndex] = forgetGate[outputIndex] * statePrev[outputIndex] +
                                         inputGate[outputIndex]  * candidateState[outputIndex];
            }

            // get the tanh(C(t))
            mTanh.ForwardActivate( stateNext, stateNextTanh, mOutputsCount );

            // finally get the next output and keep it into history
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                history[outputIndex] =
                output[outputIndex]  = outputGate[outputIndex] * stateNextTanh[outputIndex];
            }
        }
    } );
}

// Propagates error to the previous layer and calculates weights/biases gradients
void XLSTMLayer::BackwardCompute( const vector<fvector_t*>& inputs,
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
    float_t* gradWeightsX2F = gradWeights.data( );
    float_t* gradWeightsH2F = gradWeightsX2F + weightsCountInputs;

    float_t* gradWeightsX2I = gradWeightsH2F + weightsCountHistory;
    float_t* gradWeightsH2I = gradWeightsX2I + weightsCountInputs;

    float_t* gradWeightsX2Z = gradWeightsH2I + weightsCountHistory;
    float_t* gradWeightsH2Z = gradWeightsX2Z + weightsCountInputs;

    float_t* gradWeightsX2O = gradWeightsH2Z + weightsCountHistory;
    float_t* gradWeightsH2O = gradWeightsX2O + weightsCountInputs;

    // set up biases gradient pointers
    float_t* gradBiasesF = gradWeightsH2O + weightsCountHistory;
    float_t* gradBiasesI = gradBiasesF + mOutputsCount;
    float_t* gradBiasesZ = gradBiasesI + mOutputsCount;
    float_t* gradBiasesO = gradBiasesZ + mOutputsCount;

    XParallel::For( batchSize, [&]( size_t batchIndex )
    {
        float_t* stateGrad   = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_GRAD, batchIndex ) );
        float_t* historyGrad = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_GRAD, batchIndex ) );
        float_t* delta       = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_DELTA, batchIndex ) );
        float_t* dState      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_DELTA, batchIndex ) );

        for ( int sequenceIndex = (int) sequenceLen - 1; sequenceIndex >= 0; sequenceIndex-- )
        {
            size_t   sampleIndex    = batchIndex * sequenceLen + sequenceIndex;
            float_t* prevDelta      = prevDeltas[sampleIndex]->data( );

            float_t* statePrev      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_PREV, sampleIndex ) );      // C(t-1)
            float_t* stateNext      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_NEXT, sampleIndex ) );      // C(t)
            float_t* forgetGate     = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_FORGET_GATE, sampleIndex ) );     // F(t)
            float_t* inputGate      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_INPUT_GATE, sampleIndex ) );      // I(t)
            float_t* candidateState = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_CANDIDATE_STATE, sampleIndex ) ); // Z(t)
            float_t* outputGate     = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_OUTPUT_GATE, sampleIndex ) );     // O(t)
            float_t* stateNextTanh  = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_STATE_NEXT_TANH, sampleIndex ) ); // tanh(C(t))

            float_t* dCadidateState = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_CANDIDATE_STATE_DELTA, sampleIndex ) );
            float_t* dInputGate     = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_INPUT_GATE_DELTA, sampleIndex ) );
            float_t* dForgetGate    = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_FORGET_GATE_DELTA, sampleIndex ) );
            float_t* dOutputGate    = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_OUTPUT_GATE_DELTA, sampleIndex ) );

            // add history gradient from the future
            memcpy( delta, deltas[sampleIndex]->data( ), sizeof( float_t ) * mOutputsCount );
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                delta[outputIndex] += historyGrad[outputIndex];

                // precalculate other history based gradients
                dState[outputIndex]      = delta[outputIndex];
                dOutputGate[outputIndex] = delta[outputIndex];
            }

            // pass deltas backward through output gate ...
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                dState[outputIndex] *= outputGate[outputIndex];
            }
            // ... and through tanh() activation
            mTanh.BackwardActivate( stateNext, stateNextTanh, dState, dState, mOutputsCount );

            // add state gradient from the future
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                dState[outputIndex] += stateGrad[outputIndex];

                // precalculate other state based gradients
                dCadidateState[outputIndex] = dState[outputIndex];
                dInputGate[outputIndex]     = dState[outputIndex];
                dForgetGate[outputIndex]    = dState[outputIndex];
            }

            // pass state gradient backward through forget gate, so it is ready for the previous sample in the time series
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                stateGrad[outputIndex] = dState[outputIndex] * forgetGate[outputIndex];
            }
            
            // pass state gradient backward through input gate and tanh() activation to get candidate state gradient
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                dCadidateState[outputIndex] *= inputGate[outputIndex];
            }
            mTanh.BackwardActivate( candidateState, candidateState, dCadidateState, dCadidateState, mOutputsCount );

            // input gate gradients
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                dInputGate[outputIndex] *= candidateState[outputIndex];
            }
            mSigmoid.BackwardActivate( inputGate, inputGate, dInputGate, dInputGate, mOutputsCount );

            // forget gate gradients
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                dForgetGate[outputIndex] *= statePrev[outputIndex];
            }
            mSigmoid.BackwardActivate( forgetGate, forgetGate, dForgetGate, dForgetGate, mOutputsCount );

            // output gate gradients
            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                dOutputGate[outputIndex] *= stateNextTanh[outputIndex];
            }
            mSigmoid.BackwardActivate( outputGate, outputGate, dOutputGate, dOutputGate, mOutputsCount );

            // calculate gradients to pass to the previous layer of the network
            for ( size_t inputIndex = 0; inputIndex < mInputsCount; inputIndex++ )
            {
                size_t  weightIndex = inputIndex;
                float_t sum         = 0;

                for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++, weightIndex += mInputsCount )
                {
                    sum += dForgetGate[outputIndex]    * mWeightsX2F[weightIndex];
                    sum += dInputGate[outputIndex]     * mWeightsX2I[weightIndex];
                    sum += dOutputGate[outputIndex]    * mWeightsX2O[weightIndex];
                    sum += dCadidateState[outputIndex] * mWeightsX2Z[weightIndex];
                }

                prevDelta[inputIndex] = sum;
            }

            // calculate gradients to pass to the previous sample of the time series
            for ( size_t outputIndex2 = 0; outputIndex2 < mOutputsCount; outputIndex2++ )
            {
                size_t  weightIndex = outputIndex2;
                float_t sum         = 0;

                for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++, weightIndex += mOutputsCount )
                {
                    sum += dForgetGate[outputIndex]    * mWeightsH2F[weightIndex];
                    sum += dInputGate[outputIndex]     * mWeightsH2I[weightIndex];
                    sum += dOutputGate[outputIndex]    * mWeightsH2O[weightIndex];
                    sum += dCadidateState[outputIndex] * mWeightsH2Z[weightIndex];
                }

                historyGrad[outputIndex2] = sum;
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
                size_t   sampleIndex    = batchIndex * sequenceLen + sequenceIndex;
                const float_t* input    = inputs[sampleIndex]->data( );
                float_t* historyPrev    = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_HISTORY_PREV, sampleIndex ) );    // H(t-1)

                float_t* dCadidateState = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_CANDIDATE_STATE_DELTA, sampleIndex ) );
                float_t* dInputGate     = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_INPUT_GATE_DELTA, sampleIndex ) );
                float_t* dForgetGate    = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_FORGET_GATE_DELTA, sampleIndex ) );
                float_t* dOutputGate    = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_OUTPUT_GATE_DELTA, sampleIndex ) );

                float_t dInputGateVal     = dInputGate[outputIndex];
                float_t dCadidateStateVal = dCadidateState[outputIndex];
                float_t dForgetGateVal    = dForgetGate[outputIndex];
                float_t dOutputGateVal    = dOutputGate[outputIndex];

                // accumulate gradients for inputs' weights
                for ( size_t inputIndex = 0, weightIndex = weightIndexStartI; inputIndex < mInputsCount; inputIndex++, weightIndex++ )
                {
                    gradWeightsX2F[weightIndex] += dForgetGateVal    * input[inputIndex];
                    gradWeightsX2I[weightIndex] += dInputGateVal     * input[inputIndex];
                    gradWeightsX2Z[weightIndex] += dCadidateStateVal * input[inputIndex];
                    gradWeightsX2O[weightIndex] += dOutputGateVal    * input[inputIndex];
                }

                // accumulate gradients for history weights
                if ( sequenceIndex != 0 )
                {
                    for ( size_t historyIndex = 0, weightIndex = weightIndexStartH; historyIndex < mOutputsCount; historyIndex++, weightIndex++ )
                    {
                        gradWeightsH2F[weightIndex] += dForgetGateVal    * historyPrev[historyIndex];
                        gradWeightsH2I[weightIndex] += dInputGateVal     * historyPrev[historyIndex];
                        gradWeightsH2Z[weightIndex] += dCadidateStateVal * historyPrev[historyIndex];
                        gradWeightsH2O[weightIndex] += dOutputGateVal    * historyPrev[historyIndex];
                    }
                }

                // accumulate gradients for biases
                gradBiasesF[outputIndex] += dForgetGateVal;
                gradBiasesI[outputIndex] += dInputGateVal;
                gradBiasesZ[outputIndex] += dCadidateStateVal;
                gradBiasesO[outputIndex] += dOutputGateVal;
            }
        }
    } );
}

// Applies updates to the layer's weights and biases
void XLSTMLayer::UpdateWeights( const fvector_t& updates )
{
    for ( size_t i = 0, n = mAllWeights.size( ); i < n; i++ )
    {
        mAllWeights[i] += updates[i];
    }
}

// Saves layer's learnt parameters/weights
bool XLSTMLayer::SaveLearnedParams( FILE* file ) const
{
    vector<const fvector_t*> params( { &mAllWeights } );

    return SaveLearnedParamsHelper( file, LayerID::RecurrentLSTM, params );
}

// Loads layer's learnt parameters
bool XLSTMLayer::LoadLearnedParams( FILE* file )
{
    vector<fvector_t*> params( { &mAllWeights } );

    return LoadLearnedParamsHelper( file, LayerID::RecurrentLSTM, params );
}

} } // ANNT::Neuro
