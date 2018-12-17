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

#include <algorithm>
#include <functional>

#include "XNetworkTraining.hpp"
#include "XNetworkContext.hpp"
#include "../Layers/ITrainableLayer.hpp"
#include "../../Tools/XDataEncodingTools.hpp"

using namespace std;

namespace ANNT { namespace Neuro { namespace Training {

XNetworkTraining::XNetworkTraining( const shared_ptr<XNeuralNetwork>& network,
                                    const shared_ptr<INetworkOptimizer>& optimizer,
                                    const shared_ptr<ICostFunction>& costFunction ) :
    XNetworkInference( network ),
    mOptimizer( optimizer ),
    mCostFunction( costFunction ),
    mAverageWeightGradients( true ),
    mTrainingContext( true, 1 )
{
    size_t optimizerParameterVariablesCount = mOptimizer->ParameterVariablesCount( );
    size_t optimizerLayerVariablesCount     = mOptimizer->LayerVariablesCount( );

    // allocate everything, which does not depend on batch size (number of input/output samples),
    // but only depends on layers count
    // 1) weight and bias gradients (accumulated over samples during batch);
    // 2) optimizer's variables;

    for ( auto layer : *mNetwork )
    {
        size_t weightsCount = 0;

        // allocate weight and bias gradients for trainable layers
        // (for each layer, but not for each sample, since those accumulated over samples)
        if ( layer->Trainable( ) )
        {
            weightsCount = static_pointer_cast<ITrainableLayer>( layer )->WeightsCount( );
        }

        mGradWeights.push_back( fvector_t( weightsCount ) );

        // optimizer's variables ...
        mOptimizerParameterVariables.push_back( vector<fvector_t>( optimizerParameterVariablesCount ) );
        mOptimizerLayerVariables.push_back( fvector_t( optimizerLayerVariablesCount ) );

        for ( size_t i = 0; i < optimizerParameterVariablesCount; i++ )
        {
            mOptimizerParameterVariables.back( )[i] = fvector_t( weightsCount );
        }
    }
}

// Allocate the rest of vectors required for training - those which depend on the batch size
void XNetworkTraining::AllocateTrainVectors( size_t samplesCount )
{
    size_t layersCount = mNetwork->LayersCount( );

    if ( mTrainInputs.size( ) != samplesCount )
    {
        mTrainInputs.resize( samplesCount );
        mTargetOuputs.resize( samplesCount );

        mTrainOutputsStorage.resize( layersCount );
        mTrainOutputs.resize( layersCount );

        mDeltasStorage.resize( layersCount );
        mDeltas.resize( layersCount );

        // prepare output vector and deltas for all samples and for all layers
        for ( size_t layerIndex = 0; layerIndex < layersCount; layerIndex++ )
        {
            size_t layerOutputCount = mNetwork->LayerAt( layerIndex )->OutputsCount( );

            mTrainOutputsStorage[layerIndex].resize( samplesCount );
            mTrainOutputs[layerIndex].resize( samplesCount );

            mDeltasStorage[layerIndex].resize( samplesCount );
            mDeltas[layerIndex].resize( samplesCount );

            for ( size_t i = 0; i < samplesCount; i++ )
            {
                mTrainOutputsStorage[layerIndex][i] = fvector_t( layerOutputCount );
                mTrainOutputs[layerIndex][i]        = &( mTrainOutputsStorage[layerIndex][i] );

                mDeltasStorage[layerIndex][i] = fvector_t( layerOutputCount );
                mDeltas[layerIndex][i]        = &( mDeltasStorage[layerIndex][i] );
            }
        }

        // to make calculations consistant, we have deltas for inputs as well ("previous" layer of the first)
        mInputDeltasStorage.resize( samplesCount );
        mInputDeltas.resize( samplesCount );

        for ( size_t i = 0; i < samplesCount; i++ )
        {
            mInputDeltasStorage[i] = fvector_t( mNetwork->InputsCount( ) );
            mInputDeltas[i] = &( mInputDeltasStorage[i] );
        }

        // allocate new buffers for layers
        mTrainingContext.AllocateWorkingBuffers( mNetwork, samplesCount );
    }
}

// Calculate error of the last layer for each training sample
float_t XNetworkTraining::CalculateError( )
{
    vector<fvector_t>& lastOutputs = mTrainOutputsStorage.back( );
    vector<fvector_t>& lastDeltas  = mDeltasStorage.back( );
    float_t            totalCost   = 0;

    for ( size_t i = 0, n = mTrainInputs.size( ); i < n; i++ )
    {
        fvector_t& lastDelta    = lastDeltas[i];
        fvector_t& lastOutput   = lastOutputs[i];
        fvector_t& targetOutput = *mTargetOuputs[i];

        totalCost += mCostFunction->Cost( lastOutput, targetOutput );
        lastDelta  = mCostFunction->Gradient( lastOutput, targetOutput );
    }

    totalCost /= mTrainInputs.size( );

    return totalCost;
}

// Propagate error through the network starting from last layer
void XNetworkTraining::DoBackwardCompute( )
{
    size_t  layerIndex  = mNetwork->LayersCount( ) - 1;
    
    // propagate deltas for all layers except the first one
    for ( ; layerIndex > 0; layerIndex-- )
    {
        mTrainingContext.SetCurrentLayerIndex( layerIndex );

        mNetwork->LayerAt( layerIndex )->
            BackwardCompute( mTrainOutputs[layerIndex - 1], mTrainOutputs[layerIndex],
                             mDeltas[layerIndex], mDeltas[layerIndex - 1],
                             mGradWeights[layerIndex], mTrainingContext );
    }

    // now same for the first layer
    mTrainingContext.SetCurrentLayerIndex( 0 );

    mNetwork->LayerAt( 0 )->
        BackwardCompute( mTrainInputs, mTrainOutputs[0],
                         mDeltas[0], mInputDeltas,
                         mGradWeights[0], mTrainingContext );
}

// Calculate weights/biases updates from gradients and apply them
void XNetworkTraining::UpdateWeights( )
{
    auto    itLayers          = mNetwork->begin( );
    float_t batchUpdateFactor = float_t( 1 );
    
    if ( mAverageWeightGradients )
    {
        batchUpdateFactor /= mTrainInputs.size( );
    }

    for ( size_t i = 0, n = mNetwork->LayersCount( ); i < n; i++, ++itLayers )
    {
        if ( (*itLayers)->Trainable( ) )
        {
            if ( mAverageWeightGradients )
            {
                std::transform( mGradWeights[i].begin( ), mGradWeights[i].end( ), mGradWeights[i].begin( ),
                                [&]( float_t v ) -> float_t { return v * batchUpdateFactor; } );
            }

            mOptimizer->CalculateUpdatesFromGradients( mGradWeights[i], mOptimizerParameterVariables[i], mOptimizerLayerVariables[i] );

            static_pointer_cast<ITrainableLayer>( *itLayers )->UpdateWeights( mGradWeights[i] );

            // reset gradients for the next training cycle
            fill( mGradWeights[i].begin( ), mGradWeights[i].end( ), float_t( 0 ) );
        }
    }
}

// Run single training cycle
float_t XNetworkTraining::RunTraining( )
{
    float_t cost;

    // 1 - compute the network to get the actual output
    DoCompute( mTrainInputs, mTrainOutputs, mTrainingContext );

    // 2 - get error of the last layer
    cost = CalculateError( );

    // 3 - propagate the error backward through the network
    DoBackwardCompute( );

    // 4 - calculate weights/bias updates and apply those
    UpdateWeights( );

    return cost;
}

// Trains single input/output sample
float_t XNetworkTraining::TrainSample( const fvector_t& input, const fvector_t& targetOutput )
{
    float_t cost = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        AllocateTrainVectors( 1 );

        // get the single input/output into usable form
        mTrainInputs[0]  = const_cast<fvector_t*>( &input );
        mTargetOuputs[0] = const_cast<fvector_t*>( &targetOutput );

        cost = RunTraining( );
    }

    return cost;
}

// Trains single batch of prepared samples (as vectors)
float_t XNetworkTraining::TrainBatch( const vector<fvector_t>& inputs,
                                      const vector<fvector_t>& targetOutputs )
{
    float_t cost = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        AllocateTrainVectors( inputs.size( ) );

        // prepare inputs vectors and target ouputs
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            mTrainInputs[i]  = const_cast<fvector_t*>( &( inputs[i] ) );
            mTargetOuputs[i] = const_cast<fvector_t*>( &( targetOutputs[i] ) );
        }

        cost = RunTraining( );
    }

    return cost;
}

// Trains single batch of prepared samples (as pointers to vectors)
float_t XNetworkTraining::TrainBatch( const vector<fvector_t*>& inputs,
                                      const vector<fvector_t*>& targetOutputs )
{
    float_t cost = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        AllocateTrainVectors( inputs.size( ) );

        // prepare inputs vectors and target ouputs
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            mTrainInputs[i]  = inputs[i];
            mTargetOuputs[i] = targetOutputs[i];
        }

        cost = RunTraining( );
    }

    return cost;
}

// Trains single epoch using batches of the specified size (samples are provided as vectors)
float_t XNetworkTraining::TrainEpoch( const vector<fvector_t>& inputs,
                                      const vector<fvector_t>& targetOutputs,
                                      size_t batchSize, bool randomPickIntoBatch )
{
    // It is not an average cost for all samples after completion of an epoch, since that
    // requires re-testing all samples. Instead it is an average cost over all batches.
    // However, at the end of each batch the model is updated and so may improve.
    float_t averageRunningCost = 0;
    size_t  samplesCount       = inputs.size( );

    if ( samplesCount != 0 )
    {
        AllocateTrainVectors( batchSize );

        if ( ( inputs.size( ) == batchSize ) && ( !randomPickIntoBatch ) )
        {
            averageRunningCost = TrainBatch( inputs, targetOutputs );
        }
        else
        {
            size_t iterations = ( inputs.size( ) - 1 ) / batchSize + 1;
            
            for ( size_t i = 0; i < iterations; i++ )
            {
                // prepare inputs vectors and target ouputs
                for ( size_t j = 0; j < batchSize; j++ )
                {
                    size_t sampleIndex;

                    if ( !randomPickIntoBatch )
                    {
                        sampleIndex = ( i * batchSize + j ) % samplesCount;
                    }
                    else
                    {
                        sampleIndex = rand( ) % samplesCount;
                    }

                    mTrainInputs[j]  = const_cast<fvector_t*>( &( inputs[sampleIndex] ) );
                    mTargetOuputs[j] = const_cast<fvector_t*>( &( targetOutputs[sampleIndex] ) );
                }

                averageRunningCost += RunTraining( );
            }

            averageRunningCost /= iterations;
        }
    }

    return averageRunningCost;
}

// Trains single epoch using batches of the specified size (samples are provided as pointers to vectors)
float_t XNetworkTraining::TrainEpoch( const vector<fvector_t*>& inputs,
                                      const vector<fvector_t*>& targetOutputs,
                                      size_t batchSize, bool randomPickIntoBatch )
{
    float_t averageRunningCost = 0;
    size_t  samplesCount       = inputs.size( );

    if ( samplesCount != 0 )
    {
        AllocateTrainVectors( batchSize );

        if ( ( inputs.size( ) == batchSize ) && ( !randomPickIntoBatch ) )
        {
            averageRunningCost = TrainBatch( inputs, targetOutputs );
        }
        else
        {
            size_t iterations = ( inputs.size( ) - 1 ) / batchSize + 1;
            
            for ( size_t i = 0; i < iterations; i++ )
            {
                // prepare inputs vectors and target ouputs
                for ( size_t j = 0; j < batchSize; j++ )
                {
                    size_t sampleIndex;

                    if ( !randomPickIntoBatch )
                    {
                        sampleIndex = ( i * batchSize + j ) % samplesCount;
                    }
                    else
                    {
                        sampleIndex = rand( ) % samplesCount;
                    }

                    mTrainInputs[j]  = const_cast<fvector_t*>( inputs[sampleIndex] );
                    mTargetOuputs[j] = const_cast<fvector_t*>( targetOutputs[sampleIndex] );
                }

                averageRunningCost += RunTraining( );
            }

            averageRunningCost /= iterations;
        }
    }

    return averageRunningCost;
}


// Tests sample - calculates real output and provides error cost
float_t XNetworkTraining::TestSample( const fvector_t& input,
                                      const fvector_t& targetOutput,
                                      fvector_t& output )
{
    float_t cost = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        // compute the network to get the actual output
        mComputeInputs[0] = const_cast<fvector_t*>( &input );
        DoCompute( mComputeInputs, mComputeOutputs, mInferenceContext );

        output = mComputeOutputsStorage.back( )[0];
        cost   = mCostFunction->Cost( output, targetOutput );
    }

    return cost;
}

// Tests classification for the provided inputs and target labels - provides number of correctly classified
// samples and average cost (target outputs are used)
size_t XNetworkTraining::TestClassification( const std::vector<fvector_t>& inputs, const uvector_t& targetLabels,
                                             const std::vector<fvector_t>& targetOutputs, float_t* pAvgCost )
{
    size_t  correctLabelsCounter = 0;
    float_t cost = 0;

    if ( ( mNetwork->LayersCount( ) != 0 ) && ( inputs.size( ) != 0 ) )
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            mComputeInputs[0] = const_cast<fvector_t*>( &( inputs[i] ) );
            DoCompute( mComputeInputs, mComputeOutputs, mInferenceContext );

            cost += mCostFunction->Cost( mComputeOutputsStorage.back( )[0], targetOutputs[i] );

            if ( XDataEncodingTools::MaxIndex( mComputeOutputsStorage.back( )[0] ) == targetLabels[i] )
            {
                correctLabelsCounter++;
            }
        }

        cost /= inputs.size( );
    }

    if ( pAvgCost )
    {
        *pAvgCost = cost;
    }

    return correctLabelsCounter;
}

size_t XNetworkTraining::TestClassification( const std::vector<fvector_t*>& inputs, const uvector_t& targetLabels,
                                             const std::vector<fvector_t*>& targetOutputs, float_t* pAvgCost )
{
    size_t  correctLabelsCounter = 0;
    float_t cost = 0;

    if ( ( mNetwork->LayersCount( ) != 0 ) && ( inputs.size( ) != 0 ) )
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            mComputeInputs[0] = inputs[i];
            DoCompute( mComputeInputs, mComputeOutputs, mInferenceContext );

            cost += mCostFunction->Cost( mComputeOutputsStorage.back( )[0], *( targetOutputs[i] ) );

            if ( XDataEncodingTools::MaxIndex( mComputeOutputsStorage.back( )[0] ) == targetLabels[i] )
            {
                correctLabelsCounter++;
            }
        }

        cost /= inputs.size( );
    }

    if ( pAvgCost )
    {
        *pAvgCost = cost;
    }

    return correctLabelsCounter;
}

} } } // namespace ANNT::Neuro::Training
