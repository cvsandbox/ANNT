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
#include "../Layers/ITrainableLayer.hpp"

using namespace std;

namespace ANNT { namespace Neuro { namespace Training {

XNetworkTraining::XNetworkTraining( const shared_ptr<XNeuralNetwork>& network,
                                    const shared_ptr<INetworkOptimizer>& optimizer,
                                    const shared_ptr<ICostFunction>& costFunction ) :
    XNetworkComputation( network ),
    mOptimizer( optimizer ),
    mCostFunction( costFunction ),
    mAverageWeightGradients( true )
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
        size_t biasesCount  = 0;

        // allocate weight and bias gradients for trainable layers
        // (for each layer, but not for each sample, since those accumulated over samples)
        if ( layer->Trainable( ) )
        {
            shared_ptr<ITrainableLayer> trainableLayer = static_pointer_cast<ITrainableLayer>( layer );

            weightsCount = trainableLayer->WeightsCount( );
            biasesCount  = trainableLayer->BiasesCount( );
        }

        mGradWeights.push_back( vector_t( weightsCount ) );
        mGradBiases.push_back( vector_t( biasesCount ) );

        // optimizer's variables ...
        mWeightsParameterVariables.push_back( vector<vector_t>( optimizerParameterVariablesCount ) );
        mBiasesParameterVariables.push_back( vector<vector_t>( optimizerParameterVariablesCount ) );

        mWeightsLayerVariables.push_back( vector_t( optimizerLayerVariablesCount ) );
        mBiasesLayerVariables.push_back( vector_t( optimizerLayerVariablesCount ) );

        for ( size_t i = 0; i < optimizerParameterVariablesCount; i++ )
        {
            mWeightsParameterVariables.back( )[i] = vector_t( weightsCount );
            mBiasesParameterVariables.back( )[i]  = vector_t( biasesCount );
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
                mTrainOutputsStorage[layerIndex][i] = vector_t( layerOutputCount );
                mTrainOutputs[layerIndex][i]        = &( mTrainOutputsStorage[layerIndex][i] );

                mDeltasStorage[layerIndex][i] = vector_t( layerOutputCount );
                mDeltas[layerIndex][i]        = &( mDeltasStorage[layerIndex][i] );
            }
        }

        // to make calculations consistant, we have deltas for inputs as well ("previous" layer of the first)
        mInputDeltasStorage.resize( samplesCount );
        mInputDeltas.resize( samplesCount );

        for ( size_t i = 0; i < samplesCount; i++ )
        {
            mInputDeltasStorage[i] = vector_t( mNetwork->InputsCount( ) );
            mInputDeltas[i] = &( mInputDeltasStorage[i] );
        }
    }
}

// Calculate error of the last layer for each training sample
float_t XNetworkTraining::CalculateError( )
{
    vector<vector_t>& lastOutputs = mTrainOutputsStorage.back( );
    vector<vector_t>& lastDeltas  = mDeltasStorage.back( );
    float_t           cost        = 0;

    for ( size_t i = 0, n = mTrainInputs.size( ); i < n; i++ )
    {
        vector_t& lastDelta    = lastDeltas[i];
        vector_t& lastOutput   = lastOutputs[i];
        vector_t& targetOutput = *mTargetOuputs[i];

        cost     += mCostFunction->Cost( lastOutput, targetOutput );
        lastDelta = mCostFunction->Gradient( lastOutput, targetOutput );
    }

    cost /= mTrainInputs.size( );

    return cost;
}

// Propagate error through the network starting from last layer
void XNetworkTraining::DoBackwardCompute( )
{
    size_t layerIndex  = mNetwork->LayersCount( ) - 1;
    
    // propagate deltas for all layers except the first one
    for ( ; layerIndex > 0; layerIndex-- )
    {
        mNetwork->LayerAt( layerIndex )->
            BackwardCompute( mTrainOutputs[layerIndex - 1], mTrainOutputs[layerIndex],
                             mDeltas[layerIndex], mDeltas[layerIndex - 1],
                             mGradWeights[layerIndex], mGradBiases[layerIndex] );
    }

    // now same for the first layer
    mNetwork->LayerAt( 0 )->
        BackwardCompute( mTrainInputs, mTrainOutputs[0],
                         mDeltas[0], mInputDeltas,
                         mGradWeights[0], mGradBiases[0] );
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
                                std::bind1st( std::multiplies<float_t>( ), batchUpdateFactor ) );
                std::transform( mGradBiases[i].begin( ), mGradBiases[i].end( ), mGradBiases[i].begin( ),
                                std::bind1st( std::multiplies<float_t>( ), batchUpdateFactor ) );
            }

            mOptimizer->CalculateUpdatesFromGradients( mGradWeights[i], mWeightsParameterVariables[i], mWeightsLayerVariables[i] );
            mOptimizer->CalculateUpdatesFromGradients( mGradBiases [i], mBiasesParameterVariables [i], mBiasesLayerVariables [i] );

            static_pointer_cast<ITrainableLayer>( *itLayers )->UpdateWeights( mGradWeights[i], mGradBiases[i] );

            // reset gradients for the next training cycle
            fill( mGradWeights[i].begin( ), mGradWeights[i].end( ), float_t( 0 ) );
            fill( mGradBiases[i].begin( ),  mGradBiases[i].end( ),  float_t( 0 ) );
        }
    }
}

// Run single training cycle
float_t XNetworkTraining::RunTraining( )
{
    float_t cost;

    // 1 - compute the network to get the actual output
    DoCompute( mTrainInputs, mTrainOutputs );

    // 2 - get error of the last layer
    cost = CalculateError( );

    // 3 - propagate the error backward through the network
    DoBackwardCompute( );

    // 4 - calculate weights/bias updates and apply those
    UpdateWeights( );

    return cost;
}

// Train single input/output sample
float_t XNetworkTraining::TrainSample( const vector_t& input, const vector_t& targetOutput )
{
    float_t cost = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        AllocateTrainVectors( 1 );

        // get the single input/output into usable form
        mTrainInputs[0]  = const_cast<vector_t*>( &input );
        mTargetOuputs[0] = const_cast<vector_t*>( &targetOutput );

        cost = RunTraining( );
    }

    return cost;
}

// Train single batch of prepared samples (as vectors)
float_t XNetworkTraining::TrainBatch( const vector<vector_t>& inputs,
                                      const vector<vector_t>& targetOutputs )
{
    float_t cost = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        AllocateTrainVectors( inputs.size( ) );

        // prepare inputs vectors and target ouputs
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            mTrainInputs[i]  = const_cast<vector_t*>( &( inputs[i] ) );
            mTargetOuputs[i] = const_cast<vector_t*>( &( targetOutputs[i] ) );
        }

        cost = RunTraining( );
    }

    return cost;
}

// Train single batch of prepared samples (as pointers to vectors)
float_t XNetworkTraining::TrainBatch( const std::vector<vector_t*>& inputs,
                                      const std::vector<vector_t*>& targetOutputs )
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

// Test sample - calculate real output and provide error cost
float_t XNetworkTraining::TestSample( const vector_t& input,
                                      const vector_t& targetOutput,
                                      vector_t& output )
{
    float_t cost = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        // compute the network to get the actual output
        mComputeInputs[0] = const_cast<vector_t*>( &input );
        DoCompute( mComputeInputs, mComputeOutputs );

        output = mComputeOutputsStorage.back( )[0];
        cost   = mCostFunction->Cost( output, targetOutput );
    }

    return cost;
}

} } } // namespace ANNT::Neuro::Training
