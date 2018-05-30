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

#include "XNetworkComputation.hpp"
#include "../../Tools/XDataEncodingTools.hpp"

using namespace std;

namespace ANNT { namespace Neuro {

XNetworkComputation::XNetworkComputation( const shared_ptr<XNeuralNetwork>& network ) :
    mNetwork( network )
{
    mComputeInputs.resize( 1 );

    // prepare output vectors for all layers and for all samples (only one sample here for now)
    for ( auto layer : *mNetwork )
    {
        mComputeOutputsStorage.push_back( vector<vector_t>( { vector_t( layer->OutputsCount( ) ) } ) );
        mComputeOutputs.push_back( vector<vector_t*>( { &( mComputeOutputsStorage.back( )[0] ) } ) );
    }
}

// Computes output vector for the given input vector
void XNetworkComputation::Compute( const vector_t& input, vector_t& output )
{
    if ( mNetwork->LayersCount( ) != 0 )
    {
        mComputeInputs[0] = const_cast<vector_t*>( &input );

        DoCompute( mComputeInputs, mComputeOutputs );

        // copy output produced by the last layer
        output = mComputeOutputsStorage.back( )[0];
    }
}

// Runs classification for the given input - returns index of the maximum element in the corresponding output
size_t XNetworkComputation::Classify( const vector_t& input )
{
    size_t classIndex = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        mComputeInputs[0] = const_cast<vector_t*>( &input );

        DoCompute( mComputeInputs, mComputeOutputs );

        classIndex = XDataEncodingTools::MaxIndex( mComputeOutputsStorage.back( )[0] );
    }

    return classIndex;
}

// Tests classification for the provided inputs and target labels - provides number of correctly classified samples
size_t XNetworkComputation::TestClassification( const vector<vector_t>& inputs, const vector<size_t>& targetLabels )
{
    size_t correctLabelsCounter = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            mComputeInputs[0] = const_cast<vector_t*>( &( inputs[i] ) );

            DoCompute( mComputeInputs, mComputeOutputs );

            if ( XDataEncodingTools::MaxIndex( mComputeOutputsStorage.back( )[0] ) == targetLabels[i] )
            {
                correctLabelsCounter++;
            }
        }
    }

    return correctLabelsCounter;
}

// Helper method to compute output vectors for the given input vectors
void XNetworkComputation::DoCompute( const vector<vector_t*>& inputs,
                                     vector<vector<vector_t*>>& outputs )
{
    mNetwork->LayerAt( 0 )->ForwardCompute( inputs, outputs[0] );

    for ( size_t i = 1, layersCount = mNetwork->LayersCount( ); i < layersCount; i++ )
    {
        mNetwork->LayerAt( i )->ForwardCompute( outputs[i - 1], outputs[i] );
    }
}

} } // namespace ANNT::Neuro
