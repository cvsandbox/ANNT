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

#include "XNetworkInference.hpp"
#include "XNetworkContext.hpp"
#include "../../Tools/XDataEncodingTools.hpp"
#include "cstring"

using namespace std;

namespace ANNT { namespace Neuro {

XNetworkInference::XNetworkInference( const shared_ptr<XNeuralNetwork>& network ) :
    mNetwork( network )
{
    mComputeInputs.resize( 1 );

    // prepare output vectors for all layers and for all samples (only one sample here for now)
    for ( auto layer : *mNetwork )
    {
        mComputeOutputsStorage.push_back( vector<fvector_t>( { fvector_t( layer->OutputsCount( ) ) } ) );
        mComputeOutputs.push_back( vector<fvector_t*>( { &( mComputeOutputsStorage.back( )[0] ) } ) );
    }

    AllocateWorkingBuffers( mComputeMemoryBuffers, 1, false );
}

XNetworkInference::~XNetworkInference( )
{
    FreeWorkingBuffers( mComputeMemoryBuffers );
}

// Allocate working buffers for layer needing them
void XNetworkInference::AllocateWorkingBuffers( std::vector<std::vector<std::vector<void*>>>& workingBuffer, size_t batchSize, bool trainingMode )
{
    for ( auto layer : *mNetwork )
    {
        uvector_t workingMemSize = layer->WorkingMemSize( trainingMode );

        // filling this nice vector
        //
        //       -- for each layer
        //       |           -- for each requested buffer
        //       |           |           -- for each sample
        //       |           |           |       -- requested memory buffer
        //       |           |           |       |
        // std::vector<std::vector<std::vector<void*>>>

        workingBuffer.push_back( std::vector<std::vector<void*>>( workingMemSize.size( ) ) );

        for ( size_t i = 0; i < workingMemSize.size( ); i++ )
        {
            workingBuffer.back( )[i] = std::vector<void*>( );

            for ( size_t j = 0; j < batchSize; j++ )
            {
                void* memBuffer = AlignedAlloc( 32, workingMemSize[i] );

                if ( memBuffer )
                {
                    memset( memBuffer, 0, workingMemSize[i] );
                }

                workingBuffer.back( )[i].push_back( memBuffer );
            }
        }
    }
}

// Free layers' working buffers
void XNetworkInference::FreeWorkingBuffers( std::vector<std::vector<std::vector<void*>>>& workingBuffer )
{
    for ( size_t i = 0; i < workingBuffer.size( ); i++ )
    {
        for ( size_t j = 0; j < workingBuffer[i].size( ); j++ )
        {
            for ( size_t k = 0; k < workingBuffer[i][j].size( ); k++ )
            {
                AlignedFree( workingBuffer[i][j][k] );
            }
        }
    }
}

// Computes output vector for the given input vector
void XNetworkInference::Compute( const fvector_t& input, fvector_t& output )
{
    if ( mNetwork->LayersCount( ) != 0 )
    {
        mComputeInputs[0] = const_cast<fvector_t*>( &input );

        DoCompute( mComputeInputs, mComputeOutputs, mComputeMemoryBuffers );

        // copy output produced by the last layer
        output = mComputeOutputsStorage.back( )[0];
    }
}

// Runs classification for the given input - returns index of the maximum element in the corresponding output
size_t XNetworkInference::Classify( const fvector_t& input )
{
    size_t classIndex = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        mComputeInputs[0] = const_cast<fvector_t*>( &input );

        DoCompute( mComputeInputs, mComputeOutputs, mComputeMemoryBuffers );

        classIndex = XDataEncodingTools::MaxIndex( mComputeOutputsStorage.back( )[0] );
    }

    return classIndex;
}

// Tests classification for the provided inputs and target labels - provides number of correctly classified samples
size_t XNetworkInference::TestClassification( const vector<fvector_t>& inputs, const uvector_t& targetLabels )
{
    size_t correctLabelsCounter = 0;

    if ( mNetwork->LayersCount( ) != 0 )
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            mComputeInputs[0] = const_cast<fvector_t*>( &( inputs[i] ) );

            DoCompute( mComputeInputs, mComputeOutputs, mComputeMemoryBuffers );

            if ( XDataEncodingTools::MaxIndex( mComputeOutputsStorage.back( )[0] ) == targetLabels[i] )
            {
                correctLabelsCounter++;
            }
        }
    }

    return correctLabelsCounter;
}

// Helper method to compute output vectors for the given input vectors
void XNetworkInference::DoCompute( const vector<fvector_t*>& inputs,
                                     vector<vector<fvector_t*>>& outputs,
                                     vector<vector<vector<void*>>> workingBuffer,
                                     bool trainingMode )
{
    XNetworkContext ctx( trainingMode );

    ctx.SetWorkingBuffers( &( workingBuffer[0] ) );
    mNetwork->LayerAt( 0 )->ForwardCompute( inputs, outputs[0], ctx );

    for ( size_t i = 1, layersCount = mNetwork->LayersCount( ); i < layersCount; i++ )
    {
        ctx.SetWorkingBuffers( &( workingBuffer[i] ) );
        mNetwork->LayerAt( i )->ForwardCompute( outputs[i - 1], outputs[i], ctx );
    }
}

} } // namespace ANNT::Neuro
