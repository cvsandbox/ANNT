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

#include "XNetworkContext.hpp"
#include "XNeuralNetwork.hpp"
#include "cstring"

using namespace std;

namespace ANNT { namespace Neuro {

XNetworkContext::XNetworkContext( bool trainingMode, size_t sequenceLength ) :
    mTrainingMode( trainingMode ), mTrainingSequenceLength( sequenceLength ), mCurrentLayer( 0 )
{
}

XNetworkContext::~XNetworkContext( )
{
    FreeWorkingBuffers( );
}

// Allocate working buffer for laters of the network
void XNetworkContext::AllocateWorkingBuffers( const std::shared_ptr<XNeuralNetwork>& net, size_t batchSize )
{
    FreeWorkingBuffers( );

    for ( auto layer : *net )
    {
        uvector_t workingMemSize = layer->WorkingMemSize( mTrainingMode );

        // filling this nice vector
        //
        //       -- for each layer
        //       |           -- for each requested buffer
        //       |           |           -- for each sample
        //       |           |           |       -- requested memory buffer
        //       |           |           |       |
        // std::vector<std::vector<std::vector<void*>>>

        mLayersMemorySize.push_back( workingMemSize );
        mLayersMemoryBuffers.push_back( vector<vector<void*>>( workingMemSize.size( ) ) );

        for ( size_t i = 0; i < workingMemSize.size( ); i++ )
        {
            mLayersMemoryBuffers.back( )[i] = vector<void*>( );

            for ( size_t j = 0; j < batchSize; j++ )
            {
                void* memBuffer = AlignedAlloc( 32, workingMemSize[i] );

                if ( memBuffer )
                {
                    memset( memBuffer, 0, workingMemSize[i] );
                }

                mLayersMemoryBuffers.back( )[i].push_back( memBuffer );
            }
        }
    }
}

// Free layers' working buffers
void XNetworkContext::FreeWorkingBuffers( )
{
    for ( size_t i = 0; i < mLayersMemoryBuffers.size( ); i++ )
    {
        for ( size_t j = 0; j < mLayersMemoryBuffers[i].size( ); j++ )
        {
            for ( size_t k = 0; k < mLayersMemoryBuffers[i][j].size( ); k++ )
            {
                AlignedFree( mLayersMemoryBuffers[i][j][k] );
            }
        }
    }

    mLayersMemoryBuffers.clear( );
    mLayersMemorySize.clear( );
}

// Clear layers' working buffers (memset zero) 
void XNetworkContext::ResetWorkingBuffers( )
{
    for ( size_t i = 0; i < mLayersMemoryBuffers.size( ); i++ )
    {
        for ( size_t j = 0; j < mLayersMemoryBuffers[i].size( ); j++ )
        {
            for ( size_t k = 0; k < mLayersMemoryBuffers[i][j].size( ); k++ )
            {
                memset( mLayersMemoryBuffers[i][j][k], 0, mLayersMemorySize[i][j] );
            }
        }
    }
}
void XNetworkContext::ResetWorkingBuffers( uvector_t layersIndexes )
{
    for ( size_t i : layersIndexes )
    {
        for ( size_t j = 0; j < mLayersMemoryBuffers[i].size( ); j++ )
        {
            for ( size_t k = 0; k < mLayersMemoryBuffers[i][j].size( ); k++ )
            {
                memset( mLayersMemoryBuffers[i][j][k], 0, mLayersMemorySize[i][j] );
            }
        }
    }
}

} } // namespace ANNT::Neuro
