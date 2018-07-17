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

#include "XNeuralNetwork.hpp"

using namespace std;

namespace ANNT { namespace Neuro {

// Adds the specified layer to the end of layers' collection
void XNeuralNetwork::AddLayer( const shared_ptr<ILayer>& layer )
{
    if ( layer->InputsCount( ) == 0 )
    {
        if ( mLayers.empty( ) )
        {
            // TODO: error
            assert( false );
        }
        else
        {
            size_t lastOutSize = mLayers.back( )->OutputsCount( );

            layer->Initialize( lastOutSize, lastOutSize );
        }
    }

    if ( ( mLayers.empty( ) ) || ( layer->InputsCount( ) == mLayers.back( )->OutputsCount( ) ) )
    {
        mLayers.push_back( layer );
    }
    else
    {
        // TODO: error
        assert( false );
    }
}

// Saves network's learned parameters only.
// Network structure is not saved and so same network must be constructed before loading parameters
bool XNeuralNetwork::SaveLearnedParams( const string& fileName ) const
{
    FILE* file = fopen( fileName.c_str( ), "wb" );
    bool  ret = false;

    if ( file != nullptr )
    {
        // float_t can be defined to other than "float", so need to save its size and stop loading
        // saves produced by incompatible builds
        uint8_t floatTypeSize = static_cast<uint8_t>( sizeof( float_t ) );

        if ( ( fwrite( "ANNT", sizeof( char ), 4, file ) == 4 ) &&
            ( fwrite( &floatTypeSize, sizeof( floatTypeSize ), 1, file ) == 1 ) )
        {
            ret = true;

            for ( const_iterator layersIt = mLayers.begin( ); ( ret ) && ( layersIt != mLayers.end( ) ); layersIt++ )
            {
                ret = ( *layersIt )->SaveLearnedParams( file );
            }
        }

        fclose( file );
    }

    return ret;
}

// Loads network's learned parameters.
// A network of the same structure as saved must be created first, since this method loads only parameters/weights/biases.
bool XNeuralNetwork::LoadLearnedParams( const string& fileName )
{
    FILE* file = fopen( fileName.c_str( ), "rb" );
    bool  ret = false;

    if ( file != nullptr )
    {
        char    anntMagic[4];
        uint8_t floatTypeSize;

        if ( ( fread( anntMagic, sizeof( char ), 4, file ) == 4 ) &&
            ( fread( &floatTypeSize, sizeof( floatTypeSize ), 1, file ) == 1 ) &&
            ( anntMagic[0] == 'A' ) && ( anntMagic[1] == 'N' ) && ( anntMagic[2] == 'N' ) && ( anntMagic[3] == 'T' ) &&
            ( floatTypeSize == static_cast<uint8_t>( sizeof( float_t ) ) ) )
        {
            ret = true;

            for ( const_iterator layersIt = mLayers.begin( ); ( ret ) && ( layersIt != mLayers.end( ) ); layersIt++ )
            {
                ret = ( *layersIt )->LoadLearnedParams( file );
            }
        }

        fclose( file );
    }

    return ret;
}

} } // namespace ANNT::Neuro
