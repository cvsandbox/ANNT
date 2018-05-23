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

#pragma once
#ifndef ANNT_XNEURAL_NETWORK_HPP
#define ANNT_XNEURAL_NETWORK_HPP

#include <memory>
#include <assert.h>

#include "../Layers/ILayer.hpp"

namespace ANNT { namespace Neuro {

// Implementation of neural network class, which is a simple sequence of layers for now.
// No computation/training is done here - it is moved to dedicated classes which manage
// all the resources involved into that.
//
class XNeuralNetwork
{
private:
    std::vector<std::shared_ptr<ILayer>> mLayers;

public:
    typedef std::vector<std::shared_ptr<ILayer>>::iterator       iterator;
    typedef std::vector<std::shared_ptr<ILayer>>::const_iterator const_iterator;

    XNeuralNetwork( )
    {
    }

    // Reports network's inputs count (inputs count of the first layer if any)
    size_t InputsCount( ) const
    {
        return ( mLayers.empty( ) ) ? 0 : mLayers.front( )->InputsCount( );
    }

    // Reports network's outputs count (outputs count of the last layer)
    size_t OutputsCount( ) const
    {
        return ( mLayers.empty( ) ) ? 0 : mLayers.back( )->OutputsCount( );
    }

    // Reports total number of layers
    size_t LayersCount( ) const
    {
        return mLayers.size( );
    }

    // Iterators to access layers
    iterator begin( )
    {
        return mLayers.begin( );
    }
    iterator end( )
    {
        return mLayers.end( );
    }
    const_iterator begin( ) const
    {
        return mLayers.begin( );
    }
    const_iterator end( ) const
    {
        return mLayers.end( );
    }

    // Provides layer at the specified index
    std::shared_ptr<ILayer> LayerAt( size_t index ) const
    {
        return ( index < mLayers.size( ) ) ? mLayers[index] : std::shared_ptr<ILayer>( nullptr );
    }

    // Adds the specified layer to the end of layers' collection
    void AddLayer( const std::shared_ptr<ILayer>& layer )
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
};

} } // namespace ANNT::Neuro

#endif // ANNT_XNEURAL_NETWORK_HPP
