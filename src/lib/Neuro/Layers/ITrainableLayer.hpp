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
#ifndef ANNT_ITRAINABLE_LAYER_HPP
#define ANNT_ITRAINABLE_LAYER_HPP

#include "ILayer.hpp"

namespace ANNT { namespace Neuro {

class ITrainableLayer : public ILayer
{
public:
    ITrainableLayer( size_t inputsCount, size_t outputsCount ) :
        ILayer( inputsCount, outputsCount )
    {
    }

    // Reports the layer is trainable
    bool Trainable( ) const override
    {
        return true;
    }

    // Reports number of weight coefficients the layer has
    virtual size_t WeightsCount( ) const = 0;

    // Get/set layer's weights
    virtual fvector_t Weights( ) const = 0;
    virtual void SetWeights( const fvector_t& weights ) = 0;

    // Randomizes layer's weights/biases
    virtual void Randomize( ) = 0;

    // Applies updates to the layer's weights and biases
    virtual void UpdateWeights( const fvector_t& updates ) = 0;
};

} } // namespace ANNT::Neuro

#endif // ANNT_ITRAINABLE_LAYER_HPP
