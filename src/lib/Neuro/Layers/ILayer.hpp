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
#ifndef ANNT_ILAYER_HPP
#define ANNT_ILAYER_HPP

#include "../../Types/Types.hpp"
#include "../Network/XNetworkContext.hpp"
#include "LayerIDs.hpp"

namespace ANNT { namespace Neuro {

class XNeuralNetwork;

class ILayer
{
    friend class XNeuralNetwork;

protected:
    size_t mInputsCount;
    size_t mOutputsCount;

    // To be called by XNeuralNetwork to set size of layers with zero inputs/outputs,
    // which is the case for activation layers.
    virtual void Initialize( size_t inputsCount, size_t outputsCount )
    {
        mInputsCount  = inputsCount;
        mOutputsCount = outputsCount;
    }

public:
    ILayer( size_t inputsCount, size_t outputsCount )
    {
        Initialize( inputsCount, outputsCount );
    }

    virtual ~ILayer( ) { }

    // Size of input vector expected by the layer
    size_t InputsCount( ) const
    {
        return mInputsCount;
    }

    // Size of output vector produced by the layer
    size_t OutputsCount( ) const
    {
        return mOutputsCount;
    }

    // Some of the layers may need extra memory required for processing inputs or for
    // keeping state between forward and backward pass. The method below tells
    // how many buffers are required and their size.
    //
    // For example, if a layer needs two temporary buffers of type *float_t*
    // (one vector with *inputsCount* elements and another with *outputsCount* elements),
    // it may return something like this: uvector_t( { inputsCount * sizeof( float ), outputsCount * sizeof( float ) } ).
    //
    // Each individual memory buffer is 32 byte aligned, so AVX friendly.
    //
    // Number of allocated buffers equals to number of samples in a batch.
    // 
    virtual uvector_t WorkingMemSize( bool /* trainingMode */ ) const { return uvector_t( 0 ); }

    // Reports if the layer is trainable or not (has weights/biases)
    virtual bool Trainable( ) const = 0;

    // Calculates outputs for the given inputs - forward pass
    virtual void ForwardCompute( const std::vector<fvector_t*>& inputs,
                                 std::vector<fvector_t*>& outputs,
                                 const XNetworkContext& ctx ) = 0;

    // Propagates error to the previous layer and calculates weights/biases
    // gradients (in the case the layer is trainable)
    virtual void BackwardCompute( const std::vector<fvector_t*>& inputs,
                                  const std::vector<fvector_t*>& outputs,
                                  const std::vector<fvector_t*>& deltas,
                                  std::vector<fvector_t*>& prevDeltas,
                                  fvector_t& gradWeights,
                                  fvector_t& gradBiases,
                                  const XNetworkContext& ctx ) = 0;

    // Saves layer's learnt parameters/weights
    virtual bool SaveLearnedParams( FILE* /* file */ ) const { return true; }
    // Loads layer's learnt parameters
    virtual bool LoadLearnedParams( FILE* /* file */ ) { return true; }
};

} } // namespace ANNT::Neuro

#endif // ANNT_ILAYER_HPP
