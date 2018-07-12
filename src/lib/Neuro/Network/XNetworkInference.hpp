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
#ifndef ANNT_XNETWORK_COMPUTATION_HPP
#define ANNT_XNETWORK_COMPUTATION_HPP

#include <memory>
#include <vector>

#include "XNeuralNetwork.hpp"
#include "XNetworkContext.hpp"

namespace ANNT { namespace Neuro {

// Implementation of artificial neural network inference - wraps
// everything necessary to compute network's outputs for a given inputs.
//
class XNetworkInference
{
protected:
    std::shared_ptr<XNeuralNetwork>      mNetwork;

protected:
    std::vector<std::vector<fvector_t>>  mComputeOutputsStorage;
    std::vector<std::vector<fvector_t*>> mComputeOutputs;
    std::vector<fvector_t*>              mComputeInputs;

    XNetworkContext                      mInferenceContext;

public:
    // The passed network must be fully constructed at this point - no adding new layers
    XNetworkInference( const std::shared_ptr<XNeuralNetwork>& network );

    // Reset working buffers for all layers
    virtual void ResetState( )
    {
        mInferenceContext.ResetWorkingBuffers( );
    }

    // Reset working buffers for the specified layers
    virtual void ResetLayersState( uvector_t layersIndexes )
    {
        mInferenceContext.ResetWorkingBuffers( layersIndexes );
    }

    // Computes output vector for the given input vector
    void Compute( const fvector_t& input, fvector_t& output );

    // Runs classification for the given input - returns index of the maximum
    // element in the corresponding output vector
    size_t Classify( const fvector_t& input );

    // Tests classification for the provided inputs and target labels -
    // provides number of correctly classified samples
    size_t TestClassification( const std::vector<fvector_t>& inputs,
                               const uvector_t& targetLabels );

protected:

    // Helper method to compute output vectors for the given input vectors using
    // the provided storage for the intermediate outputs of all layers
    void DoCompute( const std::vector<fvector_t*>& inputs,
                    std::vector<std::vector<fvector_t*>>& outputs,
                    XNetworkContext& ctx );
};

} } // namespace ANNT::Neuro

#endif // ANNT_XNETWORK_COMPUTATION_HPP
