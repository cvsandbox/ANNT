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

namespace ANNT { namespace Neuro {

// Implementation of artificial neural network inference - wraps
// everything necessary to compute network's outputs for a given inputs.
//
class XNetworkComputation
{
protected:
    std::shared_ptr<XNeuralNetwork>     mNetwork;

protected:
    std::vector<std::vector<vector_t>>  mComputeOutputsStorage;
    std::vector<std::vector<vector_t*>> mComputeOutputs;
    std::vector<vector_t*>              mComputeInputs;

public:
    // The passed network must be fully constructed at this point - no adding new layers
    XNetworkComputation( const std::shared_ptr<XNeuralNetwork>& network );
    virtual ~XNetworkComputation( ) { }

    // Computes output vector for the given input vector
    void Compute( const vector_t& input, vector_t& output );

protected:

    // Helper method to compute output vectors for the given input vectors using
    // the provided storage for the intermediate outputs of all layers
    void DoCompute( const std::vector<vector_t*>& inputs,
                    std::vector<std::vector<vector_t*>>& outputs );
};

} } // namespace ANNT::Neuro

#endif // ANNT_XNETWORK_COMPUTATION_HPP
