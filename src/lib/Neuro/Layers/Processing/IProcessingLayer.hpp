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
#ifndef ANNT_IPROCESSING_LAYER_HPP
#define ANNT_IPROCESSING_LAYER_HPP

#include "../ILayer.hpp"

namespace ANNT { namespace Neuro {

class IProcessingLayer : public ILayer
{
public:
    IProcessingLayer( size_t inputsCount, size_t outputsCount ) :
        ILayer( inputsCount, outputsCount )
    {

    }

    // None of the processing layers have weights/biases to train
    bool Trainable( ) const override
    {
        return false;
    }

    // Calls BackwardProcess() to propagate error to the previous layer 
    void BackwardCompute( const std::vector<fvector_t*>& inputs,
                          const std::vector<fvector_t*>& outputs,
                          const std::vector<fvector_t*>& deltas,
                          std::vector<fvector_t*>& prevDeltas,
                          fvector_t& /* gradWeights */,
                          const XNetworkContext& ctx ) override
    {
        BackwardProcess( inputs, outputs, deltas, prevDeltas, ctx );
    }

    // Propagates error to the previous layer
    virtual void BackwardProcess( const std::vector<fvector_t*>& inputs,
                                  const std::vector<fvector_t*>& outputs,
                                  const std::vector<fvector_t*>& deltas,
                                  std::vector<fvector_t*>& prevDeltas,
                                  const XNetworkContext& ctx ) = 0;
};

} } // namespace ANNT::Neuro

#endif // ANNT_IPROCESSING_LAYER_HPP

