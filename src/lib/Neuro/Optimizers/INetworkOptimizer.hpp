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
#ifndef ANNT_INETWORK_OPTIMIZER_HPP
#define ANNT_INETWORK_OPTIMIZER_HPP

#include "../../Types/Types.hpp"

namespace ANNT { namespace Neuro { namespace Training {

// Common interface for algorithms calculating updates for weights/biases from their gradients
class INetworkOptimizer
{
protected:
    float mLearningRate;

public:
    INetworkOptimizer( float_t learningRate ) :
        mLearningRate( learningRate )
    {
    }

    virtual ~INetworkOptimizer( ) { }

    // Learning rate to control amount of weights/biases update
    float_t LearningRate( ) const
    {
        return mLearningRate;
    }
    virtual void SetLearningRate( float_t learningRate )
    {
        mLearningRate = learningRate;
    }

    // Variables count per learning parameter (weight/bias). This can be value of the previous update
    // like in Momentum optimizer, for example, etc.
    virtual size_t ParameterVariablesCount( ) const
    {
        return 0;
    }

    // Variables count per layer. This variables replace mutable class members, so that optimization algorithms don't
    // store any state inside. For example, Adam optimizer uses them to store b1^t and b2^t values.
    virtual size_t LayerVariablesCount( ) const
    {
        return 0;
    }

    // Calculates weights/biases updates from given gradients
    virtual void CalculateUpdatesFromGradients( fvector_t& updates, std::vector<fvector_t>& paramVariables, fvector_t& layerVariables ) = 0;
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_INETWORK_OPTIMIZER_HPP
