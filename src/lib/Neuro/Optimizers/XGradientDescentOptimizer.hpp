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
#ifndef ANNT_XGRADIENT_DESCENT_OPTIMIZER_HPP
#define ANNT_XGRADIENT_DESCENT_OPTIMIZER_HPP

#include "INetworkOptimizer.hpp"

namespace ANNT { namespace Neuro { namespace Training {

// Implementation of classical Stochastic Gradient Descent optimizer, which is
//   paramUpdate(t) = -learningRate * paramGrad(t)
// http://cs231n.github.io/neural-networks-3/#sgd
//
class XGradientDescentOptimizer : public INetworkOptimizer
{
public:
    XGradientDescentOptimizer( float_t learningRate = float_t( 0.01 ) ) :
        INetworkOptimizer( learningRate )
    {
    }

    void CalculateUpdatesFromGradients( fvector_t& updates, std::vector<fvector_t>& /* paramVariables */, fvector_t& /* layerVariables */ ) override
    {
        for ( auto& update : updates )
        {
            update *= -mLearningRate;
        }
    }
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_XGRADIENT_DESCENT_OPTIMIZER_HPP
