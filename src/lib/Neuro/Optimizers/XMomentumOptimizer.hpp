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
#ifndef ANNT_XMOMENTUM_OPTIMIZER_HPP
#define ANNT_XMOMENTUM_OPTIMIZER_HPP

#include "INetworkOptimizer.hpp"

namespace ANNT { namespace Neuro { namespace Training {

// Implementation of SGD with Momentum, which calculates updates as
//   v(t) = momentum * v(t-1) + learningRate * paramGrad
//   paramUpdate(t) = -v(t)
// http://ruder.io/optimizing-gradient-descent/index.html#momentum
//
class XMomentumOptimizer : public INetworkOptimizer
{
private:
    float_t mMomentum;

public:
    XMomentumOptimizer( float_t learningRate = float_t( 0.01 ), float_t momentum = float_t( 0.9 ) ) :
        INetworkOptimizer( learningRate ), mMomentum( momentum )
    {
    }

    // One variable per learning parameter - previous update value
    size_t ParameterVariablesCount( ) const override
    {
        return 1;
    }

    void CalculateUpdatesFromGradients( fvector_t& updates, std::vector<fvector_t>& paramVariables, fvector_t& /* layerVariables */ ) override
    {
        fvector_t& vPrev = paramVariables[0];

        for ( size_t i = 0, n = updates.size( ); i < n; i++ )
        {
            float_t vt = mMomentum * vPrev[i] + mLearningRate * updates[i];

            updates[i] = -vt;
            vPrev[i]   = vt;
        }
    }
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_XMOMENTUM_OPTIMIZER_HPP
