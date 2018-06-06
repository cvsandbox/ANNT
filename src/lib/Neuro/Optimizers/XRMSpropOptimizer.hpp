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
#ifndef ANNT_XRMSPROP_OPTIMIZER_HPP
#define ANNT_XRMSPROP_OPTIMIZER_HPP

#include "INetworkOptimizer.hpp"

namespace ANNT { namespace Neuro { namespace Training {

// Implementation of RMSprop optimization algorithm
// http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
//
class XRMSpropOptimizer : public INetworkOptimizer
{
private:
    float_t mEpsilon;
    float_t mMu;

public:
    XRMSpropOptimizer( float_t learningRate = float_t( 0.001 ), float_t mu = float_t( 0.9 ) ) :
        INetworkOptimizer( learningRate ),
        mEpsilon( float_t( 1e-8 ) ), mMu( mu )
    {
    }

    size_t ParameterVariablesCount( ) const override
    {
        return 1;
    }

    void CalculateUpdatesFromGradients( fvector_t& updates, std::vector<fvector_t>& paramVariables, fvector_t& /* layerVariables */ ) override
    {
        fvector_t& eg = paramVariables[0];

        for ( size_t i = 0, n = updates.size( ); i < n; i++ )
        {
            eg[i]       = mMu * eg[i] + ( 1 - mMu ) * updates[i] * updates[i];
            updates[i] *= -mLearningRate / std::sqrt( eg[i] + mEpsilon );
        }
    }
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_XRMSPROP_OPTIMIZER_HPP
