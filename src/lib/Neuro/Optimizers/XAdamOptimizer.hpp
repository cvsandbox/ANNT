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
#ifndef ANNT_XADAM_OPTIMIZER_HPP
#define ANNT_XADAM_OPTIMIZER_HPP

#include "INetworkOptimizer.hpp"

// Implementation of Adam otimizer
// http://ruder.io/optimizing-gradient-descent/index.html#adam
//
namespace ANNT { namespace Neuro { namespace Training {

class XAdamOptimizer : public INetworkOptimizer
{
private:
    float_t mEpsilon;
    float_t mB1;
    float_t mB2;

public:
    XAdamOptimizer( float_t learningRate = float_t( 0.001 ) ) :
        INetworkOptimizer( learningRate ), mEpsilon( float_t( 1e-8 ) ),
        mB1( float_t( 0.9 ) ), mB2( float_t( 0.999 ) )
    {
    }

    // Two variables per learning parameter, m(t) and v(t)
    size_t ParameterVariablesCount( ) const override
    {
        return 2;
    }

    // Variables to keep b1^t and b2^t values
    virtual size_t LayerVariablesCount( ) const
    {
        return 3;
    }

    void CalculateUpdatesFromGradients( fvector_t& updates, std::vector<fvector_t>& paramVariables, fvector_t& layerVariables ) override
    {
        fvector_t& mt  = paramVariables[0];
        fvector_t& vt  = paramVariables[1];
        float_t    b1t = mB1;
        float_t    b2t = mB2;

        // check if it is the first call
        if ( layerVariables[0] < float( 0.5 ) )
        {
            layerVariables[0] = float( 1.0 );
        }
        else
        {
            b1t = layerVariables[1];
            b2t = layerVariables[2];
        }

        for ( size_t i = 0, n = updates.size( ); i < n; i++ )
        {
            mt[i] = mB1 * mt[i] + ( float_t( 1 ) - mB1 ) * updates[i];
            vt[i] = mB2 * vt[i] + ( float_t( 1 ) - mB2 ) * updates[i] * updates[i];

            updates[i] = -mLearningRate * ( mt[i] / ( float_t( 1 ) - b1t ) ) /
                         std::sqrt( vt[i] / ( float_t( 1 ) - b2t ) + mEpsilon );
        }

        b1t *= mB1;
        b2t *= mB2;

        layerVariables[1] = b1t;
        layerVariables[2] = b2t;
    }
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_XADAM_OPTIMIZER_HPP
