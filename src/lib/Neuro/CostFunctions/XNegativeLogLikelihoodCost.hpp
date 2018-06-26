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
#ifndef ANNT_XNEGATIVE_LOG_LIKELIHOOD_COST_HPP
#define ANNT_XNEGATIVE_LOG_LIKELIHOOD_COST_HPP

#include "ICostFunction.hpp"

namespace ANNT { namespace Neuro { namespace Training {

// Implementation of negative log-likelihood cost function - to be used after XLogSoftMaxActivation layer,
// which produces log-probabilities
//
class XNegativeLogLikelihoodCost : public ICostFunction
{
public:

    // Calculates cost value of the specified output vector
    float_t Cost( const fvector_t& output, const fvector_t& target ) const override
    {
        size_t  length = output.size( );
        float_t cost   = float_t( 0 );

        for ( size_t i = 0; i < length; i++ )
        {
            cost += -target[i] * output[i];
        }

        return cost;
    }

    // Calculates gradient for the specified output/target pair
    fvector_t Gradient( const fvector_t& output, const fvector_t& target ) const override
    {
        size_t    length = output.size( );
        fvector_t grad( length );

        for ( size_t i = 0; i < length; i++ )
        {
            grad[i] = -target[i];
        }

        return grad;
    }
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_XNEGATIVE_LOG_LIKELIHOOD_COST_HPP
