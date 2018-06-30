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
#ifndef ANNT_XDROP_OUT_LAYER_HPP
#define ANNT_XDROP_OUT_LAYER_HPP

#include "IProcessingLayer.hpp"
#include "../../../Tools/XParallel.hpp"
#include <random>

namespace ANNT { namespace Neuro {

class XDropOutLayer : public IProcessingLayer
{
private:
    float_t mDropOutRate;

    std::mt19937                            mGenerator;
    std::uniform_real_distribution<float_t> mDistribution;

public:
    XDropOutLayer( float_t dropOutRate = float_t( 0.1f ) ) :
        IProcessingLayer( 0, 0 ),
        mDropOutRate( dropOutRate ),
        mGenerator( ), mDistribution( 0.0f, 1.0f )
    {

    }

    // Tells that we may need some extra memory for keeping drop out mask (in training mode)
    uvector_t WorkingMemSize( bool trainingMode ) const override
    {
        uvector_t workingMemSize;;

        if ( trainingMode )
        {
            workingMemSize.push_back( mOutputsCount * sizeof( float_t ) );
        }

        return workingMemSize;
    }

    // Calculates outputs for the given inputs
    void ForwardCompute( const std::vector<fvector_t*>& inputs,
                         std::vector<fvector_t*>& outputs,
                         const XNetworkContext& ctx ) override
    {
        XParallel::For( inputs.size( ), ctx.IsTraining( ), [&]( size_t i )
        {
            fvector_t& input  = *( inputs[i] );
            fvector_t& output = *( outputs[i] );

            if ( !ctx.IsTraining( ) )
            {
                output = input;
            }
            else
            {
                float_t* dropOutMask = static_cast<float_t*>( ctx.GetWorkingBuffer( 0, i ) );

                for ( size_t j = 0; j < mOutputsCount; j++ )
                {
                    dropOutMask[j] = ( mDistribution( mGenerator ) < mDropOutRate ) ? float_t( 0.0f ) : float_t( 1.0f );
                }

                for ( size_t j = 0; j < mOutputsCount; j++ )
                {
                    output[j] = input[j] * dropOutMask[j];
                }
            }
        } );
    }

    // Propagates error to the previous layer
    void BackwardProcess( const std::vector<fvector_t*>& /* inputs  */,
                          const std::vector<fvector_t*>& /* outputs */,
                          const std::vector<fvector_t*>& deltas,
                          std::vector<fvector_t*>& prevDeltas,
                          const XNetworkContext& ctx ) override
    {
        XParallel::For( deltas.size( ), ctx.IsTraining( ), [&]( size_t i )
        {
            const fvector_t& delta     = *( deltas[i] );
            fvector_t&       prevDelta = *( prevDeltas[i] );

            if ( !ctx.IsTraining( ) )
            {
                prevDelta = delta;
            }
            else
            {
                float_t* dropOutMask = static_cast<float_t*>( ctx.GetWorkingBuffer( 0, i ) );

                for ( size_t j = 0; j < mOutputsCount; j++ )
                {
                    prevDelta[j] = delta[j] * dropOutMask[j];
                }
            }
        } );
    }
};

} } // ANNT::Neuro

#endif // ANNT_XDROP_OUT_LAYER_HPP

