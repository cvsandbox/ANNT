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
#ifndef ANNT_XMAX_POOLING_HPP
#define ANNT_XMAX_POOLING_HPP

#include "IProcessingLayer.hpp"
#include "../../../Tools/XParallel.hpp"

namespace ANNT { namespace Neuro {

// Implementation of maximum pooling - outputs are maximum values of corresponding inputs from a square window
class XMaxPooling : public IProcessingLayer
{
    size_t      mInputWidth;
    size_t      mInputHeight;
    size_t      mInputDepth;
    size_t      mOutputWidth;
    size_t      mOutputHeight;

    size_t      mPoolSizeX;
    size_t      mPoolSizeY;

    size_t      mHorizontalStep;
    size_t      mVerticalStep;

    BorderMode  mBorderMode;

    std::vector<uvector_t> mOutToInMap;
    uvector_t              mInToOutMap;

public:

    XMaxPooling( size_t inputWidth, size_t inputHeight, size_t inputDepth, size_t poolSize = 2 )
        : XMaxPooling( inputWidth, inputHeight, inputDepth, poolSize, poolSize )
    { }

    XMaxPooling( size_t inputWidth, size_t inputHeight, size_t inputDepth,
                 size_t poolSize, size_t stepSize )
        : XMaxPooling( inputWidth, inputHeight, inputDepth, poolSize, poolSize, stepSize, stepSize )
    { }

    XMaxPooling( size_t inputWidth, size_t inputHeight, size_t inputDepth,
                 size_t poolSizeX, size_t poolSizeY,
                 size_t horizontalStep, size_t verticalStep,
                 BorderMode borderMode = BorderMode::Valid ) :
        IProcessingLayer( 0, 0 ),
        mInputWidth( inputWidth ), mInputHeight( inputHeight ), mInputDepth( inputDepth ),
        mOutputWidth( 0 ), mOutputHeight( 0 ),
        mPoolSizeX( poolSizeX ), mPoolSizeY( poolSizeY ),
        mHorizontalStep( horizontalStep ), mVerticalStep( verticalStep ),
        mBorderMode( borderMode )
    {
        size_t padWidth    = 0;
        size_t padHeight   = 0;

        if ( mBorderMode == BorderMode::Same )
        {
            padWidth     = poolSizeX - 1;
            padHeight    = poolSizeY - 1;
        }

        // calculation of output width/height as:
        //   outSize = ( inSize - kernelSize + padSize ) / step + 1
        mOutputWidth  = ( mInputWidth  - mPoolSizeX  + padWidth  ) / mHorizontalStep + 1;
        mOutputHeight = ( mInputHeight - mPoolSizeY  + padHeight ) / mVerticalStep   + 1;

        // total input/output size
        Initialize( mInputWidth  * mInputHeight  * mInputDepth,
                    mOutputWidth * mOutputHeight * mInputDepth );

        // build two maps:
        //   1) first tells output index for the specified input index.
        //   2) second tells indexes of inputs for the specified output;
        // An output will always have at least one input connected to it.
        // However, some inputs may not be connected at all to any of the outputs
        // (if step size is greater than pooling size).
        mInToOutMap = XDataEncodingTools::BuildPoolingInToOutMap( inputWidth, inputHeight, inputDepth, poolSizeX, poolSizeY,
                                                                  horizontalStep, verticalStep, borderMode );
        mOutToInMap = XDataEncodingTools::BuildPoolingOutToInMap( inputWidth, inputHeight, inputDepth, poolSizeX, poolSizeY,
                                                                  horizontalStep, verticalStep, borderMode );
    }

    // Tells that we may need some extra memory for keeping indexes of maximum elements (in training mode)
    uvector_t WorkingMemSize( bool /* trainingMode */ ) const override
    {
        uvector_t workingMemSize = uvector_t( 1 );

        // we don't really need this memory when doing inference only,
        // but don't want to check that always when doing forward pass
        workingMemSize[0] = mOutputsCount * sizeof( size_t);

        return workingMemSize;
    }

    // Calculates outputs for the given inputs
    void ForwardCompute( const std::vector<fvector_t*>& inputs,
                         std::vector<fvector_t*>& outputs,
                         const XNetworkContext& ctx ) override
    {
        XParallel::For( inputs.size( ), ctx.IsTraining( ), [&]( size_t i )
        {
            fvector_t& input      = *( inputs[i] );
            fvector_t& output     = *( outputs[i] );
            size_t*    maxIndexes = static_cast<size_t*>( ctx.GetWorkingBuffer( 0, i ) );

            for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
            {
                const std::vector<size_t>& outputMap = mOutToInMap[outputIndex];
                float_t                    maxValue  = std::numeric_limits<float_t>::lowest( );
                size_t                     maxIndex  = 0;

                for ( auto inputIndex : outputMap )
                {
                    if ( input[inputIndex] > maxValue )
                    {
                        maxValue = input[inputIndex];
                        maxIndex = inputIndex;
                    }
                }

                output[outputIndex]     = maxValue;
                maxIndexes[outputIndex] = maxIndex;
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
            const fvector_t& delta      = *( deltas[i] );
            fvector_t&       prevDelta  = *( prevDeltas[i] );
            size_t*          maxIndexes = static_cast<size_t*>( ctx.GetWorkingBuffer( 0, i ) );

            for ( size_t inputIndex = 0; inputIndex < mInputsCount; inputIndex++ )
            {
                if ( mInToOutMap[inputIndex] == ANNT_NOT_CONNECTED )
                {
                    prevDelta[inputIndex] = float_t( 0 );
                }
                else
                {
                    size_t outputIndex = mInToOutMap[inputIndex];

                    prevDelta[inputIndex] = ( maxIndexes[outputIndex] == inputIndex ) ? delta[outputIndex] : float_t( 0 );
                }
            }
        } );
    }
};

} } // namespace ANNT::Neuro

#endif // ANNT_XMAX_POOLING_HPP

