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

namespace ANNT { namespace Neuro {

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

    std::vector<std::vector<size_t>> mOutToInMap;
    std::vector<size_t>              mInToOutMap;
    std::vector<size_t>              mMaxIndexes;

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
        size_t leftPad     = 0;
        size_t topPad      = 0;
        size_t paddedWidth = mInputWidth;

        if ( mBorderMode == BorderMode::Same )
        {
            padWidth     = poolSizeX - 1;
            padHeight    = poolSizeY - 1;
            leftPad      = padWidth  / 2;
            topPad       = padHeight / 2;

            paddedWidth += padWidth;
        }

        // calculation of output width/height as:
        //   outSize = ( inSize - kernelSize + padSize ) / step + 1
        mOutputWidth  = ( mInputWidth  - mPoolSizeX  + padWidth  ) / mHorizontalStep + 1;
        mOutputHeight = ( mInputHeight - mPoolSizeY  + padHeight ) / mVerticalStep   + 1;

        // total input/output size
        Initialize( mInputWidth  * mInputHeight  * mInputDepth,
                    mOutputWidth * mOutputHeight * mInputDepth );

        // allocate vector to keep indexes of maximum input value for a given output
        mMaxIndexes = std::vector<size_t>( mOutputsCount );

        // build two maps:
        //   1) first tells indexes of inputs for a specified output;
        //   2) second tells output index for a specified input index.
        // An output will always have at least one input connected to it.
        // However, some inputs may not be connected at all to any of the outputs
        // (if step size is greater than pooling size).
        mOutToInMap = std::vector<std::vector<size_t>>( mOutputsCount );
        mInToOutMap = std::vector<size_t>( mInputsCount );
        std::fill( mInToOutMap.begin( ), mInToOutMap.end( ), ANNT_NOT_CONNECTED );

        for ( size_t depthIndex = 0, outputIndex = 0; depthIndex < mInputDepth; depthIndex++ )
        {
            for ( size_t outY = 0, inY = 0; outY < mOutputHeight; outY++, inY += mVerticalStep )
            {
                size_t inRowIndex = ( inY + depthIndex * mInputHeight ) * mInputWidth;

                for ( size_t outX = 0, inX = 0; outX < mOutputWidth; outX++, inX += mHorizontalStep, outputIndex++ )
                {
                    std::vector<size_t>& outputMap    = mOutToInMap[outputIndex];
                    size_t               inStartIndex = inRowIndex + inX;

                    for ( size_t poolY = 0, i = 0; poolY < mPoolSizeY; poolY++ )
                    {
                        if ( ( inY + poolY >= topPad ) &&
                             ( inY + poolY <  topPad + mInputHeight ) )
                        {
                            for ( size_t poolX = 0; poolX < mPoolSizeX; poolX++, i++ )
                            {
                                if ( ( inX + poolX >= leftPad ) &&
                                     ( inX + poolX <  leftPad + mInputWidth ) )
                                {
                                    size_t inputIndex = inStartIndex + ( poolY - topPad ) * mInputWidth + poolX - leftPad;

                                    outputMap.push_back( inputIndex );

                                    mInToOutMap[inputIndex] = outputIndex;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void ForwardCompute( const std::vector<fvector_t*>& inputs,
                         std::vector<fvector_t*>& outputs ) override
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            fvector_t& input  = *( inputs[i] );
            fvector_t& output = *( outputs[i] );

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

                output[outputIndex]      = maxValue;
                mMaxIndexes[outputIndex] = maxIndex;
            }
        }
    }
    
    void BackwardProcess( const std::vector<fvector_t*>& /* input  */,
                          const std::vector<fvector_t*>& /* output */,
                          const std::vector<fvector_t*>& deltas,
                          std::vector<fvector_t*>& prevDeltas ) const override
    {
        for ( size_t i = 0, n = deltas.size( ); i < n; i++ )
        {
            const fvector_t& delta = *( deltas[i] );
            fvector_t&       prevDelta = *( prevDeltas[i] );

            for ( size_t inputIndex = 0; inputIndex < mInputsCount; inputIndex++ )
            {
                if ( mInToOutMap[inputIndex] == ANNT_NOT_CONNECTED )
                {
                    prevDelta[inputIndex] = 0.0f;
                }
                else
                {
                    size_t outputIndex = mInToOutMap[inputIndex];

                    prevDelta[inputIndex] = ( mMaxIndexes[outputIndex] == inputIndex ) ? delta[outputIndex] : 0.0f;
                }
            }
        }
    }
};

} } // namespace ANNT::Neuro

#endif // ANNT_XMAX_POOLING_HPP

