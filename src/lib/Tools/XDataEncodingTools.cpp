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

#include "XDataEncodingTools.hpp"

using namespace std;

namespace ANNT {

// Encodes single class using one-hot encoding - a vector of all zeros except the one element set to 1,
// which index corresponds to the class value
fvector_t XDataEncodingTools::OneHotEncoding( size_t label, size_t labelsCount )
{
    fvector_t encodedClass( labelsCount, 0 );

    encodedClass[label] = 1;

    return encodedClass;
}

// Encodes a vector of classes using one-hot encoding
vector<fvector_t> XDataEncodingTools::OneHotEncoding( const uvector_t& labels, size_t labelsCount )
{
    vector<fvector_t> encodedClasses( labels.size( ) );

    for ( size_t i = 0; i < labels.size( ); i++ )
    {
        encodedClasses[i] = fvector_t( labelsCount, 0 );
        encodedClasses[i][labels[i]] = 1;
    }

    return encodedClasses;
}

// Returns index of the maximum element in the specified vector
size_t XDataEncodingTools::MaxIndex( const fvector_t& vec )
{
    size_t maxIndex = 0;
    auto   maxValue = vec[0];

    for ( size_t i = 1, n = vec.size( ); i < n; i++ )
    {
        if ( vec[i] > maxValue )
        {
            maxValue = vec[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

// Pads the specified 2D input (although it can be of certain depth) with the specified value
void XDataEncodingTools::AddPadding2d( const fvector_t& src, fvector_t& dst,
                                       size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight,
                                       size_t depth, float_t padValue )
{
    size_t dstSize = dstWidth * dstHeight * depth;

    if ( dst.size( ) != dstSize )
    {
        dst.resize( dstSize );
    }

    AddPadding2d( src.data( ), dst.data( ), srcWidth, srcHeight, dstWidth, dstHeight, depth, padValue );
}
void XDataEncodingTools::AddPadding2d( const float_t* src, float* dst,
                                       size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight,
                                       size_t depth, float_t padValue )
{
    if ( ( dstWidth >= srcWidth ) && ( dstHeight >= srcHeight ) )
    {
        size_t padWidth  = dstWidth  - srcWidth;
        size_t padHeight = dstHeight - srcHeight;
        // For even pad width/height it is distributed equally on each side.
        // However for odd value, padding goes first to right/bottom sides.
        size_t leftPad   = padWidth >> 1;
        size_t rightPad  = padWidth - leftPad;
        size_t topPad    = padHeight >> 1;
        size_t bottomPad = padHeight - topPad;

        const float* srcPtr = src;
        float*       dstPtr = dst;

        for ( size_t d = 0; d < depth; d++ )
        {
            // top padding
            for ( size_t y = 0; y < topPad; y++ )
            {
                for ( size_t x = 0; x < dstWidth; x++, dstPtr++ )
                {
                    *dstPtr = padValue;
                }
            }

            for ( size_t y = 0; y < srcHeight; y++ )
            {
                // left padding
                for ( size_t x = 0; x < leftPad; x++, dstPtr++ )
                {
                    *dstPtr = padValue;
                }

                // copying the source
                for ( size_t x = 0; x < srcWidth; x++, srcPtr++, dstPtr++ )
                {
                    *dstPtr = *srcPtr;
                }

                // right padding
                for ( size_t x = 0; x < rightPad; x++, dstPtr++ )
                {
                    *dstPtr = padValue;
                }
            }

            // bottom padding
            for ( size_t y = 0; y < bottomPad; y++ )
            {
                for ( size_t x = 0; x < dstWidth; x++, dstPtr++ )
                {
                    *dstPtr = padValue;
                }
            }
        }
    }
}

// Removes padding from the specified 2D input
void XDataEncodingTools::RemovePadding2d( const fvector_t& src, fvector_t& dst,
                                          size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight,
                                          size_t depth )
{
    size_t dstSize = dstWidth * dstHeight * depth;

    if ( dst.size( ) != dstSize )
    {
        dst.resize( dstSize );
    }

    RemovePadding2d( src.data( ), dst.data( ), srcWidth, srcHeight, dstWidth, dstHeight, depth );
}
void XDataEncodingTools::RemovePadding2d( const float_t* src, float_t* dst,
                                          size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight,
                                          size_t depth )
{
    if ( ( dstWidth <= srcWidth ) && ( dstHeight <= srcHeight ) )
    {
        size_t padWidth  = srcWidth  - dstWidth;
        size_t padHeight = srcHeight - dstHeight;
        // For even pad width/height it is distributed equally on each side.
        // However for odd value, padding goes first to right/bottom sides.
        size_t leftPad   = padWidth >> 1;
        size_t rightPad  = padWidth - leftPad;
        size_t topPad    = padHeight >> 1;
        size_t bottomPad = padHeight - topPad;

        topPad    *= srcWidth;
        bottomPad *= srcWidth;

        const float* srcPtr = src;
        float*       dstPtr = dst;

        for ( size_t d = 0; d < depth; d++ )
        {
            // skip top padding
            srcPtr += topPad;

            for ( size_t y = 0; y < dstHeight; y++ )
            {
                // skip left left padding
                srcPtr += leftPad;

                // copying the source
                for ( size_t x = 0; x < dstWidth; x++, srcPtr++, dstPtr++ )
                {
                    *dstPtr = *srcPtr;
                }

                // skip right padding
                srcPtr += rightPad;
            }

            // skip bottom padding
            srcPtr += bottomPad;
        }
    }
}

// Builds input to output index mapping for pooling operator - one to one mapping
uvector_t XDataEncodingTools::BuildPoolingInToOutMap( size_t inputWidth, size_t inputHeight, size_t inputDepth,
                                                      size_t poolSizeX, size_t poolSizeY,
                                                      size_t horizontalStep, size_t verticalStep,
                                                      BorderMode borderMode )
{
    size_t padWidth    = 0;
    size_t padHeight   = 0;
    size_t leftPad     = 0;
    size_t topPad      = 0;

    if ( borderMode == BorderMode::Same )
    {
        padWidth     = poolSizeX - 1;
        padHeight    = poolSizeY - 1;
        leftPad      = padWidth  / 2;
        topPad       = padHeight / 2;
    }

    // calculation of output width/height as:
    //   outSize = ( inSize - kernelSize + padSize ) / step + 1
    size_t outputWidth  = ( inputWidth  - poolSizeX + padWidth )  / horizontalStep + 1;
    size_t outputHeight = ( inputHeight - poolSizeY + padHeight ) / verticalStep   + 1;

    size_t inputsCount  = inputWidth * inputHeight * inputDepth;

    // build the map providing output index for the given input index
    uvector_t inToOutMap = uvector_t( inputsCount );

    std::fill( inToOutMap.begin( ), inToOutMap.end( ), ANNT_NOT_CONNECTED );

    for ( size_t depthIndex = 0, outputIndex = 0; depthIndex < inputDepth; depthIndex++ )
    {
        for ( size_t outY = 0, inY = 0; outY < outputHeight; outY++, inY += verticalStep )
        {
            size_t inRowIndex = ( inY + depthIndex * inputHeight ) * inputWidth;

            for ( size_t outX = 0, inX = 0; outX < outputWidth; outX++, inX += horizontalStep, outputIndex++ )
            {
                size_t inStartIndex = inRowIndex + inX;

                for ( size_t poolY = 0, i = 0; poolY < poolSizeY; poolY++ )
                {
                    if ( ( inY + poolY >= topPad ) &&
                         ( inY + poolY <  topPad + inputHeight ) )
                    {
                        for ( size_t poolX = 0; poolX < poolSizeX; poolX++, i++ )
                        {
                            if ( ( inX + poolX >= leftPad ) &&
                                 ( inX + poolX <  leftPad + inputWidth ) )
                            {
                                size_t inputIndex = inStartIndex + ( poolY - topPad ) * inputWidth + poolX - leftPad;

                                inToOutMap[inputIndex] = outputIndex;
                            }
                        }
                    }
                }
            }
        }
    }

    return inToOutMap;
}

// Builds output index to input indexes mapping for pooling operator - 1 to many mapping
vector<uvector_t> XDataEncodingTools::BuildPoolingOutToInMap( size_t inputWidth, size_t inputHeight, size_t inputDepth,
                                                              size_t poolSizeX, size_t poolSizeY,
                                                              size_t horizontalStep, size_t verticalStep,
                                                              BorderMode borderMode )
{
    size_t padWidth    = 0;
    size_t padHeight   = 0;
    size_t leftPad     = 0;
    size_t topPad      = 0;

    if ( borderMode == BorderMode::Same )
    {
        padWidth     = poolSizeX - 1;
        padHeight    = poolSizeY - 1;
        leftPad      = padWidth  / 2;
        topPad       = padHeight / 2;
    }

    // calculation of output width/height as:
    //   outSize = ( inSize - kernelSize + padSize ) / step + 1
    size_t outputWidth  = ( inputWidth  - poolSizeX + padWidth  ) / horizontalStep + 1;
    size_t outputHeight = ( inputHeight - poolSizeY + padHeight ) / verticalStep   + 1;
    size_t outputsCount = outputWidth * outputHeight * inputDepth;

    vector<uvector_t> outToInMap = vector<uvector_t>( outputsCount );

    for ( size_t depthIndex = 0, outputIndex = 0; depthIndex < inputDepth; depthIndex++ )
    {
        for ( size_t outY = 0, inY = 0; outY < outputHeight; outY++, inY += verticalStep )
        {
            size_t inRowIndex = ( inY + depthIndex * inputHeight ) * inputWidth;

            for ( size_t outX = 0, inX = 0; outX < outputWidth; outX++, inX += horizontalStep, outputIndex++ )
            {
                std::vector<size_t>& outputMap    = outToInMap[outputIndex];
                size_t               inStartIndex = inRowIndex + inX;

                for ( size_t poolY = 0, i = 0; poolY < poolSizeY; poolY++ )
                {
                    if ( ( inY + poolY >= topPad ) &&
                         ( inY + poolY <  topPad + inputHeight ) )
                    {
                        for ( size_t poolX = 0; poolX < poolSizeX; poolX++, i++ )
                        {
                            if ( ( inX + poolX >= leftPad ) &&
                                 ( inX + poolX <  leftPad + inputWidth ) )
                            {
                                size_t inputIndex = inStartIndex + ( poolY - topPad ) * inputWidth + poolX - leftPad;

                                outputMap.push_back( inputIndex );
                            }
                        }
                    }
                }
            }
        }
    }

    return outToInMap;
}

} // namespace ANNT
