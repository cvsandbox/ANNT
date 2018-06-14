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
#ifndef ANNT_XDATA_ENCODING_TOOLS_HPP
#define ANNT_XDATA_ENCODING_TOOLS_HPP

#include "../Types/Types.hpp"

namespace ANNT {

// Collection of tools to encode/decode data to/from formats expected/produced by ANNs
class XDataEncodingTools
{
private:
    XDataEncodingTools( );

public:
    // Encodes single class/label using one-hot encoding - a vector of all zeros except the one element set to 1,
    // which index corresponds to the class value
    static fvector_t OneHotEncoding( size_t label, size_t labelsCount );

    // Encodes a vector of labels using one-hot encoding
    static std::vector<fvector_t> OneHotEncoding( const uvector_t& labels, size_t labelsCount );

    // Returns index of the maximum element in the specified vector
    static size_t MaxIndex( const fvector_t& vec );

    // Pads the specified 2D input (although it can be of certain depth) with the specified value
    static void AddPadding2d( const fvector_t& src, fvector_t& dst,
                              size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight,
                              size_t depth, float_t padValue );
    static void AddPadding2d( const float_t* src, float_t* dst,
                              size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight,
                              size_t depth, float_t padValue );

    // Removes padding from the specified 2D input (although it can be of certain depth)
    static void RemovePadding2d( const fvector_t& src, fvector_t& dst,
                                 size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight,
                                 size_t depth );
    static void RemovePadding2d( const float_t* src, float_t* dst,
                                 size_t srcWidth, size_t srcHeight, size_t dstWidth, size_t dstHeight,
                                 size_t depth );

    // Builds input to output index mapping for pooling operator - one to one mapping
    static uvector_t BuildPoolingInToOutMap( size_t inputWidth, size_t inputHeight, size_t inputDepth,
                                             size_t poolSizeX, size_t poolSizeY,
                                             size_t horizontalStep, size_t verticalStep,
                                             BorderMode borderMode = BorderMode::Valid );

    // Builds output index to input indexes mapping for pooling operator - 1 to many mapping
    static std::vector<uvector_t> BuildPoolingOutToInMap( size_t inputWidth, size_t inputHeight, size_t inputDepth,
                                                          size_t poolSizeX, size_t poolSizeY,
                                                          size_t horizontalStep, size_t verticalStep,
                                                          BorderMode borderMode = BorderMode::Valid );
};

} // namespace ANNT

#endif // ANNT_XDATA_ENCODING_TOOLS_HPP
