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
    static vector_t OneHotEncoding( size_t label, size_t labelsCount );

    // Encodes a vector of labels using one-hot encoding
    static std::vector<vector_t> OneHotEncoding( const std::vector<size_t>& labels, size_t labelsCount );

    // Returns index of the maximum element in the specified vector
    static size_t MaxIndex( const vector_t& vec );
};

} // namespace ANNT

#endif // ANNT_XDATA_ENCODING_TOOLS_HPP
