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
vector_t XDataEncodingTools::OneHotEncoding( size_t label, size_t labelsCount )
{
    vector_t encodedClass( labelsCount, 0 );

    encodedClass[label] = 1;

    return encodedClass;
}

// Encodes a vector of classes using one-hot encoding
vector<vector_t> XDataEncodingTools::OneHotEncoding( const vector<size_t>& labels, size_t labelsCount )
{
    vector<vector_t> encodedClasses( labels.size( ) );

    for ( size_t i = 0; i < labels.size( ); i++ )
    {
        encodedClasses[i] = vector_t( labelsCount, 0 );
        encodedClasses[i][labels[i]] = 1;
    }

    return encodedClasses;
}

// Returns index of the maximum element in the specified vector
size_t XDataEncodingTools::MaxIndex( const vector_t& vec )
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

} // namespace ANNT
