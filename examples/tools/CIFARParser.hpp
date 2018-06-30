/*
    ANNT - Artificial Neural Networks C++ library

    CIFAR-10 data set parser

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
#ifndef ANNT_CIFAR10_PARSER_HPP
#define ANNT_CIFAR10_PARSER_HPP

#include <string>
#include "Types/Types.hpp"

namespace ANNT {

// Helper function to load images and labels from CIFAR-10 dataset
// https://www.cs.toronto.edu/~kriz/cifar.html
//
class CIFARParser
{
private:
    CIFARParser( );

public:

    // Loads images and labels from the specified CIFAR-10 dataset file (appends to the provided vectors)
    static bool LoadDataSet( const std::string& fileName, uvector_t& labels, std::vector<fvector_t>& images,
                             float_t scaleMin, float_t scaleMax );
};

} // namespace ANNT

#endif // ANNT_CIFAR10_PARSER_HPP
