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
#ifndef ANNT_ICOST_FUNCTION_HPP
#define ANNT_ICOST_FUNCTION_HPP

#include "../../Types/Types.hpp"

namespace ANNT { namespace Neuro { namespace Training {

// Cost functions' interface for calculating cost and gradient
// for a specified output/target pair
class ICostFunction
{
public:
    virtual ~ICostFunction( ) { }

    // Calculates cost value of the specified output vector
    virtual float_t Cost( const fvector_t& output, const fvector_t& target ) const = 0;

    // Calculates gradient for the specified output/target pair
    virtual fvector_t Gradient( const fvector_t& output, const fvector_t& target ) const = 0;
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_ICOST_FUNCTION_HPP
