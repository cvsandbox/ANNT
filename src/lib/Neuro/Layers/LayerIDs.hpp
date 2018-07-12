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
#ifndef ANNT_LAYER_IDS_HPP
#define ANNT_LAYER_IDS_HPP

namespace ANNT { namespace Neuro {

enum class LayerID
{
    Unknown            = 0,

    FullyConnected     = 1,
    Convolution        = 2,
    RecurrentBasic     = 3,
    RecurrentLSTM      = 4,
    RecurrentGRU       = 5,

    Sigmoid            = 1000,
    Tanh               = 1001,
    Relu               = 1002,
    LeakyRelu          = 1003,
    Elu                = 1004,
    Softmax            = 1005,
    LogSoftmax         = 1006,

    MaxPooling         = 2001,
    AveragePooling     = 2002,
    DropOut            = 2003,
    BatchNormalization = 2004,
};

} } // namespace ANNT::Neuro

#endif // ANNT_LAYER_IDS_HPP
