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

#include "Types/Types.hpp"

/* Classes used for artificial neural networks inference */

#include "Neuro/Layers/XFullyConnectedLayer.hpp"

#include "Neuro/Layers/Activations/XSigmoidActivation.hpp"
#include "Neuro/Layers/Activations/XTanhActivation.hpp"
#include "Neuro/Layers/Activations/XReLuActivation.hpp"
#include "Neuro/Layers/Activations/XLeakyReLuActivation.hpp"
#include "Neuro/Layers/Activations/XEluActivation.hpp"
#include "Neuro/Layers/Activations/XSoftMaxActivation.hpp"

#include "Neuro/Network/XNeuralNetwork.hpp"
#include "Neuro/Network/XNetworkComputation.hpp"

/* Classes used for artificial neural networks training */

#include "Neuro/CostFunctions/XMSECost.hpp"
#include "Neuro/CostFunctions/XAbsoluteCost.hpp"
#include "Neuro/CostFunctions/XCrossEntropyCost.hpp"
#include "Neuro/CostFunctions/XBinaryCrossEntropyCost.hpp"

#include "Neuro/Optimizers/XGradientDescentOptimizer.hpp"
#include "Neuro/Optimizers/XMomentumOptimizer.hpp"
#include "Neuro/Optimizers/XNesterovMomentumOptimizer.hpp"
#include "Neuro/Optimizers/XAdagradOptimizer.hpp"
#include "Neuro/Optimizers/XAdamOptimizer.hpp"
#include "Neuro/Optimizers/XRMSpropOptimizer.hpp"

#include "Neuro/Network/XNetworkTraining.hpp"
