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
#ifndef ANNT_XNETWORK_TRAINING_HPP
#define ANNT_XNETWORK_TRAINING_HPP

#include <memory>
#include <vector>

#include "XNetworkComputation.hpp"
#include "../Optimizers/INetworkOptimizer.hpp"
#include "../CostFunctions/ICostFunction.hpp"

namespace ANNT { namespace Neuro { namespace Training {

// Implementation of artificial neural network training with
// error back propagation algorithm - wraps memory buffers and
// infrastructure required for training a network: run forward
// pass, calculate error, propagate it backward through the
// network calculating gradients of weights/biases and applying
// them.
//
class XNetworkTraining : public XNetworkComputation
{
private:
    std::shared_ptr<INetworkOptimizer>  mOptimizer;
    std::shared_ptr<ICostFunction>      mCostFunction;
    bool                                mAverageWeightGradients;

private:
    // storage and pointers for outputs computed during training
    std::vector<std::vector<vector_t>>  mTrainOutputsStorage;
    std::vector<std::vector<vector_t*>> mTrainOutputs;

    // storade and pointers to compute deltas for each layer
    std::vector<std::vector<vector_t>>  mDeltasStorage;
    std::vector<std::vector<vector_t*>> mDeltas;

    // storage and pointers "input deltas"
    // no needed, just to make calculations consistent
    std::vector<vector_t>               mInputDeltasStorage;
    std::vector<vector_t*>              mInputDeltas;

    // vectors used to assemble pointers to training samples (input/outputs)
    std::vector<vector_t*>              mTrainInputs;
    std::vector<vector_t*>              mTargetOuputs;

    // weights/biases gradients for all layers
    std::vector<vector_t>               mGradWeights;
    std::vector<vector_t>               mGradBiases;

    // vectors with parameter variables for optimizer
    std::vector<std::vector<vector_t>>  mWeightsParameterVariables;
    std::vector<std::vector<vector_t>>  mBiasesParameterVariables;

    // vectors with layer variables for optimizer
    std::vector<vector_t>               mWeightsLayerVariables;
    std::vector<vector_t>               mBiasesLayerVariables;

public:

    XNetworkTraining( const std::shared_ptr<XNeuralNetwork>& network,
                      const std::shared_ptr<INetworkOptimizer>& optimizer,
                      const std::shared_ptr<ICostFunction>& costFunction
                      );

    // Provides access to the weights/biases optimizer
    std::shared_ptr<INetworkOptimizer> Optimizer( ) const
    {
        return mOptimizer;
    }

    // Provides access to the cost function used for error calculation
    std::shared_ptr<ICostFunction> CostFunction( ) const
    {
        return mCostFunction;
    }

    // Average or not weights/biases gradients when running in batch mode
    bool AverageWeightGradients( ) const
    {
        return mAverageWeightGradients;
    }
    void SetAverageWeightGradients( bool average )
    {
        mAverageWeightGradients = average;
    }

    // Trains single input/output sample
    float_t TrainSample( const vector_t& input, const vector_t& targetOutput );

    // Trains single batch of prepared samples (as vectors)
    float_t TrainBatch( const std::vector<vector_t>& inputs,
                        const std::vector<vector_t>& targetOutputs );

    // Trains single batch of prepared samples (as pointers to vectors)
    float_t TrainBatch( const std::vector<vector_t*>& inputs,
                        const std::vector<vector_t*>& targetOutputs );

    // Trains single epoch using batches of the specified size
    float_t TrainEpoch( const std::vector<vector_t>& inputs, 
                        const std::vector<vector_t>& targetOutputs,
                        size_t batchSize,
                        bool randomPickIntoBatch = false );

    // Tests sample - calculates real output and provides error cost
    float_t TestSample( const vector_t& input,
                        const vector_t& targetOutput,
                        vector_t& output );

    // Tests classification for the provided inputs and target labels -
    // provides number of correctly classified samples and average cost (target outputs are used)
    size_t TestClassification( const std::vector<vector_t>& inputs,
                               const std::vector<size_t>& targetLabels,
                               const std::vector<vector_t>& targetOutputs,
                               float_t* pAvgCost );

private:

    float_t RunTraining( );
    float_t CalculateError( );
    void    DoBackwardCompute( );
    void    UpdateWeights( );
    void    AllocateTrainVectors( size_t samplesCount );
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_XNETWORK_TRAINING_HPP
