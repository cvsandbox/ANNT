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
#ifndef ANNT_XCONVOLUTION_LAYER_HPP
#define ANNT_XCONVOLUTION_LAYER_HPP

#include "ITrainableLayer.hpp"

namespace ANNT { namespace Neuro {

class XConvolutionLayer : public ITrainableLayer
{
private:
    size_t      mInputWidth;
    size_t      mInputHeight;
    size_t      mInputDepth;
    size_t      mOutputWidth;
    size_t      mOutputHeight;
    size_t      mKernelWidth;
    size_t      mKernelHeight;
    size_t      mKernelsCount;
    size_t      mHorizontalStep;
    size_t      mVerticalStep;
    BorderMode  mBorderMode;

    size_t      mPadWidth;
    size_t      mPadHeight;

    vector_t    mKernelsWeights;
    vector_t    mKernelsBiases;

    std::vector<vector_t> mPaddedInputs;
    std::vector<vector_t> mPaddedPrevDeltas;

public:
    XConvolutionLayer( size_t inputWidth,  size_t inputHeight,  size_t inputDepth,
                       size_t kernelWidth, size_t kernelHeight, size_t kernelsCount,
                       BorderMode borderMode = BorderMode::Valid,
                       size_t horizontalStep = 1, size_t verticalStep = 1 );

    // Reports number of weight coefficients the layer has
    size_t WeightsCount( ) const override
    {
        return mKernelsWeights.size( );
    }

    // Reports number of bias coefficients the layer has
    virtual size_t BiasesCount( ) const override
    {
        return mKernelsBiases.size( );
    }

    // Randomizes layer's weights, clears biases
    void Randomize( ) override;

    // Calculates outputs for the given inputs
    void ForwardCompute( const std::vector<vector_t*>& inputs,
                         std::vector<vector_t*>& outputs ) override;

    // Propagates error to the previous layer and calculates weights/biases gradients
    void BackwardCompute( const std::vector<vector_t*>& inputs,
                          const std::vector<vector_t*>& outputs,
                          const std::vector<vector_t*>& deltas,
                          std::vector<vector_t*>& prevDeltas,
                          vector_t& gradWeights,
                          vector_t& gradBiases ) override;

    // Applies updates to the layer's weights and biases
    void UpdateWeights( const vector_t& weightsUpdate,
                        const vector_t& biasesUpdate ) override;

private:

    void InputPadding( const std::vector<vector_t*>& input, std::vector<vector_t>& padded );
    void DeltasUnpadding( std::vector<vector_t>& deltas, std::vector<vector_t*>& unpadded );
};

} } // namespace ANNT::Neuro

#endif // ANNT_XCONVOLUTION_LAYER_HPP
