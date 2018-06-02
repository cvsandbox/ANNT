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

#include "XConvolutionLayer.hpp"

using namespace std;

namespace ANNT { namespace Neuro {

XConvolutionLayer::XConvolutionLayer( size_t inputWidth, size_t inputHeight, size_t inputDepth,
                                      size_t kernelWidth, size_t kernelHeight, size_t kernelsCount,
                                      BorderMode borderMode, size_t horizontalStep, size_t verticalStep ) :
                                      ITrainableLayer( 0, 0 ),
    mInputWidth( inputWidth ), mInputHeight( inputHeight ), mInputDepth( inputDepth ),
    mOutputWidth( 0 ), mOutputHeight( 0 ),
    mKernelWidth( kernelWidth ), mKernelHeight( kernelHeight ), mKernelsCount( kernelsCount ),
    mHorizontalStep( horizontalStep ), mVerticalStep( verticalStep ), mBorderMode( borderMode ), 
    mPadWidth( 0 ), mPadHeight( 0 )
{
    // use input padding, if border handling mode is set to produce same size output
    // (same ouput size will be only when step size is 1; output is always smaller for larger steps)
    if ( mBorderMode == BorderMode::Same )
    {
        mPadWidth  = mKernelWidth  - 1;
        mPadHeight = mKernelHeight - 1;

        //size_t paddedInputSize = ( mInputWidth + mPadWidth ) * ( mInputHeight + mPadHeight ) * mInputDepth;

        // allocate memory for padded input data and deltas of the previous layer
        //mPaddedInput      = vector<float>( paddedInputSize );
        //mPaddedPrevDeltas = vector<float>( paddedInputSize );
    }

    // calculation of output width/height as:
    //   outSize = ( inSize - kernelSize + padSize ) / step + 1
    mOutputWidth  = ( mInputWidth  - mKernelWidth  + mPadWidth  ) / mHorizontalStep + 1;
    mOutputHeight = ( mInputHeight - mKernelHeight + mPadHeight ) / mVerticalStep   + 1;

    // total input/output size
    Initialize( mInputWidth  * mInputHeight  * mInputDepth,
                mOutputWidth * mOutputHeight * mKernelsCount );

    mKernelsWeights = vector_t( mKernelWidth * mKernelHeight * mInputDepth * mKernelsCount );
    mKernelsBiases  = vector_t( mKernelsCount );

    Randomize( );
}

// Randomizes layer's weights, clears biases
void XConvolutionLayer::Randomize( )
{
    float halfRange = sqrt( 3.0f / ( mKernelWidth * mKernelHeight * mInputDepth ) );

    for ( auto& w : mKernelsWeights )
    {
        w = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRange ) - halfRange;
    }
    for ( auto& b : mKernelsBiases )
    {
        b = 0;
    }
}

// Calculates outputs for the given inputs
void XConvolutionLayer::ForwardCompute( const vector<vector_t*>& inputs,
                                        vector<vector_t*>& outputs )
{
    const float_t* kernelsData  = mKernelsWeights.data( );

    // kernel size in 2D and 3D spaces
    size_t  kernelSize2D = mKernelWidth * mKernelHeight;
    size_t  kernelSize3D = kernelSize2D * mInputDepth;

    // will be using either original input width/heigh or padded
    size_t  inputWidth   = mInputWidth;
    size_t  inputHeight  = mInputHeight;

    // decide if raw data to be used or padded
    if ( mBorderMode == BorderMode::Same )
    {
        InputPadding( inputs, mPaddedInputs );
        //inputData    = mPaddedInput.data( );
        inputWidth  += mPadWidth;
        inputHeight += mPadHeight;
    }

    size_t  inputRowInc     = inputWidth * mVerticalStep;
    // gap size after processing one input row with kernel to the next row to be processed
    size_t  inputNextRowGap = inputWidth - mKernelWidth;

    // process all samples
    for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
    {
        const float_t* inputData  = inputs[i]->data( );
        float_t*       outputData = outputs[i]->data( );

        if ( mBorderMode == BorderMode::Same )
        {
            inputData = mPaddedInputs[i].data( );
        }

        // clear the output
        fill( outputData, outputData + mOutputsCount, float_t( 0 ) );

        // go through all kernels to build output feature maps
        for ( size_t kernelIndex = 0; kernelIndex < mKernelsCount; kernelIndex++ )
        {
            float_t* outputBase = outputData + kernelIndex * mOutputWidth * mOutputHeight;
            float_t  biasValue = mKernelsBiases[kernelIndex];

            // go through all input layers (or feature maps produced by previous layers)
            for ( size_t inputDepthIndex = 0; inputDepthIndex < mInputDepth; inputDepthIndex++ )
            {
                const float_t* inputBase = inputData + inputDepthIndex * inputWidth * inputHeight;
                // get the 2D kernel for current input/output map combination
                const float_t* kernelBase = kernelsData + kernelIndex * kernelSize3D + inputDepthIndex * kernelSize2D;

                // calculate output contributions for the current input map
                for ( size_t oy = 0; oy < mOutputHeight; oy++ )
                {
                    const float_t* inputRow = inputBase + oy * inputRowInc;
                    float_t*       outputRow = outputBase + oy * mOutputWidth;

                    for ( size_t ox = 0; ox < mOutputWidth; ox++ )
                    {
                        const float_t* kernelPtr = kernelBase;
                        const float_t* inputPtr  = inputRow;
                        float_t        sum       = float_t( 0 );

                        // "convolve" input with the kernel
                        // (we actually do cross-correlation since it does not matter for CNN)
                        for ( size_t ky = 0; ky < mKernelHeight; ky++ )
                        {
                            for ( size_t kx = 0; kx < mKernelWidth; kx++ )
                            {
                                sum += *inputPtr * *kernelPtr;
                                kernelPtr++;
                                inputPtr++;
                            }
                            // since input pointer was already shifted horizontally,
                            // we need to align it back to the start of the next row
                            inputPtr += inputNextRowGap;
                        }

                        *outputRow += sum;
                        *outputRow += biasValue;

                        // shift output/input row pointers to the next position of the sliding kernel's window
                        outputRow++;
                        inputRow += mHorizontalStep;
                    }
                }
            }
        }
    }
}

// Propagates error to the previous layer and calculates weights/biases gradients
void XConvolutionLayer::BackwardCompute( const vector<vector_t*>& inputs,
                                         const vector<vector_t*>& /* outputs */,
                                         const vector<vector_t*>& deltas,
                                         vector<vector_t*>& prevDeltas,
                                         vector_t& gradWeights,
                                         vector_t& gradBiases )
{
    const float* kernelsData     = mKernelsWeights.data( );
    //const float* deltaData       = delta.data( );
    //const float* inputData       = input.data( );
    //float*       prevDeltaData   = prevDelta.data( );
    float*       gradWeightsData = gradWeights.data( );

    size_t       outputSize      = mOutputWidth * mOutputHeight;

    // kernel size if 2D and 3D spaces
    size_t       kernelSize2D  = mKernelWidth * mKernelHeight;
    size_t       kernelSize3D  = kernelSize2D * mInputDepth;

    // will be using either original input width/heigh or padded
    size_t       inputWidth    = mInputWidth;
    size_t       inputHeight   = mInputHeight;

    // decide if raw data to be used or padded
    if ( mBorderMode == BorderMode::Same )
    {
        //prevDeltaData = mPaddedPrevDeltas.data( );
        //inputData     = mPaddedInput.data( );
        inputWidth   += mPadWidth;
        inputHeight  += mPadHeight;

        //fill( mPaddedPrevDeltas.begin( ), mPaddedPrevDeltas.end( ), 0.0f );
    }
    else
    {
        //fill( prevDelta.begin( ), prevDelta.end( ), 0.0f );
    }

    size_t inputRowInc         = inputWidth * mVerticalStep;
    // gap size after processing one row of previous deltas to the next row to be processed
    size_t prevDeltaNextRowGap = inputWidth - mKernelWidth;
    
    // 1 - first propagate deltas to the previous layer
    for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
    {
        const float_t* deltaData     = deltas[i]->data( );
        const float_t* inputData     = inputs[i]->data( );
        float_t*       prevDeltaData = prevDeltas[i]->data( );

        if ( mBorderMode == BorderMode::Same )
        {
            prevDeltaData = mPaddedPrevDeltas[i].data( );
            inputData     = mPaddedInputs[i].data( );
        }

        fill( prevDeltaData, prevDeltaData + mInputsCount, float_t( 0 ) );

        // go through all input feature maps (which are the outputs of the previous layer)
        for ( size_t inputDepthIndex = 0; inputDepthIndex < mInputDepth; inputDepthIndex++ )
        {
            float_t* prevDeltaBase = prevDeltaData + inputDepthIndex * inputWidth * inputHeight;

            // go through all kernels, which were applied to the feature map
            for ( size_t kernelIndex = 0; kernelIndex < mKernelsCount; kernelIndex++ )
            {
                const float_t* deltaBase = deltaData + kernelIndex * outputSize;
                // get the 2D kernel for then current input/output map combination
                const float_t* kernelBase = kernelsData + kernelIndex * kernelSize3D + inputDepthIndex * kernelSize2D;

                // go through the current deltas of the output produced by the current kernel
                for ( size_t oy = 0; oy < mOutputHeight; oy++ )
                {
                    const float_t* deltaPtr     = deltaBase     + oy * mOutputWidth;
                    float_t*       prevDeltaRow = prevDeltaBase + oy * inputRowInc;

                    for ( size_t ox = 0; ox < mOutputWidth; ox++ )
                    {
                        const float_t* kernelPtr    = kernelBase;
                        float_t*       prevDeltaPtr = prevDeltaRow;

                        // go through the kernel at current image position
                        for ( size_t ky = 0; ky < mKernelHeight; ky++ )
                        {
                            for ( size_t kx = 0; kx < mKernelWidth; kx++ )
                            {
                                *prevDeltaPtr += *deltaPtr * *kernelPtr;

                                kernelPtr++;
                                prevDeltaPtr++;
                            }
                            // since previous delta pointer was already shifted horizontally,
                            // we need to align it back to the start of the next row
                            prevDeltaPtr += prevDeltaNextRowGap;
                        }

                        // shift current/previous delta pointers to the next position of the sliding kernel's window
                        deltaPtr++;
                        prevDeltaRow += mHorizontalStep;
                    }
                }
            }
        }
    }

    // 2 - accumulate weights' difference

    // go through all input feature maps
    for ( size_t inputDepthIndex = 0; inputDepthIndex < mInputDepth; inputDepthIndex++ )
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            const float_t* deltaData = deltas[i]->data( );
            const float_t* inputData = inputs[i]->data( );

            if ( mBorderMode == BorderMode::Same )
            {
                inputData = mPaddedInputs[i].data( );
            }

            const float_t* inputBase = inputData + inputDepthIndex * inputWidth * inputHeight;

            // go through all kernels, which were applied to the feature map
            for ( size_t kernelIndex = 0; kernelIndex < mKernelsCount; kernelIndex++ )
            {
                const float_t* deltaBase = deltaData + kernelIndex * outputSize;
                // get the 2D portion of weights' gradients for the current input/output map combination
                float_t* gradWeightsPtr  = gradWeightsData + kernelIndex * kernelSize3D + inputDepthIndex * kernelSize2D;

                // calculate gradients for each weight (kernel element)
                for ( size_t ky = 0; ky < mKernelHeight; ky++ )
                {
                    for ( size_t kx = 0; kx < mKernelWidth; kx++ )
                    {
                        float_t sum = float_t( 0 );

                        // multiply output deltas by corresponding inputs
                        for ( size_t oy = 0; oy < mOutputHeight; oy++ )
                        {
                            const float_t* deltaPtr = deltaBase + oy * mOutputWidth;
                            const float_t* inputPtr = inputBase + oy * inputRowInc + ky * inputWidth + kx;

                            for ( size_t ox = 0; ox < mOutputWidth; ox++ )
                            {
                                sum += *deltaPtr * *inputPtr;

                                deltaPtr++;
                                inputPtr += mHorizontalStep;
                            }
                        }

                        *gradWeightsPtr += sum;
                        gradWeightsPtr++;
                    }
                }
            }
        }
    }

    // 3 - accumulate baises' difference
    for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
    {
        const float* deltaPtr = deltas[i]->data( );

        // go through all kernels
        for ( size_t kernelIndex = 0; kernelIndex < mKernelsCount; kernelIndex++ )
        {
            float sum = 0.0f;

            for ( size_t outputIndex = 0; outputIndex < outputSize; outputIndex++ )
            {
                sum += *deltaPtr;
                deltaPtr++;
            }

            gradBiases[kernelIndex] += sum;
        }
    }

    // do unpadding of previous deltas
    if ( mBorderMode == BorderMode::Same )
    {
        DeltasUnpadding( mPaddedPrevDeltas, prevDeltas );
    }
}

// Applies updates to the layer's weights and biases
void XConvolutionLayer::UpdateWeights( const vector_t& weightsUpdate,
                                       const vector_t& biasesUpdate )
{
    for ( size_t i = 0, n = mKernelsWeights.size( ); i < n; i++ )
    {
        mKernelsWeights[i] += weightsUpdate[i];
    }
    for ( size_t i = 0, n = mKernelsBiases.size( ); i < n; i++ )
    {
        mKernelsBiases[i] += biasesUpdate[i];
    }
}

void XConvolutionLayer::InputPadding( const std::vector<vector_t*>& input, std::vector<vector_t>& padded )
{
    // For odd size kernels, padding is distributed equally on each side.
    // However for even size kernels, padding goes first to right/bottom sides.
    size_t lefPad      = ( mKernelWidth  - 1 ) / 2;
    size_t rightPad    = ( mKernelWidth  - 1 ) - lefPad;
    size_t topPad      = ( mKernelHeight - 1 ) / 2;
    size_t bottomPad   = ( mKernelHeight - 1 ) - topPad;
    size_t paddedWidth = mInputWidth + mPadWidth;

    topPad    *= paddedWidth;
    bottomPad *= paddedWidth;

    size_t paddedInputSize = ( mInputWidth + mPadWidth ) * ( mInputHeight + mPadHeight ) * mInputDepth;

    // make sure padded vector has enough space
    if ( ( padded.size( ) == 0 ) ||
         ( padded.size( ) != input.size( ) ) ||
         ( padded[0].size( ) != paddedInputSize ) )
    {
        padded.resize( input.size( ) );

        for ( size_t i = 0, n = input.size( ); i < n; i++ )
        {
            padded[i] = vector_t( paddedInputSize );
        }
    }

    for ( size_t i = 0, n = input.size( ); i < n; i++ )
    {
        const float* ptrInput  = input[i]->data( );
        float*       ptrPadded = padded[i].data( );

        for ( size_t d = 0; d < mInputDepth; d++ )
        {
            ptrPadded += topPad;

            for ( size_t y = 0; y < mInputHeight; y++ )
            {
                ptrPadded += lefPad;

                for ( size_t x = 0; x < mInputWidth; x++, ptrInput++, ptrPadded++ )
                {
                    *ptrPadded = *ptrInput;
                }

                ptrPadded += rightPad;
            }

            ptrPadded += bottomPad;
        }
    }
}

void XConvolutionLayer::DeltasUnpadding( std::vector<vector_t>& deltas, std::vector<vector_t*>& unpadded )
{
    size_t lefPad      = ( mKernelWidth  - 1 ) / 2;
    size_t rightPad    = ( mKernelWidth  - 1 ) - lefPad;
    size_t topPad      = ( mKernelHeight - 1 ) / 2;
    size_t bottomPad   = ( mKernelHeight - 1 ) - topPad;
    size_t paddedWidth = mInputWidth + mPadWidth;

    topPad    *= paddedWidth;
    bottomPad *= paddedWidth;

    // make sure unpadded vector has enough space
    if ( ( unpadded.size( ) == deltas.size( ) ) &&
         ( unpadded[0]->size( ) == mInputsCount ) )
    {
        for ( size_t i = 0, n = deltas.size( ); i < n; i++ )
        {
            const float_t* ptrPadded = deltas[i].data( );
            float_t*       ptrUnpadded = unpadded[i]->data( );

            for ( size_t d = 0; d < mInputDepth; d++ )
            {
                ptrPadded += topPad;

                for ( size_t y = 0; y < mInputHeight; y++ )
                {
                    ptrPadded += lefPad;

                    for ( size_t x = 0; x < mInputWidth; x++, ptrPadded++, ptrUnpadded++ )
                    {
                        *ptrUnpadded = *ptrPadded;
                    }

                    ptrPadded += rightPad;
                }

                ptrPadded += bottomPad;
            }
        }
    }
}

} } // namespace ANNT::Neuro
