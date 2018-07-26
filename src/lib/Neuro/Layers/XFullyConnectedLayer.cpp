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

#include "XFullyConnectedLayer.hpp"
#include "../../Tools/XParallel.hpp"
#include "../../Tools/XVectorize.hpp"

using namespace std;

namespace ANNT { namespace Neuro {

XFullyConnectedLayer::XFullyConnectedLayer( size_t inputsCount, size_t outputsCount ) :
    ITrainableLayer( inputsCount, outputsCount ),
    mAllWeights( inputsCount * outputsCount + outputsCount )
{
    // set up weights/biases pointers
    mWeights = mAllWeights.data( );
    mBiases  = mWeights + mInputsCount * mOutputsCount;

    Randomize( );
}

// Randomizes layer's weights, clears biases
void XFullyConnectedLayer::Randomize( )
{
    float_t halfRange = sqrt( float_t( 3 ) / mInputsCount );

    for ( size_t i = 0, n = mInputsCount * mOutputsCount; i < n; i++ )
    {
        mWeights[i] = ( static_cast<float_t>( rand( ) ) / RAND_MAX ) * ( float_t( 2 ) * halfRange ) - halfRange;
    }
    for ( size_t i = 0; i < mOutputsCount; i++ )
    {
        mBiases[i] = 0;
    }
}

// Calculates outputs for the given inputs
void XFullyConnectedLayer::ForwardCompute( const vector<fvector_t*>& inputs,
                                           vector<fvector_t*>& outputs,
                                           const XNetworkContext& ctx )
{
    XParallel::For( inputs.size( ), ctx.IsTraining( ), [&]( size_t i )
    {
        const float_t* weights = mWeights;
        const float_t* input   = inputs[i]->data( );
        fvector_t&     output  = *( outputs[i] );

        for ( size_t otputIndex = 0; otputIndex < mOutputsCount; otputIndex++ )
        {
            output[otputIndex] = XVectorize::Dot( input, weights, mInputsCount ) + mBiases[otputIndex];

            weights += mInputsCount;
        }
    } );
}

// Propagates error to the previous layer and calculates weights/biases gradients
void XFullyConnectedLayer::BackwardCompute( const vector<fvector_t*>& inputs,
                                            const vector<fvector_t*>& /* outputs */,
                                            const vector<fvector_t*>& deltas,
                                            vector<fvector_t*>& prevDeltas,
                                            fvector_t& gradWeights,
                                            const XNetworkContext& ctx )
{
    // set up weights/biases gradients pointers
    float_t*  gradWeightsData = gradWeights.data( );
    float_t*  gradBiasesData  = gradWeightsData + mInputsCount * mOutputsCount;

    // 1 - first propagate deltas to the previous layer
    XParallel::For( inputs.size( ), ctx.IsTraining( ), [&]( size_t i )
    {
        fvector_t&       prevDelta = *( prevDeltas[i] );
        const fvector_t& delta     = *( deltas[i] );

        for ( size_t inputIndex = 0; inputIndex < mInputsCount; inputIndex++ )
        {
            size_t  weightIndex = inputIndex;
            float_t sum         = 0;

            for ( size_t otputIndex = 0; otputIndex < mOutputsCount; otputIndex++, weightIndex += mInputsCount )
            {
                sum += delta[otputIndex] * mWeights[weightIndex];
            }

            prevDelta[inputIndex] = sum;
        }
    } );

    // 2 - accumulate weights' difference
    XParallel::For( mOutputsCount, ctx.IsTraining( ), [&]( size_t outputIndex )
    {
        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            const fvector_t& input      = *( inputs[i] );
            float_t          deltaValue = ( *( deltas[i] ) )[outputIndex];

            for ( size_t inputIndex = 0, weightIndex = outputIndex * mInputsCount; inputIndex < mInputsCount; inputIndex++, weightIndex++ )
            {
                gradWeightsData[weightIndex] += deltaValue * input[inputIndex];
            }
        }
    } );

    // 3 - accumulate baises' difference
    for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
    {
        const fvector_t& delta = *( deltas[i] );

        for ( size_t outputIndex = 0; outputIndex < mOutputsCount; outputIndex++ )
        {
            gradBiasesData[outputIndex] += delta[outputIndex];
        }
    }
}

// Applies updates to the layer's weights and biases
void XFullyConnectedLayer::UpdateWeights( const fvector_t& updates )
{
    for ( size_t i = 0, n = mAllWeights.size( ); i < n; i++ )
    {
        mAllWeights[i] += updates[i];
    }
}

// Saves layer's learnt parameters/weights
bool XFullyConnectedLayer::SaveLearnedParams( FILE* file ) const
{
    vector<const fvector_t*> params( { &mAllWeights } );

    return SaveLearnedParamsHelper( file, LayerID::FullyConnected, params );
}

// Loads layer's learnt parameters
bool XFullyConnectedLayer::LoadLearnedParams( FILE* file )
{
    vector<fvector_t*> params( { &mAllWeights } );

    return LoadLearnedParamsHelper( file, LayerID::FullyConnected, params );
}

} } // namespace ANNT::Neuro
