/*
    ANNT - Artificial Neural Networks C++ library

    XOR example with Fully Connected ANN

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

#include <stdio.h>
#include <vector>

#include "ANNT.hpp"

// If defined, batch training is used - all samples are given to network before updating it.
// If commented, on-line training is done - samples are provided randomly one by one.
//#define USE_BATCH_TRAINING

using namespace std;
using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

// Helper function to print values of the vector
void PrintVector( const fvector_t& vec )
{
    printf( "{ " );
    for ( size_t i = 0; i < vec.size( ); i++ )
    {
        printf( "%5.2f ", static_cast<float>( vec[i] ) );
    }
    printf( "}" );
}

// Helper function to print specified inputs and corresponding computed outputs
void TestNetwork( const shared_ptr<XNeuralNetwork>& net, const vector<fvector_t>& inputs )
{
    XNetworkInference netInference( net );
    fvector_t         output( net->OutputsCount( ) );

    for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
    {
        netInference.Compute( inputs[i], output );

        PrintVector( inputs[i] );
        printf( " -> " );
        PrintVector( output );
        printf( "\n" );
    }

    printf( "\n" );
}

// Example application's entry point
int main( int /* argc */, char** /* argv */ )
{
    printf( "XOR example with Fully Connected ANN \n\n" );

    // prepare XOR training data, inputs encoded as -1 and 1, while outputs as 0, 1
    vector<fvector_t> inputs;
    vector<fvector_t> targetOutputs;

    inputs.push_back( { -1.0f, -1.0f } ); /* -> */ targetOutputs.push_back( { 0.0f } );
    inputs.push_back( {  1.0f, -1.0f } ); /* -> */ targetOutputs.push_back( { 1.0f } );
    inputs.push_back( { -1.0f,  1.0f } ); /* -> */ targetOutputs.push_back( { 1.0f } );
    inputs.push_back( {  1.0f,  1.0f } ); /* -> */ targetOutputs.push_back( { 0.0f } );

    // Prepare 2 layer ANN.
    // A single layer/neuron is enough for AND or OR functions, but XOR needs two layers.
    shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

    net->AddLayer( make_shared<XFullyConnectedLayer>( 2, 2 ) );
    net->AddLayer( make_shared<XTanhActivation>( ) );
    net->AddLayer( make_shared<XFullyConnectedLayer>( 2, 1 ) );
    net->AddLayer( make_shared<XSigmoidActivation>( ) );

    // create training context with Nesterov optimizer and Binary Cross Entropy cost function
    XNetworkTraining netTraining( net,
                                  make_shared<XNesterovMomentumOptimizer>( 0.1f ),
                                  make_shared<XBinaryCrossEntropyCost>( ) );

    // average weight/bias gradients over batch
    netTraining.SetAverageWeightGradients( true );

    printf( "Network output before training: \n" );
    TestNetwork( net, inputs );

    // train the neural network
#ifndef USE_BATCH_TRAINING
    printf( "Cost of each sample: \n" );
    for ( size_t i = 0; i < 80 * 2; i++ )
    {
        size_t sample = rand( ) % inputs.size( );
        auto   cost   = netTraining.TrainSample( inputs[sample], targetOutputs[sample] );

        printf( "%.04f ", static_cast<float>( cost ) );

        if ( ( i % 8 ) == 7 )
            printf( "\n" );
    }
#else
    printf( "Cost of each batch: \n" );
    for ( size_t i = 0; i < 80; i++ )
    {
        auto cost = netTraining.TrainBatch( inputs, targetOutputs );

        printf( "%.04f ", cost );

        if ( ( i % 8 ) == 7 )
            printf( "\n" );
    }
#endif
    printf( "\n" );

    printf( "Network output after training: \n" );
    TestNetwork( net, inputs );

    return 0;
}
