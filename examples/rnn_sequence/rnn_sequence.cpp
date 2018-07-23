/*
    ANNT - Artificial Neural Networks C++ library

    Sequence prediction with Recurrent ANN

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

#if defined(_MSC_VER) && defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <vector>

#include "ANNT.hpp"

using namespace std;
using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

// Number of steps in sequnces used in this example
#define STEPS_PER_SEQUENCE (10)

// Helper function to show target sequence and the one predicted with the specified network
void ShowPredictedSequences( shared_ptr<XNeuralNetwork>& net, const vector<fvector_t>& inputs, const vector<fvector_t>& outputs, size_t sequenceCount )
{
    XNetworkInference netInference( net );
    fvector_t         output( 10 );

    for ( size_t i = 0; i < sequenceCount; i++ )
    {
        string targetSequence;
        string producedSequence;

        for ( size_t step = 0; step < STEPS_PER_SEQUENCE; step++ )
        {
            size_t sampleIndex = i * STEPS_PER_SEQUENCE + step;

            if ( step == 0 )
            {
                targetSequence   += static_cast<char>( XDataEncodingTools::MaxIndex( inputs[sampleIndex] ) + '0' );
                producedSequence += targetSequence[0];
            }

            netInference.Compute( inputs[sampleIndex], output );

            targetSequence   += static_cast<char>( XDataEncodingTools::MaxIndex( outputs[sampleIndex] ) + '0' );
            producedSequence += static_cast<char>( XDataEncodingTools::MaxIndex( output ) + '0' );
        }

        netInference.ResetState( );

        printf( "Target sequence:   %s \n", targetSequence.c_str( ) );
        printf( "Produced sequence: %s ", producedSequence.c_str( ) );
        printf( "%s \n", ( targetSequence == producedSequence ) ? "Good" : "Bad" );
        printf( "\n" );
    }
}

// Example application's entry point
int main( int /* argc */, char** /* argv */ )
{
#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemState memState;
    _CrtMemCheckpoint( &memState );
#endif

    printf( "Sequence prediction with Recurrent ANN \n\n" );

    //_CrtSetBreakAlloc( 159 );

    {
        // 10 sequences to train
        vector<uvector_t> sequences(
        {
            uvector_t( { 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1 } ),
            uvector_t( { 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 2 } ),
            uvector_t( { 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 3 } ),
            uvector_t( { 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4 } ),
            uvector_t( { 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5 } ),

            uvector_t( { 6, 0, 1, 2, 4, 4, 4, 6, 7, 8, 6 } ),
            uvector_t( { 7, 0, 1, 2, 4, 4, 4, 6, 7, 8, 7 } ),
            uvector_t( { 8, 0, 1, 2, 4, 4, 4, 6, 7, 8, 8 } ),
            uvector_t( { 9, 0, 1, 2, 4, 4, 4, 6, 7, 8, 9 } ),
            uvector_t( { 0, 0, 1, 2, 4, 4, 4, 6, 7, 8, 0 } )
        } );

        // make sure all sequences have valid number of steps/transitions,
        // which means number of values is greater by 1
        for ( size_t i = 0, n = sequences.size( ); i < n; i++ )
        {
            assert( sequences[i].size( ) == STEPS_PER_SEQUENCE + 1 );
        }

        vector<fvector_t> inputs;
        vector<fvector_t> outputs;

        // Create inputs/outputs from above sequences using one-hot encoding.
        for ( size_t i = 0, n = sequences.size( ); i < n; i++ )
        {
            for ( size_t j = 0; j <= STEPS_PER_SEQUENCE; j++ )
            {
                fvector_t oneHotEncoded = XDataEncodingTools::OneHotEncoding( sequences[i][j], 10 );

                // inputs get all elements of the sequence except the last one
                if ( j != STEPS_PER_SEQUENCE )
                {
                    inputs.push_back( oneHotEncoded );
                }
                // output get all except the first one
                if ( j != 0 )
                {
                    outputs.push_back( oneHotEncoded );
                }
            }
        }

        // prepare a recurrent ANN
        shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

        // basic recurrent network
        net->AddLayer( make_shared<XRecurrentLayer>( 10, 20 ) );
        net->AddLayer( make_shared<XRecurrentLayer>( 20, 30 ) );
        net->AddLayer( make_shared<XFullyConnectedLayer>( 30, 10 ) );
        net->AddLayer( make_shared<XSoftMaxActivation>( ) );

        // similar result using LSTM layer - one recurrent layer is enough as it is more complex
        /*
        net->AddLayer( make_shared<XLSTMLayer>( 10, 20 ) );
        net->AddLayer( make_shared<XFullyConnectedLayer>( 20, 10 ) );
        net->AddLayer( make_shared<XSoftMaxActivation>( ) );
        */

        // or same can be done using GRU layer
        /*
        net->AddLayer( make_shared<XGRULayer>( 10, 20 ) );
        net->AddLayer( make_shared<XFullyConnectedLayer>( 20, 10 ) );
        net->AddLayer( make_shared<XSoftMaxActivation>( ) );
        */

        // create training context with Adam optimizer and Cross Entropy cost function
        XNetworkTraining netTraining( net,
                                      make_shared<XAdamOptimizer>( 0.05f ),
                                      make_shared<XCrossEntropyCost>( ) );

        netTraining.SetAverageWeightGradients( false );
        // since we are dealing with recurrent network, we need to tell trainer the length of time series
        netTraining.SetTrainingSequenceLength( STEPS_PER_SEQUENCE );

        // show target and predicted sequnces before training
        printf( "Before training: \n" );
        ShowPredictedSequences( net, inputs, outputs, sequences.size( ) );

        // run training epochs providing all data as single batch
        for ( size_t i = 1; i <= 100; i++ )
        {
            auto cost = netTraining.TrainBatch( inputs, outputs );
            printf( "%0.4f ", static_cast<float>( cost ) );

            // we can reset individual layers, however just reset all so we don't need to think about
            // layers' indexes and network structure (we expect to have only recurrent layers with state
            // for this sample)
            netTraining.ResetState( );
            // or reset only recurrent layers
            // netTraining.ResetLayersState( { 0, 1 } ); // for basic recurrent
            // netTraining.ResetLayersState( { 0 } );    // for LSTM and GRU

            if ( ( i % 10 ) == 0 )
            {
                printf( "\n" );
            }
        }
        printf( "\n\n" );

        // show target and predicted sequnces after training completes
        printf( "After training: \n" );
        ShowPredictedSequences( net, inputs, outputs, sequences.size( ) );
    }

#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemDumpAllObjectsSince( &memState );
#endif

    return 0;
}
