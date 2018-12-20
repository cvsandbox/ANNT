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

// Number of steps in sequences used in this example
#define STEPS_PER_SEQUENCE (10)

// Types of supported recurrent layers
enum class RecurrentLayerType
{
    Basic = 0,
    LSTM  = 1,
    GRU   = 2
};

// Some training parameters for this example application
typedef struct TrainingParamsStruct
{
    float              LearningRate;
    size_t             EpochsCount;
    RecurrentLayerType RecurrrentType;

    TrainingParamsStruct( ) :
        LearningRate( 0.01f ), EpochsCount( 150 ), RecurrrentType( RecurrentLayerType::Basic )
    {
    }
}
TrainingParams;

// Parse command line to get some of the training parameters
static void ParseCommandLine( int argc, char** argv, TrainingParams* trainingParams )
{
    bool showUsage = false;

    if ( argv == nullptr )
    {
        return;
    }

    for ( int i = 1; i < argc; i++ )
    {
        bool   parsed = false;
        size_t paramLen = strlen( argv[i] );

        if ( paramLen >= 2 )
        {
            char* paramStart = &( argv[i][1] );

            if ( ( argv[i][0] == '-' ) || ( argv[i][0] == '/' ) )
            {
                if ( ( strstr( paramStart, "ec:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    if ( sscanf( &( argv[i][4] ), "%zu", &trainingParams->EpochsCount ) == 1 )
                    {
                        parsed = true;
                    }
                }
                else if ( ( strstr( paramStart, "lr:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    if ( sscanf( &( argv[i][4] ), "%f", &trainingParams->LearningRate ) == 1 )
                    {
                        parsed = true;
                    }
                }
                else if ( ( strstr( paramStart, "type:" ) == paramStart ) && ( paramLen == 7 ) )
                {
                    int layerType = argv[i][6] - '0';

                    if ( ( layerType >= 0 ) && ( layerType <= 2 ) )
                    {
                        trainingParams->RecurrrentType = static_cast<RecurrentLayerType>( layerType );
                        parsed = true;
                    }
                }
            }
        }

        if ( !parsed )
        {
            showUsage = true;
        }
    }

    if ( showUsage )
    {
        printf( "Failed parsing some of the parameters \n\n" );

        printf( "Available parameters are:\n" );
        printf( "  -ec:<> - epochs count; \n" );
        printf( "  -lr:<> - learning rate; \n" );
        printf( "  -type:<> - recurrent layer type: \n" );
        printf( "                 0 - basic ( default ); \n" );
        printf( "                 1 - LSTM; \n" );
        printf( "                 2 - GRU. \n" );
        printf( "\n" );
    }
}

// Helper function to show target sequence and the one predicted with the specified network
static void ShowPredictedSequences( shared_ptr<XNeuralNetwork>& net, const vector<fvector_t>& inputs, const vector<fvector_t>& outputs, size_t sequenceCount )
{
    XNetworkInference netInference( net );
    fvector_t         input( 10 );
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

                // get the first sample of the sequence
                input = inputs[sampleIndex];
            }

            netInference.Compute( input, output );

            targetSequence   += static_cast<char>( XDataEncodingTools::MaxIndex( outputs[sampleIndex] ) + '0' );
            producedSequence += static_cast<char>( XDataEncodingTools::MaxIndex( output ) + '0' );

            // use network's output as the next input
            input = output;
        }

        netInference.ResetState( );

        printf( "Target sequence:   %s \n", targetSequence.c_str( ) );
        printf( "Produced sequence: %s ", producedSequence.c_str( ) );
        printf( "%s \n", ( targetSequence == producedSequence ) ? "Good" : "Bad" );
        printf( "\n" );
    }
}

// Example application's entry point
int main( int argc, char** argv )
{
#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemState memState;
    _CrtMemCheckpoint( &memState );
#endif

    printf( "Sequence prediction with Recurrent ANN \n\n" );

    //_CrtSetBreakAlloc( 159 );

    {
        TrainingParams trainingParams;

        // check if any of the defaults are overridden
        ParseCommandLine( argc, argv, &trainingParams );

        printf( "Learning rate  : %0.4f \n", trainingParams.LearningRate );
        printf( "Epochs count   : %zu \n", trainingParams.EpochsCount );
        printf( "Recurrent type : %s \n", ( trainingParams.RecurrrentType == RecurrentLayerType::GRU ) ? "GRU" :
                                          ( trainingParams.RecurrrentType == RecurrentLayerType::LSTM ) ? "LSTM" : "basic" );
        printf( "\n" );

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
        switch ( trainingParams.RecurrrentType )
        {
        case RecurrentLayerType::Basic:
        default:
            net->AddLayer( make_shared<XRecurrentLayer>( 10, 20 ) );
            break;

        case RecurrentLayerType::LSTM:
            net->AddLayer( make_shared<XLSTMLayer>( 10, 20 ) );
            break;

        case RecurrentLayerType::GRU:
            net->AddLayer( make_shared<XGRULayer>( 10, 20 ) );
            break;
        }

        // complete the network with fully connecte layer and soft max activation
        net->AddLayer( make_shared<XFullyConnectedLayer>( 20, 10 ) );
        net->AddLayer( make_shared<XSoftMaxActivation>( ) );

        // create training context with Adam optimizer and Cross Entropy cost function
        XNetworkTraining netTraining( net,
                                      make_shared<XAdamOptimizer>( trainingParams.LearningRate ),
                                      make_shared<XCrossEntropyCost>( ) );

        netTraining.SetAverageWeightGradients( false );
        // since we are dealing with recurrent network, we need to tell trainer the length of time series
        netTraining.SetTrainingSequenceLength( STEPS_PER_SEQUENCE );

        // show target and predicted sequnces before training
        printf( "Before training: \n" );
        ShowPredictedSequences( net, inputs, outputs, sequences.size( ) );

        // run training epochs providing all data as single batch
        for ( size_t i = 1; i <= trainingParams.EpochsCount; i++ )
        {
            auto cost = netTraining.TrainBatch( inputs, outputs );
            printf( "%0.4f ", static_cast<float>( cost ) );

            // we can reset individual layers, however just reset all so we don't need to think about
            // layers' indexes and network structure (we expect to have only recurrent layers with state
            // for this sample)
            netTraining.ResetState( );
            // or reset only recurrent layers
            // netTraining.ResetLayersState( { 0 } );

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
