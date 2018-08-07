/*
    ANNT - Artificial Neural Networks C++ library

    Regression example with Fully Connected ANN

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
#include <string.h>
#include <vector>

#include "ANNT.hpp"

using namespace std;
using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

// Some training parameters for this example application
typedef struct TrainingParamsStruct
{
    string         InputDataFile;
    string         OutputDataFile;
    vector<size_t> HiddenLayers;
    float          LearningRate;
    size_t         EpochsCount;
    size_t         BatchSize;

    TrainingParamsStruct( ) :
        InputDataFile( "data/parabola.csv" ), OutputDataFile( "data/parabola-out.csv" ),
        HiddenLayers( { 10 } ), LearningRate( 0.01f ), EpochsCount( 1000 ), BatchSize( 10 )

        // Some of the network structures to try with provided training sample data
        // line     : HiddenLayers( { } )
        // parabola : HiddenLayers( { 10 } )
        // sine     : HiddenLayers( { 20 } )
        // sine-inc : HiddenLayers( { 20 } )
    {
    }
}
TrainingParams;

// Parse command line to get some of the training parameters
static void ParseCommandLine( int argc, char** argv, TrainingParams* trainingParams )
{
    bool showUsage           = false;
    bool outputFileSpecified = false;

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
                else if ( ( strstr( paramStart, "bs:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    if ( sscanf( &( argv[i][4] ), "%zu", &trainingParams->BatchSize ) == 1 )
                    {
                        if ( trainingParams->BatchSize == 0 )
                        {
                            trainingParams->BatchSize = 1;
                        }
                        parsed = true;
                    }
                }
                else if ( ( strstr( paramStart, "hn:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    parsed = true;
                    trainingParams->HiddenLayers.clear( );

                    if ( ( paramLen != 5 ) || ( argv[i][4] != '0' ) )
                    {
                        size_t neuronsCount;
                        char*  parsePtr = &( argv[i][4] );

                        for ( ; ; )
                        {
                            if ( ( sscanf( parsePtr, "%zu", &neuronsCount ) == 1 ) && ( neuronsCount != 0 ) )
                            {
                                trainingParams->HiddenLayers.push_back( neuronsCount );

                                parsePtr = strchr( parsePtr, ':' );

                                if ( parsePtr != nullptr )
                                {
                                    parsePtr++;
                                }
                                else
                                {
                                    break;
                                }
                            }
                            else
                            {
                                parsed = false;
                                break;
                            }
                        }
                    }
                }
                else if ( ( strstr( paramStart, "in:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    trainingParams->InputDataFile = string( &( argv[i][4] ) );

                    if ( !outputFileSpecified )
                    {
                        size_t dotPos =  trainingParams->InputDataFile.rfind( '.' );

                        if ( dotPos == string::npos )
                        {
                            trainingParams->OutputDataFile = trainingParams->InputDataFile + "-out";
                        }
                        else
                        {
                            trainingParams->OutputDataFile = trainingParams->InputDataFile.substr( 0, dotPos ) + "-out" + trainingParams->InputDataFile.substr( dotPos );
                        }
                    }

                    parsed = true;
                }
                else if ( ( strstr( paramStart, "out:" ) == paramStart ) && ( paramLen > 5 ) )
                {
                    trainingParams->OutputDataFile = string( &( argv[i][5] ) );
                    outputFileSpecified = true;
                    parsed = true;
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
        printf( "  -bs:<> - batch size; \n" );
        printf( "  -lr:<> - learning rate; \n" );
        printf( "  -hn:<X[:X]> - number of neurons in hidden layers; examples: \n" );
        printf( "           0 - no hidden layers; \n" );
        printf( "           10 - single hidden layer with 10 neurons; \n" );
        printf( "           20:10 - two hidden layers - 20 neurons in the first and 10 in the second; \n" );
        printf( "  -in:<> - file name to read input training data from; \n" );
        printf( "  -out:<> - file name to write predicted results to. \n" );
        printf( "\n" );
    }
}

// Load data from the specified file
bool LoadData( const string& fileName, vector<fvector_t>& inputs, vector<fvector_t>& outputs, vector<fvector_t>& noisyOutputs  )
{
    bool  ret  = false;
    FILE* file = fopen( fileName.c_str( ), "r" );

    if ( file )
    {
        char buff[256];

        inputs.clear( );
        outputs.clear( );
        noisyOutputs.clear( );

        while ( fgets( buff, 256, file ) != nullptr )
        {
            float x, y, yNoisy;

            if ( sscanf( buff, "%f,%f,%f", &x, &y, &yNoisy ) == 3 )
            {
                inputs.push_back( fvector_t( { x } ) );
                outputs.push_back( fvector_t( { y } ) );
                noisyOutputs.push_back( fvector_t( { yNoisy } ) );
            }
        }

        fclose( file );
        ret = ( !inputs.empty( ) );
    }

    return ret;
}

// Save data to the specified file - original data plus outputs predicted by the network
bool SaveData( const string& fileName, const vector<fvector_t>& inputs, const vector<fvector_t>& outputs,
                                       const vector<fvector_t>& noisyOutputs, const vector<fvector_t> networkOutputs )
{
    bool  ret  = false;
    FILE* file = fopen( fileName.c_str( ), "w" );

    if ( file )
    {
        for ( size_t i = 0; i < inputs.size( ); i++ )
        {
            fprintf( file, "%f,%f,%f,%f\n", inputs[i][0], outputs[i][0], noisyOutputs[i][0], networkOutputs[i][0] );
        }

        fclose( file );
        ret = true;
    }

    return ret;
}

// Example application's entry point
int main( int argc, char** argv )
{
    printf( "Regression example with Fully Connected ANN \n\n" );

    {
        TrainingParams trainingParams;

        // check if any of the defaults are overridden
        ParseCommandLine( argc, argv, &trainingParams );

        printf( "Input data file  : %s \n", trainingParams.InputDataFile.c_str( ) );
        printf( "Output data file : %s \n", trainingParams.OutputDataFile.c_str( ) );
        printf( "Learning rate    : %0.4f \n", trainingParams.LearningRate );
        printf( "Epochs count     : %zu \n", trainingParams.EpochsCount );
        printf( "Batch size       : %zu \n", trainingParams.BatchSize );
        printf( "Hidden neurons   : " );

        if ( trainingParams.HiddenLayers.empty( ) )
        {
            printf( "none" );
        }
        else
        {
            for ( size_t i = 0; i < trainingParams.HiddenLayers.size( ); i++ )
            {
                if ( i != 0 )
                {
                    printf( ":" );
                }
                printf( "%zu", trainingParams.HiddenLayers[i] );
            }
        }
        printf( "\n\n" );

        // load training data from the specified file
        vector<fvector_t> inputs, outputs, noisyOutputs;

        if ( !LoadData( trainingParams.InputDataFile, inputs, outputs, noisyOutputs ) )
        {
            printf( "Error: failed loading training data \n\n" );
            return -1;
        }

        size_t samplesCount = inputs.size( );
        printf( "Loaded %zu training samples \n\n", samplesCount );

        // get pointers to inputs/outputs for shuffling
        vector<fvector_t*> ptrInputs( samplesCount ), ptrTargetOutputs( samplesCount );

        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            ptrInputs[i]        = &( inputs[i] );
            ptrTargetOutputs[i] = &( noisyOutputs[i] );
        }

        // prepare fully connected ANN of the specified structure
        shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );
        size_t                     inputsCount = 1;

        for ( size_t neuronsCount : trainingParams.HiddenLayers )
        {
            net->AddLayer( make_shared<XFullyConnectedLayer>( inputsCount, neuronsCount ) );
            net->AddLayer( make_shared<XSigmoidActivation>( ) );

            inputsCount = neuronsCount;
        }

        // add output layer
        net->AddLayer( make_shared<XFullyConnectedLayer>( inputsCount, 1 ) );

        // create training context with Nesterov optimizer and MSE cost function
        XNetworkTraining netTraining( net,
                                      make_shared<XNesterovMomentumOptimizer>( trainingParams.LearningRate ),
                                      make_shared<XMSECost>( ) );

        for ( size_t epoch = 1; epoch <= trainingParams.EpochsCount; epoch++ )
        {
            // shuffle data
            for ( size_t i = 0; i < samplesCount / 2; i++ )
            {
                int swapIndex1 = rand( ) % samplesCount;
                int swapIndex2 = rand( ) % samplesCount;

                std::swap( ptrInputs[swapIndex1], ptrInputs[swapIndex2] );
                std::swap( ptrTargetOutputs[swapIndex1], ptrTargetOutputs[swapIndex2] );
            }

            auto cost = netTraining.TrainEpoch( ptrInputs, ptrTargetOutputs, trainingParams.BatchSize );

            if ( ( epoch % 10 ) == 0 )
            {
                printf( "%0.4f ", static_cast<float>( cost ) );
            }

            if ( ( epoch % 100 ) == 0 )
            {
                printf( "\n" );
            }
        }

        // get outputs produced by the trained network
        vector<fvector_t> networkOutputs( samplesCount );

        for ( size_t i = 0; i < samplesCount; i++ )
        {
            fvector_t output( 1 );

            netTraining.Compute( inputs[i], output );
            networkOutputs[i] = output;
        }

        SaveData( trainingParams.OutputDataFile, inputs, outputs, noisyOutputs, networkOutputs );
    }

    return 0;
}
