/*
    Time Series Prediction example with Recurrent ANN

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
    size_t         PredictionSize;
    size_t         SequenceSize;

    TrainingParamsStruct( ) :
        InputDataFile( "data/series1.csv" ), OutputDataFile( "data/series1-out.csv" ),
        HiddenLayers( { 30 } ), LearningRate( 0.05f ), EpochsCount( 1000 ),
        PredictionSize( 5 ), SequenceSize( 10 )
    {
    }
}
TrainingParams;

// Parse command line to get some of the training parameters
static void ParseCommandLine( int argc, char** argv, TrainingParams* trainingParams )
{
    bool showUsage = false;
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
                else if ( ( strstr( paramStart, "ps:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    if ( sscanf( &( argv[i][4] ), "%zu", &trainingParams->PredictionSize ) == 1 )
                    {
                        if ( trainingParams->PredictionSize == 0 )
                        {
                            trainingParams->PredictionSize = 1;
                        }
                        parsed = true;
                    }
                }
                else if ( ( strstr( paramStart, "ss:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    if ( sscanf( &( argv[i][4] ), "%zu", &trainingParams->SequenceSize ) == 1 )
                    {
                        if ( trainingParams->SequenceSize == 0 )
                        {
                            trainingParams->SequenceSize = 1;
                        }
                        parsed = true;
                    }
                }
                else if ( ( strstr( paramStart, "hn:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    size_t neuronsCount;
                    char*  parsePtr = &( argv[i][4] );

                    parsed = true;
                    trainingParams->HiddenLayers.clear( );

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
                else if ( ( strstr( paramStart, "in:" ) == paramStart ) && ( paramLen > 4 ) )
                {
                    trainingParams->InputDataFile = string( &( argv[i][4] ) );

                    if ( !outputFileSpecified )
                    {
                        size_t dotPos = trainingParams->InputDataFile.rfind( '.' );

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
        printf( "  -lr:<> - learning rate; \n" );
        printf( "  -ps:<> - prediction size; \n" );
        printf( "  -ss:<> - size of overlapping sequnces to generate for training; \n" );
        printf( "  -hn:<X[:X]> - number of neurons in hidden recurrent layers; examples: \n" );
        printf( "           10 - single hidden layer with 10 neurons; \n" );
        printf( "           20:10 - two hidden layers - 20 neurons in the first and 10 in the second; \n" );
        printf( "  -in:<> - file name to read input training data from; \n" );
        printf( "  -out:<> - file name to write predicted results to. \n" );
        printf( "\n" );
    }
}

// Load time series data points from the specified file
bool LoadData( const string& fileName, fvector_t& timeSeries )
{
    bool  ret  = false;
    FILE* file = fopen( fileName.c_str( ), "r" );

    if ( file )
    {
        char buff[256];

        timeSeries.clear( );

        while ( fgets( buff, 256, file ) != nullptr )
        {
            float x;

            if ( sscanf( buff, "%f", &x ) == 1 )
            {
                timeSeries.push_back( x );
            }
            else
            {
                printf( "|%s| \n", buff );
            }
        }

        fclose( file );
        ret = ( !timeSeries.empty( ) );
    }

    return ret;
}

// Save original time series, output of the network produced for the training set and predictions (output of the network for inputs not included into training)
bool SaveData( const string& fileName, const fvector_t& timeSeries, const fvector_t& networkOutput, const fvector_t& networkPrediction )
{
    bool  ret = false;
    FILE* file = fopen( fileName.c_str( ), "w" );

    if ( file )
    {
        for ( size_t i = 0; i < timeSeries.size( ); i++ )
        {
            if ( i == 0 )
            {
                fprintf( file, "%f,,\n", timeSeries[i] );
            }
            else if ( i < timeSeries.size( ) - networkPrediction.size( ) )
            {
                fprintf( file, "%f,%f,\n", timeSeries[i], networkOutput[i - 1] );
            }
            else
            {
                fprintf( file, "%f,,%f\n", timeSeries[i], networkPrediction[i + networkPrediction.size( ) - timeSeries.size( )] );
            }
        }

        fclose( file );
        ret = true;
    }

    return ret;
}

// Example application's entry point
int main( int argc, char** argv )
{
    printf( "Time Series Prediction example with Recurrent ANN \n\n" );

    {
        TrainingParams trainingParams;

        // check if any of the defaults are overridden
        ParseCommandLine( argc, argv, &trainingParams );

        printf( "Input data file  : %s \n", trainingParams.InputDataFile.c_str( ) );
        printf( "Output data file : %s \n", trainingParams.OutputDataFile.c_str( ) );
        printf( "Learning rate    : %0.4f \n", trainingParams.LearningRate );
        printf( "Epochs count     : %zu \n", trainingParams.EpochsCount );
        printf( "Sequence size    : %zu \n", trainingParams.SequenceSize );
        printf( "Prediction size  : %zu \n", trainingParams.PredictionSize );
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

        // load time series data
        fvector_t timeSeries;

        if ( !LoadData( trainingParams.InputDataFile, timeSeries ) )
        {
            printf( "Error: failed loading time series data \n\n" );
            return -1;
        }

        printf( "Loaded %zu time series data points \n\n", timeSeries.size( ) );

        // make sure there are enough data points in the time series for the specified options
        if ( trainingParams.PredictionSize + trainingParams.SequenceSize > timeSeries.size( ) / 2 )
        {
            printf( "Not enough data points in the time series. Must be at least twice of the Sequence Size and Prediction Size. \n\n" );
            return -1;
        }

        // create training samples - certain number of overlapping sequences of the specified size
        size_t            sequncesCount = timeSeries.size( ) - trainingParams.PredictionSize - trainingParams.SequenceSize;
        vector<fvector_t> inputs, outputs;

        for ( size_t i = 0; i < sequncesCount; i++ )
        {
            for ( size_t j = 0; j < trainingParams.SequenceSize; j++ )
            {
                inputs.push_back( { timeSeries[i + j] } );
                outputs.push_back( { timeSeries[i + j + 1] } );
            }
        }

        printf( "Created %zu training sequences, %zu training samples total \n\n", sequncesCount, inputs.size( ) );

        // prepare recurrent ANN to train
        shared_ptr<XNeuralNetwork> net         = make_shared<XNeuralNetwork>( );
        size_t                     inputsCount = 1;

        for ( size_t neuronsCount : trainingParams.HiddenLayers )
        {
            net->AddLayer( make_shared<XGRULayer>( inputsCount, neuronsCount ) );
            net->AddLayer( make_shared<XTanhActivation>( ) );

            inputsCount = neuronsCount;
        }
        // add fully connected output layer
        net->AddLayer( make_shared<XFullyConnectedLayer>( inputsCount, 1 ) );

        // create training context with Nesterov optimizer and MSE cost function
        XNetworkTraining netTraining( net,
                                      make_shared<XNesterovMomentumOptimizer>( trainingParams.LearningRate ),
                                      make_shared<XMSECost>( ) );

        netTraining.SetTrainingSequenceLength( trainingParams.SequenceSize );

        for ( size_t epoch = 1; epoch <= trainingParams.EpochsCount; epoch++ )
        {
            auto cost = netTraining.TrainBatch( inputs, outputs );

            netTraining.ResetState( );

            if ( ( epoch % 10 ) == 0 )
            {
                printf( "%0.4f ", static_cast<float>( cost ) );
            }

            if ( ( epoch % 100 ) == 0 )
            {
                printf( "\n" );
            }
        }
        printf( "\n" );

        // get outputs produced by the trained network - predict single point only to see the fit is good in any way
        fvector_t networkOutput;
        fvector_t input( 1 );
        fvector_t output( 1 );

        for ( size_t i = 0; i < timeSeries.size( ) - trainingParams.PredictionSize - 1; i++ )
        {
            input[0] = timeSeries[i]; // here input always comes from original time series

            netTraining.Compute( input, output );
            networkOutput.push_back( output[0] );
        }

        // now predict some points, which were excluded from training
        fvector_t networkPrediction( trainingParams.PredictionSize );
        ANNT::float_t error, minError = ANNT::float_t( 0.0 ), maxError = ANNT::float_t( 0.0 ), avgError = ANNT::float_t( 0.0 );

        // don't reset state of the recurrent network, just feed it the next point from the time series
        input[0] = timeSeries[timeSeries.size( ) - trainingParams.PredictionSize - 1];

        for ( size_t i = 0, j = timeSeries.size( ) - trainingParams.PredictionSize; i < trainingParams.PredictionSize; i++, j++ )
        {
            netTraining.Compute( input, output );
            networkPrediction[i] = output[0];

            // use just predicted point as the new input
            input[0] = output[0];

            // find prediction error
            error = fabs( output[0] - timeSeries[j] );
            avgError += error;

            if ( i == 0 )
            {
                minError = maxError = error;
            }
            else if ( error < minError )
            {
                minError = error;
            }
            else if ( error > maxError )
            {
                maxError = error;
            }
        }

        avgError /= trainingParams.PredictionSize;
        printf( "Prediction error: min = %0.4f, max = %0.4f, avg = %0.4f \n",
                static_cast< float >( minError ), static_cast< float >( maxError ), static_cast< float >( avgError ) );

        // save training/prediction results into CSV file
        SaveData( trainingParams.OutputDataFile, timeSeries, networkOutput, networkPrediction );
    }

    return 0;
}
