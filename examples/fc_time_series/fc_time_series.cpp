/*
    Time Series Prediction example with Fully Connected ANN

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
    size_t         BatchSize;
    size_t         WindowSize;

    TrainingParamsStruct( ) :
        InputDataFile( "data/series3.csv" ), OutputDataFile( "data/series3-out.csv" ),
        HiddenLayers( { 10 } ), LearningRate( 0.01f ), EpochsCount( 1000 ), BatchSize( 10 ),
        WindowSize( 5 )
    {
    }
}
TrainingParams;

// Parse command line to get some of the training parameters
static void ParseCommandLine( int argc, char** argv, TrainingParams* trainingParams )
{
    bool showUsage           = false;
    bool outputFileSpecified = false;


}

// ?
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

//
bool SaveData( const string& fileName, const fvector_t& timeSeries, const fvector_t predictions, size_t windowSize )
{
    bool  ret  = false;
    FILE* file = fopen( fileName.c_str( ), "w" );

    if ( file )
    {
        for ( size_t i = 0; i < timeSeries.size( ); i++ )
        {
            if ( i < windowSize )
            {
                fprintf( file, "%f,\n", timeSeries[i] );
            }
            else
            {
                fprintf( file, "%f,%f\n", timeSeries[i], predictions[i - windowSize] );
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
    printf( "Time Series Prediction example with Fully Connected ANN \n\n" );

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

        // load time series data
        fvector_t timeSeries;

        if ( !LoadData( trainingParams.InputDataFile, timeSeries ) )
        {
            printf( "Error: failed loading time series data \n\n" );
            return -1;
        }

        printf( "Loaded %zu time series data points \n\n", timeSeries.size( ) );

        //
        if ( trainingParams.WindowSize > timeSeries.size( ) / 2 )
        {
            printf( "Not enough data points in the time series. Must be at least twice of the Window Size. \n\n" );
        }

        // create training inputs/outputs
        size_t            samplesCount = timeSeries.size( ) - trainingParams.WindowSize;
        vector<fvector_t> inputs, outputs;

        for ( size_t i = 0; i < samplesCount; i++ )
        {
            fvector_t input( trainingParams.WindowSize );

            for ( size_t j = 0; j < trainingParams.WindowSize; j++ )
            {
                input[j] = timeSeries[i + j];
            }

            inputs.push_back( input );
            outputs.push_back( { timeSeries[i + trainingParams.WindowSize] /* * 10 */ } );
        }

        // ??

        // get pointers to inputs/outputs for shuffling
        vector<fvector_t*> ptrInputs( samplesCount ), ptrTargetOutputs( samplesCount );

        for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
        {
            ptrInputs[i]        = &( inputs[i] );
            ptrTargetOutputs[i] = &( outputs[i] );
        }

        // prepare fully connected ANN of the specified structure
        shared_ptr<XNeuralNetwork> net         = make_shared<XNeuralNetwork>( );
        size_t                     inputsCount = trainingParams.WindowSize;

        for ( size_t neuronsCount : trainingParams.HiddenLayers )
        {
            net->AddLayer( make_shared<XFullyConnectedLayer>( inputsCount, neuronsCount ) );
            net->AddLayer( make_shared<XTanhActivation>( ) );

            inputsCount = neuronsCount;
        }

        // add output layer
        net->AddLayer( make_shared<XFullyConnectedLayer>( inputsCount, 1 ) );

        // create training context with Nesterov optimizer and Cross Entropy cost function
        shared_ptr<XNetworkTraining> netTraining = make_shared<XNetworkTraining>( net,
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

            auto cost = netTraining->TrainEpoch( ptrInputs, ptrTargetOutputs, trainingParams.BatchSize );

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
        fvector_t predictions( samplesCount );

        for ( size_t i = 0; i < samplesCount; i++ )
        {
            fvector_t output( 1 );

            netTraining->Compute( inputs[i], output );
            predictions[i] = output[0];// / 10;
        }

        SaveData( trainingParams.OutputDataFile, timeSeries, predictions, trainingParams.WindowSize );
    }

    return 0;
}

