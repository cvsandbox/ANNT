/*
    ANNT - Artificial Neural Networks C++ library

    MNIST handwritten digits classification example with Recurrent ANN

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
#include <chrono>

#include "ANNT.hpp"
#include "MNISTParser.hpp"

using namespace std;
using namespace std::chrono;

using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

#define MNIST_IMAGE_WIDTH  (28)
#define MNIST_IMAGE_HEIGHT (28)

// Names of MNIST data files.
// Available at: http://yann.lecun.com/exdb/mnist/
static const char* MNIST_TRAIN_LABELS_FILE = "data/train-labels.idx1-ubyte";
static const char* MNIST_TRAIN_IMAGES_FILE = "data/train-images.idx3-ubyte";
static const char* MNIST_TEST_LABELS_FILE  = "data/t10k-labels.idx1-ubyte";
static const char* MNIST_TEST_IMAGES_FILE  = "data/t10k-images.idx3-ubyte";

// Extract sequence elements (rows of MNIST images) from pointers to complete samples
static void ExtractSamplesAsSequence( const vector<fvector_t*>& inputPtrs, const vector<fvector_t*>& outputPtrs,
                                      vector<fvector_t>& inputSequence, vector<fvector_t>& outputSequence,
                                      size_t samplesToExtract, size_t startIndex, size_t sequenceLength )
{
    size_t i             = startIndex;
    size_t totalSamples  = inputPtrs.size( );
    size_t sequenceWidth = inputPtrs[i]->size( ) / sequenceLength;

    inputSequence.clear( );
    outputSequence.clear( );

    if ( sequenceWidth * sequenceLength == inputPtrs[i]->size( ) )
    {
        for ( size_t j = 0; j < samplesToExtract; j++ )
        {
            fvector_t* samplePtr = inputPtrs[i];

            for ( size_t k = 0, z = 0; k < sequenceLength; k++ )
            {
                fvector_t sequenceRow( sequenceWidth );

                for ( size_t x = 0; x < sequenceWidth; x++, z++ )
                {
                    sequenceRow[x] = (*samplePtr)[z];
                }

                inputSequence.push_back( sequenceRow );
                outputSequence.push_back( *( outputPtrs[i] ) );
            }

            i = ( i + 1 ) % totalSamples;
        }
    }
}

// Extract sequence elements (rows of MNIST images) from samples
static void ExtractSamplesAsSequence( const vector<fvector_t>& inputs, const vector<fvector_t>& outputs,
                                      vector<fvector_t>& inputSequence, vector<fvector_t>& outputSequence,
                                      size_t samplesToExtract, size_t startIndex, size_t sequenceLength )
{
    size_t i             = startIndex;
    size_t totalSamples  = inputs.size( );
    size_t sequenceWidth = inputs[i].size( ) / sequenceLength;

    inputSequence.clear( );
    outputSequence.clear( );

    if ( sequenceWidth * sequenceLength == inputs[i].size( ) )
    {
        for ( size_t j = 0; j < samplesToExtract; j++ )
        {
            const fvector_t& sample = inputs[i];

            for ( size_t k = 0, z = 0; k < sequenceLength; k++ )
            {
                fvector_t sequenceRow( sequenceWidth );

                for ( size_t x = 0; x < sequenceWidth; x++, z++ )
                {
                    sequenceRow[x] = sample[z];
                }

                inputSequence.push_back( sequenceRow );
                outputSequence.push_back( outputs[i]);
            }

            i = ( i + 1 ) % totalSamples;
        }
    }
}

// Helper function to extract validation samples out of MNIST training data
template <typename T> vector<T> ExtractValidationSamples( vector<T>& allSamples )
{
    vector<T> validationSamples( allSamples.size( ) / 6 );
    size_t    start = allSamples.size( ) - validationSamples.size( );

    std::move( allSamples.begin( ) + start, allSamples.end( ), validationSamples.begin( ) );

    allSamples.erase( allSamples.begin( ) + start, allSamples.end( ) );

    return validationSamples;
}

// Test classificication accuracy of the recurrent network
size_t TestClassification( shared_ptr<XNeuralNetwork>& net, shared_ptr<ICostFunction> costFunction,
                           const vector<fvector_t>& inputs, const uvector_t& targetLabels,
                           const vector<fvector_t>& targetOutputs, float* pAvgCost )
{
    XNetworkInference netInference( net );
    vector<fvector_t> sequenceInputs;
    vector<fvector_t> sequenceOutputs;
    fvector_t         output( 10 );
    size_t            correctLabelsCounter = 0;
    float             cost = 0;

    for ( size_t i = 0; i < inputs.size( ); i++ )
    {
        ExtractSamplesAsSequence( inputs, targetOutputs, sequenceInputs, sequenceOutputs, 1, i, MNIST_IMAGE_WIDTH );

        for ( size_t j = 0; j < MNIST_IMAGE_HEIGHT; j++ )
        {
            netInference.Compute( sequenceInputs[j], output );
        }

        if ( XDataEncodingTools::MaxIndex( output ) == targetLabels[i] )
        {
            correctLabelsCounter++;
        }

        // get cost only of the final output after entire image was presented to the network row by row
        cost += static_cast<float>( costFunction->Cost( output, targetOutputs[i] ) );

        netInference.ResetState( );
    }

    if ( pAvgCost != nullptr )
    {
        *pAvgCost = cost / inputs.size( );
    }

    return correctLabelsCounter;
}

// Example application's entry point
int main( int argc, char** argv )
{
#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemState memState;
    _CrtMemCheckpoint( &memState );
#endif

    printf( "MNIST handwritten digits classification example with Recurrent ANN \n\n" );

    {
        uvector_t         trainLabels;
        vector<fvector_t> trainImages;
        uvector_t         testLabels;
        vector<fvector_t> testImages;

        // load training data set
        if ( !MNISTParser::LoadLabels( MNIST_TRAIN_LABELS_FILE, trainLabels ) )
        {
            printf( "Failed loading training labels database \n\n" );
            return -1;
        }

        if ( !MNISTParser::LoadImages( MNIST_TRAIN_IMAGES_FILE, trainImages, -1, 1, 0, 0 ) )
        {
            printf( "Failed loading training images database \n\n" );
            return -2;
        }

        // load test data set
        if ( !MNISTParser::LoadLabels( MNIST_TEST_LABELS_FILE, testLabels ) )
        {
            printf( "Failed loading test labels database \n\n" );
            return -3;
        }

        if ( !MNISTParser::LoadImages( MNIST_TEST_IMAGES_FILE, testImages, -1, 1, 0, 0 ) )
        {
            printf( "Failed loading test images database \n\n" );
            return -4;
        }

        // make sure we've got same number of labels as images
        if ( trainImages.size( ) != trainLabels.size( ) )
        {
            printf( "Size missmatch for training images and labels data sets \n\n" );
            return -5;
        }
        if ( testImages.size( ) != testLabels.size( ) )
        {
            printf( "Size missmatch for test images and labels data sets \n\n" );
            return -6;
        }

        printf( "Loaded %zu training data samples \n", trainLabels.size( ) );
        printf( "Loaded %zu test data samples \n\n", testLabels.size( ) );

        // extract validation data set out of training set
        uvector_t         validationLabels = ExtractValidationSamples( trainLabels );
        vector<fvector_t> validationImages = ExtractValidationSamples( trainImages );

        printf( "Samples usage: training = %zu, validation = %zu, test = %zu \n\n",
            trainLabels.size( ), validationLabels.size( ), testLabels.size( ) );

        // perform one hot encoding for all labels
        vector<fvector_t> encodedTrainLabels      = XDataEncodingTools::OneHotEncoding( trainLabels, 10 );
        vector<fvector_t> encodedValidationLabels = XDataEncodingTools::OneHotEncoding( validationLabels, 10 );
        vector<fvector_t> encodedTestLabels       = XDataEncodingTools::OneHotEncoding( testLabels, 10 );

        // take pointers to original inputs/outputs, so those could be shuffled 
        size_t             samplesCount = trainImages.size( );
        vector<fvector_t*> trainingInputsPtr( samplesCount );
        vector<fvector_t*> trainingOutputsPtr( samplesCount );

        for ( size_t i = 0; i < samplesCount; i++ )
        {
            trainingInputsPtr[i]  = &( trainImages[i] );
            trainingOutputsPtr[i] = &( encodedTrainLabels[i] );
        }

        // default training parameters
        Helpers::TrainingParams trainingParams;

        trainingParams.LearningRate = 0.001f;
        trainingParams.EpochsCount  = 20;
        trainingParams.BatchSize    = 48;

        // parse command line for any overrides
        Helpers::ParseTrainingParamsCommandLine( argc, argv, &trainingParams );

        // log current settings
        Helpers::PrintTrainingParams( &trainingParams );

        // prepare a recurrent ANN
        shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );
        
        // XLSTMLayer vs XGRULayer
        // number of inputs as per image width
        // net->AddLayer( make_shared<XLSTMLayer>( MNIST_IMAGE_WIDTH, 56 ) );
        net->AddLayer( make_shared<XGRULayer>( MNIST_IMAGE_WIDTH, 56 ) );
        net->AddLayer( make_shared<XFullyConnectedLayer>( 56, 10 ) );
        net->AddLayer( make_shared<XSoftMaxActivation>( ) );

        // create training context with Adam optimizer and Cross Entropy cost function
        XNetworkTraining netTraining( net,
                                      make_shared<XAdamOptimizer>( trainingParams.LearningRate ),
                                      make_shared<XCrossEntropyCost>( ) );

        netTraining.SetAverageWeightGradients( false );
        netTraining.SetTrainingSequenceLength( MNIST_IMAGE_HEIGHT ); /* sequence length as per image height */

        // load network parameters from the previous save file
        if ( !trainingParams.NetworkInputFileName.empty( ) )
        {
            if ( !net->LoadLearnedParams( trainingParams.NetworkInputFileName ) )
            {
                printf( "Failed loading network's parameters \n\n" );
            }
        }

        // training performance
        steady_clock::time_point timeStartForAll = steady_clock::now( );
        steady_clock::time_point timeStart;
        long long                timeTaken;
        float                    cost;
        size_t                   correct;
        float                    lastValidationAccuracy = 0.0f;

        // check classification error before starting training
        if ( trainingParams.RunPreTrainingTest )
        {
            timeStart = steady_clock::now( );
            correct   = TestClassification( net, netTraining.CostFunction( ),
                                            trainImages, trainLabels, encodedTrainLabels, &cost );
            timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

            printf( "Before training: accuracy = %0.2f%% (%zu/%zu), cost = %0.4f, %0.3fs \n\n",
                    static_cast<float>( correct ) / trainImages.size( ) * 100,
                    correct, trainImages.size( ), cost,
                    static_cast<float>( timeTaken ) / 1000 );
        }

        // run the specified number of epochs
        size_t iterationsPerEpoch   = ( samplesCount - 1 ) / trainingParams.BatchSize + 1;
        size_t batchCostOutputFreq  = iterationsPerEpoch / 80;
        int    progressStringLength = 0;

        if ( batchCostOutputFreq == 0 )
        {
            batchCostOutputFreq = 1;
        }

        vector<fvector_t> inputs;
        vector<fvector_t> outputs;

        for ( size_t epoch = 0; epoch < trainingParams.EpochsCount; epoch++ )
        {
            printf( "Epoch %3zu : ", epoch + 1 );
            if ( !trainingParams.ShowIntermediateBatchCosts )
            {
                // show progress bar only
                putchar( '[' );
            }
            else
            {
                printf( "\n" );
            }

            // shuffle training samples
            for ( size_t i = 0; i < samplesCount / 2; i++ )
            {
                int swapIndex1 = rand( ) % samplesCount;
                int swapIndex2 = rand( ) % samplesCount;

                std::swap( trainingInputsPtr[swapIndex1], trainingInputsPtr[swapIndex2] );
                std::swap( trainingOutputsPtr[swapIndex1], trainingOutputsPtr[swapIndex2] );
            }

            // start of epoch timing
            timeStart = steady_clock::now( );

            for ( size_t iteration = 0; iteration < iterationsPerEpoch; iteration++ )
            {
                // prepare batch inputs and ouputs
                ExtractSamplesAsSequence( trainingInputsPtr, trainingOutputsPtr, inputs, outputs,
                                          trainingParams.BatchSize, iteration * trainingParams.BatchSize, MNIST_IMAGE_WIDTH );

                auto batchCost = netTraining.TrainBatch( inputs, outputs );
                netTraining.ResetState( );

                // erase previous progress if any 
                Helpers::EraseTrainingProgress( progressStringLength );

                // show cost of some batches or progress bar only
                if ( !trainingParams.ShowIntermediateBatchCosts )
                {
                    Helpers::UpdateTrainingPogressBar( iteration, iteration + 1, iterationsPerEpoch, 50, '=' );
                }
                else
                {
                    if ( ( ( iteration + 1 ) % batchCostOutputFreq ) == 0 )
                    {
                        printf( "%0.4f ", static_cast<float>( batchCost ) );

                        if ( ( ( iteration + 1 ) % ( batchCostOutputFreq * 8 ) ) == 0 )
                        {
                            printf( "\n" );
                        }
                    }
                }

                // show current progress of the epoch
                progressStringLength = Helpers::ShowTrainingProgress( iteration + 1, iterationsPerEpoch );
            }

            Helpers::EraseTrainingProgress( progressStringLength );
            progressStringLength = 0;

            // end of epoch timing
            timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

            // output time spent on training
            if ( !trainingParams.ShowIntermediateBatchCosts )
            {
                printf( "] " );
            }
            else
            {
                printf( "\nTime taken : " );
            }
            printf( "%0.3fs\n", static_cast<float>( timeTaken ) / 1000 );

            float validationAccuracy = 0.0f;

            // get classification error on training data after completion of an epoch
            if ( !trainingParams.RunValidationOnly )
            {
                timeStart = steady_clock::now( );
                correct   = TestClassification( net, netTraining.CostFunction( ),
                                                trainImages, trainLabels, encodedTrainLabels, &cost );
                timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

                printf( "Training accuracy = %0.2f%% (%zu/%zu), cost = %0.4f, %0.3fs \n",
                        static_cast<float>( correct ) / trainImages.size( ) * 100,
                        correct, trainImages.size( ), cost,
                        static_cast<float>( timeTaken ) / 1000 );
            }

            // use validation set to check classification error on data not included into training
            timeStart = steady_clock::now( );
            correct   = TestClassification( net, netTraining.CostFunction( ),
                                            validationImages, validationLabels, encodedValidationLabels, &cost );
            timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

            printf( "Validation accuracy = %0.2f%% (%zu/%zu), cost = %0.4f, %0.3fs \n",
                    static_cast<float>( correct ) / validationImages.size( ) * 100,
                    correct, validationImages.size( ), static_cast<float>( cost ),
                    static_cast<float>( timeTaken ) / 1000 );

            validationAccuracy = static_cast<float>( correct ) / validationImages.size( );

            // save network at the end of epoch
            if ( trainingParams.SaveMode == NetworkSaveMode::OnEpochEnd )
            {
                net->SaveLearnedParams( trainingParams.NetworkOutputFileName );
            }
            else if ( ( trainingParams.SaveMode == NetworkSaveMode::OnValidationImprovement ) &&
                      ( validationAccuracy > lastValidationAccuracy ) )
            {
                net->SaveLearnedParams( trainingParams.NetworkOutputFileName );
                lastValidationAccuracy = validationAccuracy;
            }
        }

        // final test on test data
        timeStart = steady_clock::now( );
        correct   = TestClassification( net, netTraining.CostFunction( ),
                                        testImages, testLabels, encodedTestLabels, &cost );
        timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

        printf( "\nTest accuracy = %0.2f%% (%zu/%zu), cost = %0.4f, %0.3fs \n",
                static_cast<float>( correct ) / testImages.size( ) * 100,
                correct, testImages.size( ), static_cast<float>( cost ),
                static_cast<float>( timeTaken ) / 1000 );

        // total time taken by the training
        timeTaken = duration_cast<seconds>( steady_clock::now( ) - timeStartForAll ).count( );
        printf( "\nTotal time taken : %ds (%0.2fmin) \n", static_cast<int>( timeTaken ), static_cast<float>( timeTaken ) / 60 );

        // save network when training is done
        if ( trainingParams.SaveMode == NetworkSaveMode::OnTrainingEnd )
        {
            net->SaveLearnedParams( trainingParams.NetworkOutputFileName );
        }
    }

#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemDumpAllObjectsSince( &memState );
#endif

    return 0;
}
