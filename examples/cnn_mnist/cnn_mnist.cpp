/*
    ANNT - Artificial Neural Networks C++ library

    MNIST handwritten digits classification example with Convolutional ANN

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
#include "MNISTParser.hpp"

using namespace std;

using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

// Names of MNIST data files.
// Available at: http://yann.lecun.com/exdb/mnist/
static const char* MNIST_TRAIN_LABELS_FILE = "data/train-labels.idx1-ubyte";
static const char* MNIST_TRAIN_IMAGES_FILE = "data/train-images.idx3-ubyte";
static const char* MNIST_TEST_LABELS_FILE  = "data/t10k-labels.idx1-ubyte";
static const char* MNIST_TEST_IMAGES_FILE  = "data/t10k-images.idx3-ubyte";

// Helper function to extract validation samples out of MNIST training data
template <typename T> vector<T> ExtractValidationSamples( vector<T>& allSamples )
{
    vector<T> validationSamples( allSamples.size( ) / 6 );
    size_t    start = allSamples.size( ) - validationSamples.size( );

    std::move( allSamples.begin( ) + start, allSamples.end( ), validationSamples.begin( ) );

    allSamples.erase( allSamples.begin( ) + start, allSamples.end( ) );

    return validationSamples;
}

// Example application's entry point
int main( int /* argc */, char** /* argv */ )
{
    printf( "MNIST handwritten digits classification example with Convolutional ANN \n\n" );

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

    if ( !MNISTParser::LoadImages( MNIST_TRAIN_IMAGES_FILE, trainImages, -1, 1, 2, 2 ) )
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

    if ( !MNISTParser::LoadImages( MNIST_TEST_IMAGES_FILE, testImages, -1, 1, 2, 2 ) )
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

    printf( "Loaded %u training data samples \n", trainLabels.size( )  );
    printf( "Loaded %u test data samples \n\n", testLabels.size( ) );

    // extract validation data set out of training set
    uvector_t         validationLabels = ExtractValidationSamples( trainLabels );
    vector<fvector_t> validationImages = ExtractValidationSamples( trainImages );

    printf( "Samples usage: training = %u, validation = %u, test = %u \n\n",
            trainLabels.size( ), validationLabels.size( ), testLabels.size( ) );

    // perform one hot encoding for all labels
    vector<fvector_t> encodedTrainLabels      = XDataEncodingTools::OneHotEncoding( trainLabels, 10 );
    vector<fvector_t> encodedValidationLabels = XDataEncodingTools::OneHotEncoding( validationLabels, 10 );
    vector<fvector_t> encodedTestLabels       = XDataEncodingTools::OneHotEncoding( testLabels, 10 );

    // prepare a convolutional ANN
    shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

    net->AddLayer( make_shared<XConvolutionLayer>( 32, 32, 1, 5, 5, 6 ) );
    net->AddLayer( make_shared<XReLuActivation>( ) );
    net->AddLayer( make_shared<XAveragePooling>( 28, 28, 6, 2 ) );

    net->AddLayer( make_shared<XConvolutionLayer>( 14, 14, 6, 5, 5, 16 ) );
    net->AddLayer( make_shared<XReLuActivation>( ) );
    net->AddLayer( make_shared<XAveragePooling>( 10, 10, 16, 2 ) );

    net->AddLayer( make_shared<XConvolutionLayer>( 5, 5, 16, 5, 5, 120 ) );
    net->AddLayer( make_shared<XReLuActivation>( ) );

    net->AddLayer( make_shared<XFullyConnectedLayer>( 120, 10 ) );
    net->AddLayer( make_shared<XSigmoidActivation>( ) );
    net->AddLayer( make_shared<XSoftMaxActivation>( ) );

    // create training context with Nesterov optimizer and Binary Cross Entropy cost function
    XNetworkTraining netCtx( net,
                             make_shared<XAdamOptimizer>( 0.002f ),
                             make_shared<XBinaryCrossEntropyCost>( ) );

    // check classification error on the training data with random model
    ANNT::float_t cost    = 0;
    size_t        correct = 0;

    correct = netCtx.TestClassification( trainImages, trainLabels, encodedTrainLabels, &cost );

    printf( "Before training: accuracy = %0.2f%% (%u/%u), cost = %0.4f \n", static_cast<float>( correct ) / trainImages.size( ) * 100,
            correct, trainImages.size( ), static_cast<float>( cost ) );

    //
    vector<fvector_t*> inputs( 50 );
    vector<fvector_t*> outputs( 50 );

    // train the neural network
    for ( size_t i = 0; i < 20; i++ )
    {
        printf( "epoch %u \n", i + 1 );
        for ( size_t j = 0; j < 1000; j++ )
        {
            for ( size_t k = 0; k < 50; k++ )
            {
                size_t s = rand( ) % trainImages.size( );

                inputs[k]  = &( trainImages[s] );
                outputs[k] = &( encodedTrainLabels[s] );
            }

            auto batchCost = netCtx.TrainBatch( inputs, outputs );

            if ( ( j % 10 ) == 0 )
            {
                printf( "%0.4f ", static_cast<float>( batchCost ) );
            }
            if ( ( j % 80 ) == 0 )
            {
                printf( "\n" );
            }
        }

        printf( "\n" );

        correct = netCtx.TestClassification( trainImages, trainLabels, encodedTrainLabels, &cost );

        printf( "Epoch %3u : accuracy = %0.2f%% (%5u/%5u), cost = %0.4f \n", i + 1, static_cast<float>( correct ) / trainImages.size( ) * 100,
                correct, trainImages.size( ), static_cast<float>( cost ) );
    }

	return 0;
}

