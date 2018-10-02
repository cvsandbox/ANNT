/*
    ANNT - Artificial Neural Networks C++ library

    MNIST handwritten digits classification example with Fully Connected ANN

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

// Helper function to extract validation samples (20%) out of MNIST training data
template <typename T> vector<T> ExtractValidationSamples( vector<T>& allSamples )
{
    vector<T> validationSamples( allSamples.size( ) / 6 );
    size_t    start = allSamples.size( ) - validationSamples.size( );

    std::move( allSamples.begin( ) + start, allSamples.end( ), validationSamples.begin( ) );

    allSamples.erase( allSamples.begin( ) + start, allSamples.end( ) );

    return validationSamples;
}

// Example application's entry point
int main( int argc, char** argv )
{
    printf( "MNIST handwritten digits classification example with Fully Connected ANN \n\n" );

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

    printf( "Loaded %zu training data samples \n", trainLabels.size( )  );
    printf( "Loaded %zu test data samples \n\n", testLabels.size( ) );

    // extract validation data set out of training set
    uvector_t         validationLabels = ExtractValidationSamples( trainLabels );
    vector<fvector_t> validationImages = ExtractValidationSamples( trainImages );

    printf( "Samples usage: training = %zu, validation = %zu, test = %zu \n\n",
            trainLabels.size( ), validationLabels.size( ), testLabels.size( ) );

    // perform one hot encoding for all labels
    vector<fvector_t> encodedTrainLabels      = XDataEncodingTools::OneHotEncoding( trainLabels,      10 );
    vector<fvector_t> encodedValidationLabels = XDataEncodingTools::OneHotEncoding( validationLabels, 10 );
    vector<fvector_t> encodedTestLabels       = XDataEncodingTools::OneHotEncoding( testLabels,       10 );

    // prepare a 3 layer ANN
    shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

    net->AddLayer( make_shared<XFullyConnectedLayer>( trainImages[0].size( ), 300 ) );
    net->AddLayer( make_shared<XTanhActivation>( ) );
    net->AddLayer( make_shared<XFullyConnectedLayer>( 300, 100 ) );
    net->AddLayer( make_shared<XTanhActivation>( ) );
    net->AddLayer( make_shared<XFullyConnectedLayer>( 100, 10 ) );
    net->AddLayer( make_shared<XSoftMaxActivation>( ) );

    // create training context with Adam optimizer and Cross Entropy cost function
    shared_ptr<XNetworkTraining> netTraining = make_shared<XNetworkTraining>( net,
                                               make_shared<XAdamOptimizer>( 0.001f ),
                                               make_shared<XCrossEntropyCost>( ) );

    // using the helper for training ANN to do classification
    XClassificationTrainingHelper trainingHelper( netTraining, argc, argv );
    trainingHelper.SetValidationSamples( validationImages, encodedValidationLabels, validationLabels );
    trainingHelper.SetTestSamples( testImages, encodedTestLabels, testLabels );

    // 20 epochs, 50 samples in batch
    trainingHelper.RunTraining( 20, 50, trainImages, encodedTrainLabels, trainLabels );

    return 0;
}
