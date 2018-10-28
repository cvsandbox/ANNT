/*
    ANNT - Artificial Neural Networks C++ library

    CIFAR-10 dataset classification example with Convolutional ANN

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
#include "CIFARParser.hpp"

using namespace std;

using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

// Names of MNIST data files.
// Available at: https://www.cs.toronto.edu/~kriz/cifar.html
static const char* CIFAR10_TRAIN_FILE_1 = "data/data_batch_1.bin";
static const char* CIFAR10_TRAIN_FILE_2 = "data/data_batch_2.bin";
static const char* CIFAR10_TRAIN_FILE_3 = "data/data_batch_3.bin";
static const char* CIFAR10_TRAIN_FILE_4 = "data/data_batch_4.bin";
static const char* CIFAR10_TRAIN_FILE_5 = "data/data_batch_5.bin";
static const char* CIFAR10_TEST_FILE    = "data/test_batch.bin";

// Helper function to extract validation samples out of MNIST training data
template <typename T> vector<T> ExtractValidationSamples( vector<T>& allSamples )
{
    vector<T> validationSamples( allSamples.size( ) / 8 );
    size_t    start = allSamples.size( ) - validationSamples.size( );

    std::move( allSamples.begin( ) + start, allSamples.end( ), validationSamples.begin( ) );

    allSamples.erase( allSamples.begin( ) + start, allSamples.end( ) );

    return validationSamples;
}

// Example application's entry point
int main( int argc, char** argv )
{
    printf( "CIFAR-10 dataset classification example with Convolutional ANN \n\n" );

    uvector_t         trainLabels;
    vector<fvector_t> trainImages;
    uvector_t         testLabels;
    vector<fvector_t> testImages;

    // load training data set
    if ( ( !CIFARParser::LoadDataSet( CIFAR10_TRAIN_FILE_1, trainLabels, trainImages, -1, 1 ) ) ||
         ( !CIFARParser::LoadDataSet( CIFAR10_TRAIN_FILE_2, trainLabels, trainImages, -1, 1 ) ) ||
         ( !CIFARParser::LoadDataSet( CIFAR10_TRAIN_FILE_3, trainLabels, trainImages, -1, 1 ) ) ||
         ( !CIFARParser::LoadDataSet( CIFAR10_TRAIN_FILE_4, trainLabels, trainImages, -1, 1 ) ) ||
         ( !CIFARParser::LoadDataSet( CIFAR10_TRAIN_FILE_5, trainLabels, trainImages, -1, 1 ) ) )
    {
        printf( "Failed loading training dataset\n\n" );
        return -1;
    }

    if ( !CIFARParser::LoadDataSet( CIFAR10_TEST_FILE, testLabels, testImages, -1, 1 ) )
    {
        printf( "Failed loading test dataset\n\n" );
        return -1;
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

    // prepare a convolutional ANN
    shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

    // since ReLU(MaxPool)=MaxPool(ReLU), we use the first - minor optimization though
    net->AddLayer( make_shared<XConvolutionLayer>( 32, 32, 3, 5, 5, 32, BorderMode::Same ) );
    net->AddLayer( make_shared<XMaxPooling>( 32, 32, 32, 2 ) );
    net->AddLayer( make_shared<XReLuActivation>( ) );
    net->AddLayer( make_shared<XBatchNormalization>( 16, 16, 32 ) );

    net->AddLayer( make_shared<XConvolutionLayer>( 16, 16, 32, 5, 5, 32, BorderMode::Same ) );
    net->AddLayer( make_shared<XMaxPooling>( 16, 16, 32, 2 ) );
    net->AddLayer( make_shared<XReLuActivation>( ) );
    net->AddLayer( make_shared<XBatchNormalization>( 8, 8, 32 ) );

    net->AddLayer( make_shared<XConvolutionLayer>( 8, 8, 32, 5, 5, 64, BorderMode::Same ) );
    net->AddLayer( make_shared<XMaxPooling>( 8, 8, 64, 2 ) );
    net->AddLayer( make_shared<XReLuActivation>( ) );
    net->AddLayer( make_shared<XBatchNormalization>( 4, 4, 64 ) );

    net->AddLayer( make_shared<XFullyConnectedLayer>( 4 * 4 * 64, 64 ) );
    net->AddLayer( make_shared<XReLuActivation>( ) );
    net->AddLayer( make_shared<XBatchNormalization>( 64, 1, 1 ) );

    net->AddLayer( make_shared<XFullyConnectedLayer>( 64, 10 ) );
    net->AddLayer( make_shared<XLogSoftMaxActivation>( ) );

    // create training context with Adam optimizer and Negative Log Likelihood cost function (since we use Log-Softmax)
    shared_ptr<XNetworkTraining> netTraining = make_shared<XNetworkTraining>( net,
                                               make_shared<XAdamOptimizer>( 0.001f ),
                                               make_shared<XNegativeLogLikelihoodCost>( ) );

    // using the helper for training ANN to do classification
    XClassificationTrainingHelper trainingHelper( netTraining, argc, argv );
    trainingHelper.SetValidationSamples( validationImages, encodedValidationLabels, validationLabels );
    trainingHelper.SetTestSamples( testImages, encodedTestLabels, testLabels );

    // 20 epochs, 50 samples in batch
    trainingHelper.RunTraining( 20, 50, trainImages, encodedTrainLabels, trainLabels );

    return 0;
}
