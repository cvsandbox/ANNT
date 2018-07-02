/*
    ANNT - Artificial Neural Networks C++ library

    Iris flower classification with Fully Connected ANN

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
#include <ctype.h>
#include <vector>
#include <map>

#include "ANNT.hpp"

using namespace std;
using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

// Name of the Iris data file, which is publicly available using the link below
// http://archive.ics.uci.edu/ml/datasets/Iris
static const char* IRIS_DATA_FILE = "data/iris.data";

// Helper function to load Iris data set
bool LoadData( vector<fvector_t>& attributes, uvector_t& labels )
{
    bool  ret  = false;
    FILE* file = fopen( IRIS_DATA_FILE, "r" );

    if ( file )
    {
        char     buff[256];
        size_t   labelsCounter = 0;

        map<string, size_t> labelsMap;

        while ( fgets( buff, 256, file ) != nullptr )
        {
            size_t len = strlen( buff );

            while ( ( len > 0 ) && ( isspace( buff[len - 1] ) ) )
            {
                buff[--len] = '\0';
            }

            if ( len != 0 )
            {
                float attr1, attr2, attr3, attr4;
                char* classNamePtr = strrchr( buff, ',' );

                if ( ( sscanf( buff, "%f,%f,%f,%f,", &attr1, &attr2, &attr3, &attr4 ) == 4 ) && ( classNamePtr != nullptr ) )
                {
                    classNamePtr++;

                    size_t labelId = labelsCounter;
                    auto   labelIt = labelsMap.find( classNamePtr );

                    if ( labelIt != labelsMap.end( ) )
                    {
                        labelId = labelIt->second;
                    }
                    else
                    {
                        labelsMap.insert( pair<string, size_t>( classNamePtr, labelsCounter ) );
                        labelsCounter++;
                    }

                    // Iris data set has only 3 classes, so ignore anything else
                    if ( labelId <= 2 )
                    {
                        attributes.push_back( fvector_t( { attr1, attr2, attr3, attr4 } ) );
                        labels.push_back( labelId );
                    }
                }
            }
        }

        fclose( file );
        ret = true;
    }

    return ret;
}

// Helper function to extract test samples (20%) out of Iris data
template <typename T> vector<T> ExtractTestSamples( vector<T>& allSamples )
{
    vector<T> testSamples( 30 );

    // Iris flower dataset contains 150 samples - 3 classes of 50 samples each.
    // Classes are not mixed and follow one after another. So take 10 samples
    // from each class and move those into test data set.

    std::move( allSamples.begin( ) +  40, allSamples.begin( ) +  50, testSamples.begin( ) );
    std::move( allSamples.begin( ) +  90, allSamples.begin( ) + 100, testSamples.begin( ) + 10 );
    std::move( allSamples.begin( ) + 140, allSamples.begin( ) + 150, testSamples.begin( ) + 20 );

    allSamples.erase( allSamples.begin( ) + 140, allSamples.begin( ) + 150 );
    allSamples.erase( allSamples.begin( ) +  90, allSamples.begin( ) + 100 );
    allSamples.erase( allSamples.begin( ) +  40, allSamples.begin( ) +  50 );

    return testSamples;
}

// Example application's entry point
int main( int argc, char** argv )
{
    printf( "Iris classification example with Fully Connected ANN \n\n" );

    vector<fvector_t> trainAttributes;
    uvector_t         trainLabels;

    if ( !LoadData( trainAttributes, trainLabels ) )
    {
        printf( "Failed loading Iris database \n\n" );
        return -1;
    }

    printf( "Loaded %zu data samples \n\n", trainAttributes.size( ) );

    // make sure we have expected number of samples
    if ( trainAttributes.size( ) != 150 )
    {
        printf( "The data set is expected to provide 150 samples \n\n" );
        return -2;
    }

    // split the data set into two: training (120 samples) and test (30 samples)
    vector<fvector_t> testAttributes = ExtractTestSamples( trainAttributes );
    uvector_t         testLabels     = ExtractTestSamples( trainLabels );

    printf( "Using %zu samples for training and %zu samples for test \n\n", trainAttributes.size( ), testAttributes.size( ) );

    // perform one hot encoding of train/test labels
    vector<fvector_t> encodedTrainLabels = XDataEncodingTools::OneHotEncoding( trainLabels, 3 );
    vector<fvector_t> encodedTestLabels  = XDataEncodingTools::OneHotEncoding( testLabels, 3 );

    // prepare a 3 layer ANN
    shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

    net->AddLayer( make_shared<XFullyConnectedLayer>( 4, 10 ) );
    net->AddLayer( make_shared<XTanhActivation>( ) );
    net->AddLayer( make_shared<XFullyConnectedLayer>( 10, 10 ) );
    net->AddLayer( make_shared<XTanhActivation>( ) );
    net->AddLayer( make_shared<XFullyConnectedLayer>( 10, 3 ) );
    net->AddLayer( make_shared<XSigmoidActivation>( ) );
  
    // create training context with Nesterov optimizer and Cross Entropy cost function
    shared_ptr<XNetworkTraining> netTraining = make_shared<XNetworkTraining>( net,
                                               make_shared<XNesterovMomentumOptimizer>( 0.01f ),
                                               make_shared<XCrossEntropyCost>( ) );

    // using the helper for training ANN to do classification
    XClassificationTrainingHelper trainingHelper( netTraining, argc, argv );
    trainingHelper.SetTestSamples( testAttributes, encodedTestLabels, testLabels );

    // 40 epochs, 10 samples in batch
    trainingHelper.RunTraining( 40, 10, trainAttributes, encodedTrainLabels, trainLabels );

    return 0;
}
