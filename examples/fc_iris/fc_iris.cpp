// fc_iris.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <iterator>
#include <vector>
#include <map>

#include "ANNT.hpp"

using namespace std;
using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

// Name of the Iris data file, which is publicly  available using the link below
// http://archive.ics.uci.edu/ml/datasets/Iris
static const char* IRIS_DATA_FILE = "data/iris.data";

// Helper function to load Iris data set
bool LoadData( vector<vector_t>& attributes, vector<uint16_t>& classes )
{
    bool  ret  = false;
    FILE* file = fopen( IRIS_DATA_FILE, "r" );

    if ( file )
    {
        char     buff[256];
        uint16_t classCounter = 0;

        map<string, uint16_t> classMap;

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

                    uint16_t classId = classCounter;
                    auto     classIt = classMap.find( classNamePtr );

                    if ( classIt != classMap.end( ) )
                    {
                        classId = classIt->second;
                    }
                    else
                    {
                        classMap.insert( pair<string, uint16_t>( classNamePtr, classCounter ) );
                        classCounter++;
                    }

                    // Iris data set has only 3 classes, so ignore anything else
                    if ( classId <= 2 )
                    {
                        attributes.push_back( vector_t( { attr1, attr2, attr3, attr4 } ) );
                        classes.push_back( classId );
                    }
                }
            }
        }

        fclose( file );
        ret = true;
    }

    return ret;
}

vector<vector_t> OneHotEncoding( const vector<uint16_t>& classes, uint16_t classesCount )
{
    vector<vector_t> encodedClasses( classes.size( ) );

    for ( size_t i = 0; i < classes.size( ); i++ )
    {
        encodedClasses[i] = vector_t( classesCount );
        encodedClasses[i][classes[i]] = 1;
    }

    return encodedClasses;
}

uint16_t MaxIndex( const vector_t& vec )
{
    uint16_t maxIndex = 0;
    auto     maxValue = vec[0];

    for ( size_t i = 1, n = vec.size( ); i < n; i++ )
    {
        if ( vec[i] > maxValue )
        {
            maxValue = vec[i];
            maxIndex = static_cast<uint16_t>( i );
        }
    }

    return maxIndex;
}

int main( int /* argc */, char** /* argv */ )
{
    printf( "Iris classification example with Fully Connected ANN \n\n" );

    vector<vector_t> attributes;
    vector<uint16_t> classes;
    vector<vector_t> encodedClasses;

    if ( !LoadData( attributes, classes ) )
    {
        printf( "Failed loading Iris database \n\n" );
        return -1;
    }

    printf( "Loaded %d data samples \n\n", attributes.size( ) );

    encodedClasses = OneHotEncoding( classes, 3 );

    // prepare a 3 layer ANN
    shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

    net->AddLayer( make_shared<XFullyConnectedLayer>( 4, 10 ) );
    net->AddLayer( make_shared<XTanhActivation>( ) );
    net->AddLayer( make_shared<XFullyConnectedLayer>( 10, 10 ) );
    net->AddLayer( make_shared<XTanhActivation>( ) );
    net->AddLayer( make_shared<XFullyConnectedLayer>( 10, 3 ) );
    net->AddLayer( make_shared<XSigmoidActivation>( ) );
    //net->AddLayer( make_shared<XSoftMaxActivation>( ) );
  
    // create training context with Nesterov optimizer and MSE cost function
    XNetworkTraining netCtx( net,
                             //make_shared<XAdamOptimizer>( 0.01f ),
                             make_shared<XAdamOptimizer>( 0.0075f ),
                             //make_shared<XNesterovMomentumOptimizer>( 0.001f ),
                             make_shared<XBinaryCrossEntropyCost>( ) );
                    
    printf( "Cost of each sample: \n" );
    for ( size_t i = 0; i < 150 * 20; i++ )
    {
        size_t sample = rand( ) % attributes.size( );
        auto   cost   = netCtx.TrainSample( attributes[sample], encodedClasses[sample] );

        printf( "%.04f ", static_cast<float>( cost ) );

        if ( ( i % 8 ) == 7 )
            printf( "\n" );
    }
    printf( "\n" );

    float    cost = 0.0f;
    uint16_t correct = 0;
    vector_t output;

    for ( size_t i = 0; i < attributes.size( ); i++ )
    {
        cost += netCtx.TestSample( attributes[i], encodedClasses[i], output );

        uint16_t maxIndex = MaxIndex( output );

        if ( maxIndex == classes[i] )
        {
            correct++;
        }
    }

    cost /= attributes.size( );

    printf( "Correct classifications: %u \n", correct );
    printf( "Average cost: %f \n", cost );

    return 0;
}

