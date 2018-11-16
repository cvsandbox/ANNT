/*
    ANNT - Artificial Neural Networks C++ library

    Names generation example with Recurrent ANN

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
#include <algorithm>
#include <time.h>

#include "ANNT.hpp"

using namespace std;

using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

// File name to load training dataset from
static const char* FNAME_CITIES = "data/cities.txt";

// Some ANN training parameters
#define EPOCHS_COUNT    (20)
#define BATCH_SIZE      (48)
#define LEARNING_RATE   (0.001f)

// 26 labels for A-Z, 3 labels for '-', '.' and space and 1 for string terminator
#define LABELS_COUNT    (30)

// Number of random characters to put at the start of the generated word
#define INITIAL_RANDOM_CHAR_COUNT (1)
// Controls how often to extra random characters - every Nth.
#define RANDOM_CHAR_CYCLE         (5)

// Maximum length of generated word (in the case if ANN fails to complete it with null terminator)
#define MAX_GENERATED_LENGTH      (80)

// Number of words to generate before and after training. Before training is mostly to
// see that neural network produces some garbage. After training, however, it should
// produce something, which looks more like artificially generated "words".
#define BEFORE_TRAINING_GENERATE_COUNT  (10)
#define AFTER_TRAINING_GENERATE_COUNT   (50)

// For example, if the above two #defines are set to 2 and 4 respectively, then
// generated words will have the next pattern (R - random, N - added by ANN):
// RRNNNRNNNRNNNR ....

// Load collection of words from the specified file
static vector<string> LoadVocabulary( string fileName )
{
    vector<string> words;
    FILE*          file = fopen( fileName.c_str( ), "r" );

    if ( file == nullptr )
    {
        printf( "Error: failed opening words file \n\n" );
    }
    else
    {
        char buff[256];

        while ( fgets( buff, 256, file ) != nullptr )
        {
            size_t len = strlen( buff );

            while ( ( len > 0 ) && ( isspace( buff[len - 1] ) ) )
            {
                len--;
                buff[len] = '\0';
            }

            if ( len != 0 )
            {
                for ( size_t i = 0; i < len; i++ )
                {
                    buff[i] = static_cast<char>( toupper( buff[i] ) );

                    // allow only characters A-Z, dot, dash and space
                    if ( ( ( buff[i] < 'A' ) || ( buff[i] > 'Z' ) ) && ( buff[i] != '.' ) && ( buff[i] != '-' ) && ( buff[i] != ' ' ) )
                    {
                        printf( "Warning: found unsupported character '%c' in word '%s' \n\n", buff[i], buff );
                        buff[i] = ' ';
                    }
                }

                words.push_back( buff );
            }
        }

        if ( words.empty( ) )
        {
            printf( "Error: did not find any words \n\n" );
        }
    }

    return words;
}

// Get maximum length of strings in the specified vector
static size_t MaxWordLength( const vector<string>& words )
{
    size_t maxLen = 0;

    for ( size_t i = 0, n = words.size( ); i < n; i++ )
    {
        size_t len = words[i].length( );

        if ( len > maxLen )
        {
            maxLen = len;
        }
    }

    return maxLen;
}

// Convert character to label to be encoded into ANN's input
static size_t CharToLabel( char c )
{
    size_t label = 0; // default to null terminator

    if ( ( c >= 'A' ) && ( c <= 'Z' ) )
    {
        label = c - 'A' + 1;
    }
    else if ( ( c >= 'a' ) && ( c <= 'z' ) )
    {
        label = c - 'a' + 1;
    }
    else if ( c == '.' )
    {
        label = 27;
    }
    else if ( c == '-' )
    {
        label = 28;
    }
    else if ( c != '\0' ) // anything else is treated as space
    {
        label = 29;
    }

    return label;
}

// Convert label decoded from ANN's output into character
static char LabelToChar( size_t label )
{
    char c = '\0';

    if ( ( label >= 1 ) && ( label <= 26 ) )
    {
        c = static_cast<char>( 'A' + ( label - 1 ) );
    }
    else if ( label == 27 )
    {
        c = '.';
    }
    else if ( label == 28 )
    {
        c = '-';
    }
    else if ( label == 29 )
    {
        c = ' ';
    }

    return c;
}

// Extract training sequence elements from training words
static void ExtractSamplesAsSequence( const vector<string>& words,
                                      vector<fvector_t>& inputSequence, vector<fvector_t>& outputSequence,
                                      size_t samplesToExtract, size_t startIndex, size_t sequenceLength )
{
    size_t i            = startIndex;
    size_t totalSamples = words.size( );

    inputSequence.clear( );
    outputSequence.clear( );

    for ( size_t j = 0; j < samplesToExtract; j++ )
    {
        const string& word     = words[i];
        size_t        len      = word.length( );
        char          prevChar = word[0];

        for ( size_t k = 1; k <= sequenceLength; k++ )
        {
            inputSequence.push_back( XDataEncodingTools::OneHotEncoding( CharToLabel( prevChar ), LABELS_COUNT ) );
            prevChar = ( k < len ) ? word[k] : '\0';
            outputSequence.push_back( XDataEncodingTools::OneHotEncoding( CharToLabel( prevChar ), LABELS_COUNT ) );
        }

        i = ( i + 1 ) % totalSamples;
    }
}

// Generates specified number of words using recurrent neural network
static void GenerateWords( shared_ptr<XNeuralNetwork> net, const vector<string>& existingWords, size_t toGenerate )
{
    XNetworkInference netInference( net );
    fvector_t         input;
    fvector_t         output;

    srand( static_cast<int>( time( nullptr ) ) );

    for ( size_t i = 0; i < toGenerate; i++ )
    {
        char    nextChar      = static_cast<char>( 'A' + ( rand( ) % 26 ) );
        bool    keepUpperCase = true;
        string  word;

        while ( ( nextChar != '\0' ) && ( word.length( ) < MAX_GENERATED_LENGTH ) )
        {
            word += ( keepUpperCase ) ? nextChar : static_cast<char>( tolower( nextChar ) );

            keepUpperCase = ( ( nextChar == ' ' ) || ( nextChar == '-' ) || ( nextChar == '.' ) );

            if ( // put random character after space or dash
                 ( nextChar == ' ' ) || ( nextChar == '-' ) ||
                 // make sure we have enough random characters at the start
                 ( word.length( ) < INITIAL_RANDOM_CHAR_COUNT ) ||
                 // keep repeating random characters after every Nth
                 ( ( RANDOM_CHAR_CYCLE > 1 ) && ( ( ( word.length( ) - INITIAL_RANDOM_CHAR_COUNT + 1 ) % RANDOM_CHAR_CYCLE ) == 0 ) ) )
            {
                nextChar = static_cast<char>( 'A' + ( rand( ) % 26 ) );
            }
            else
            {
                // generate next character using the ANN
                input = XDataEncodingTools::OneHotEncoding( CharToLabel( nextChar ), LABELS_COUNT );

                netInference.Compute( input, output );

                nextChar = LabelToChar( XDataEncodingTools::MaxIndex( output ) );
            }
        }
        netInference.ResetState( );

        printf( "%s - ", word.c_str( ) );

        if ( std::find( existingWords.begin( ), existingWords.end( ), word ) == existingWords.end( ) )
        {
            printf( "New word \n" );
        }
        else
        {
            printf( "Training word \n" );
        }
    }
}

// Example application's entry point
int main( int /* argc */, char** /* argv */ )
{
#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemState memState;
    _CrtMemCheckpoint( &memState );
#endif

    printf( "Names generation example with Recurrent ANN \n\n" );

    {
        vector<string> trainingWords = LoadVocabulary( FNAME_CITIES );

        if ( !trainingWords.empty( ) )
        {
            size_t samplesCount  = trainingWords.size( );
            size_t maxWordLength = MaxWordLength( trainingWords );

            printf( "Loaded %zu words for training \n", samplesCount );
            printf( "Maximum word length: %zu \n\n", maxWordLength );

            // prepare a recurrent ANN
            shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

            net->AddLayer( make_shared<XGRULayer>( LABELS_COUNT, 60 ) );
            net->AddLayer( make_shared<XFullyConnectedLayer>( 60, LABELS_COUNT ) );
            net->AddLayer( make_shared<XSoftMaxActivation>( ) );

            // generate some words using untrained ANN - should produce some garbage
            #if BEFORE_TRAINING_GENERATE_COUNT > 0
                printf( "Generated words before training: \n" );
                GenerateWords( net, trainingWords, BEFORE_TRAINING_GENERATE_COUNT );
                printf( "\n" );
            #endif

            // create training context with Adam optimizer and Cross Entropy cost function
            XNetworkTraining netTraining( net,
                                          make_shared<XAdamOptimizer>( LEARNING_RATE ),
                                          make_shared<XCrossEntropyCost>( ) );

            netTraining.SetAverageWeightGradients( false );
            netTraining.SetTrainingSequenceLength( maxWordLength ); /* sequence length as per the longest word */

            // run the specified number of epochs
            size_t iterationsPerEpoch  = ( samplesCount - 1 ) / BATCH_SIZE + 1;
            size_t batchCostOutputFreq = iterationsPerEpoch / 10;

            if ( batchCostOutputFreq == 0 )
            {
                batchCostOutputFreq = 1;
            }

            vector<fvector_t> inputs;
            vector<fvector_t> outputs;

            for ( size_t epoch = 0; epoch < EPOCHS_COUNT; epoch++ )
            {
                printf( "Epoch %zu \n", epoch + 1 );

                // shuffle training samples
                for ( size_t i = 0; i < samplesCount / 2; i++ )
                {
                    int swapIndex1 = rand( ) % samplesCount;
                    int swapIndex2 = rand( ) % samplesCount;

                    std::swap( trainingWords[swapIndex1], trainingWords[swapIndex2] );
                }

                for ( size_t iteration = 0; iteration < iterationsPerEpoch; iteration++ )
                {
                    // prepare batch inputs and ouputs
                    ExtractSamplesAsSequence( trainingWords, inputs, outputs, BATCH_SIZE, iteration * BATCH_SIZE, maxWordLength );

                    auto batchCost = netTraining.TrainBatch( inputs, outputs );
                    netTraining.ResetState( );

                    if ( ( ( iteration + 1 ) % batchCostOutputFreq ) == 0 )
                    {
                        printf( "%0.4f ", static_cast<float>( batchCost ) );
                    }
                }
                printf( "\n" );
            }
            printf( "\n" );

            // generate some words using trained ANN - should produce something more interesting
            #if AFTER_TRAINING_GENERATE_COUNT > 0
                printf( "Generated words after training: \n" );
                GenerateWords( net, trainingWords, AFTER_TRAINING_GENERATE_COUNT );
                printf( "\n" );
            #endif
        }
    }

#if defined(_MSC_VER) && defined(_DEBUG)
    _CrtMemDumpAllObjectsSince( &memState );
#endif

    return 0;
}
