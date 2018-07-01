/*
    ANNT - Artificial Neural Networks C++ library

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

#include "XClassificationTrainingHelper.hpp"

#include <chrono>

using namespace std;
using namespace std::chrono;

namespace ANNT { namespace Neuro { namespace Training {

XClassificationTrainingHelper::XClassificationTrainingHelper( const shared_ptr<XNetworkTraining>& networkTraining ) :
    mNetworkTraining( networkTraining ),
    mEpochSelectionMode( EpochSelectionMode::Shuffle ),
    mRunPreTrainingTest( true ), mRunValidationOnly( false ),
    mShowIntermediateBatchCosts( false )
{

}

// Sets validation samples to use for validating classification after each training epch
void XClassificationTrainingHelper::SetValidationSamples( const vector<fvector_t>& validationInputs,
                                                          const vector<fvector_t>& validationOutputs,
                                                          const uvector_t& validationLabels )
{
    size_t samplesCount = validationInputs.size( );

    mValidationInputs.resize( validationInputs.size( ) );
    mValidationOutputs.resize( validationOutputs.size( ) );
    mValidationLabels = validationLabels;

    for ( size_t i = 0; i < samplesCount; i++ )
    {
        mValidationInputs[i]  = const_cast<fvector_t*>( &( validationInputs[i] ) );
        mValidationOutputs[i] = const_cast<fvector_t*>( &( validationOutputs[i] ) );
    }
}

// Sets test samples to use for testing classification after training is complete
void XClassificationTrainingHelper::SetTestSamples( const std::vector<fvector_t>& testInputs,
                                                    const std::vector<fvector_t>& testOutputs,
                                                    const uvector_t& testLabels )
{
    size_t samplesCount = testInputs.size( );

    mTestInputs.resize( testInputs.size( ) );
    mTestOutputs.resize( testOutputs.size( ) );
    mTestLabels = testLabels;

    for ( size_t i = 0; i < samplesCount; i++ )
    {
        mTestInputs[i] = const_cast<fvector_t*>( &( testInputs[i] ) );
        mTestOutputs[i] = const_cast<fvector_t*>( &( testOutputs[i] ) );
    }
}

// Helper function to show some progress bar on stdout
static void UpdatePogressBar( size_t lastProgress, size_t currentProgress, size_t totalSteps, size_t barLength, char barChar )
{
    size_t barsDone = lastProgress    * barLength / totalSteps;
    size_t barsNeed = currentProgress * barLength / totalSteps;

    while ( barsDone++ != barsNeed )
    {
        putchar( barChar );
    }
    fflush( stdout );
}

// Helper function to show/erase training epoch progress (%)
static void EraseProgress( int stringLength )
{
    while ( stringLength > 0 )
    {
        printf( "\b \b" );
        stringLength--;
    }
}
static int ShowProgress( size_t currentProgress, size_t totalSteps )
{
    int printed = printf( "<%d%%>", static_cast<int>( currentProgress * 100 / totalSteps ) );
    fflush( stdout );
    return printed;
}

// Runs training loop providing progress to stdout
void XClassificationTrainingHelper::RunTraining( size_t epochs, size_t batchSize,
                                                 const vector<fvector_t>& trainingInputs,
                                                 const vector<fvector_t>& trainingOutputs,
                                                 const uvector_t& trainingLabels )
{
    vector<fvector_t*> trainingInputsPtr( trainingInputs.size( ) );
    vector<fvector_t*> trainingOutputsPtr( trainingOutputs.size( ) );
    vector<fvector_t*> trainingInputsBatch( batchSize );
    vector<fvector_t*> trainingOutputsBatch( batchSize );

    size_t             samplesCount       = trainingInputs.size( );
    size_t             iterationsPerEpoch = ( samplesCount - 1 ) / batchSize + 1;

    float_t            cost;
    size_t             correct;

    steady_clock::time_point timeStartForAll = steady_clock::now( );
    steady_clock::time_point timeStart;
    long long                timeTaken;

    size_t             batchCostOutputFreq = iterationsPerEpoch / 80;

    int                progressStringLength = 0;

    if ( batchCostOutputFreq == 0 )
    {
        batchCostOutputFreq = 1;
    }

    // take pointers to original inputs/outputs, so those could be shuffled 
    for ( size_t i = 0; i < samplesCount; i++ )
    {
        trainingInputsPtr[i]  = const_cast<fvector_t*>( &( trainingInputs[i]  ) );
        trainingOutputsPtr[i] = const_cast<fvector_t*>( &( trainingOutputs[i] ) );
    }

    // check classification error before starting training
    if ( mRunPreTrainingTest )
    {
        timeStart = steady_clock::now( );
        correct   = mNetworkTraining->TestClassification( trainingInputs, trainingLabels, trainingOutputs, &cost );
        timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

        printf( "Before training: accuracy = %0.2f%% (%zu/%zu), cost = %0.4f, %0.3fs \n\n",
                static_cast<float>( correct ) / trainingInputs.size( ) * 100,
                correct, trainingInputs.size( ), static_cast<float>( cost ),
                static_cast<float>( timeTaken ) / 1000 );
    }

    // run the specified number of epochs
    for ( size_t epoch = 0; epoch < epochs; epoch++ )
    {
        printf( "Epoch %3zu : ", epoch + 1 );
        if ( !mShowIntermediateBatchCosts )
        {
            // show progress bar only
            putchar( '[' );
        }
        else
        {
            printf( "\n" );
        }

        // shuffle samples if required
        if ( mEpochSelectionMode == EpochSelectionMode::Shuffle )
        {
            for ( size_t i = 0; i < samplesCount / 2; i++ )
            {
                int swapIndex1 = rand( ) % samplesCount;
                int swapIndex2 = rand( ) % samplesCount;

                std::swap( trainingInputsPtr[swapIndex1],  trainingInputsPtr[swapIndex2]  );
                std::swap( trainingOutputsPtr[swapIndex1], trainingOutputsPtr[swapIndex2] );
            }
        }

        // start of epoch timing
        timeStart = steady_clock::now( );

        for ( size_t iteration = 0; iteration < iterationsPerEpoch; iteration++ )
        {
            // prepare batch inputs and ouputs
            for ( size_t i = 0; i < batchSize; i++ )
            {
                size_t sampleIndex;

                if ( mEpochSelectionMode == EpochSelectionMode::RandomPick )
                {
                    sampleIndex = rand( ) % samplesCount;
                }
                else
                {
                    sampleIndex = ( iteration * batchSize + i ) % samplesCount;
                }

                trainingInputsBatch[i]  = trainingInputsPtr[sampleIndex];
                trainingOutputsBatch[i] = trainingOutputsPtr[sampleIndex];
            }

            float_t batchCost = mNetworkTraining->TrainBatch( trainingInputsBatch, trainingOutputsBatch );

            // erase previous progress if any 
            EraseProgress( progressStringLength );

            // show cost of some batches or progress bar only
            if ( !mShowIntermediateBatchCosts )
            {
                UpdatePogressBar( iteration, iteration + 1, iterationsPerEpoch, 50, '=' );
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
            progressStringLength = ShowProgress( iteration + 1, iterationsPerEpoch );
        }

        EraseProgress( progressStringLength );
        progressStringLength = 0;

        // end of epoch timing
        timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

        // output time spent on training
        if ( !mShowIntermediateBatchCosts )
        {
            printf( "] " );
        }
        else
        {
            printf( "\nTime taken : " );
        }
        printf( "%0.3fs\n", static_cast<float>( timeTaken ) / 1000 );

        // get classification error on training data after completion of an epoch
        if ( ( !mRunValidationOnly ) || ( mValidationInputs.size( ) == 0 ) )
        {
            timeStart = steady_clock::now( );
            correct   = mNetworkTraining->TestClassification( trainingInputs, trainingLabels, trainingOutputs, &cost );
            timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

            printf( "Training accuracy = %0.2f%% (%zu/%zu), cost = %0.4f, %0.3fs \n",
                    static_cast<float>( correct ) / trainingInputs.size( ) * 100,
                    correct, trainingInputs.size( ), static_cast<float>( cost ),
                    static_cast<float>( timeTaken ) / 1000 );
        }

        // use validation set to check classification error on data not included into training
        if ( mValidationInputs.size( ) != 0 )
        {
            timeStart = steady_clock::now( );
            correct   = mNetworkTraining->TestClassification( mValidationInputs, mValidationLabels, mValidationOutputs, &cost );
            timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

            printf( "Validation accuracy = %0.2f%% (%zu/%zu), cost = %0.4f, %0.3fs \n",
                    static_cast<float>( correct ) / mValidationInputs.size( ) * 100,
                    correct, mValidationInputs.size( ), static_cast<float>( cost ),
                    static_cast<float>( timeTaken ) / 1000 );
        }
    }

    // final test on test data
    if ( mTestInputs.size( ) != 0 )
    {
        timeStart = steady_clock::now( );
        correct   = mNetworkTraining->TestClassification( mTestInputs, mTestLabels, mTestOutputs, &cost );
        timeTaken = duration_cast<milliseconds>( steady_clock::now( ) - timeStart ).count( );

        printf( "\nTest accuracy = %0.2f%% (%zu/%zu), cost = %0.4f, %0.3fs \n",
            static_cast<float>( correct ) / mTestInputs.size( ) * 100,
            correct, mTestInputs.size( ), static_cast<float>( cost ),
            static_cast<float>( timeTaken ) / 1000 );
    }

    // total time taken by the training
    timeTaken = duration_cast<seconds>( steady_clock::now( ) - timeStartForAll ).count( );
    printf( "\nTotal time taken : %ds (%0.2fmin) \n", static_cast<int>( timeTaken ), static_cast<float>( timeTaken ) / 60 );
}

} } } // namespace ANNT::Neuro::Training
