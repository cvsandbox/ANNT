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

#pragma once
#ifndef ANNT_XBATCH_NORMALIZATION_HPP
#define ANNT_XBATCH_NORMALIZATION_HPP

#include "IProcessingLayer.hpp"
#include "../../../Tools/XParallel.hpp"
#include "../../../Tools/XVectorize.hpp"
#include <cstring>
#include <numeric>

namespace ANNT { namespace Neuro {

// Implementation of the layer performaing normalization over batch data
class XBatchNormalization : public IProcessingLayer
{
private:
    bool      mFirstUpdate;

    size_t    mSpatialSize;
    size_t    mInputDepth;

    float_t   mMomentum;
    float_t   mEpsilon;

    // learn't mean and std.dev.
    fvector_t mMean;
    fvector_t mStdDev;

    enum
    {
        BUFFER_INDEX_LEARNT_VARIANCE        = 0, // must stay alive, so buffer can not be reused in backward pass
        BUFFER_INDEX_BATCH_MEAN             = 1,
        BUFFER_INDEX_BATCH_VARIANCE         = 2,
        BUFFER_INDEX_BATCH_STD_DEV          = 3,
        BUFFER_INDEX_DELTAS_DOT_OUTPUT_MEAN = 1,
        BUFFER_INDEX_DELTAS_MEAN            = 2,
    };

public:
    XBatchNormalization( size_t inputWidth, size_t inputHeight, size_t inputDepth, float_t momentum = float_t( 0.999 ) ) :
        IProcessingLayer( inputWidth * inputHeight * inputDepth, inputWidth * inputHeight * inputDepth ),
        mFirstUpdate( true ),
        mSpatialSize( inputWidth * inputHeight ), mInputDepth( inputDepth ),
        mMomentum( momentum ), mEpsilon( float_t( 0.00001 ) )
    {
        mMean   = fvector_t( mInputDepth, float_t( 0.0f ) );
        mStdDev = fvector_t( mInputDepth, float_t( 1.0f ) );
    }

    // Tells that we may need some extra memory for keeping temporary calculations
    uvector_t WorkingMemSize( bool /* trainingMode */ ) const override
    {
        uvector_t workingMemSize( 4 );

        // these buffers we don't need per sample - only per batch, so there is a bit of memory waste

        // forward pass - learning variance
        workingMemSize[BUFFER_INDEX_LEARNT_VARIANCE] = mInputDepth *  sizeof( float_t );

        // forward pass - batch mean while training
        // backward pass - mean of deltas[i]*output[i]
        workingMemSize[BUFFER_INDEX_BATCH_MEAN] = mInputDepth *  sizeof( float_t );

        // forward pass - batch variance while training
        // backward pass - mean of deltas
        workingMemSize[BUFFER_INDEX_BATCH_VARIANCE] = mInputDepth *  sizeof( float_t );

        // forward pass - batch std.dev while training
        workingMemSize[BUFFER_INDEX_BATCH_STD_DEV] = mInputDepth *  sizeof( float_t );

        return workingMemSize;
    }

    // Calculates outputs for the given inputs
    void ForwardCompute( const std::vector<fvector_t*>& inputs,
                         std::vector<fvector_t*>& outputs,
                         const XNetworkContext& ctx ) override
    {
        float_t* batchMean      = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_BATCH_MEAN, 0 ) );
        float_t* batchVariance  = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_BATCH_VARIANCE, 0 ) );
        float_t* batchStdDEv    = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_BATCH_STD_DEV, 0 ) );

        float_t* learntMean     = mMean.data( );
        float_t* learntVariance = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_LEARNT_VARIANCE, 0 ) );
        float_t* learntStdDev   = mStdDev.data( );

        float_t* meanToUse      = ( ctx.IsTraining( ) ) ? batchMean   : learntMean;
        float_t* stdDevToUse    = ( ctx.IsTraining( ) ) ? batchStdDEv : learntStdDev;

        if ( ctx.IsTraining( ) )
        {
            CalculateMeanAndVariance( inputs, batchMean, batchVariance );
            CalculateStdDev( batchVariance, batchStdDEv );
        }

        XParallel::For( inputs.size( ), ctx.IsTraining( ), [&]( size_t i )
        {
            for ( size_t depthIndex = 0; depthIndex < mInputDepth; depthIndex++ )
            {
                float_t meanValue   = meanToUse[depthIndex];
                float_t stdDevValue = stdDevToUse[depthIndex];

                const float_t* input  = inputs[i]->data( )  + depthIndex * mSpatialSize;
                float_t*       output = outputs[i]->data( ) + depthIndex * mSpatialSize;

                for ( size_t spatialIndex = 0; spatialIndex < mSpatialSize; spatialIndex++, input++, output++ )
                {
                    *output = ( *input - meanValue ) / stdDevValue;
                }
            }
        } );

        if ( ctx.IsTraining( ) )
        {
            if ( mFirstUpdate )
            {
                memcpy( learntMean, batchMean, mInputDepth * sizeof( float_t ) );
                memcpy( learntVariance, batchVariance, mInputDepth * sizeof( float_t ) );

                mFirstUpdate = false;
            }
            else
            {
                float_t antiMomentum = float_t( 1 ) - mMomentum;

                // update learnt mean and variance
                for ( size_t depthIndex = 0; depthIndex < mInputDepth; depthIndex++ )
                {
                    learntMean[depthIndex]     = mMomentum * learntMean[depthIndex]     + antiMomentum * batchMean[depthIndex];
                    learntVariance[depthIndex] = mMomentum * learntVariance[depthIndex] + antiMomentum * batchVariance[depthIndex];
                }
            }

            // calculate std dev on learnt variance
            CalculateStdDev( learntVariance, learntStdDev );
        }
    }

    // Propagates error to the previous layer
    void BackwardProcess( const std::vector<fvector_t*>& /* inputs  */,
                          const std::vector<fvector_t*>& outputs,
                          const std::vector<fvector_t*>& deltas,
                          std::vector<fvector_t*>& prevDeltas,
                          const XNetworkContext& ctx ) override
    {
        float_t* deltasDotOutputsMean = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_DELTAS_DOT_OUTPUT_MEAN, 0 ) );
        float_t* deltasMean           = static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_DELTAS_MEAN, 0 ) );

        const float_t* stdDevToUse = ( ctx.IsTraining( ) ) ? static_cast<float_t*>( ctx.GetWorkingBuffer( BUFFER_INDEX_BATCH_STD_DEV, 0 ) ) : mStdDev.data( );

        // calculate mean for delta[i]*output[i]
        XParallel::For( mInputDepth, mInputDepth != 1, [&]( size_t depthIndex )
        {
            deltasDotOutputsMean[depthIndex] = float_t( 0 );

            for ( size_t i = 0, n = outputs.size( ); i < n; i++ )
            {
                const float_t* output = outputs[i]->data( ) + depthIndex * mSpatialSize;
                const float_t* delta  = deltas[i]->data( )  + depthIndex * mSpatialSize;

                deltasDotOutputsMean[depthIndex] += XVectorize::Dot( delta, output, mSpatialSize ) / mSpatialSize;
            }

            deltasDotOutputsMean[depthIndex] /= outputs.size( );
        } );

        CalculateMean( deltas, deltasMean );

        XParallel::For( outputs.size( ), ctx.IsTraining( ), [&]( size_t i )
        {
            const fvector_t& output    = *( outputs[i] );
            const fvector_t& delta     = *( deltas[i] );
            fvector_t&       prevDelta = *( prevDeltas[i] );

            for ( size_t depthIndex = 0; depthIndex < mInputDepth; depthIndex++ )
            {
                for ( size_t spatialIndex = 0, j = depthIndex * mSpatialSize; spatialIndex < mSpatialSize; spatialIndex++, j++ )
                {
                    prevDelta[j] = ( delta[j] - deltasMean[depthIndex] - deltasDotOutputsMean[depthIndex] * output[j] ) / stdDevToUse[depthIndex];
                }
            }
        } );
    }

    // Saves layer's learnt parameters/weights
    bool SaveLearnedParams( FILE* file ) const override
    {
        std::vector<const fvector_t*> params( { &mMean, &mStdDev } );

        return SaveLearnedParamsHelper( file, LayerID::BatchNormalization, params );
    }

    // Loads layer's learnt parameters
    bool LoadLearnedParams( FILE* file ) override
    {
        std::vector<fvector_t*> params( { &mMean, &mStdDev } );

        return LoadLearnedParamsHelper( file, LayerID::BatchNormalization, params );
    }

private:

    void CalculateMean( const std::vector<fvector_t*>& inputs, float_t* mean ) const
    {
        //for ( size_t depthIndex = 0; depthIndex < mInputDepth; depthIndex++ )
        XParallel::For( mInputDepth, mInputDepth != 1, [&]( size_t depthIndex )
        {
            mean[depthIndex] = float_t( 0 );

            for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
            {
                mean[depthIndex] += std::accumulate( inputs[i]->begin( ) +   depthIndex       * mSpatialSize,
                                                     inputs[i]->begin( ) + ( depthIndex + 1 ) * mSpatialSize,
                                                     float_t( 0 ) ) / mSpatialSize;
            }

            mean[depthIndex] /= inputs.size( );
        } );
    }

    void CalculateMeanAndVariance( const std::vector<fvector_t*>& inputs, float_t* mean, float_t* variance ) const
    {
        CalculateMean( inputs, mean );

        XParallel::For( mInputDepth, mInputDepth != 1, [&]( size_t depthIndex )
        {
            float_t meanValue = mean[depthIndex];
            float_t diff;

            variance[depthIndex] = float_t( 0 );

            for ( size_t i = 0, n = inputs.size( ); i < n; i++ )
            {
                const float_t* input = inputs[i]->data( ) + depthIndex * mSpatialSize;
                float_t        sum   = float_t( 0 );

                for ( size_t spatialIndex = 0; spatialIndex < mSpatialSize; spatialIndex++, input++ )
                {
                    diff = *input - meanValue;
                    sum += diff * diff;
                }

                variance[depthIndex] += sum / mSpatialSize;
            }

            variance[depthIndex] /= inputs.size( );
        } );
    }

    void CalculateStdDev( const float_t* variance, float_t* stdDev )
    {
        for ( size_t depthIndex = 0; depthIndex < mInputDepth; depthIndex++ )
        {
            stdDev[depthIndex] = std::sqrt( variance[depthIndex] + mEpsilon );
        }
    }
};

} } // ANNT::Neuro

#endif // ANNT_XBATCH_NORMALIZATION_HPP

