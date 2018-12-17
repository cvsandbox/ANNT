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
#ifndef ANNT_XNETWORK_CONTEXT_HPP
#define ANNT_XNETWORK_CONTEXT_HPP

#include "../../Types/Types.hpp"

namespace ANNT { namespace Neuro {

class XNeuralNetwork;
class XNetworkInference;

namespace Training
{
    class XNetworkTraining;
}

// The class encapsulates some context passed to layers by inference/training classes
class XNetworkContext
{
    friend class XNetworkInference;
    friend class Training::XNetworkTraining;

private:

    bool    mTrainingMode;
    size_t  mTrainingSequenceLength;   // length of sequences used to train recurrent networks
    size_t  mCurrentLayer;


    std::vector<std::vector<std::vector<void*>>> mLayersMemoryBuffers;
    std::vector<uvector_t>                       mLayersMemorySize;

public:

    XNetworkContext( bool trainingMode ) :
        XNetworkContext( trainingMode, 1 )
    { }

    XNetworkContext( bool trainingMode, size_t sequenceLength );
    ~XNetworkContext( );

    // Checks if network is being trained
    bool IsTraining( ) const { return mTrainingMode; }

    // Get/set length of training sequences used for recurrent networks
    size_t TrainingSequenceLength( ) const
    {
        return mTrainingSequenceLength;
    }
    void SetTrainingSequenceLength( size_t sequenceLength )
    {
        mTrainingSequenceLength = sequenceLength;
    }

    // Provides specified working buffer for the sample index
    void* GetWorkingBuffer( size_t buffer, size_t sample ) const
    {
        return mLayersMemoryBuffers[mCurrentLayer][buffer][sample];
    }

protected:

    // Allocate working buffer for layers of the network
    void AllocateWorkingBuffers( const std::shared_ptr<XNeuralNetwork>& net, size_t batchSize );

    // Clear layers' working buffers (memset zero)
    void ResetWorkingBuffers( );
    void ResetWorkingBuffers( uvector_t layersIndexes );

    // Set current layer index, so that correct working buffer could be provided
    void SetCurrentLayerIndex( size_t currentLayer )
    {
        mCurrentLayer = currentLayer;
    }

private:

    // Free layers' working buffers
    void FreeWorkingBuffers( );
};

} } // namespace ANNT::Neuro

#endif // ANNT_XNETWORK_CONTEXT_HPP
