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

#include <vector>

namespace ANNT { namespace Neuro {

class XNetworkComputation;

namespace Training
{
    class XNetworkTraining;
}

// The class encapsulates some context passed to layers by inference/training classes
class XNetworkContext
{
    friend class XNetworkComputation;
    friend class Training::XNetworkTraining;

private:

    bool                             mTrainingMode;
    std::vector<std::vector<void*>>* mBuffers;

public:

    XNetworkContext( bool trainingMode ) :
        mTrainingMode( trainingMode ), mBuffers( nullptr )
    { }

    // Checks if network is being trained
    bool IsTraining( ) const { return mTrainingMode; }

    // Provides specified working buffer for the sample index
    void* GetWorkingBuffer( size_t buffer, size_t sample ) const
    {
        return (*mBuffers)[buffer][sample];
    }

protected:

    // Set working buffers for the layer this context is to be used with
    void SetWorkingBuffers( std::vector<std::vector<void*>>* buffers )
    {
        mBuffers = buffers;
    }
};

} } // namespace ANNT::Neuro

#endif // ANNT_XNETWORK_CONTEXT_HPP
