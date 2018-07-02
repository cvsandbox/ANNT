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
#ifndef ANNT_XCLASSIFICATION_TRAINING_HELPER_HPP
#define ANNT_XCLASSIFICATION_TRAINING_HELPER_HPP

#include "XNetworkTraining.hpp"

namespace ANNT { namespace Neuro { namespace Training {

// A helper class which encapsulates training task of a classification problem
class XClassificationTrainingHelper
{
private:
    std::shared_ptr<XNetworkTraining>  mNetworkTraining;
    EpochSelectionMode                 mEpochSelectionMode;

    bool                               mRunPreTrainingTest;
    bool                               mRunValidationOnly;
    bool                               mShowIntermediateBatchCosts;

    std::vector<fvector_t*>            mValidationInputs;
    std::vector<fvector_t*>            mValidationOutputs;
    uvector_t                          mValidationLabels;

    std::vector<fvector_t*>            mTestInputs;
    std::vector<fvector_t*>            mTestOutputs;
    uvector_t                          mTestLabels;

    int    mArgc;
    char** mArgv;

public:
    XClassificationTrainingHelper( const std::shared_ptr<XNetworkTraining>& networkTraining,
                                   int argc = 0, char** argv = nullptr );

    // Get/set the mode of selecting data samples while running training epoch
    EpochSelectionMode SamplesSelectionMode( ) const
    {
        return mEpochSelectionMode;
    }
    void SetSamplesSelectionMode( EpochSelectionMode selectionMode )
    {
        mEpochSelectionMode = selectionMode;
    }

    // Run or not pre training test on training data to see the initial classification error
    bool RunPreTrainingTest( ) const
    {
        return mRunPreTrainingTest;
    }
    void SetRunPreTrainingTest( bool runIt )
    {
        mRunPreTrainingTest = runIt;
    }

    // Run validation only after each training epoch or classification test on training set as well
    bool RunValidationOnly( ) const
    {
        return mRunValidationOnly;
    }
    void SetRunValidationOnly( bool validationOnly )
    {
        mRunValidationOnly = validationOnly;
    }

    // Show cost of some training batches or progress bar
    bool ShowIntermediateBatchCosts( ) const
    {
        return mShowIntermediateBatchCosts;
    }
    void SetShowIntermediateBatchCosts( bool showBatchCost )
    {
        mShowIntermediateBatchCosts = showBatchCost;
    }

    // Sets validation samples to use for validating classification after each training epch
    // (takes pointers of inputs/outputs, so original data must stay alive)
    void SetValidationSamples( const std::vector<fvector_t>& validationInputs,
                               const std::vector<fvector_t>& validationOutputs,
                               const uvector_t& validationLabels );

    // Sets test samples to use for testing classification after training is complete
    // (takes pointers of inputs/outputs, so original data must stay alive)
    void SetTestSamples( const std::vector<fvector_t>& testInputs,
                         const std::vector<fvector_t>& testOutputs,
                         const uvector_t& testLabels );

    // Runs training loop providing progress to stdout
    void RunTraining( size_t epochs, size_t batchSize,
                      const std::vector<fvector_t>& trainingInputs,
                      const std::vector<fvector_t>& trainingOutputs,
                      const uvector_t& trainingLabels );
};

} } } // namespace ANNT::Neuro::Training

#endif // ANNT_XCLASSIFICATION_TRAINING_HELPER_HPP
