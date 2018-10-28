# CIFAR-10 classification example with Convolutional ANN

This example performs classification of color images from [CIFAR-10 dataset]( https://www.cs.toronto.edu/~kriz/cifar.html) using convolutional ANN - 3 convolutional layers followed by 2 fully connected layers. All convolution layers are followed by Max Pooling. Batch Normalization is also put in between every layer. The structure of the network can be described as:

```
Conv(32x32x3,  5x5x32, BorderMode::Same) -> ReLU -> MaxPool -> BatchNorm
Conv(16x16x32, 5x5x32, BorderMode::Same) -> ReLU -> MaxPool -> BatchNorm
Conv(8x8x32,   5x5x64, BorderMode::Same) -> ReLU -> MaxPool -> BatchNorm
FC(1024, 64) -> ReLU -> BatchNorm
FC(64, 10) -> SoftMax
```

Translating the above neural network’s structure into the code gives the next result. Note: since ReLU(MaxPool) produces same result as MaxPool(ReLU), we use the first as it reduces ReLU computation by 75% (although very negligible compared to the rest of the network).

```C++
// prepare a convolutional ANN
shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

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
```

Similar to other classification examples, the application creates training context specifying cost function and weights’ optimizer. It is then passed to classification training helper class, which runs the training/validation loop and performs the final test.

```C++
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
```

## Sample output

```
CIFAR-10 dataset classification example with Convolutional ANN

Loaded 50000 training data samples
Loaded 10000 test data samples

Samples usage: training = 43750, validation = 6250, test = 10000

Learning rate: 0.0010, Epochs: 20, Batch Size: 50

Before training: accuracy = 9.91% (4336/43750), cost = 2.3293, 844.825s

Epoch   1 : [==================================================] 1725.516s
Training accuracy = 48.25% (21110/43750), cost = 1.9622, 543.087s
Validation accuracy = 47.46% (2966/6250), cost = 2.0036, 77.284s
Epoch   2 : [==================================================] 1742.268s
Training accuracy = 54.38% (23793/43750), cost = 1.3972, 568.358s
Validation accuracy = 52.93% (3308/6250), cost = 1.4675, 76.287s
...
Epoch  19 : [==================================================] 1642.750s
Training accuracy = 90.34% (39522/43750), cost = 0.2750, 599.431s
Validation accuracy = 69.07% (4317/6250), cost = 1.2472, 81.053s
Epoch  20 : [==================================================] 1708.940s
Training accuracy = 91.27% (39931/43750), cost = 0.2484, 578.551s
Validation accuracy = 69.15% (4322/6250), cost = 1.2735, 81.037s

Test accuracy = 68.34% (6834/10000), cost = 1.3218, 122.455s

Total time taken : 48304s (805.07min)
```

## Command line options

Some of the useful command line options are:
* -ec:<> - epochs count;
* -bs:<> - batch size;
* -lr:<> - learning rate;
* -fout:<> - file name to save network's weights;
* -fin:<> - file name to load network's weights from (further training with different hyperparameters, for example).

The full list can be obtained by running the application with '-?' option.
