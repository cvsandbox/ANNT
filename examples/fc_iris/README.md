# Iris flower classification with Fully Connected ANN

This example performs a simple multiclass classification of [Iris flowers](../data/iris/). First it creates a simple 3 layers network – 10 neurons in each hidden layer and 3 neurons in the output layer (as we have 3 classes):

```C++
// prepare a 3 layer ANN
shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

net->AddLayer( make_shared<XFullyConnectedLayer>( 4, 10 ) );
net->AddLayer( make_shared<XTanhActivation>( ) );
net->AddLayer( make_shared<XFullyConnectedLayer>( 10, 10 ) );
net->AddLayer( make_shared<XTanhActivation>( ) );
net->AddLayer( make_shared<XFullyConnectedLayer>( 10, 3 ) );
net->AddLayer( make_shared<XSigmoidActivation>( ) );
```

Since we have multiple classes, the Cross Entropy (multiclass one, not the binary) cost function is used:
```C++
// create training context with Nesterov optimizer and Cross Entropy cost function
shared_ptr<XNetworkTraining> netTraining = make_shared<XNetworkTraining>( net,
                                           make_shared<XNesterovMomentumOptimizer>( 0.01f ),
                                           make_shared<XCrossEntropyCost>( ) );
```

Unlike with [XOR example](../fc_xor), this one does not run training loop by itself. Instead a helper class used, which does all the job – training, validation, testing, etc. and provides a way of customizing some training parameters from command line:

```C++
// using the helper for training ANN to do classification
XClassificationTrainingHelper trainingHelper( netTraining, argc, argv );
trainingHelper.SetTestSamples( testAttributes, encodedTestLabels, testLabels );

// 40 epochs, 10 samples in batch
trainingHelper.RunTraining( 40, 10, trainAttributes, encodedTrainLabels, trainLabels );
```

## Sample output

```
Iris classification example with Fully Connected ANN

Loaded 150 data samples

Using 120 samples for training and 30 samples for test

Learning rate: 0.0100, Epochs: 40, Batch Size: 10

Before training: accuracy = 33.33% (40/120), cost = 0.5627, 0.000s

Epoch   1 : [==================================================] 0.005s
Training accuracy = 33.33% (40/120), cost = 0.3154, 0.000s
Epoch   2 : [==================================================] 0.003s
Training accuracy = 86.67% (104/120), cost = 0.1649, 0.000s

...

Epoch  40 : [==================================================] 0.006s
Training accuracy = 93.33% (112/120), cost = 0.0064, 0.000s

Test accuracy = 96.67% (29/30), cost = 0.0064, 0.000s

Total time taken : 0s (0.00min)
```

## Command line options
Some of the useful command line options are:
* -ec:<> - epochs count;
* -bs:<> - batch size;
* -lr:<> - learning rate.

The full list can be obtained by running the application with ‘-?’ option.
