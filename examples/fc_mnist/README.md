# MNIST classification example with Fully Connected ANN

This example performs classification of [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) using fully connected 3 layers ANN - 300 neurons in the first hidden layer, 100 neurons in the second and finally 10 neurons in the output layer.

```C++
// prepare a 3 layer ANN
shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

net->AddLayer( make_shared<XFullyConnectedLayer>( trainImages[0].size( ), 300 ) );
net->AddLayer( make_shared<XTanhActivation>( ) );
net->AddLayer( make_shared<XFullyConnectedLayer>( 300, 100 ) );
net->AddLayer( make_shared<XTanhActivation>( ) );
net->AddLayer( make_shared<XFullyConnectedLayer>( 100, 10 ) );
net->AddLayer( make_shared<XSoftMaxActivation>( ) );
```

It uses Adam optimizer with Cross Entropy cost function for training then:
```C++
// create training context with Adam optimizer and Cross Entropy cost function
shared_ptr<XNetworkTraining> netTraining = make_shared<XNetworkTraining>( net,
                                           make_shared<XAdamOptimizer>( 0.001f ),
                                           make_shared<XCrossEntropyCost>( ) );
```

Similar to [Iris classification example](../fc_iris), this example trusts the training routine to a helper class, which gets training/validation/test data and runs the training loop for the specified number of epochs.

```C++
// using the helper for training ANN to do classification
XClassificationTrainingHelper trainingHelper( netTraining, argc, argv );
trainingHelper.SetValidationSamples( validationImages, encodedValidationLabels, validationLabels );
trainingHelper.SetTestSamples( testImages, encodedTestLabels, testLabels );

// 20 epochs, 50 samples in batch
trainingHelper.RunTraining( 20, 50, trainImages, encodedTrainLabels, trainLabels );
```

## Sample output

```
MNIST handwritten digits classification example with Fully Connected ANN

Loaded 60000 training data samples
Loaded 10000 test data samples

Samples usage: training = 50000, validation = 10000, test = 10000

Learning rate: 0.0010, Epochs: 20, Batch Size: 50

Before training: accuracy = 10.17% (5087/50000), cost = 2.4892, 2.377s

Epoch   1 : [==================================================] 59.215s
Training accuracy = 92.83% (46414/50000), cost = 0.2349, 3.654s
Validation accuracy = 93.15% (9315/10000), cost = 0.2283, 0.636s
Epoch   2 : [==================================================] 61.675s
Training accuracy = 94.92% (47459/50000), cost = 0.1619, 2.685s
Validation accuracy = 94.91% (9491/10000), cost = 0.1693, 0.622s
...
Epoch  19 : [==================================================] 59.822s
Training accuracy = 96.81% (48404/50000), cost = 0.0978, 2.976s
Validation accuracy = 95.88% (9588/10000), cost = 0.1491, 0.527s
Epoch  20 : [==================================================] 87.108s
Training accuracy = 97.77% (48883/50000), cost = 0.0688, 2.823s
Validation accuracy = 96.60% (9660/10000), cost = 0.1242, 0.658s

Test accuracy = 96.55% (9655/10000), cost = 0.1146, 0.762s

Total time taken : 1067s (17.78min)
```

## Command line options
Some of the useful command line options are:
* -ec:<> - epochs count;
* -bs:<> - batch size;
* -lr:<> - learning rate.

The full list can be obtained by running the application with '-?' option.

