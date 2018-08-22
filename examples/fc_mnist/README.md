# MNIST classification example with Fully Connected ANN

This example performs classification of MNIST handwritten digits using fully connected 3 layers ANN - 300 neurons in the first hidden layer, 100 neurons in the second and finally 10 neurons in the output layer.

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

Before training: accuracy = 15.18% (7591/50000), cost = 2.3780, 2.495s

Epoch   1 : [==================================================] 44.302s
Training accuracy = 90.94% (45471/50000), cost = 0.3016, 3.210s
Validation accuracy = 91.57% (9157/10000), cost = 0.2851, 0.608s
Epoch   2 : [==================================================] 42.233s
Training accuracy = 92.39% (46196/50000), cost = 0.2413, 2.754s
Validation accuracy = 92.78% (9278/10000), cost = 0.2325, 0.548s
...
Epoch  19 : [==================================================] 69.953s
Training accuracy = 95.69% (47845/50000), cost = 0.1370, 2.909s
Validation accuracy = 95.10% (9510/10000), cost = 0.1689, 0.541s
Epoch  20 : [==================================================] 44.607s
Training accuracy = 96.51% (48254/50000), cost = 0.1162, 2.768s
Validation accuracy = 95.81% (9581/10000), cost = 0.1489, 0.541s

Test accuracy = 95.53% (9553/10000), cost = 0.1501, 0.551s

Total time taken : 1111s (18.52min)
```

## Command line options
Some of the useful command line options are:
* -ec:<> - epochs count;
* -bs:<> - batch size;
* -lr:<> - learning rate.

The full list can be obtained by running the application with '-?' option.

