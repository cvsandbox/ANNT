# MNIST classification example with Convolutional ANN

This example performs classification of [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) using convolutional ANN - 3 convolutional layers followed by a single fully connected layer. The first two convolutional layers are followed ReLU activation function and average pooling layer, while the third convolutional layer is followed by ReLU only. The structure of network can be described as:

```
Conv(32x32x1, 5x5x6  ) -> ReLU -> AvgPool
Conv(14x14x6, 5x5x16 ) -> ReLU -> AvgPool
Conv(5x5x16,  5x5x120) -> ReLU
FC(120, 10) -> SoftMax
```

Although the first convolutional layer produces 6 feature maps the second convolutional layer does not use them all in its 16 convolutions. The first 6 convolutions of the second layer use different patterns of 3 feature maps produced by the first layer. The next 9 convolutions use different patterns of 4 feature maps. Finally, the last convolution uses all 6 feature maps of the first layer. This is done to make sure that different feature maps of the second layer are not all based on the same input feature maps.

```C++
// connection table to specify wich feature maps of the first convolution layer
// to use for feature maps produced by the second layer
vector<bool> connectionTable( {
    true,  true,  true,  false, false, false,
    false, true,  true,  true,  false, false,
    false, false, true,  true,  true,  false,
    false, false, false, true,  true,  true,
    true,  false, false, false, true,  true,
    true,  true,  false, false, false, true,
    true,  true,  true,  true,  false, false,
    false, true,  true,  true,  true,  false,
    false, false, true,  true,  true,  true,
    true,  false, false, true,  true,  true,
    true,  true,  false, false, true,  true,
    true,  true,  true,  false, false, true,
    true,  true,  false, true,  true,  false,
    false, true,  true,  false, true,  true,
    true,  false, true,  true,  false, true,
    true,  true,  true,  true,  true,  true
} );

// prepare a convolutional ANN
shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

net->AddLayer( make_shared<XConvolutionLayer>( 32, 32, 1, 5, 5, 6 ) );
net->AddLayer( make_shared<XReLuActivation>( ) );
net->AddLayer( make_shared<XAveragePooling>( 28, 28, 6, 2 ) );

net->AddLayer( make_shared<XConvolutionLayer>( 14, 14, 6, 5, 5, 16, connectionTable ) );
net->AddLayer( make_shared<XReLuActivation>( ) );
net->AddLayer( make_shared<XAveragePooling>( 10, 10, 16, 2 ) );

net->AddLayer( make_shared<XConvolutionLayer>( 5, 5, 16, 5, 5, 120 ) );
net->AddLayer( make_shared<XReLuActivation>( ) );

net->AddLayer( make_shared<XFullyConnectedLayer>( 120, 10 ) );
net->AddLayer( make_shared<XLogSoftMaxActivation>( ) );
```

Similar to the [FC version of MNIST classification](../fc_mnist), this example delegates the training routine to a helper class, which gets training/validation/test data sets and runs the training loop for the specified number of epochs.

## Sample output

```
MNIST handwritten digits classification example with Convolution ANN

Loaded 60000 training data samples
Loaded 10000 test data samples

Samples usage: training = 50000, validation = 10000, test = 10000

Learning rate: 0.0020, Epochs: 20, Batch Size: 50

Before training: accuracy = 5.00% (2500/50000), cost = 2.3175, 34.324s

Epoch   1 : [==================================================] 123.060s
Training accuracy = 97.07% (48536/50000), cost = 0.0878, 32.930s
Validation accuracy = 97.49% (9749/10000), cost = 0.0799, 6.825s
Epoch   2 : [==================================================] 145.140s
Training accuracy = 97.87% (48935/50000), cost = 0.0657, 36.821s
Validation accuracy = 97.94% (9794/10000), cost = 0.0669, 5.939s
...
Epoch  19 : [==================================================] 101.305s
Training accuracy = 99.75% (49877/50000), cost = 0.0077, 26.094s
Validation accuracy = 98.96% (9896/10000), cost = 0.0684, 6.345s
Epoch  20 : [==================================================] 104.519s
Training accuracy = 99.73% (49865/50000), cost = 0.0107, 28.545s
Validation accuracy = 99.02% (9902/10000), cost = 0.0718, 7.885s

Test accuracy = 99.01% (9901/10000), cost = 0.0542, 5.910s

Total time taken : 3187s (53.12min)
```

## Command line options
Some of the useful command line options are:
* -ec:<> - epochs count;
* -bs:<> - batch size;
* -lr:<> - learning rate;
* -fout:<> - file name to save network's weights;
* -fin:<> - file name to load network's weights from (further training with different hyperparameters, for example).

The full list can be obtained by running the application with '-?' option.
