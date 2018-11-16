# MNIST classification example with Recurrent ANN

This example performs classification of [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/) using recurrent ANN – GRU layers with 56 neurons followed by fully connected layer with 10 neurons.
```C++
// prepare a recurrent ANN
shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

net->AddLayer( make_shared<XGRULayer>( MNIST_IMAGE_WIDTH, 56 ) );
net->AddLayer( make_shared<XFullyConnectedLayer>( 56, 10 ) );
net->AddLayer( make_shared<XSoftMaxActivation>( ) );
```

Unlike with fully connected or convolutional neural networks, the 28x28 MNST images are not presented to recurrent network at once as a single input sample. Instead, each image is fed to network row by row, i.e. 28 vectors of 28 values are given one after another. The final output of the network is then used to obtain classification result. The code for this may look something like this:
```C++
XNetworkInference netInference( net );
vector<fvector_t> sequenceInputs;
fvector_t         output( 10 );

// prepare images rows as vector of vectors - sequenceInputs
// ...

// feed MNIST image to network row by row
for ( size_t j = 0; j < MNIST_IMAGE_HEIGHT; j++ )
{
    netInference.Compute( sequenceInputs[j], output );
}

// get the classification label of the image (0-9)
size_t label = XDataEncodingTools::MaxIndex( output );

// reset network inference so it is ready to classify another image
netInference.ResetState( );
```

Unlike other classification examples, this sample application does not use helper class to run neural network’s training loop. This is due to the fact that each MNIST image needs to be presented as a sequence of rows and so extra care must be done in handling training samples. As the result, the example code is a bit lengthy. But it follows the standard pattern of other sample applications, so should be readable enough.

# Sample output

```
MNIST handwritten digits classification example with Recurrent ANN

Loaded 60000 training data samples
Loaded 10000 test data samples

Samples usage: training = 50000, validation = 10000, test = 10000

Learning rate: 0.0010, Epochs: 20, Batch Size: 48

Before training: accuracy = 9.70% (4848/50000), cost = 2.3851, 18.668s

Epoch   1 : [==================================================] 77.454s
Training accuracy = 90.81% (45407/50000), cost = 0.3224, 24.999s
Validation accuracy = 91.75% (9175/10000), cost = 0.2984, 3.929s
Epoch   2 : [==================================================] 90.788s
Training accuracy = 94.05% (47027/50000), cost = 0.2059, 20.189s
Validation accuracy = 94.30% (9430/10000), cost = 0.2017, 4.406s
...
Epoch  19 : [==================================================] 52.225s
Training accuracy = 98.87% (49433/50000), cost = 0.0369, 23.995s
Validation accuracy = 98.03% (9803/10000), cost = 0.0761, 4.030s
Epoch  20 : [==================================================] 84.035s
Training accuracy = 98.95% (49475/50000), cost = 0.0332, 39.265s
Validation accuracy = 98.04% (9804/10000), cost = 0.0745, 7.464s

Test accuracy = 97.79% (9779/10000), cost = 0.0824, 7.747s

Total time taken : 1864s (31.07min)
```

## Command line options
Some of the useful command line options are:
* -ec:<> - epochs count;
* -bs:<> - batch size;
* -lr:<> - learning rate;
* -fout:<> - file name to save network's weights;
* -fin:<> - file name to load network's weights from (further training with different hyperparameters, for example).

The full list can be obtained by running the application with '-?' option.
