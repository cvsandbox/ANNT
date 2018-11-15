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
