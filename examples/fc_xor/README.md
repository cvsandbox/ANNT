# XOR example with Fully Connected ANN

This example provides the simplest problem to solve with an Artificial Neural Network – XOR function. Having the simplest dataset provided by the XOR function, the example shows the smallest possible network to correctly classify inputs.

## XOR function
| x1 | x2 |  y |
| -- | -- | -- |
|  0 |  0 |  0 |
|  0 |  1 |  1 |
|  1 |  0 |  1 |
|  1 |  1 |  0 |

Since the above data set is not linearly separable, one single neuron is not enough to correctly classify it. And so, a two-layer network is used – 2 neurons in the first layer and 1 neuron in the output layer.

```C++
shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

net->AddLayer( make_shared<XFullyConnectedLayer>( 2, 2 ) );
net->AddLayer( make_shared<XTanhActivation>( ) );
net->AddLayer( make_shared<XFullyConnectedLayer>( 2, 1 ) );
net->AddLayer( make_shared<XSigmoidActivation>( ) );
```

The network is then trained either iteratively providing single random example for each weights update or in batch mode, when entire data set is provided on each iteration.

## Sample output

```
XOR example with Fully Connected ANN

Network output before training:
{ -1.00 -1.00 } -> {  0.54 }
{  1.00 -1.00 } -> {  0.47 }
{ -1.00  1.00 } -> {  0.53 }
{  1.00  1.00 } -> {  0.46 }

Cost of each sample:
0.6262 0.5256 0.4130 1.1266 0.9358 0.7967 0.6214 1.1063
...
0.0254 0.0188 0.0246 0.0184 0.0184 0.0209 0.0179 0.0172

Network output after training:
{ -1.00 -1.00 } -> {  0.02 }
{  1.00 -1.00 } -> {  0.98 }
{ -1.00  1.00 } -> {  0.98 }
{  1.00  1.00 } -> {  0.02 }
```

## Playing with the example

Few things can be done to experiment with the example:
* Comment the first layer (and its activation function) to get just single neuron – the network will fail classifying **XOR** function;
* With the single neuron network, change data set so it looks like **AND** or **OR** function – the single neuron will be able learn classifying it;
* Switch between on-line stochastic learning and batch learning and see if makes any difference.


