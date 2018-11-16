# Sequence prediction with Recurrent ANN

This example application is a very simple introduction to recurrent neural networks. It attempts to predict next digit of a sequence. The training set contains 10 sequences shown below:

```
1 0 1 2 3 4 5 6 7 8 1
2 0 1 2 3 4 5 6 7 8 2
3 0 1 2 3 4 5 6 7 8 3
4 0 1 2 3 4 5 6 7 8 4
5 0 1 2 3 4 5 6 7 8 5

6 0 1 2 4 4 4 6 7 8 6
7 0 1 2 4 4 4 6 7 8 7
8 0 1 2 4 4 4 6 7 8 8
9 0 1 2 4 4 4 6 7 8 9
0 0 1 2 4 4 4 6 7 8 0
```

The first 5 sequences are almost identical, only the first and the last digits are different. Same about the other 5 sequences, only the pattern in the middle has changed, but still the same in all those sequences. The task for a trained network is to first output '0' for any of the provided inputs, then it needs to output '1' when presented with '0', then '2' when presented with '1', and then it needs to output '3' or '4' when presented with '2'. Wait a second. How should it know what to choose, '3' or '4' if the input is the same? Yes, something fully connected network would fail to digest. However recurrent networks have internal state (memory, so to speak). This state should tell the network to look not only at the currently provided input, but also at what was there before. And since the first digit of every sequence is different, it makes them unique. So, all the network needs to do is to look few steps behind (sometimes more than few).

To approach this prediction task, a simple 2-layer network is used - first layer is recurrent, while the second is fully connected. As we have 10 possible digits in our sequences and those are one-hot encoded, the network has 10 inputs and 10 outputs.

```C++
// prepare a recurrent ANN
shared_ptr<XNeuralNetwork> net = make_shared<XNeuralNetwork>( );

// basic recurrent network
switch ( trainingParams.RecurrrentType )
{
case RecurrentLayerType::Basic:
default:
    net->AddLayer( make_shared<XRecurrentLayer>( 10, 20 ) );
    break;

case RecurrentLayerType::LSTM:
    net->AddLayer( make_shared<XLSTMLayer>( 10, 20 ) );
    break;

case RecurrentLayerType::GRU:
    net->AddLayer( make_shared<XGRULayer>( 10, 20 ) );
    break;
}

// complete the network with fully connecte layer and soft max activation
net->AddLayer( make_shared<XFullyConnectedLayer>( 20, 10 ) );
net->AddLayer( make_shared<XSoftMaxActivation>( ) );
```

Since each sequence has 10 transitions between digits, it will make 10 input/output one-hot encoded training samples for each. In total – 100 training samples. Those can all be fed to network in a single batch. However, to make it all work correctly the network must be told about the length of the sequence, so that back propagation through time could do it all right. 

```C++
// create training context with Adam optimizer and Cross Entropy cost function
XNetworkTraining netTraining( net,
                              make_shared<XAdamOptimizer>( trainingParams.LearningRate ),
                              make_shared<XCrossEntropyCost>( ) );

netTraining.SetAverageWeightGradients( false );
// since we are dealing with recurrent network, we need to tell trainer the length of time series
netTraining.SetTrainingSequenceLength( STEPS_PER_SEQUENCE );

// run training epochs providing all data as single batch
for ( size_t i = 1; i <= trainingParams.EpochsCount; i++ )
{
    auto cost = netTraining.TrainBatch( inputs, outputs );
    printf( "%0.4f ", static_cast<float>( cost ) );

    // reset state before the next batch/epoch
    netTraining.ResetState( );
}
```

# Sample output

The sample output below shows that untrained network generates something random, but not the next digit of the sequence. While the trained network is able to reconstruct all of the sequences. Replace the recurrent layer with fully connected one and it will ruin everything - trained or not the network will fail.

```
Sequence prediction with Recurrent ANN

Learning rate  : 0.0100
Epochs count   : 150
Recurrent type : basic

Before training:
Target sequence:   10123456781
Produced sequence: 13032522355 Bad

Target sequence:   20123456782
Produced sequence: 20580425851 Bad

Target sequence:   30123456783
Produced sequence: 33036525351 Bad

Target sequence:   40123456784
Produced sequence: 49030522355 Bad

Target sequence:   50123456785
Produced sequence: 52030522855 Bad

Target sequence:   60124446786
Produced sequence: 69036525251 Bad

Target sequence:   70124446787
Produced sequence: 71436521251 Bad

Target sequence:   80124446788
Produced sequence: 85036525251 Bad

Target sequence:   90124446789
Produced sequence: 97036525251 Bad

Target sequence:   00124446780
Produced sequence: 00036525251 Bad

2.3539 2.1571 1.9923 1.8467 1.7097 1.5770 1.4487 1.3262 1.2111 1.1050
...
0.0014 0.0014 0.0014 0.0014 0.0014 0.0014 0.0013 0.0013 0.0013 0.0013


After training:
Target sequence:   10123456781
Produced sequence: 10123456781 Good

Target sequence:   20123456782
Produced sequence: 20123456782 Good

Target sequence:   30123456783
Produced sequence: 30123456783 Good

Target sequence:   40123456784
Produced sequence: 40123456784 Good

Target sequence:   50123456785
Produced sequence: 50123456785 Good

Target sequence:   60124446786
Produced sequence: 60124446786 Good

Target sequence:   70124446787
Produced sequence: 70124446787 Good

Target sequence:   80124446788
Produced sequence: 80124446788 Good

Target sequence:   90124446789
Produced sequence: 90124446789 Good

Target sequence:   00124446780
Produced sequence: 00124446780 Good
```
