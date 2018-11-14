# ANNT examples

The folder provides number of example applications, which demonstrate usage of the ANNT library for creating artificial neural networks of different architectures and applying them to different tasks.

**Note**: none of the examples pretend they provide the best architecture for a chosen problem, or the best learning approach/parameters, data pre-processing, etc. In fact, none of the examples say that using artificial neural networks for a given problem is the best/right way of solving it. All examples are purely to demonstrate usage of the ANNT library, not for guiding on how to solve certain problem.

## Examples list

* Fully Connected networks
  - [fc_xor](fc_xor/) - binary classification of XOR function;
  - [fc_iris](fc_iris/) - Iris flower multiclass classification;
  - [fc_regression](fc_regression/) - approximation of single argument function (regression);
  - [fc_time_series](fc_time_series/) - times series prediction using past values;
  - [fc_mnist](fc_mnist/) - MNIST handwritten digits classification;
* Convolutional networks
  - [cnn_mnist](cnn_mnist/) - MNIST handwritten digits classification;
  - [cnn_cifar10](cnn_cifar10/) - CIFAR-10 color images classification;
* Recurrent networks
  - [rnn_time_series](rnn_time_series/) - times series prediction;
  - [rnn_sequence](rnn_sequence/) - sequence prediction (one-hot encoded labels);
  - [rnn_mnist](rnn_mnist/) - MNIST handwritten digits classification;
  - [rnn_words](rnn_words/) - generating names of cities.

## Building and running examples

Each example application is provided with MSVC solution file and GCC Makefile. Building with MSVC will make sure the ANNT library is built as well. Building with GCC requires the library to be built first. The *make* subfolder provides solution and makefile to build entire examples collection.

Once example application is built, its binaries are put into */build/`compiler`/`mode`/bin* folder. If it uses any of the provided [data](data/) sets, required files will by copied automatically into *data* subfolder. Alternatively, external data sets must be downloaded manually and put into that folder.

Run all applications from the *bin* subfolder of the build output.
