![ANNT](images/annt.png)
# ANNT

ANNT is a lightweight C++ library implementing some of the common architectures of artificial neural networks:
* Fully connected;
* Convolutional;
* Simple recurrent networks (RNN), Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU).

The library comes with a number of sample applications, which demonstrate its usage in solving different tasks. The provided samples cover all of the implemented neural network architectures.

## Building the code

The code comes with MSVC solution files ([src/make/msvc](src/make/msvc)) and GCC make file ([src/make/gcc](src/make/gcc)). On Windows, simply open solution file in MS Visual Studio and hit build. On Linux just use "make" command to build the library. In both cases the built library ends up in the "/build/compiler/mode" folder, which also contains the necessary header files.

## Additional reading

The ANNT library and some of the theory about artificial neural networks is described in a set of articles published on Code Project:
* [Feed forward fully connected neural networks](https://www.codeproject.com/Articles/1261763/ANNT-Feed-forward-fully-connected-neural-networks);
* [Convolutional neural networks](https://www.codeproject.com/Articles/1264962/ANNT-Convolutional-neural-networks);
* [Recurrent neural networks](https://www.codeproject.com/Articles/1272354/ANNT-Recurrent-neural-networks).


