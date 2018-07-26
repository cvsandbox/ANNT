# ANNT source files

# search path for source files
VPATH = ../../lib \
        ../../lib/Types \
        ../../lib/Tools \
        ../../lib/Neuro/Layers \
        ../../lib/Neuro/Network

# source files
SRC = ANNT.cpp \
      XAlignedAllocator.cpp \
      XCpu.cpp \
      XAvxVectorTools.cpp \
      XSseVectorTools.cpp \
      XVectorTools.cpp \
      XVectorize.cpp \
      XDataEncodingTools.cpp \
      XFullyConnectedLayer.cpp \
      XConvolutionLayer.cpp \
      XRecurrentLayer.cpp \
      XLSTMLayer.cpp \
	  XGRULayer.cpp \
      XNeuralNetwork.cpp \
      XNetworkContext.cpp \
      XNetworkInference.cpp \
      XNetworkTraining.cpp \
      XClassificationTrainingHelper.cpp
