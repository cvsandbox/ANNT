# ANNT source files

# search path for source files
VPATH = ../../lib \
        ../../lib/Tools \
        ../../lib/Neuro/Layers \
        ../../lib/Neuro/Network

# source files
SRC = ANNT.cpp \
      XCpu.cpp \
      XAvxVectorTools.cpp \
      XSseVectorTools.cpp \
      XVectorTools.cpp \
      XDataEncodingTools.cpp \
      XFullyConnectedLayer.cpp \
      XConvolutionLayer.cpp \
      XNetworkInference.cpp \
      XNetworkTraining.cpp

OBJ = $(SRC:.cpp=.o)
