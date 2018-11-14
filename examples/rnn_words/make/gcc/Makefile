# rnn_words example makefile

include ../src.mk
include ../../../../settings/gcc/compiler_cpp.mk

OUT = rnn_words

include ../../../../settings/gcc/build_app.mk

post_build: $(OUT)
	mkdir -p ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data
	cp ../../../data/words/*.txt ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data

