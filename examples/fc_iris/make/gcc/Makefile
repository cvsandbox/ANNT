# vectorization test makefile

include ../../../../settings/gcc/compiler_cpp.mk
include ../src.mk

OUT = fc_iris

include ../../../../settings/gcc/build_app.mk

post_build: $(OUT)
	mkdir -p ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data
	cp ../../iris.data ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data

