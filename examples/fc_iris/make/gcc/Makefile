# fc_iris example makefile

include ../src.mk
include ../../../../settings/gcc/compiler_cpp.mk

OUT = fc_iris

include ../../../../settings/gcc/build_app.mk

post_build: $(OUT)
	mkdir -p ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data
	cp ../../../data/iris/iris.data ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data

