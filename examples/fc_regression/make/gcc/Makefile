# fc_regression example makefile

include ../src.mk
include ../../../../settings/gcc/compiler_cpp.mk

OUT = fc_regression

include ../../../../settings/gcc/build_app.mk

post_build: $(OUT)
	mkdir -p ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data
	cp ../../*.csv ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data

