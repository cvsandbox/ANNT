# fc_time_series example makefile

include ../src.mk
include ../../../../settings/gcc/compiler_cpp.mk

OUT = fc_time_series

include ../../../../settings/gcc/build_app.mk

post_build: $(OUT)
	mkdir -p ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data
	cp ../../../data/time-series/*.csv ../../../../build/$(TARGET)/$(BUILD_TYPE)/bin/data

