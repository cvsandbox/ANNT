# ANNT makefile

include ../../../settings/gcc/compiler_cpp.mk
include ../src.mk

OUT = libannt.a

CFLAGS += -mavx

include ../../../settings/gcc/build_lib.mk

post_build: $(OUT)
	mkdir -p ../../../build/$(TARGET)/$(BUILD_TYPE)/include
	cd ../../lib ; find . -name '*.hpp' -exec cp -r --parents \{\} ../../build/$(TARGET)/$(BUILD_TYPE)/include/ \;

