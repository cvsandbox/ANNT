# ANNT makefile

include ../src.mk
include ../../../settings/gcc/compiler_cpp.mk

OUT = libannt.a

XAvxVectorTools.o: CFLAGS += -mavx
XSseVectorTools.o: CFLAGS += -msse2

include ../../../settings/gcc/build_lib.mk

post_build: $(OUT)
	mkdir -p ../../../build/$(TARGET)/$(BUILD_TYPE)/include
	cd ../../lib ; find . -name '*.hpp' -exec cp -r --parents \{\} ../../build/$(TARGET)/$(BUILD_TYPE)/include/ \;

