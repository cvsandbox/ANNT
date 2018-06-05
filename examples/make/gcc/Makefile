# Makefile to build all examples

SELF_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

EXAMPLES = fc_xor \
           fc_iris \
           fc_mnist

all:
	$(foreach ex,$(EXAMPLES), cd "$(SELF_DIR)../../$(ex)/make/gcc"; make ; cd "$(SELF_DIR)" ;)

debug:
	$(foreach ex,$(EXAMPLES), cd "$(SELF_DIR)../../$(ex)/make/gcc"; make debug ; cd "$(SELF_DIR)" ;)

clean:
	$(foreach ex,$(EXAMPLES), cd "$(SELF_DIR)../../$(ex)/make/gcc"; make clean ; cd "$(SELF_DIR)" ;)

