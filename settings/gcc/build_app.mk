# Executable application build steps

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

# output folder
OUT_FOLDER = "$(SELF_DIR)../../build/$(TARGET)/$(BUILD_TYPE)/bin/"

# additional include folders
INCLUDES += -I$(SELF_DIR)../../build/$(TARGET)/$(BUILD_TYPE)/include/

CFLAGS += $(INCLUDES)

# additional libraries
LIBS += -lannt
LIBDIR += -L$(SELF_DIR)../../build/$(TARGET)/$(BUILD_TYPE)/lib/

LDFLAGS += $(LIBDIR) $(LIBS)

# build targets
all: build post_build

debug: build post_build
 
%.o: %.cpp
	$(COMPILER) $(CFLAGS) -c $^ -o $@

$(OUT): $(OBJ)
	$(COMPILER) -o $@ $(OBJ) $(LDFLAGS)

build: $(OUT)
	mkdir -p $(OUT_FOLDER)
	cp -f $(OUT) $(OUT_FOLDER)

post_build: $(OUT)

clean:
	rm -f $(OBJ) $(OUT)

