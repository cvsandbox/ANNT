# Static library build steps

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

# output folder
OUT_FOLDER = "$(SELF_DIR)../../build/$(TARGET)/$(BUILD_TYPE)/lib/"

# build targets
all: build post_build

debug: build post_build

%.o: %.cpp
	$(COMPILER) $(CFLAGS) -c $^ -o $@

$(OUT): $(OBJ)
	$(ARCHIVER) rcs $(OUT) $(OBJ)

build: $(OUT)
	mkdir -p $(OUT_FOLDER)
	cp -f $(OUT) $(OUT_FOLDER)

post_build: $(OUT)

clean:
	rm -f $(OBJ) $(OUT)

