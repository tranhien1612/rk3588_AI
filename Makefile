CXX      ?= g++
BUILD    := build
TARGET   := $(BUILD)/gimbal_vision
SRCS     := $(wildcard src/*.cpp) $(wildcard src/*.cc)
OBJS     := $(patsubst src/%.cpp,$(BUILD)/%.o,$(filter %.cpp,$(SRCS))) $(patsubst src/%.cc,$(BUILD)/%.o,$(filter %.cc,$(SRCS)))
PKGS     := gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0 librga
CXXFLAGS := -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod -mtune=cortex-a76 -Wall -Wno-unused-result \
            -Iinclude -Ithird_party -Ithird_party/rknn -Ithird_party/onnxruntime/include -I/usr/include/rga $(shell pkg-config --cflags $(PKGS))
LDFLAGS  := -Lthird_party/rknn -lrknnrt -Lthird_party/onnxruntime/lib -lonnxruntime \
            -Wl,-rpath,'$$ORIGIN/../third_party/rknn' -Wl,-rpath,'$$ORIGIN/../third_party/onnxruntime/lib' \
            $(shell pkg-config --libs $(PKGS)) -lpthread

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(BUILD)/%.o: src/%.cc
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
.PHONY: all clean
