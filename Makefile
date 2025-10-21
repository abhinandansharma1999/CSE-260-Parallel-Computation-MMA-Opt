# Compiler and flags
MY_OPT := 
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17  -ggdb -O3 $(MY_OPT)
CXXFLAGS += -march=armv8.2-a+sve
# Target executable name
TARGET := mm

# Source files (all .cpp in current directory and matrix subdirectory)
SRCS := $(wildcard *.cpp) $(wildcard dgemm/*.cpp) $(wildcard matrix/*.cpp) $(wildcard utils/*.cpp) $(wildcard cse260_hw1/*.cpp)

# Object files
OBJS := $(SRCS:.cpp=.o)

LD := g++

LDFLAGS := -lopenblas

# Default rule
all: $(TARGET)


# Explicit link step
$(TARGET): $(OBJS)
	$(LD) -o $@ $^ $(LDFLAGS) -lpthread -lm


# Compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET) matrix/*.o utils/*.o dgemm/*.o

# Phony targets
.PHONY: all clean
