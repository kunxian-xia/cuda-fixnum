CXX ?= g++
GENCODES ?= 60 

INCLUDE_DIRS = -I./src
NVCC_FLAGS = -ccbin $(CXX) -std=c++11 -Xcompiler -Wall,-Wextra
NVCC_OPT_FLAGS = -DNDEBUG
NVCC_TEST_FLAGS = -lineinfo
NVCC_DBG_FLAGS = -g -G
NVCC_LIBS = -lstdc++
NVCC_TEST_LIBS = -lgtest

all:
	@echo "Please run 'make check' or 'make bench'."

tests/test-suite: tests/test-suite.cu
	nvcc $(NVCC_TEST_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) $(NVCC_TEST_LIBS) -o $@ $<

check: tests/test-suite
	@./tests/test-suite

bench/bench: bench/bench.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

bench: bench/bench

main: main.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

word: word.cu quadratic_ext.cu
	nvcc $(NVCC_DBG_FLAGS) $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

main2: main2.cu
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

.PHONY: clean
clean:
	$(RM) tests/test-suite bench/bench word main
