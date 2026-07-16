# LS-MPS GPU Minimal Implementation Makefile
# Target: NVIDIA A100 (sm_80), CUDA 12.6.0

NVCC := nvcc
ARCH := -arch=sm_80
CUDA_FLAGS := -O3 -use_fast_math --expt-relaxed-constexpr -lineinfo
LDFLAGS := -lcusparse -lcublas

TARGET := lsmps
SRCDIR := src
OBJDIR := obj

SRCS := $(SRCDIR)/lsmps.cu
OBJS := $(OBJDIR)/lsmps.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(ARCH) $(LDFLAGS) -o $@ $^

$(OBJDIR)/lsmps.o: $(SRCDIR)/lsmps.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(ARCH) $(CUDA_FLAGS) -c -o $@ $<

run-small: $(TARGET)
	./$(TARGET) config/small.yaml

run-benchmark: $(TARGET)
	./$(TARGET) config/benchmark.yaml

profile: $(TARGET)
	nsys profile --trace=cuda,nvtx ./$(TARGET) config/benchmark.yaml

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean run-small run-benchmark profile
