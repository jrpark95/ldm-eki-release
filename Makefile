NVCC = /usr/local/cuda/bin/nvcc
# Optimized flags for faster compilation:
# - Removed -DCRAM_DEBUG (CRAM not used)
# - Using -O2 instead of -O3 (faster compilation, minimal runtime difference)
# - Parallel build enabled (use: make -j4)
NVCCFLAGS = -w -O2 -arch=sm_61 -I./src/include -I./src/kernels

# Enable parallel builds
MAKEFLAGS += -j$(shell nproc)

ifeq ($(OS),Windows_NT)
    OS_DETECTED := Windows
    PATH_SEP := /
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS_DETECTED := Linux
        # Removed -fPIC, -lcublas, -lmpi (not actively used)
        # Note: mpiRank/mpiSize variables exist in code but MPI functions are not called
        NVCCFLAGS += -Xcompiler -fopenmp -lgomp
        PATH_SEP := /
    endif
endif

OBJ_PATH = .$(PATH_SEP)objectfiles

TARGET = ldm
TARGET_EKI = ldm-eki
TARGET_RECEPTOR_DEBUG = ldm-receptor-debug

SRCS = src/main.cu
SRCS_EKI = src/main_eki.cu
SRCS_RECEPTOR_DEBUG = src/main_receptor_debug.cu
SRCS_CPP = src/memory_doctor.cpp
OBJS = $(addprefix $(OBJ_PATH)$(PATH_SEP), $(SRCS:.cu=.o))
OBJS_EKI = $(addprefix $(OBJ_PATH)$(PATH_SEP), $(SRCS_EKI:.cu=.o)) $(addprefix $(OBJ_PATH)$(PATH_SEP), $(SRCS_CPP:.cpp=.o))
OBJS_RECEPTOR_DEBUG = $(addprefix $(OBJ_PATH)$(PATH_SEP), $(SRCS_RECEPTOR_DEBUG:.cu=.o))

all: $(TARGET_EKI)

all-targets: $(TARGET) $(TARGET_EKI) $(TARGET_RECEPTOR_DEBUG)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

$(TARGET_EKI): $(OBJS_EKI)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS_EKI)

$(TARGET_RECEPTOR_DEBUG): $(OBJS_RECEPTOR_DEBUG)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS_RECEPTOR_DEBUG)

$(OBJ_PATH)$(PATH_SEP)%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_PATH)$(PATH_SEP)%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

clean:
ifeq ($(OS_DETECTED),Windows)
	del /Q /F $(subst /,\,$(OBJS)) $(subst /,\,$(OBJS_EKI)) $(subst /,\,$(OBJS_RECEPTOR_DEBUG)) $(TARGET).exe $(TARGET_EKI).exe $(TARGET_RECEPTOR_DEBUG).exe $(TARGET).exp $(TARGET).lib $(TARGET_EKI).exp $(TARGET_EKI).lib $(TARGET_RECEPTOR_DEBUG).exp $(TARGET_RECEPTOR_DEBUG).lib
else
	rm -f $(OBJS) $(OBJS_EKI) $(OBJS_RECEPTOR_DEBUG) $(TARGET) $(TARGET_EKI) $(TARGET_RECEPTOR_DEBUG)
endif

$(OBJS) $(OBJS_EKI) $(OBJS_RECEPTOR_DEBUG): | $(OBJ_PATH)

$(OBJ_PATH):
	mkdir -p $(OBJ_PATH)
