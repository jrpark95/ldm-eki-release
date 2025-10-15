NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -w -O3 -arch=sm_61 -DCRAM_DEBUG -I./src/include -I./src/kernels

MPI_INC = /usr/local/openmpi/include
MPI_LIB = /usr/local/openmpi/lib

ifeq ($(OS),Windows_NT)
    OS_DETECTED := Windows
    PATH_SEP := /
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS_DETECTED := Linux
        NVCCFLAGS += -Xcompiler -fPIC -Xcompiler -fopenmp -I$(MPI_INC) -L$(MPI_LIB) -lmpi -lcublas -lgomp
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
