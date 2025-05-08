NVCC := nvcc
STD  := -std=c++11
OPT  := -O3
OMP  := -Xcompiler -fopenmp
ARCH := -arch=sm_60

SRC := jac3d.cu
BIN := jac3d

.PHONY: all float double clean

all: double

double: CFLAGS := $(STD) $(OPT) $(ARCH) $(OMP) -DUSE_DOUBLE
double: $(BIN)

float: CFLAGS := $(STD) $(OPT) $(ARCH) $(OMP)
float: clean $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f $(BIN)
