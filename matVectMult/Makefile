# Compiler and Flags
NVCC = nvcc
CXX = g++

# Architecture
ARCH = sm_80

# Source and output
SRC = matVecMult.cu
OUT = exec

# Targets
all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) -arch=$(ARCH) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)

run: $(OUT)
	./$(OUT)
