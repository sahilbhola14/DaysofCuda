# Notes:
# -g: Host code debug symbols (No performance degradation)
# -G: Device code debug symbols (Huge slowdown)
# -lineinfo: Device embed source line info (Minimal slowdown)

# compiler
NVCC 	:= nvcc
# architecture
ARCH 	:= sm_80
# files
# SRC 	:= vecAdd.cu
SRC 	:= vecAdd_pinned.cu
OUT 	:= run
# build
DEBUG_FLAGS    := -O0 -g -G -lineinfo
RELEASE_FLAGS  := -O3
PROFILE_FLAGS  := -O2 -g -lineinfo
# default  target
all: release
# release build
release:
	$(NVCC) -arch=$(ARCH) $(RELEASE_FLAGS) $(SRC) -o $(OUT)
# debug build
debug:
	$(NVCC) -arch=$(ARCH) $(DEBUG_FLAGS) $(SRC) -o $(OUT)
# profile
profile:
	$(NVCC) -arch=$(ARCH) $(PROFILE_FLAGS) $(SRC) -o $(OUT)
# run
run: release
	./$(OUT)
# nsight system (system profiling)
nsys: profile
	nsys profile --trace=cuda --stats=true --force-overwrite=true -o nsys_report ./$(OUT)
	nsys stats --force-export=true nsys_report.nsys-rep > nsys_report.txt
	@echo "Readable sytem summary written to nsys_report.txt"
# nsight compute
ncu: profile
	ncu --set full --force-overwrite -o ncu_report ./$(OUT)
	ncu --import ncu_report.ncu-rep > ncu_report.txt
	@echo "Readable compute summary written to ncu_report.txt"

# both reports
report: nsys ncu
# clean
clean:
	rm -f $(OUT) *.qdrep *.nsys-rep *.sqlite nsys_report.txt *.ncu-rep ncu_report.txt
