NVCC = nvcc
NVCCFLAGS = -pg -m64
LINKFLAGS = -pg -m64

GCC = g++
GCCFLAGS = -Wall -pedantic -pg -m64

IN_DIR = src
OUT_DIR = bin
CSOURCES = tester.cpp
CUSOURCES = spline.cu solve_system.cu
COBJ = tester.o
CUOBJ = spline.obj solve_system.obj
OUT_CUOBJ = $(addprefix $(OUT_DIR)/,$(CUOBJ))
OUT_COBJ = $(addprefix $(OUT_DIR)/,$(COBJ))

all: splinetest
	
splinetest: bin $(OUT_CUOBJ) $(OUT_COBJ)
	$(NVCC) $(LINKFLAGS) $(OUT_CUOBJ) $(OUT_COBJ) -o splinetest

objects: bin $(CUOBJ)
	echo
	
$(OUT_DIR)/%.obj: $(IN_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT_DIR)/%.o: $(IN_DIR)/%.cpp
	$(GCC) $(GCCFLAGS) -c $< -o $@

bin:
	mkdir bin
	
clean:
	rm bin/*.o bin/*.obj splinetest
