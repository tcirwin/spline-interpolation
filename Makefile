NVCC = nvcc
NVCCFLAGS = -m64 -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
LINKFLAGS = -m64 -O3

GCC = g++
GCCFLAGS = -g -O3 -Wall -pedantic -m64

IN_DIR = src
OUT_DIR = bin
OUT_OPENGL_DIR = bin/opengl
COBJ = tester.o print_curve.o init_curves.o
COBJ_W = opengl/draw_window.o opengl/draw_spline.o init_curves.o
CUOBJ = spline.obj solve_system.obj
CPUOBJ = JacobiMethod.o spline_cpu.o
OUT_CUOBJ = $(addprefix $(OUT_DIR)/,$(CUOBJ))
OUT_COBJ = $(addprefix $(OUT_DIR)/,$(COBJ))
OUT_COBJ_W = $(addprefix $(OUT_DIR)/,$(COBJ_W))
OUT_CPUOBJ = $(addprefix $(OUT_DIR)/,$(CPUOBJ))

all: splinetest splinetest_cpu windowtest windowtest_cpu
osx: splinetest windowtest_osx splinetest_cpu windowtest_osx_cpu

splinetest: $(OUT_DIR) $(OUT_CUOBJ) $(OUT_COBJ)
	$(NVCC) $(LINKFLAGS) $(OUT_CUOBJ) $(OUT_COBJ) -o splinetest

splinetest_cpu: $(OUT_DIR) $(OUT_CPUOBJ) $(OUT_COBJ)
	$(GCC) $(LINKFLAGS) $(OUT_CPUOBJ) $(OUT_COBJ) -fopenmp -o splinetest_cpu

windowtest: $(OUT_DIR) $(OUT_OPENGL_DIR) $(OUT_CUOBJ) $(OUT_COBJ_W)
	$(NVCC) $(LINKFLAGS) $(OUT_CUOBJ) $(OUT_COBJ_W) -lGL -lGLU -lglut -o windowtest

windowtest_osx: $(OUT_DIR) $(OUT_OPENGL_DIR) $(OUT_CUOBJ) $(OUT_COBJ_W)
	$(NVCC) $(LINKFLAGS) $(OUT_CUOBJ) $(OUT_COBJ_W) -Xlinker -framework,OpenGL,-framework,GLUT -o windowtest_osx

windowtest_osx_cpu: $(OUT_DIR) $(OUT_OPENGL_DIR) $(OUT_CUOBJ) $(OUT_COBJ_W)
	$(GCC) $(LINKFLAGS) $(OUT_CPUOBJ) $(OUT_COBJ_W) -fopenmp -framework OpenGL -framework GLUT -o windowtest_osx_cpu

windowtest_cpu: $(OUT_DIR) $(OUT_OPENGL_DIR) $(OUT_CPUOBJ) $(OUT_COBJ_W)
	$(GCC) $(LINKFLAGS) $(OUT_CPUOBJ) $(OUT_COBJ_W) -lGL -lGLU -lglut -fopenmp -o windowtest_cpu

objects: $(OUT_DIR) $(CUOBJ)
	echo

$(OUT_DIR)/%.obj: $(IN_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OUT_DIR)/%.o: $(IN_DIR)/%.cpp
	$(GCC) $(GCCFLAGS) -fopenmp -c $< -o $@

$(OUT_DIR):
	mkdir $(OUT_DIR)

$(OUT_OPENGL_DIR):
	mkdir $(OUT_OPENGL_DIR)

clean:
	rm bin/*.o bin/*.obj bin/opengl/*.o splinetest windowtest
