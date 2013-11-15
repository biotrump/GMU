##################################################
# nastaveni
CFLAGS_COMMON=
CC=gcc
CXX=g++

# INTEL OR AMD
#OPENCLROOT = $(AMDAPPSDKROOT)
# OPENCLROOT = $(INTELSDKROOT)

#ATI local installation
#MOPENCL_LIBS = -L"$(OPENCLROOT)/lib/x86" -static -lOpenCL
#MOPENCL_FLAGS = -I"$(OPENCLROOT)/include/"

# CUDA
#MOPENCL_LIBS=-L$(CUDA_LIB_PATH) -static -lOpenCL
#MOPENCL_FLAGS=-I$(CUDA_INC_PATH)

#CVT_linux
#MOPENCL_LIBS=-L/usr/lib/ -static -lOpenCL
#MOPENCL_FLAGS=-I/usr/local/share/cuda/include/

#Linux standard
MOPENCL_LIBS=-lOpenCL
MOPENCL_FLAGS=

##################################################
# nastaveni knihoven s OpenGL - mingw
ifeq ($(OS), Windows_NT)
	CFLAGS=$(CFLAGS_COMMON) -DWIN32 -I. -I./ext/include/ $(MOPENCL_FLAGS)
	LIBS=-L./ext/lib/ -static -lSDL -lSDLmain -lSDL_image $(MOPENCL_LIBS)
	OUTPEXE = bin/gmu3.exe
	RM=del
	COPY = copy
else
	CFLAGS=$(CFLAGS_COMMON) $(MOPENCL_FLAGS) `sdl-config --cflags` -I.
	LIBS= `sdl-config --libs` -lSDL_image -lXmu -lX11 -lXi -L/usr/X11R6/lib $(MOPENCL_LIBS)
	OUTPEXE = gmu3
	RM=rm -f
	COPY = cp
endif

CXXFLAGS=$(CFLAGS)

DEPS=sdlwrapper.o sdlwrapper.h error.o error.h

.PHONY: all clean

all: $(OUTPEXE)

clean:
	$(RM) *.o
	$(RM) $(OUTPEXE)
	$(RM) *~

$(OUTPEXE): main.cpp $(DEPS)
	$(CXX) -o $@ main.cpp $(DEPS) $(CFLAGS) $(LIBS)


