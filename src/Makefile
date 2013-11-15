##################################################
# nastaveni
CFLAGS_COMMON=
CC=gcc
CXX=g++

#Linux standard
MOPENCL_LIBS=-lOpenCL
MOPENCL_FLAGS=

##################################################
# nastaveni knihoven s OpenGL - mingw
ifeq ($(OS), Windows_NT)
	CFLAGS=$(CFLAGS_COMMON) -DWIN32 -I. -I./ext/include/ $(MOPENCL_FLAGS)
	LIBS=-L./ext/lib/ -static -lSDL -lSDLmain -lSDL_image $(MOPENCL_LIBS)
	OUTPEXE=gmu.exe
	RM=del
	COPY=copy
else
	CFLAGS=$(CFLAGS_COMMON) $(MOPENCL_FLAGS) `sdl-config --cflags` -I.
	LIBS=`sdl-config --libs` -lSDL_image -lXmu -lX11 -lXi -L/usr/X11R6/lib $(MOPENCL_LIBS)
	OUTPEXE=gmu
	RM=rm -f
	COPY=cp
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


