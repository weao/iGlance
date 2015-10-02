CVFLAGS = `pkg-config --cflags opencv`
CVLIBS = `pkg-config --libs opencv`
LIBS = -L /usr/local/lib
DEPS = Sighter.h Sighter.cpp Pupiler.cpp Pupiler.h main.cpp MLClassifier.h MLClassifier.cpp
INC = -I /usr/local/include
CPP = main.cpp MLClassifier.cpp Sighter.cpp Pupiler.cpp

build : $(CPP) Makefile $(DEPS)
	g++ $(CPP) $(CVFLAGS) $(CVLIBS) $(LIBS) -o iglance
