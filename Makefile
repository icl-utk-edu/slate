
include make.inc

all:
	$(CPP) $(CCFLAGS) $(INC) -o app app.cc $(LIB) $(LIBS)
