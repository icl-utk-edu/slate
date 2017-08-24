
CC  = gcc
CPP = g++

CFLAGS  = -O3 -std=c99
CCFLAGS = -O3 -std=c++11 -fopenmp 

CUDA = /usr/local/cuda
MPI  = /sw/openmpi/2.1.1-thread-multiple

INC = -I$(MPI)/include \
      -I$(MKLROOT)/include

LIB = -L$(MPI)/lib -lmpi \
      -L${MKLROOT}/lib \
      -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
      -L$(CUDA)/lib64 -lcublas
