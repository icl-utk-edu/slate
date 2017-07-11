
CC  = gcc
CPP = g++

CFLAGS  = -O3 -std=c99
CCFLAGS = -O3 -std=c++11 -fopenmp

MPI = /opt/openmpi

INC = -I$(MPI)/include \
      -I$(MKLROOT)/include

LIB = -L$(MPI)/lib -lmpi \
      -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
      -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
