
CC  = gcc
CPP = g++

CFLAGS  = -O3 -std=c99
CCFLAGS = -O3 -std=c++11 -fopenmp

INC = -I$(MKLROOT)/include
LIB = -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
      -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
