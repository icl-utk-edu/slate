
CC  = mpicc
CPP = mpicxx

CFLAGS  = -O3 -std=c99
CCFLAGS = -O3 -std=c++17 -fopenmp 

#MPI  = /sw/openmpi/2.1.1-thread-multiple
CUDA = /usr/local/cuda

INC = -I$(MPI)/include \
      -I$(MKLROOT)/include

LIB = -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
      -L$(CUDA)/lib64 -lcublas -lcudart \
      -lmpi
