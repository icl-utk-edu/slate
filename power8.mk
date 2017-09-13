# On summitdev:
# Currently Loaded Modules:
#   1) hsi/5.0.2.p5   3) DefApps    5) gcc/7.1.1-20170802               7) essl/5.5.0-20161110
#   2) xalt/0.7.5     4) tmux/2.2   6) spectrum_mpi/10.1.0.3-20170501   8) cuda/8.0.54
CC  = mpicc
CPP = mpic++

CFLAGS  = -O3 -std=c99 -DESSL -DNOBATCH
CCFLAGS = -O3 -std=c++11 -fopenmp -DESSL -DNOBATCH


ESSLROOT = $(OLCF_ESSL_ROOT)
INC =  	-I ~/lapack-3.7.0-US/CBLAS/include \
	-I ~/lapack-3.7.0-US/LAPACKE/include \
	-I$(OLCF_ESSL_ROOT)/include \
	-I${CUDA_DIR}/include

LIB =   -L${ESSLROOT}/lib64 /sw/summitdev/essl/5.5.0-20161110/lib64/libessl.so \
	/sw/summitdev/xl/20161123/xlf/15.1.5/lib/libxlf90.so \
	/sw/summitdev/xl/20161123/xlf/15.1.5/lib/libxlfmath.so \
	~/lapack-3.7.0-US/liblapacke.a \
	~/lapack-3.7.0-US/liblapack.a \
	-lpthread -lm -ldl \
	-L${CUDA_DIR}/lib64 -lcublas -lcudart
