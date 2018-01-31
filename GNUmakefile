
CFLAGS   = -O3 -std=c99
CXXFLAGS = -O3 -std=c++11

#---------------------------------------------
# if OpenMP
ifeq (${openmp},1)
	CXXFLAGS += -fopenmp
else
	SRC += slate_NoOpenmp.cc
endif

#------------------------------------------------------
# if MPI
ifeq (${mpi},1)
	CFLAGS   += -DSLATE_WITH_MPI
	CXXFLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi
# if Spectrum MPI
else ifeq (${spectrum},1)
	CFLAGS   += -DSLATE_WITH_MPI
	CXXFLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi_ibm
else
	SRC += slate_NoMpi.cc
endif

#-----------------------------------------------------------------------------
# if MKL 
ifeq (${mkl},1)
	CXXFLAGS += -DSLATE_WITH_MKL
	# if Linux
	ifeq (${linux},1)
		LIB += -L${MKLROOT}/lib \
		       -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	# if MacOS
	else ifeq (${macos},1)
		LIB += -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
		       -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	endif
# if ESSL
else ifeq (${essl},1)
	CXXFLAGS += -DSLATE_WITH_ESSL
	LIB += -lessl -llapack
endif

#-----------------------------------------
# if CUDA
ifeq (${cuda},1)
	CXXFLAGS += -DSLATE_WITH_CUDA
	LIB += -lcublas -lcudart
else
	SRC += slate_NoCuda.cc
	SRC += slate_NoCublas.cc
endif

CXXFLAGS += -I./blaspp/include
CXXFLAGS += -I./lapackpp/include

LIB += -L./lapackpp/lib -llapackpp

#-------------------------------------------------------------------------------
SRC += slate_Debug.cc \
       slate_Matrix_gemm.cc \
       slate_Matrix_potrf.cc \
       slate_Matrix_syrk.cc \
       slate_Matrix_trsm.cc \
       slate_Matrix.cc \
       slate_Memory.cc \
       slate_Tile.cc \
       slate_potrf.cc

OBJ = $(SRC:.cc=.o)

all: potrf

potrf: $(OBJ) potrf.o trace/trace.o
	$(CXX) $(CXXFLAGS) $(OBJ) potrf.o trace/trace.o $(LIB) -o $@

clean:
	rm -f $(OBJ)
	rm -f potrf potrf.o trace_*.svg

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
