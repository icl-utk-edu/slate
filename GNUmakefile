# Relies on settings in environment. These can be set by modules.
# Set compiler by $CC and $CXX.
# Add include directories to $CPATH for MPI, CUDA, MKL, etc.
# Add lib directories to $LIBRARY_PATH for MPI, CUDA, MKL, etc.
# At runtime, these lib directories need to be in $LD_LIBRARY_PATH,
# or on MacOS, $DYLD_LIBRARY_PATH.
#
# Set options on command line or in make.inc file:
# mpi=1         for MPI (-lmpi)
# spectrum=1    for IBM Spectrum MPI (-lmpi_ibm)
# mkl=1         for Intel MKL. $MKLROOT must also be set.
# cuda=1        for CUDA
# openmp=1      for OpenMP
# shared=1      for shared library (libslate.so); otherwise static (libslate.a)

top ?= .
-include ${top}/make.inc

CXXFLAGS += -O3 -std=c++11 -Wall -pedantic -MMD

pwd = ${shell pwd}

#-------------------------------------------------------------------------------
# if shared
ifeq (${shared},1)
	CXXFLAGS += -fPIC
	LDFLAGS  += -fPIC
endif

#-------------------------------------------------------------------------------
# if OpenMP
ifeq (${openmp},1)
	CXXFLAGS += -fopenmp
	LDFLAGS  += -fopenmp
else
	lib_src += slate_openmp_stubs.cc
endif

#-------------------------------------------------------------------------------
# if MPI
ifeq (${mpi},1)
	CXXFLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi
# if Spectrum MPI
else ifeq (${spectrum},1)
	CXXFLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi_ibm
else
	lib_src += slate_mpi_stubs.cc
endif

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
# if CUDA
ifeq (${cuda},1)
	CXXFLAGS += -DSLATE_WITH_CUDA
	LIB += -lcublas -lcudart
else
	lib_src += slate_cuda_stubs.cc
	lib_src += slate_cublas_stubs.cc
endif

#-------------------------------------------------------------------------------
# SLATE libraries
CXXFLAGS += -I${top}
CXXFLAGS += -I${top}/blaspp/include
CXXFLAGS += -I${top}/lapackpp/include
CXXFLAGS += -I./blaspp/test  # for blas_flops.hh

LIB += -L${top}/lapackpp/lib -Wl,-rpath,${pwd}/${top}/lapackpp/lib -llapackpp

#-------------------------------------------------------------------------------
# Files

# types and classes
lib_src += \
       slate_Debug.cc \
       slate_Memory.cc \
       slate_trace_Trace.cc \
       slate_types.cc \

# internal
lib_src += \
       slate_internal_comm.cc \
       slate_internal_gemm.cc \
       slate_internal_hemm.cc \
       slate_internal_her2k.cc \
       slate_internal_herk.cc \
       slate_internal_potrf.cc \
       slate_internal_symm.cc \
       slate_internal_syr2k.cc \
       slate_internal_syrk.cc \
       slate_internal_trmm.cc \
       slate_internal_trsm.cc \
       slate_internal_util.cc \

# driver
lib_src += \
       slate_gemm.cc \
       slate_hemm.cc \
       slate_her2k.cc \
       slate_herk.cc \
       slate_potrf.cc \
       slate_symm.cc \
       slate_syr2k.cc \
       slate_syrk.cc \
       slate_trmm.cc \
       slate_trsm.cc \

test_src = \
       test_internal_blas.cc \
       test_memory.cc \
       test_matrix.cc \
       test_tile.cc \
       test_tile_blas.cc \

lib_obj  = $(lib_src:.cc=.o)
test_obj = $(test_src:.cc=.o)
dep      = $(lib_src:.cc=.d) $(test_src:.cc=.d)

test = $(basename $(test_src))


#-------------------------------------------------------------------------------
# Rules
.DELETE_ON_ERROR:
.SUFFIXES:
.PHONY: all docs libs clean

all: $(test)

docs:
	doxygen docs/doxygen/doxyfile.conf

lib:
	mkdir lib

# shared or static library
ifeq (${shared},1)
    lib_so = ${top}/lib/libslate.so

    libs = $(lib_so)

    $(lib_so): $(lib_obj) | lib
		$(CXX) $(LDFLAGS) $^ $(LIB) -shared -o $@
else
    lib_a = ${top}/lib/libslate.a

    libs = $(lib_a)

    $(lib_a): $(lib_obj) | lib
		ar cr $@ $^
		ranlib $@
endif

libs: $(libs)

$(test): %: %.o $(libs)
	$(CXX) $(LDFLAGS) $< -L${top}/lib -Wl,-rpath,${pwd}/lib -lslate $(LIB) -o $@

clean:
	rm -f $(lib_obj) $(test_obj) $(test) trace_*.svg 

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# preprocess source
%.i: %.cc
	$(CXX) $(CXXFLAGS) -E $< -o $@

# precompile header to check for errors
%.gch: %.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include ${dep}

echo:
	@echo "openmp   = '${openmp}'"
	@echo "mpi      = '${mpi}'"
	@echo "spectrum = '${spectrum}'"
	@echo "linux    = '${linux}'"
	@echo "macos    = '${macos}'"
	@echo "mkl      = '${mkl}'"
	@echo "essl     = '${essl}'"
	@echo "cuda     = '${cuda}'"
	@echo "shared   = '${shared}'"
	@echo
	@echo "lib_src  = ${lib_src}"
	@echo "lib_obj  = ${lib_obj}"
	@echo "test_src = ${test_src}"
	@echo "test_obj = ${test_obj}"
	@echo "test     = ${test}"
	@echo "dep      = ${dep}"
	@echo
	@echo "lib_a    = ${lib_a}"
	@echo "lib_so   = ${lib_so}"
	@echo "libs     = ${libs}"
	@echo
	@echo "CXX      = ${CXX}"
	@echo "CXXFLAGS = ${CXXFLAGS}"
	@echo "LDFLAGS  = ${LDFLAGS}"
	@echo "LIB      = ${LIB}"
