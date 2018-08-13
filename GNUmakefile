# Relies on settings in environment. These can be set by modules or in make.inc.
# Set compiler by $CXX; usually want CXX=mpicxx.
# Add include directories to $CPATH or $CXXFLAGS for MPI, CUDA, MKL, etc.
# Add lib directories to $LIBRARY_PATH or $LDFLAGS for MPI, CUDA, MKL, etc.
# At runtime, these lib directories need to be in $LD_LIBRARY_PATH,
# or on MacOS, $DYLD_LIBRARY_PATH, or set as rpaths in $LDFLAGS.
#
# Set options on command line or in make.inc file:
# mpi=1           for MPI (-lmpi).
# spectrum=1      for IBM Spectrum MPI (-lmpi_ibm).
#
# mkl=1           for Intel MKL. Additional sub-options:
#   mkl_intel=1     for Intel MKL with Intel Fortran conventions; otherwise uses
#                   GNU conventions. Auto-detected if CXX=icpc or on MacOS.
#   mkl_threaded=1  for multi-threaded Intel MKL.
#   ilp64=1         for ILP64. Currently only with Intel MKL.
# essl=1          for IBM ESSL.
# openblas=1      for OpenBLAS.
#
# cuda=1          for CUDA.
# openmp=1        for OpenMP.
# static=1        for static library (libslate.a);
#                 otherwise shared library (libslate.so).
#
# cuda_arch="ARCH" for CUDA architectures, where ARCH is one or more of:
#                     kepler maxwell pascal volta sm_XX
#                  and sm_XX is a CUDA architecture (see nvcc -h).

-include make.inc

NVCC ?= nvcc

CXXFLAGS += -O3 -std=c++11 -Wall -pedantic -MMD
NVCCFLAGS += -O3 -std=c++11 --compiler-options '-Wall -Wno-unused-function'

# auto-detect OS
# $OSTYPE may not be exported from the shell, so echo it
ostype = $(shell echo $${OSTYPE})
ifneq ($(findstring darwin, $(ostype)),)
	# MacOS is darwin
	macos = 1
endif

#-------------------------------------------------------------------------------
# if shared
ifneq ($(static),1)
	CXXFLAGS += -fPIC
	LDFLAGS  += -fPIC
	NVCCFLAGS += --compiler-options '-fPIC'
endif

#-------------------------------------------------------------------------------
# if OpenMP
ifeq ($(openmp),1)
	CXXFLAGS += -fopenmp
	LDFLAGS  += -fopenmp
else
	lib_src += slate_openmp_stubs.cc
endif

#-------------------------------------------------------------------------------
# if MPI
ifeq ($(mpi),1)
	FLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi
# if Spectrum MPI
else ifeq ($(spectrum),1)
	FLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi_ibm
else
	lib_src += slate_mpi_stubs.cc
endif

#-------------------------------------------------------------------------------
# ScaLAPACK, by default
scalapack = -lscalapack

# BLAS and LAPACK
# if MKL
ifeq ($(mkl_threaded),1)
	mkl = 1
endif
ifeq ($(mkl_intel),1)
	mkl = 1
endif
ifeq ($(mkl),1)
	FLAGS += -DSLATE_WITH_MKL
	# Auto-detect whether to use Intel or GNU conventions.
	# Won't detect if CXX = mpicxx.
	ifeq ($(CXX),icpc)
		mkl_intel = 1
	endif
	ifeq ($(macos),1)
		# MKL on MacOS (version 20180001) has only Intel Fortran version
		mkl_intel = 1
	endif
	ifeq ($(mkl_intel),1)
		# use Intel Fortran conventions
		ifeq ($(ilp64),1)
			LIB += -lmkl_intel_ilp64
		else
			LIB += -lmkl_intel_lp64
		endif

		# if threaded, use Intel OpenMP (iomp5)
		ifeq ($(mkl_threaded),1)
			LIB += -lmkl_intel_thread
		else
			LIB += -lmkl_sequential
		endif
	else
		# use GNU Fortran conventions
		ifeq ($(ilp64),1)
			LIB += -lmkl_gf_ilp64
		else
			LIB += -lmkl_gf_lp64
		endif

		# if threaded, use GNU OpenMP (gomp)
		ifeq ($(mkl_threaded),1)
			LIB += -lmkl_gnu_thread
		else
			LIB += -lmkl_sequential
		endif
	endif

	LIB += -lmkl_core -lpthread -lm -ldl

	# MKL on MacOS doesn't include ScaLAPACK; use default
	ifneq ($(macos),1)
		ifeq ($(ilp64),1)
			scalapack = -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64
		else
			scalapack = -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
		endif
	endif
# if ESSL
else ifeq ($(essl),1)
	FLAGS += -DSLATE_WITH_ESSL
	LIB += -lessl -llapack
# if OpenBLAS
else ifeq ($(openblas),1)
	FLAGS += -DSLATE_WITH_OPENBLAS
	LIB += -lopenblas
endif

#-------------------------------------------------------------------------------
# if CUDA
ifeq ($(cuda),1)
	FLAGS += -DSLATE_WITH_CUDA
	LIB += -lcublas -lcudart
else
	lib_src += slate_cuda_stubs.cc
	lib_src += slate_cublas_stubs.cc
endif

#-------------------------------------------------------------------------------
# Generate flags for which CUDA architectures to build.
# cuda_arch_ is a local copy to modify.
cuda_arch ?= kepler pascal
cuda_arch_ = $(cuda_arch)
ifneq ($(findstring kepler, $(cuda_arch_)),)
	cuda_arch_ += sm_30
endif
ifneq ($(findstring maxwell, $(cuda_arch_)),)
	cuda_arch_ += sm_50
endif
ifneq ($(findstring pascal, $(cuda_arch_)),)
	cuda_arch_ += sm_60
endif
ifneq ($(findstring volta, $(cuda_arch_)),)
	cuda_arch_ += sm_70
endif

# CUDA architectures that nvcc supports
sms = 30 32 35 37 50 52 53 60 61 62 70 72

# code=sm_XX is binary, code=compute_XX is PTX
gencode_sm      = -gencode arch=compute_$(sm),code=sm_$(sm)
gencode_compute = -gencode arch=compute_$(sm),code=compute_$(sm)

# Get gencode options for all sm_XX in cuda_arch_.
nv_sm      = $(filter %, $(foreach sm, $(sms),$(if $(findstring sm_$(sm), $(cuda_arch_)),$(gencode_sm))))
nv_compute = $(filter %, $(foreach sm, $(sms),$(if $(findstring sm_$(sm), $(cuda_arch_)),$(gencode_compute))))

ifeq ($(nv_sm),)
    $(warning No valid CUDA architectures found in cuda_arch = $(cuda_arch).)
else
	# Get last option (last 2 words) of nv_compute.
	nwords  = $(words $(nv_compute))
	nwords_1 = $(shell expr $(nwords) - 1)
	nv_compute_last = $(wordlist $(nwords_1), $(nwords), $(nv_compute))
endif

# Use all sm_XX (binary), and the last compute_XX (PTX) for forward compatibility.
NVCCFLAGS += $(nv_sm) $(nv_compute_last)

#-------------------------------------------------------------------------------
# MacOS needs shared library's path set
ifeq ($(macos),1)
   install_name = -install_name @rpath/$(notdir $@)
else
   install_name =
endif

#-------------------------------------------------------------------------------
# Files

# types and classes
lib_src += \
       slate_Debug.cc \
       slate_Exception.cc \
       slate_Memory.cc \
       slate_trace_Trace.cc \
       slate_types.cc \

# internal
lib_src += \
       slate_internal_comm.cc \
       slate_internal_gemm.cc \
       slate_internal_genorm.cc \
       slate_internal_getrf.cc \
       slate_internal_hemm.cc \
       slate_internal_her2k.cc \
       slate_internal_herk.cc \
       slate_internal_potrf.cc \
       slate_internal_symm.cc \
       slate_internal_synorm.cc \
       slate_internal_syr2k.cc \
       slate_internal_syrk.cc \
       slate_internal_trmm.cc \
       slate_internal_trsm.cc \
       slate_internal_trnorm.cc \
       slate_internal_util.cc \

# device
ifeq ($(cuda),1)
    lib_src += \
           slate_device_genorm.cu \
           slate_device_synorm.cu \
           slate_device_trnorm.cu
endif

# driver
lib_src += \
       slate_gemm.cc \
       slate_getrf.cc \
       slate_hemm.cc \
       slate_her2k.cc \
       slate_herk.cc \
       slate_norm.cc \
       slate_potrf.cc \
       slate_symm.cc \
       slate_syr2k.cc \
       slate_syrk.cc \
       slate_trmm.cc \
       slate_trsm.cc \

# main tester
test_src += \
        test/test.cc       \
        test/test_gemm.cc  \
        test/test_trmm.cc  \
        test/test_symm.cc  \
        test/test_syrk.cc  \
        test/test_syr2k.cc \
        test/test_trsm.cc  \
        test/test_potrf.cc \
        test/test_hemm.cc  \
        test/test_her2k.cc \
        test/test_herk.cc  \
        test/test_genorm.cc  \
        test/test_synorm.cc  \
        test/test_trnorm.cc  \
        test/test_getrf.cc \

# Compile fixes for ScaLAPACK routines if Fortran compiler $(FC) exists.
# Note that 'make' sets $(FC) to f77 by default.
FORTRAN = $(shell which $(FC))
ifneq ($(FORTRAN),)
    test_src += \
        test/pslange.f \
        test/pdlange.f \
        test/pclange.f \
        test/pzlange.f \
        test/pslansy.f \
        test/pdlansy.f \
        test/pclansy.f \
        test/pzlansy.f \
        test/pslantr.f \
        test/pdlantr.f \
        test/pclantr.f \
        test/pzlantr.f
endif

# unit testers
unit_src = \
        unit_test/test_Tile.cc \
        unit_test/test_Matrix.cc \
        unit_test/test_Memory.cc \
        unit_test/test_norm.cc \

# unit test framework
unit_test_obj = \
        unit_test/unit_test.o

lib_obj   = $(addsuffix .o, $(basename $(lib_src)))
test_obj  = $(addsuffix .o, $(basename $(test_src)))
unit_obj  = $(addsuffix .o, $(basename $(unit_src)))
dep       = $(addsuffix .d, $(basename $(lib_src) $(test_src) $(unit_src) \
                                       $(unit_test_obj)))

test      = test/test
unit_test = $(basename $(unit_src))

#-------------------------------------------------------------------------------
# SLATE specific flags and libraries
# FLAGS accumulates definitions, include dirs, etc. for both CXX and NVCC.
FLAGS += -I.
FLAGS += -I./blaspp/include
FLAGS += -I./lapackpp/include

CXXFLAGS  += $(FLAGS)
NVCCFLAGS += $(FLAGS)

# libraries to create libslate.so
LDFLAGS  += -L./blaspp/lib -Wl,-rpath,$(abspath ./blaspp/lib)
LDFLAGS  += -L./lapackpp/lib -Wl,-rpath,$(abspath ./lapackpp/lib)
LIB      := -lblaspp -llapackpp $(LIB)

# additional flags and libraries for testers
$(test_obj): CXXFLAGS += -I./blaspp/test    # for blas_flops.hh
$(test_obj): CXXFLAGS += -I./lapackpp/test  # for lapack_flops.hh
$(test_obj): CXXFLAGS += -I./libtest

TEST_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
TEST_LDFLAGS += -L./libtest -Wl,-rpath,$(abspath ./libtest)
TEST_LIB     += -lslate -ltest $(scalapack)

UNIT_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
UNIT_LIB     += -lslate

#-------------------------------------------------------------------------------
# Rules
.DELETE_ON_ERROR:
.SUFFIXES:
.PHONY: all docs lib test unit_test clean distclean
.DEFAULT_GOAL := all

all: lib test unit_test scalapack_api lapack_api

docs:
	doxygen docs/doxygen/doxyfile.conf

#-------------------------------------------------------------------------------
# LAPACK++ library
liblapackpp_src = $(wildcard lapackpp/include/*.h \
                             lapackpp/include/*.hh \
                             lapackpp/src/*.cc)

ifeq ($(static),1)
	liblapackpp = lapackpp/lib/liblapackpp.a
else
	liblapackpp = lapackpp/lib/liblapackpp.so
endif

$(liblapackpp): $(liblapackpp_src)
	cd lapackpp && $(MAKE) lib

#-------------------------------------------------------------------------------
# BLAS++ library
libblaspp_src = $(wildcard blaspp/include/*.h \
                           blaspp/include/*.hh \
                           blaspp/src/*.cc)

ifeq ($(static),1)
	libblaspp = blaspp/lib/libblaspp.a
else
	libblaspp = blaspp/lib/libblaspp.so
endif

$(libblaspp): $(libblaspp_src)
	cd blaspp && $(MAKE) lib

#-------------------------------------------------------------------------------
# libtest library
libtest_src = $(wildcard libtest/*.hh libtest/*.cc)

ifeq ($(static),1)
	libtest = libtest/libtest.a
else
	libtest = libtest/libtest.so
endif

$(libtest): $(libtest_src)
	cd libtest && $(MAKE) lib

#-------------------------------------------------------------------------------
# libslate library
lib_a  = ./lib/libslate.a
lib_so = ./lib/libslate.so

$(lib_a): $(lib_obj) $(libblaspp) $(liblapackpp)
	mkdir -p lib
	-rm $@
	ar cr $@ $(lib_obj)
	ranlib $@

$(lib_so): $(lib_obj) $(libblaspp) $(liblapackpp)
	mkdir -p lib
	$(CXX) $(LDFLAGS) \
		$(lib_obj) \
		$(LIB) \
		-shared $(install_name) -o $@

ifeq ($(static),1)
    lib = $(lib_a)
else
    lib = $(lib_so)
endif

lib: $(lib)

#-------------------------------------------------------------------------------
# main tester
test: $(test)

test/clean:
	rm -f $(test) $(test_obj)

$(test): $(test_obj) $(lib) $(libtest)
	$(CXX) $(TEST_LDFLAGS) $(LDFLAGS) $(test_obj) \
		$(TEST_LIB) $(LIB) -o $@

#-------------------------------------------------------------------------------
# unit testers
unit_test: $(unit_test)

unit_test/clean:
	rm -f $(unit_test) $(unit_obj) $(unit_test_obj)

$(unit_test): %: %.o $(unit_test_obj) $(lib)
	$(CXX) $(UNIT_LDFLAGS) $(LDFLAGS) $< \
		$(unit_test_obj) $(UNIT_LIB) $(LIB) -o $@

#-------------------------------------------------------------------------------
# scalapack_api library
scalapack_api = lib/libslate_scalapack_api.so

scalapack_api_src += \
                     scalapack_api/scalapack_gemm.cc \
                     scalapack_api/scalapack_syrk.cc \
                     scalapack_api/scalapack_symm.cc \
                     scalapack_api/scalapack_trsm.cc \
                     scalapack_api/scalapack_syr2k.cc \
                     scalapack_api/scalapack_trmm.cc \
                     scalapack_api/scalapack_hemm.cc \
                     scalapack_api/scalapack_herk.cc \
                     scalapack_api/scalapack_her2k.cc \
                     scalapack_api/scalapack_lange.cc \
                     scalapack_api/scalapack_lansy.cc \
                     scalapack_api/scalapack_lantr.cc \

scalapack_api_obj = $(addsuffix .o, $(basename $(scalapack_api_src)))

SCALAPACK_API_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
SCALAPACK_API_LIB     += -lslate $(scalapack)

scalapack_api: lib $(scalapack_api)

scalapack_api/clean:
	rm -f $(scalapack_api) $(scalapack_api_obj)

$(scalapack_api): $(scalapack_api_obj) $(lib)
	$(CXX) $(SCALAPACK_API_LDFLAGS) $(LDFLAGS) $(scalapack_api_obj) \
		$(SCALAPACK_API_LIB) $(LIB) -shared $(install_name) -o $@

#-------------------------------------------------------------------------------
# lapack_api library
lapack_api = lib/libslate_lapack_api.so

lapack_api_src += \
		lapack_api/lapack_gemm.cc \
		lapack_api/lapack_hemm.cc \
		lapack_api/lapack_symm.cc \
		lapack_api/lapack_trmm.cc \
		lapack_api/lapack_trsm.cc \
		lapack_api/lapack_herk.cc \
		lapack_api/lapack_syrk.cc \

lapack_api_obj = $(addsuffix .o, $(basename $(lapack_api_src)))

LAPACK_API_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
LAPACK_API_LIB     += -lslate 

lapack_api: lib $(lapack_api)

lapack_api/clean:
	rm -f $(lapack_api) $(lapack_api_obj)

$(lapack_api): $(lapack_api_obj) $(lib)
	$(CXX) $(LAPACK_API_LDFLAGS) $(LDFLAGS) $(lapack_api_obj) \
		$(LAPACK_API_LIB) $(LIB) -shared $(install_name) -o $@

#-------------------------------------------------------------------------------
# general rules
clean: test/clean unit_test/clean
	rm -f $(lib_a) $(lib_so) $(lib_obj)
	rm -f trace_*.svg

distclean: clean
	rm -f $(dep)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.f
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# preprocess source
%.i: %.cc
	$(CXX) $(CXXFLAGS) -E $< -o $@

# precompile header to check for errors
%.gch: %.hh
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(dep)

#-------------------------------------------------------------------------------
# debugging
echo:
	@echo "openmp        = '$(openmp)'"
	@echo "mpi           = '$(mpi)'"
	@echo "spectrum      = '$(spectrum)'"
	@echo "macos         = '$(macos)'"
	@echo "mkl_intel     = '$(mkl_intel)'"
	@echo "ilp64         = '$(ilp64)'"
	@echo "mkl           = '$(mkl)'"
	@echo "mkl_threaded  = '$(mkl_threaded)'"
	@echo "essl          = '$(essl)'"
	@echo "cuda          = '$(cuda)'"
	@echo "static        = '$(static)'"
	@echo
	@echo "lib_a         = $(lib_a)"
	@echo "lib_so        = $(lib_so)"
	@echo "lib           = $(lib)"
	@echo
	@echo "lib_obj       = $(lib_obj)"
	@echo
	@echo "test_src      = $(test_src)"
	@echo
	@echo "test_obj      = $(test_obj)"
	@echo
	@echo "test          = $(test)"
	@echo
	@echo "unit_src      = $(unit_src)"
	@echo
	@echo "unit_obj      = $(unit_obj)"
	@echo
	@echo "unit_test_obj = $(unit_test_obj)"
	@echo
	@echo "unit_test     = $(unit_test)"
	@echo
	@echo "dep           = $(dep)"
	@echo
	@echo "CXX           = $(CXX)"
	@echo "CXXFLAGS      = $(CXXFLAGS)"
	@echo
	@echo "NVCC          = $(NVCC)"
	@echo "NVCCFLAGS     = $(NVCCFLAGS)"
	@echo "cuda_arch     = $(cuda_arch)"
	@echo "cuda_arch_    = $(cuda_arch_)"
	@echo "sms           = $(sms)"
	@echo "nv_sm         = $(nv_sm)"
	@echo "nv_compute    = $(nv_compute)"
	@echo "nwords        = $(nwords)"
	@echo "nwords_1      = $(nwords_1)"
	@echo "nv_compute_last = $(nv_compute_last)"
	@echo
	@echo "FC            = $(FC)"
	@echo "FCFLAGS       = $(FCFLAGS)"
	@echo
	@echo "LDFLAGS       = $(LDFLAGS)"
	@echo "LIB           = $(LIB)"
	@echo
	@echo "TEST_LDFLAGS  = $(TEST_LDFLAGS)"
	@echo "TEST_LIB      = $(TEST_LIB)"
	@echo
	@echo "UNIT_LDFLAGS  = $(UNIT_LDFLAGS)"
	@echo "UNIT_LIB      = $(UNIT_LIB)"
