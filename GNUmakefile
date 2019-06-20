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
#   openmpi=1       for OpenMPI BLACS.
#   intelmpi=1      for Intel MPI BLACS (default).
# essl=1          for IBM ESSL.
# openblas=1      for OpenBLAS.
#
# cuda=1          for CUDA.
# openmp=1        for OpenMP.
# static=1        for static library (libslate.a);
#                 otherwise shared library (libslate.so).
#
# cuda_arch="ARCH" for CUDA architectures, where ARCH is one or more of:
#                     kepler maxwell pascal volta turing sm_XX
#                  and sm_XX is a CUDA architecture (see nvcc -h).

-include make.inc

NVCC ?= nvcc

CXXFLAGS  += -O3 -std=c++11 -Wall -pedantic -MMD
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
    libslate_src += src/stubs/openmp_stubs.cc
endif

#-------------------------------------------------------------------------------
# if MPI
ifeq ($(mpi),1)
    LIBS  += -lmpi
# if Spectrum MPI
else ifeq ($(spectrum),1)
    LIBS  += -lmpi_ibm
else
    FLAGS += -DSLATE_NO_MPI
    libslate_src += src/stubs/mpi_stubs.cc
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
            LIBS += -lmkl_intel_ilp64
        else
            LIBS += -lmkl_intel_lp64
        endif

        # if threaded, use Intel OpenMP (iomp5)
        ifeq ($(mkl_threaded),1)
            LIBS += -lmkl_intel_thread
        else
            LIBS += -lmkl_sequential
        endif
    else
        # use GNU Fortran conventions
        ifeq ($(ilp64),1)
            LIBS += -lmkl_gf_ilp64
        else
            LIBS += -lmkl_gf_lp64
        endif

        # if threaded, use GNU OpenMP (gomp)
        ifeq ($(mkl_threaded),1)
            LIBS += -lmkl_gnu_thread
        else
            LIBS += -lmkl_sequential
        endif
    endif

    LIBS += -lmkl_core -lpthread -lm -ldl

    # MKL on MacOS doesn't include ScaLAPACK; use default.
    # For others, link with appropriate version of ScaLAPACK and BLACS.
    ifneq ($(macos),1)
        ifeq ($(openmpi),1)
            ifeq ($(ilp64),1)
                scalapack = -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64
            else
                scalapack = -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64
            endif
        else
            ifeq ($(ilp64),1)
                scalapack = -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64
            else
                scalapack = -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
            endif
        endif
    endif
# if ESSL
else ifeq ($(essl),1)
    FLAGS += -DSLATE_WITH_ESSL
    LIBS += -lessl -llapack
# if OpenBLAS
else ifeq ($(openblas),1)
    FLAGS += -DSLATE_WITH_OPENBLAS
    LIBS += -lopenblas
endif

#-------------------------------------------------------------------------------
# if CUDA
ifeq ($(cuda),1)
    LIBS += -lcublas -lcudart
else
    FLAGS += -DSLATE_NO_CUDA
    libslate_src += src/stubs/cuda_stubs.cc
    libslate_src += src/stubs/cublas_stubs.cc
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
ifneq ($(findstring turing, $(cuda_arch_)),)
    cuda_arch_ += sm_75
endif

# CUDA architectures that nvcc supports
sms = 30 32 35 37 50 52 53 60 61 62 70 72 75

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
libslate_src += \
        src/aux/Debug.cc \
        src/aux/Exception.cc \
        src/core/Memory.cc \
        src/aux/Trace.cc \
        src/core/types.cc \

# internal
libslate_src += \
        src/internal/internal_comm.cc \
        src/internal/internal_gecopy.cc \
        src/internal/internal_gbnorm.cc \
        src/internal/internal_geadd.cc \
        src/internal/internal_gemm.cc \
        src/internal/internal_gemm_A.cc \
        src/internal/internal_genorm.cc \
        src/internal/internal_geqrf.cc \
        src/internal/internal_geset.cc \
        src/internal/internal_getrf.cc \
        src/internal/internal_hemm.cc \
        src/internal/internal_henorm.cc \
        src/internal/internal_her2k.cc \
        src/internal/internal_herk.cc \
        src/internal/internal_potrf.cc \
        src/internal/internal_swap.cc \
        src/internal/internal_symm.cc \
        src/internal/internal_synorm.cc \
        src/internal/internal_syr2k.cc \
        src/internal/internal_syrk.cc \
        src/internal/internal_trmm.cc \
        src/internal/internal_trnorm.cc \
        src/internal/internal_trsm.cc \
        src/internal/internal_trtri.cc \
        src/internal/internal_trtrm.cc \
        src/internal/internal_ttmqr.cc \
        src/internal/internal_ttqrt.cc \
        src/internal/internal_unmqr.cc \
        src/internal/internal_util.cc \
        src/internal/internal_transpose.cc \
        src/internal/internal_tzcopy.cc \

# device
ifeq ($(cuda),1)
    libslate_src += \
            src/cuda/device_geadd.cu \
            src/cuda/device_gecopy.cu \
            src/cuda/device_genorm.cu \
            src/cuda/device_geset.cu \
            src/cuda/device_henorm.cu \
            src/cuda/device_synorm.cu \
            src/cuda/device_trnorm.cu \
            src/cuda/device_transpose.cu \
            src/cuda/device_tzcopy.cu \

endif

# driver
libslate_src += \
        src/colNorms.cc \
        src/copy.cc \
        src/gbmm.cc \
        src/gbsv.cc \
        src/gbtrf.cc \
        src/gbtrs.cc \
        src/geadd.cc \
        src/gels.cc \
        src/gemm.cc \
        src/geqrf.cc \
        src/gesv.cc \
        src/gesvMixed.cc \
        src/getrf.cc \
        src/getri.cc \
        src/getrs.cc \
        src/hemm.cc \
        src/her2k.cc \
        src/herk.cc \
        src/hesv.cc \
        src/hetrf.cc \
        src/hetrs.cc \
        src/norm.cc \
        src/posv.cc \
        src/posvMixed.cc \
        src/potrf.cc \
        src/potri.cc \
        src/potrs.cc \
        src/symm.cc \
        src/syr2k.cc \
        src/syrk.cc \
        src/tbsm.cc \
        src/trmm.cc \
        src/trsm.cc \
        src/trtri.cc \
        src/trtrm.cc \
        src/unmqr.cc \

# main tester
test_src += \
        test/test.cc \
        test/test_gbmm.cc \
        test/test_gbnorm.cc \
        test/test_gbsv.cc \
        test/test_gels.cc \
        test/test_gemm.cc \
        test/test_genorm.cc \
        test/test_geqrf.cc \
        test/test_gesv.cc \
        test/test_hemm.cc \
        test/test_henorm.cc \
        test/test_her2k.cc \
        test/test_herk.cc \
        test/test_hesv.cc \
        test/test_posv.cc \
        test/test_symm.cc \
        test/test_synorm.cc \
        test/test_syr2k.cc \
        test/test_syrk.cc \
        test/test_tbsm.cc \
        test/test_trmm.cc \
        test/test_trnorm.cc \
        test/test_trsm.cc \
        test/test_potri.cc

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
        test/pzlantr.f \

endif

# unit testers
unit_src = \
        unit_test/test_BandMatrix.cc \
        unit_test/test_HermitianMatrix.cc \
        unit_test/test_Matrix.cc \
        unit_test/test_Memory.cc \
        unit_test/test_SymmetricMatrix.cc \
        unit_test/test_TrapezoidMatrix.cc \
        unit_test/test_TriangularMatrix.cc \
        unit_test/test_Tile.cc \
        unit_test/test_Tile_kernels.cc \
        unit_test/test_norm.cc \

# unit test framework
unit_test_obj = \
        unit_test/unit_test.o

libslate_obj = $(addsuffix .o, $(basename $(libslate_src)))
test_obj     = $(addsuffix .o, $(basename $(test_src)))
unit_obj     = $(addsuffix .o, $(basename $(unit_src)))
dep          = $(addsuffix .d, $(basename $(libslate_src) $(test_src) \
                                          $(unit_src) $(unit_test_obj)))

test      = test/test
unit_test = $(basename $(unit_src))

#-------------------------------------------------------------------------------
# SLATE specific flags and libraries
# FLAGS accumulates definitions, include dirs, etc. for both CXX and NVCC.
# FLAGS += -I.
FLAGS += -I./blaspp/include
FLAGS += -I./lapackpp/include
FLAGS += -I./include
FLAGS += -I./src

CXXFLAGS  += $(FLAGS)
NVCCFLAGS += $(FLAGS)

# libraries to create libslate.so
LDFLAGS  += -L./blaspp/lib -Wl,-rpath,$(abspath ./blaspp/lib)
LDFLAGS  += -L./lapackpp/lib -Wl,-rpath,$(abspath ./lapackpp/lib)
LIBS     := -lblaspp -llapackpp $(LIBS)

# additional flags and libraries for testers
$(test_obj): CXXFLAGS += -I./libtest

TEST_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
TEST_LDFLAGS += -L./libtest -Wl,-rpath,$(abspath ./libtest)
TEST_LIBS    += -lslate -ltest $(scalapack)

UNIT_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
UNIT_LIBS    += -lslate

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
libslate_a  = ./lib/libslate.a
libslate_so = ./lib/libslate.so

$(libslate_a): $(libslate_obj) $(libblaspp) $(liblapackpp)
	mkdir -p lib
	-rm $@
	ar cr $@ $(libslate_obj)
	ranlib $@

$(libslate_so): $(libslate_obj) $(libblaspp) $(liblapackpp)
	mkdir -p lib
	$(CXX) $(LDFLAGS) \
		$(libslate_obj) \
		$(LIBS) \
		-shared $(install_name) -o $@

ifeq ($(static),1)
    libslate = $(libslate_a)
else
    libslate = $(libslate_so)
endif

lib: $(libslate)

#-------------------------------------------------------------------------------
# main tester
test: $(test)

test/clean:
	rm -f $(test) $(test_obj)

$(test): $(test_obj) $(libslate) $(libtest)
	$(CXX) $(TEST_LDFLAGS) $(LDFLAGS) $(test_obj) \
		$(TEST_LIBS) $(LIBS) \
		-o $@

#-------------------------------------------------------------------------------
# unit testers
unit_test: $(unit_test)

unit_test/clean:
	rm -f $(unit_test) $(unit_obj) $(unit_test_obj)

$(unit_test): %: %.o $(unit_test_obj) $(libslate)
	$(CXX) $(UNIT_LDFLAGS) $(LDFLAGS) $< \
		$(unit_test_obj) $(UNIT_LIBS) $(LIBS)  \
		-o $@

#-------------------------------------------------------------------------------
# scalapack_api library
scalapack_api = lib/libslate_scalapack_api.so

scalapack_api_src += \
        scalapack_api/scalapack_gemm.cc \
        scalapack_api/scalapack_hemm.cc \
        scalapack_api/scalapack_her2k.cc \
        scalapack_api/scalapack_herk.cc \
        scalapack_api/scalapack_lange.cc \
        scalapack_api/scalapack_lansy.cc \
        scalapack_api/scalapack_lantr.cc \
        scalapack_api/scalapack_potrf.cc \
        scalapack_api/scalapack_getrf.cc \
        scalapack_api/scalapack_symm.cc \
        scalapack_api/scalapack_syr2k.cc \
        scalapack_api/scalapack_syrk.cc \
        scalapack_api/scalapack_trmm.cc \
        scalapack_api/scalapack_trsm.cc \
        scalapack_api/scalapack_getrs.cc \
        scalapack_api/scalapack_gesv.cc \
        scalapack_api/scalapack_lanhe.cc \
        scalapack_api/scalapack_posv.cc \
        scalapack_api/scalapack_gels.cc

scalapack_api_obj = $(addsuffix .o, $(basename $(scalapack_api_src)))

dep += $(addsuffix .d, $(basename $(scalapack_api_src)))

SCALAPACK_API_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
SCALAPACK_API_LIBS    += -lslate $(scalapack)

scalapack_api: $(scalapack_api)

scalapack_api/clean:
	rm -f $(scalapack_api) $(scalapack_api_obj)

$(scalapack_api): $(scalapack_api_obj) $(libslate)
	$(CXX) $(SCALAPACK_API_LDFLAGS) $(LDFLAGS) $(scalapack_api_obj) \
		$(SCALAPACK_API_LIBS) $(LIBS) -shared $(install_name) -o $@

#-------------------------------------------------------------------------------
# lapack_api library
lapack_api = lib/libslate_lapack_api.so

lapack_api_src += \
        lapack_api/lapack_gemm.cc \
        lapack_api/lapack_hemm.cc \
        lapack_api/lapack_her2k.cc \
        lapack_api/lapack_herk.cc \
        lapack_api/lapack_lange.cc \
        lapack_api/lapack_lansy.cc \
        lapack_api/lapack_lantr.cc \
        lapack_api/lapack_potrf.cc \
        lapack_api/lapack_getrf.cc \
        lapack_api/lapack_symm.cc \
        lapack_api/lapack_syr2k.cc \
        lapack_api/lapack_syrk.cc \
        lapack_api/lapack_trmm.cc \
        lapack_api/lapack_trsm.cc \
        lapack_api/lapack_slate.cc \
        lapack_api/lapack_getrs.cc \
        lapack_api/lapack_lanhe.cc \


lapack_api_obj = $(addsuffix .o, $(basename $(lapack_api_src)))

dep += $(addsuffix .d, $(basename $(lapack_api_src)))

LAPACK_API_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
LAPACK_API_LIBS    += -lslate

lapack_api: $(lapack_api)

lapack_api/clean:
	rm -f $(lapack_api) $(lapack_api_obj)

$(lapack_api): $(lapack_api_obj) $(libslate)
	$(CXX) $(LAPACK_API_LDFLAGS) $(LDFLAGS) $(lapack_api_obj) \
		$(LAPACK_API_LIBS) $(LIBS) -shared $(install_name) -o $@

#-------------------------------------------------------------------------------
# general rules
clean: test/clean unit_test/clean scalapack_api/clean lapack_api/clean
	rm -f $(libslate_a) $(libslate_so) $(libslate_obj)
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
	@echo "libblaspp     = $(libblaspp)"
	@echo "liblapackpp   = $(liblapackpp)"
	@echo "libtest       = $(libtest)"
	@echo
	@echo "libslate_a    = $(libslate_a)"
	@echo "libslate_so   = $(libslate_so)"
	@echo "libslate      = $(libslate)"
	@echo
	@echo "libslate_obj  = $(libslate_obj)"
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
	@echo "LIBS          = $(LIBS)"
	@echo
	@echo "TEST_LDFLAGS  = $(TEST_LDFLAGS)"
	@echo "TEST_LIBS     = $(TEST_LIBS)"
	@echo
	@echo "UNIT_LDFLAGS  = $(UNIT_LDFLAGS)"
	@echo "UNIT_LIBS     = $(UNIT_LIBS)"
