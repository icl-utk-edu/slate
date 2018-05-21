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

-include make.inc

CXXFLAGS += -O3 -std=c++11 -Wall -pedantic -MMD

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
	CXXFLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi
# if Spectrum MPI
else ifeq ($(spectrum),1)
	CXXFLAGS += -DSLATE_WITH_MPI
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
	CXXFLAGS += -DSLATE_WITH_MKL
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
	CXXFLAGS += -DSLATE_WITH_ESSL
	LIB += -lessl -llapack
# if OpenBLAS
else ifeq ($(openblas),1)
	CXXFLAGS += -DSLATE_WITH_OPENBLAS
	LIB += -lopenblas
endif

#-------------------------------------------------------------------------------
# if CUDA
ifeq ($(cuda),1)
	CXXFLAGS += -DSLATE_WITH_CUDA
	LIB += -lcublas -lcudart
else
	lib_src += slate_cuda_stubs.cc
	lib_src += slate_cublas_stubs.cc
endif

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
       slate_Memory.cc \
       slate_trace_Trace.cc \
       slate_types.cc \

# internal
lib_src += \
       slate_internal_comm.cc \
       slate_internal_gemm.cc \
       slate_internal_genorm.cc \
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
       slate_genorm.cc \
       slate_hemm.cc \
       slate_her2k.cc \
       slate_herk.cc \
       slate_potrf.cc \
       slate_symm.cc \
       slate_syr2k.cc \
       slate_syrk.cc \
       slate_trmm.cc \
       slate_trsm.cc \

# main tester
test_src = \
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

#unit testers
unit_src = \
        unit_test/test_Tile.cc \
        unit_test/test_Memory.cc \

# unit test framework
unit_test_obj = \
        unit_test/unit_test.o

lib_obj   = $(lib_src:.cc=.o)
test_obj  = $(test_src:.cc=.o)
unit_obj  = $(unit_src:.cc=.o)
dep       = $(lib_src:.cc=.d) $(test_src:.cc=.d) $(unit_src:.cc=.d) \
            $(unit_test_obj:.o=.d)

test      = test/test
unit_test = $(basename $(unit_src))

#-------------------------------------------------------------------------------
# SLATE specific flags and libraries
CXXFLAGS += -I.
CXXFLAGS += -I./blaspp/include
CXXFLAGS += -I./lapackpp/include

# libraries to create libslate.so
LDFLAGS  += -L./lapackpp/lib -Wl,-rpath,$(abspath ./lapackpp/lib)
LIB      := -llapackpp $(LIB)

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

all: lib test unit_test

docs:
	doxygen docs/doxygen/doxyfile.conf

#-------------------------------------------------------------------------------
# libslate library
lib_a  = ./lib/libslate.a
lib_so = ./lib/libslate.so

$(lib_a): $(lib_obj)
	mkdir -p lib
	-rm $@
	ar cr $@ $^
	ranlib $@

$(lib_so): $(lib_obj)
	mkdir -p lib
	$(CXX) $(LDFLAGS) \
		$^ \
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

$(test): $(test_obj) $(lib)
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
# scalapack_compat library
scalapack_compat = lib/libslate_scalapack_compat.so

scalapack_compat_src = \
                     scalapack_compat/scalapack_compat_gemm.cc \
                     scalapack_compat/scalapack_compat_syrk.cc \
                     scalapack_compat/scalapack_compat_symm.cc \
                     scalapack_compat/scalapack_compat_trsm.cc \
                     scalapack_compat/scalapack_compat_syr2k.cc \
                     scalapack_compat/scalapack_compat_trmm.cc 

scalapack_compat_obj = $(scalapack_compat_src:.cc=.o) 

SCALAPACK_COMPAT_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
SCALAPACK_COMPAT_LIB     += -lslate $(scalapack)

scalapack_compat: lib $(scalapack_compat)

scalapack_compat/clean:
	rm -f $(scalapack_compat) $(scalapack_compat_obj)

$(scalapack_compat): $(scalapack_compat_obj) $(lib)
	$(CXX) $(SCALAPACK_COMPAT_LDFLAGS) $(LDFLAGS) $^ \
		$(SCALAPACK_COMPAT_LIB) $(LIB) -shared $(install_name) -o $@

#-------------------------------------------------------------------------------
# general rules
clean: test/clean unit_test/clean
	rm -f $(lib_a) $(lib_so) $(lib_obj)
	rm -f trace_*.svg

distclean: clean
	rm -f $(dep)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

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
	@echo "LDFLAGS       = $(LDFLAGS)"
	@echo "LIB           = $(LIB)"
	@echo
	@echo "TEST_LDFLAGS  = $(TEST_LDFLAGS)"
	@echo "TEST_LIB      = $(TEST_LIB)"
	@echo
	@echo "UNIT_LDFLAGS  = $(UNIT_LDFLAGS)"
	@echo "UNIT_LIB      = $(UNIT_LIB)"
