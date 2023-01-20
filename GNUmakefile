# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# See INSTALL.md for documentation.
#
# Set only_unit=1 to avoid compiling most of the SLATE library,
# which isn't needed by most unit testers (except test_lq, test_qr).
# Useful to avoid expensive recompilation when debugging headers.
#
# Sort lists alphabetically and end with \ to avoid merge conflicts.
# The "# End." comment avoids the next line being appended accidentally.

-include make.inc

#-------------------------------------------------------------------------------
# Error for obsolete settings. Remove 2023-06.
ifneq ($(spectrum),)
    $(error ERROR: Variable `spectrum=$(spectrum)` is obsolete; use `mpi = spectrum`)
endif
ifneq ($(mkl),)
    $(error ERROR: Variable `mkl=$(mkl)` is obsolete; use `blas = mkl`)
endif
ifneq ($(essl),)
    $(error ERROR: Variable `essl=$(essl)` is obsolete; use `blas = essl`)
endif
ifneq ($(openblas),)
    $(error ERROR: Variable `openblas=$(openblas)` is obsolete; use `blas = openblas`)
endif
ifneq ($(mkl_threaded),)
    $(error ERROR: Variable `mkl_threaded=$(mkl_threaded)` is obsolete; use `blas_threaded = $(mkl_threaded)`)
endif
ifeq ($(mkl_intel),1)
    $(error ERROR: Variable `mkl_intel=$(mkl_intel)` is obsolete; use `blas_fortran = ifort`)
endif
ifeq ($(ilp64),1)
    $(error ERROR: Variable `ilp64=$(ilp64)` is obsolete; use `blas_int = int64`)
    blas_int ?= int64
endif
ifeq ($(cuda),1)
    $(error ERROR: Variable `cuda=$(cuda)` is obsolete; use `gpu_backend = cuda`)
endif
ifeq ($(hip),1)
    $(error ERROR: Variable `hip=$(hip)` is obsolete; use `gpu_backend = hip`)
endif

#-------------------------------------------------------------------------------
# Define functions.

# Get parent directory, stripping trailing /.
dir_strip = $(patsubst %/,%,$(dir $(1)))

#-------------------------------------------------------------------------------
# Set defaults
# Do all ?= before strip!
prefix          ?= /opt/slate

blas_int        ?= int
blas_threaded   ?= 0
openmp          ?= 1
c_api           ?= 0
fortran_api     ?= 0

NVCC            ?= nvcc
HIPCC           ?= hipcc
hipify          ?= hipify-perl
md5sum          ?= tools/md5sum.pl

gpu_backend     ?= auto

# Strip whitespace from variables, in case make.inc had trailing spaces.
mpi             := $(strip $(mpi))
blas            := $(strip $(blas))
blas_int        := $(strip $(blas_int))
blas_threaded   := $(strip $(blas_threaded))
blas_fortran    := $(strip $(blas_fortran))
mkl_blacs       := $(strip $(mkl_blacs))
openmp          := $(strip $(openmp))
static          := $(strip $(static))
gpu_backend     := $(strip $(gpu_backend))
cuda_arch       := $(strip $(cuda_arch))
hip_arch        := $(strip $(hip_arch))
prefix          := $(strip $(prefix))
c_api           := $(strip $(c_api))
fortran_api     := $(strip $(fortran_api))

#-------------------------------------------------------------------------------
# Export variables to sub-make for testsweeper, BLAS++, LAPACK++.
export CXX blas blas_int blas_threaded openmp static gpu_backend

CXXFLAGS   += -O3 -std=c++17 -Wall -Wshadow -pedantic -MMD
NVCCFLAGS  += -O3 -std=c++11 --compiler-options '-Wall -Wno-unused-function'
HIPCCFLAGS += -std=c++11 -DTCE_HIP -fno-gpu-rdc

force: ;

# Auto-detect CUDA, HIP.
ifneq ($(filter-out auto cuda hip none, $(gpu_backend)),)
    $(error ERROR: gpu_backend = $(gpu_backend) is unknown)
endif

cuda = 0
ifneq ($(filter auto cuda, $(gpu_backend)),)
    NVCC_which := $(shell which $(NVCC) 2>/dev/null)
    ifneq ($(NVCC_which),)
        cuda = 1
        CUDA_DIR ?= $(call dir_strip, $(call dir_strip, $(NVCC_which)))
    else ifeq ($(gpu_backend),cuda)
        $(error ERROR: gpu_backend = $(gpu_backend), but NVCC = $(NVCC) not found)
    endif
endif

hip = 0
ifneq ($(cuda),1)
    ifneq ($(filter auto hip, $(gpu_backend)),)
        HIPCC_which = $(shell which $(HIPCC) 2>/dev/null)
        ifneq ($(HIPCC_which),)
            hip = 1
            ROCM_DIR ?= $(call dir_strip, $(call dir_strip, $(HIPCC_which)))
        else ifeq ($(gpu_backend),hip)
            $(error ERROR: gpu_backend = $(gpu_backend), but HIPCC = $(HIPCC) not found)
        endif
    endif
endif

# Default LD=ld won't work; use CXX. Can override in make.inc or environment.
ifeq ($(origin LD),default)
    LD = $(CXX)
endif

# auto-detect OS
# $OSTYPE may not be exported from the shell, so echo it
ostype := $(shell echo $${OSTYPE})
ifneq ($(findstring darwin, $(ostype)),)
    # MacOS is darwin
    macos = 1
endif

# Check if Fortran compiler exists.
# Note that 'make' sets $(FC) to f77 by default.
have_fortran := $(shell which $(FC) 2>/dev/null)
ifeq ($(have_fortran),)
    fortran_api = 0
endif

# Fortran API depends on C API.
ifneq ($(c_api),1)
    fortran_api = 0
endif

#-------------------------------------------------------------------------------
# if shared
ifneq ($(static),1)
    CXXFLAGS   += -fPIC
    LDFLAGS    += -fPIC
    FCFLAGS    += -fPIC
    NVCCFLAGS  += --compiler-options '-fPIC'
    HIPCCFLAGS += -fPIC
    lib_ext = so
else
    lib_ext = a
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
ifneq ($(filter mpi%,$(CXX)),)
    # CXX = mpicxx, mpic++, ...
    # Generic MPI via compiler wrapper. No flags to set.
else ifeq ($(mpi),cray)
    # Cray MPI via compiler wrapper. No flags to set.
else ifeq ($(mpi),1)
    # Generic MPI.
    LIBS  += -lmpi
else ifeq ($(mpi),spectrum)
    # IBM Spectrum MPI
    LIBS  += -lmpi_ibm
else
    FLAGS += -DSLATE_NO_MPI
    libslate_src += src/stubs/mpi_stubs.cc
    fortran_api = 0
endif

#-------------------------------------------------------------------------------
# BLAS and LAPACK
# todo: really should get these libraries from BLAS++ and LAPACK++.
# If using shared libraries, and Fortran files that directly call BLAS are
# removed, BLAS++ would pull in the BLAS library for us.

ifeq ($(blas),mkl)
    # Intel MKL
    # Auto-detect whether to use Intel or GNU conventions.
    # Won't detect if CXX = mpicxx.
    ifeq ($(CXX),icpc)
        blas_fortran = ifort
    endif
    ifeq ($(macos),1)
        # MKL on MacOS (version 20180001) has only Intel Fortran version
        blas_fortran = ifort
    endif
    ifeq ($(blas_fortran),ifort)
        # use Intel Fortran conventions
        ifeq ($(blas_int),int64)
            LIBS += -lmkl_intel_ilp64
        else
            LIBS += -lmkl_intel_lp64
        endif

        # if threaded, use Intel OpenMP (iomp5)
        ifeq ($(blas_threaded),1)
            LIBS += -lmkl_intel_thread
        else
            LIBS += -lmkl_sequential
        endif
    else
        # use GNU Fortran conventions
        ifeq ($(blas_int),int64)
            LIBS += -lmkl_gf_ilp64
        else
            LIBS += -lmkl_gf_lp64
        endif

        # if threaded, use GNU OpenMP (gomp)
        ifeq ($(blas_threaded),1)
            LIBS += -lmkl_gnu_thread
        else
            LIBS += -lmkl_sequential
        endif
    endif

    LIBS += -lmkl_core -lpthread -lm -ldl

    # MKL on MacOS doesn't include ScaLAPACK; use default.
    # For others, link with appropriate version of ScaLAPACK and BLACS.
    ifneq ($(macos),1)
        ifeq ($(mkl_blacs),openmpi)
            ifeq ($(blas_int),int64)
                SCALAPACK_LIBRARIES ?= -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64
            else
                SCALAPACK_LIBRARIES ?= -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64
            endif
        else
            ifeq ($(blas_int),int64)
                SCALAPACK_LIBRARIES ?= -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64
            else
                SCALAPACK_LIBRARIES ?= -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
            endif
        endif
    endif
else ifeq ($(blas),essl)
    # IBM ESSL
    # todo threaded, int64
    # hmm... likely LAPACK won't be int64 even if ESSL is.
    LIBS += -lessl -llapack
else ifeq ($(blas),openblas)
    # OpenBLAS
    LIBS += -lopenblas
else ifeq ($(blas),libsci)
    # Cray LibSci
    # no LIBS to add
    SCALAPACK_LIBRARIES ?=
else
    $(error ERROR: unknown `blas=$(blas)`. Set blas to one of mkl, essl, openbblas, libsci.)
endif

# If not set by user or above, set default.
SCALAPACK_LIBRARIES ?= -lscalapack

#-------------------------------------------------------------------------------
# if CUDA
ifeq ($(cuda),1)
    # Generate flags for which CUDA architectures to build.
    # cuda_arch_ is a local copy to modify.
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
    ifneq ($(findstring ampere, $(cuda_arch_)),)
        cuda_arch_ += sm_80
    endif
    ifneq ($(findstring hopper, $(cuda_arch_)),)
        cuda_arch_ += sm_90
    endif

    # Extract CUDA sm architectures.
    sms = $(sort $(patsubst sm_%, %, $(filter sm_%, $(cuda_arch_))))

    # Generate nvcc gencode options for all sm_XY in cuda_arch_.
    # code=sm_XX is binary, code=compute_XX is PTX
    nv_sm      = $(foreach sm, $(sms),-gencode arch=compute_$(sm),code=sm_$(sm))
    nv_compute = $(foreach sm, $(sms),-gencode arch=compute_$(sm),code=compute_$(sm))

    ifeq ($(sms),)
        # Error if cuda_arch is not empty and sms is empty.
        ifneq ($(cuda_arch),)
            $(error ERROR: unknown `cuda_arch=$(cuda_arch)`. Set cuda_arch to one or more of kepler, maxwell, pascal, volta, turing, ampere, hopper, or valid sm_XY from nvcc -h)
        endif
    else
        # Get last option (last 2 words) of nv_compute.
        nwords := $(words $(nv_compute))
        nwords_1 := $(shell expr $(nwords) - 1)
        nv_compute_last := $(wordlist $(nwords_1), $(nwords), $(nv_compute))
    endif

    # Use all sm_XX (binary), and the last compute_XX (PTX) for forward compatibility.
    NVCCFLAGS += $(nv_sm) $(nv_compute_last)
    LIBS += -lcusolver -lcublas -lcudart
endif

#-------------------------------------------------------------------------------
# if HIP
ifeq ($(hip),1)
    # Generate flags for which HIP architectures to build.
    # hip_arch_ is a local copy to modify.
    hip_arch_ = $(hip_arch)
    ifneq ($(findstring mi25, $(hip_arch_)),)
        hip_arch_ += gfx900
    endif
    ifneq ($(findstring mi50, $(hip_arch_)),)
        hip_arch_ += gfx906
    endif
    ifneq ($(findstring mi100, $(hip_arch_)),)
        hip_arch_ += gfx908
    endif
    ifneq ($(findstring mi200, $(hip_arch_)),)
        hip_arch_ += gfx90a
    endif

    # Extract AMD gfx architectures.
    gfx = $(sort $(filter gfx%, $(hip_arch_)))
    ifeq ($(gfx),)
        # Error if hip_arch is not empty and gfx is empty.
        ifneq ($(hip_arch),)
            $(error ERROR: unknown `hip_arch=$(hip_arch)`. Set hip_arch to one or more of mi25, mi50, mi100, or valid gfxXYZ. See https://llvm.org/docs/AMDGPUUsage.html)
        endif
    endif

    # Generate hipcc target options for all gfx in hip_arch_.
    amdgpu_targets = $(foreach arch, $(gfx),--amdgpu-target=$(arch))
    HIPCCFLAGS += $(amdgpu_targets)
    FLAGS += -D__HIP_PLATFORM_HCC__
    LIBS += -L$(ROCM_DIR)/lib -lrocsolver -lrocblas -lamdhip64

    # ROCm 4.0 has errors in its headers that produce excessive warnings.
    CXXFLAGS := $(filter-out -pedantic, $(CXXFLAGS))
    CXXFLAGS += -Wno-unused-result
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
libslate_src += \
        src/auxiliary/Debug.cc \
        src/auxiliary/Trace.cc \
        src/core/Memory.cc \
        src/core/types.cc \
        src/version.cc \
        # End. Add alphabetically.

# internal
libslate_src += \
        src/internal/internal_comm.cc \
        src/internal/internal_transpose.cc \
        src/internal/internal_util.cc \
        # End. Add alphabetically.

# Most unit testers don't need the whole library, only the above subset.
ifneq ($(only_unit),1)
    libslate_src += \
        src/device/dev_gescale_row_col.cc \
        src/internal/internal_copyhb2st.cc \
        src/internal/internal_copytb2bd.cc \
        src/internal/internal_gbnorm.cc \
        src/internal/internal_geadd.cc \
        src/internal/internal_gebr.cc \
        src/internal/internal_gecopy.cc \
        src/internal/internal_gemm.cc \
        src/internal/internal_gemmA.cc \
        src/internal/internal_genorm.cc \
        src/internal/internal_geqrf.cc \
        src/internal/internal_he2hb_gemm.cc \
        src/internal/internal_he2hb_hemm.cc \
        src/internal/internal_he2hb_her2k_offdiag_ranks.cc \
        src/internal/internal_he2hb_trmm.cc \
        src/internal/internal_gescale.cc \
        src/internal/internal_gescale_row_col.cc \
        src/internal/internal_geset.cc \
        src/internal/internal_getrf.cc \
        src/internal/internal_getrf_nopiv.cc \
        src/internal/internal_getrf_tntpiv.cc \
        src/internal/internal_hbnorm.cc \
        src/internal/internal_hebr.cc \
        src/internal/internal_hegst.cc \
        src/internal/internal_hemm.cc \
        src/internal/internal_hemmA.cc \
        src/internal/internal_henorm.cc \
        src/internal/internal_her2k.cc \
        src/internal/internal_herk.cc \
        src/internal/internal_hettmqr.cc \
        src/internal/internal_norm1est.cc \
        src/internal/internal_potrf.cc \
        src/internal/internal_swap.cc \
        src/internal/internal_symm.cc \
        src/internal/internal_synorm.cc \
        src/internal/internal_syr2k.cc \
        src/internal/internal_syrk.cc \
        src/internal/internal_trmm.cc \
        src/internal/internal_trnorm.cc \
        src/internal/internal_trsm.cc \
        src/internal/internal_trsmA.cc \
        src/internal/internal_trtri.cc \
        src/internal/internal_trtrm.cc \
        src/internal/internal_ttlqt.cc \
        src/internal/internal_ttmlq.cc \
        src/internal/internal_ttmqr.cc \
        src/internal/internal_ttqrt.cc \
        src/internal/internal_tzadd.cc \
        src/internal/internal_tzcopy.cc \
        src/internal/internal_tzscale.cc \
        src/internal/internal_tzset.cc \
        src/internal/internal_unmlq.cc \
        src/internal/internal_unmqr.cc \
        src/internal/internal_unmtr_hb2st.cc \
        # End. Add alphabetically.
endif

# device
cuda_src := \
        src/cuda/device_geadd.cu \
        src/cuda/device_gecopy.cu \
        src/cuda/device_genorm.cu \
        src/cuda/device_gescale.cu \
        src/cuda/device_gescale_row_col.cu \
        src/cuda/device_geset.cu \
        src/cuda/device_henorm.cu \
        src/cuda/device_synorm.cu \
        src/cuda/device_transpose.cu \
        src/cuda/device_trnorm.cu \
        src/cuda/device_tzadd.cu \
        src/cuda/device_tzcopy.cu \
        src/cuda/device_tzscale.cu \
        src/cuda/device_tzset.cu \
        # End. Add alphabetically.

cuda_hdr := \
        src/cuda/device_util.cuh

hip_src := $(patsubst src/cuda/%.cu,src/hip/%.hip.cc,$(cuda_src))
hip_hdr := $(patsubst src/cuda/%.cuh,src/hip/%.hip.hh,$(cuda_hdr))

ifeq ($(cuda),1)
    libslate_src += $(cuda_src)
endif

ifeq ($(hip),1)
    libslate_src += ${hip_src}
endif

# driver
ifneq ($(only_unit),1)
    libslate_src += \
        src/add.cc \
        src/bdsqr.cc \
        src/cholqr.cc \
        src/colNorms.cc \
        src/copy.cc \
        src/gbmm.cc \
        src/gbsv.cc \
        src/gbtrf.cc \
        src/gbtrs.cc \
        src/ge2tb.cc \
        src/gecondest.cc \
        src/gelqf.cc \
        src/gels.cc \
        src/gels_cholqr.cc \
        src/gels_qr.cc \
        src/gemm.cc \
        src/gemmA.cc \
        src/gemmC.cc \
        src/geqrf.cc \
        src/gesv.cc \
        src/gesvMixed.cc \
        src/gesv_nopiv.cc \
        src/gesvd.cc \
        src/getrf.cc \
        src/getrf_nopiv.cc \
        src/getrf_tntpiv.cc \
        src/getri.cc \
        src/getriOOP.cc \
        src/getrs.cc \
        src/getrs_nopiv.cc \
        src/hb2st.cc \
        src/hbmm.cc \
        src/he2hb.cc \
        src/heev.cc \
        src/hegst.cc \
        src/hegv.cc \
        src/hemm.cc \
        src/hemmA.cc \
        src/hemmC.cc \
        src/her2k.cc \
        src/herk.cc \
        src/hesv.cc \
        src/hetrf.cc \
        src/hetrs.cc \
        src/norm.cc \
        src/pbsv.cc \
        src/pbtrf.cc \
        src/pbtrs.cc \
        src/posv.cc \
        src/posvMixed.cc \
        src/potrf.cc \
        src/potri.cc \
        src/potrs.cc \
        src/print.cc \
        src/scale.cc \
        src/scale_row_col.cc \
        src/set.cc \
        src/steqr2.cc \
        src/sterf.cc \
        src/symm.cc \
        src/syr2k.cc \
        src/syrk.cc \
        src/tb2bd.cc \
        src/tbsm.cc \
        src/tbsmPivots.cc \
        src/trcondest.cc \
        src/trmm.cc \
        src/trsm.cc \
        src/trsmA.cc \
        src/trsmB.cc \
        src/trtri.cc \
        src/trtrm.cc \
        src/unmlq.cc \
        src/unmqr.cc \
        src/unmtr_hb2st.cc \
        src/unmtr_he2hb.cc \
        src/work/work_trmm.cc \
        src/work/work_trsm.cc \
        src/work/work_trsmA.cc \
        # End. Add alphabetically.
endif

ifneq ($(have_fortran),)
    libslate_src += \
        src/ssteqr2.f \
        src/dsteqr2.f \
        src/csteqr2.f \
        src/zsteqr2.f \
        # End. Add alphabetically, by base name after precision.
else
    $(error ERROR: Fortran compiler FC='$(FC)' not found. Set FC to a Fortran compiler (mpif90, gfortran, ifort, xlf, ftn, ...). We hope to eventually remove this requirement.)
endif

# C API
ifeq ($(c_api),1)
    libslate_src += \
        src/c_api/matrix.cc \
        src/c_api/util.cc \
        src/c_api/wrappers.cc \
        src/c_api/wrappers_precisions.cc \
        # End. Add alphabetically.
endif

# Fortran module
ifeq ($(fortran_api),1)
    libslate_src += \
        src/fortran/slate_module.f90 \
        # End. Add alphabetically.
endif

# main tester
tester_src += \
        test/matrix_generator.cc \
        test/matrix_params.cc \
        test/test.cc \
        test/test_add.cc \
        test/test_bdsqr.cc \
        test/test_copy.cc \
        test/test_gbmm.cc \
        test/test_gbnorm.cc \
        test/test_gbsv.cc \
        test/test_ge2tb.cc \
        test/test_gecondest.cc \
        test/test_gelqf.cc \
        test/test_gels.cc \
        test/test_gemm.cc \
        test/test_genorm.cc \
        test/test_geqrf.cc \
        test/test_gesv.cc \
        test/test_gesvd.cc \
        test/test_getri.cc \
        test/test_hb2st.cc \
        test/test_hbmm.cc \
        test/test_hbnorm.cc \
        test/test_he2hb.cc \
        test/test_heev.cc \
        test/test_hegst.cc \
        test/test_hegv.cc \
        test/test_hemm.cc \
        test/test_henorm.cc \
        test/test_her2k.cc \
        test/test_herk.cc \
        test/test_hesv.cc \
        test/test_pbsv.cc \
        test/test_posv.cc \
        test/test_potri.cc \
        test/test_scale.cc \
        test/test_scale_row_col.cc \
        test/test_set.cc \
        test/test_steqr2.cc \
        test/test_sterf.cc \
        test/test_symm.cc \
        test/test_synorm.cc \
        test/test_syr2k.cc \
        test/test_syrk.cc \
        test/test_tb2bd.cc \
        test/test_tbsm.cc \
        test/test_trcondest.cc \
        test/test_trmm.cc \
        test/test_trnorm.cc \
        test/test_trsm.cc \
        test/test_trtri.cc \
        test/test_unmqr.cc \
        test/test_unmtr_hb2st.cc \
        test/test_unmtr_he2hb.cc \
        # End. Add alphabetically.

# Compile fixes for ScaLAPACK routines if Fortran compiler $(FC) exists.
ifneq (${SCALAPACK_LIBRARIES},none)
ifneq ($(have_fortran),)
    tester_src += \
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
        # End. Add alphabetically, by base name after precision.
endif
endif

# unit testers
unit_src = \
    unit_test/test_BandMatrix.cc \
    unit_test/test_HermitianMatrix.cc \
    unit_test/test_LockGuard.cc \
    unit_test/test_Matrix.cc \
    unit_test/test_Memory.cc \
    unit_test/test_OmpSetMaxActiveLevels.cc \
    unit_test/test_SymmetricMatrix.cc \
    unit_test/test_Tile.cc \
    unit_test/test_Tile_kernels.cc \
    unit_test/test_TrapezoidMatrix.cc \
    unit_test/test_TriangularBandMatrix.cc \
    unit_test/test_TriangularMatrix.cc \
    unit_test/test_geadd.cc \
    unit_test/test_gecopy.cc \
    unit_test/test_gescale.cc \
    unit_test/test_geset.cc \
    unit_test/test_internal_blas.cc \
    unit_test/test_norm.cc \
    # End. Add alphabetically.

ifneq ($(only_unit),1)
unit_src += \
    unit_test/test_lq.cc \
    unit_test/test_qr.cc \
    # End. Add alphabetically.
endif

# unit test framework
unit_test_obj = \
        unit_test/unit_test.o

libslate_obj = $(addsuffix .o, $(basename $(libslate_src)))
tester_obj   = $(addsuffix .o, $(basename $(tester_src)))
unit_obj     = $(addsuffix .o, $(basename $(unit_src)))
dep          = $(addsuffix .d, $(basename $(libslate_src) $(tester_src) \
                                          $(unit_src) $(unit_test_obj)))

tester    = test/tester
unit_test = $(basename $(unit_src))

# For `tester --debug`, lldb may need test.o compiled with -O0 (after -O3)
# to see variable `i`.
test/test.o: CXXFLAGS += -O0

#-------------------------------------------------------------------------------
# Get Mercurial id, and make version.o depend on it via .id file.

ifneq ($(wildcard .git),)
    id := $(shell git rev-parse --short HEAD)
    src/version.o: CXXFLAGS += -DSLATE_ID='"$(id)"'
endif

last_id := $(shell [ -e .id ] && cat .id || echo 'NA')
ifneq ($(id),$(last_id))
    .id: force
endif

.id:
	echo $(id) > .id

src/version.o: .id

#-------------------------------------------------------------------------------
# SLATE specific flags and libraries
# FLAGS accumulates definitions, include dirs, etc. for both CXX and NVCC.
# FLAGS += -I.
FLAGS += -I./blaspp/include
FLAGS += -I./lapackpp/include
FLAGS += -I./include
FLAGS += -I./src

CXXFLAGS   += $(FLAGS)
NVCCFLAGS  += $(FLAGS)
HIPCCFLAGS += $(FLAGS)

# libraries to create libslate.so
LDFLAGS  += -L./blaspp/lib
LDFLAGS  += -L./lapackpp/lib
LIBS     := -lblaspp -llapackpp $(LIBS)

# additional flags and libraries for testers
$(tester_obj):    CXXFLAGS += -I./testsweeper
$(unit_obj):      CXXFLAGS += -I./testsweeper
$(unit_test_obj): CXXFLAGS += -I./testsweeper

TEST_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
TEST_LDFLAGS += -L./testsweeper -Wl,-rpath,$(abspath ./testsweeper)
TEST_LDFLAGS += -Wl,-rpath,$(abspath ./blaspp/lib)
TEST_LDFLAGS += -Wl,-rpath,$(abspath ./lapackpp/lib)
TEST_LIBS    += -lslate -ltestsweeper
ifneq (${SCALAPACK_LIBRARIES},none)
    TEST_LIBS += ${SCALAPACK_LIBRARIES}
    CXXFLAGS  += -DSLATE_HAVE_SCALAPACK
endif

UNIT_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
UNIT_LDFLAGS += -L./testsweeper -Wl,-rpath,$(abspath ./testsweeper)
UNIT_LDFLAGS += -Wl,-rpath,$(abspath ./blaspp/lib)
UNIT_LDFLAGS += -Wl,-rpath,$(abspath ./lapackpp/lib)
UNIT_LIBS    += -lslate -ltestsweeper

#-------------------------------------------------------------------------------
# Rules
.DELETE_ON_ERROR:
.SUFFIXES:
.PHONY: all docs hooks lib test tester unit_test clean distclean testsweeper blaspp lapackpp
.DEFAULT_GOAL := all

all: lib unit_test hooks
install: lib

ifneq ($(only_unit),1)
    all: tester lapack_api
    install: lapack_api
    ifneq (${SCALAPACK_LIBRARIES},none)
        all: scalapack_api
        install: scalapack_api
    endif
endif

install:
	cd blaspp   && $(MAKE) install prefix=${prefix}
	@echo
	cd lapackpp && $(MAKE) install prefix=${prefix}
	@echo
	mkdir -p $(DESTDIR)$(prefix)/include/slate/internal
	mkdir -p $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)
	cp include/slate/*.hh          $(DESTDIR)$(prefix)/include/slate
	cp include/slate/internal/*.hh $(DESTDIR)$(prefix)/include/slate/internal
	cp lib/lib*                    $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)
	if [ $(c_api) -eq 1 ]; then \
		mkdir -p $(DESTDIR)$(prefix)/include/slate/c_api; \
		cp include/slate/c_api/*.h  $(DESTDIR)$(prefix)/include/slate/c_api; \
		cp include/slate/c_api/*.hh $(DESTDIR)$(prefix)/include/slate/c_api; \
	fi
	if [ ${fortran_api} -eq 1 ]; then \
		cp slate.mod                $(DESTDIR)$(prefix)/include/; \
	fi

uninstall:
	cd blaspp   && $(MAKE) uninstall prefix=${prefix}
	@echo
	cd lapackpp && $(MAKE) uninstall prefix=${prefix}
	@echo
	$(RM) -r $(DESTDIR)$(prefix)/include/slate
	$(RM)    $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)/libslate*

docs:
	doxygen docs/doxygen/doxyfile.conf
	@echo "------------------------------------------------------------"
	@echo "Errors:"
	perl -pe 's@^/.*?slate/@@' docs/doxygen/errors.txt

#-------------------------------------------------------------------------------
# C API
ifeq ($(c_api),1)
    include/slate/c_api/wrappers.h: src/c_api/wrappers.cc
		python tools/c_api/generate_wrappers.py $< $@ \
			src/c_api/wrappers_precisions.cc

    include/slate/c_api/matrix.h: include/slate/Tile.hh
		python tools/c_api/generate_matrix.py $< $@ \
			src/c_api/matrix.cc

    include/slate/c_api/util.hh: include/slate/c_api/types.h
		python tools/c_api/generate_util.py $< $@ \
			src/c_api/util.cc

    src/c_api/wrappers_precisions.cc: include/slate/c_api/wrappers.h
    src/c_api/matrix.cc: include/slate/c_api/matrix.h
    src/c_api/util.cc: include/slate/c_api/util.hh
    src/c_api/wrappers.o: include/slate/c_api/wrappers.h

    generate: include/slate/c_api/wrappers.h
    generate: include/slate/c_api/matrix.h
    generate: include/slate/c_api/util.hh
endif

#-------------------------------------------------------------------------------
# Fortran module
ifeq ($(fortran_api),1)
    src/fortran/slate_module.f90: include/slate/c_api/wrappers.h \
                                  include/slate/c_api/types.h \
                                  include/slate/c_api/matrix.h
		python tools/fortran/generate_fortran_module.py $^ --output $@

    generate: src/fortran/slate_module.f90
endif

#-------------------------------------------------------------------------------
# testsweeper library
testsweeper_src = $(wildcard testsweeper/*.hh testsweeper/*.cc)

testsweeper = testsweeper/libtestsweeper.$(lib_ext)

$(testsweeper): $(testsweeper_src)
	cd testsweeper && $(MAKE) lib

testsweeper: $(testsweeper)

#-------------------------------------------------------------------------------
# BLAS++ library
libblaspp_src = $(wildcard blaspp/include/*.h \
                           blaspp/include/*.hh \
                           blaspp/src/*.cc)

libblaspp = blaspp/lib/libblaspp.$(lib_ext)

# dependency on testsweeper serializes compiles
$(libblaspp): $(libblaspp_src) | $(testsweeper)
	cd blaspp && $(MAKE) lib

blaspp: $(libblaspp)

#-------------------------------------------------------------------------------
# LAPACK++ library
liblapackpp_src = $(wildcard lapackpp/include/*.h \
                             lapackpp/include/*.hh \
                             lapackpp/src/*.cc)

liblapackpp = lapackpp/lib/liblapackpp.$(lib_ext)

# dependency on testsweeper, BLAS++ serializes compiles
$(liblapackpp): $(liblapackpp_src) | $(testsweeper) $(libblaspp)
	cd lapackpp && $(MAKE) lib

lapackpp: $(liblapackpp)

#-------------------------------------------------------------------------------
# libslate library
libslate_a  = lib/libslate.a
libslate_so = lib/libslate.so
libslate    = lib/libslate.$(lib_ext)

$(libslate_a): $(libslate_obj)
	mkdir -p lib
	-rm $@
	ar cr $@ $(libslate_obj)
	ranlib $@

$(libslate_so): $(libslate_obj)
	mkdir -p lib
	$(LD) $(LDFLAGS) \
		$(libslate_obj) \
		$(LIBS) \
		-shared $(install_name) -o $@

src: $(libslate)

#-------------------------------------------------------------------------------
# headers
# precompile headers to verify self-sufficiency
headers     = $(wildcard include/slate/*.hh \
                         include/slate/internal/*.hh \
                         test/*.hh \
                         include/slate/c_api/*.h \
                         include/slate/c_api/*.hh)

headers_gch = $(addsuffix .gch, $(basename $(headers)))

headers: $(headers_gch)

# sub-directory rules
include: headers

include/clean:
	$(RM) include/*/*.gch test/*.gch

#-------------------------------------------------------------------------------
# main tester
# Note 'test' is sub-directory rule; 'tester' is CMake-compatible rule.
test: $(tester)
tester: $(tester)

test/clean:
	rm -f $(tester) $(tester_obj)

$(tester): $(tester_obj) $(libslate) $(testsweeper)
	$(LD) $(TEST_LDFLAGS) $(LDFLAGS) $(tester_obj) \
		$(TEST_LIBS) $(LIBS) \
		-o $@

test/check: check
unit_test/check: check

check: test unit_test
	cd test; python run_tests.py --quick gesv posv gels heev gesvd
	cd unit_test; python run_tests.py

#-------------------------------------------------------------------------------
# unit testers
unit_test: $(unit_test)

unit_test/clean:
	rm -f $(unit_test) $(unit_obj) $(unit_test_obj)

$(unit_test): %: %.o $(unit_test_obj) $(libslate)
	$(LD) $(UNIT_LDFLAGS) $(LDFLAGS) $< \
		$(unit_test_obj) $(UNIT_LIBS) $(LIBS)  \
		-o $@

#-------------------------------------------------------------------------------
# scalapack_api library
scalapack_api_a  = lib/libslate_scalapack_api.a
scalapack_api_so = lib/libslate_scalapack_api.so
scalapack_api    = lib/libslate_scalapack_api.$(lib_ext)

scalapack_api_src += \
        scalapack_api/scalapack_gels.cc \
        scalapack_api/scalapack_gemm.cc \
        scalapack_api/scalapack_gesv.cc \
        scalapack_api/scalapack_gesvMixed.cc \
        scalapack_api/scalapack_getrf.cc \
        scalapack_api/scalapack_getrs.cc \
        scalapack_api/scalapack_hemm.cc \
        scalapack_api/scalapack_her2k.cc \
        scalapack_api/scalapack_herk.cc \
        scalapack_api/scalapack_lange.cc \
        scalapack_api/scalapack_lanhe.cc \
        scalapack_api/scalapack_lansy.cc \
        scalapack_api/scalapack_lantr.cc \
        scalapack_api/scalapack_posv.cc \
        scalapack_api/scalapack_potrf.cc \
        scalapack_api/scalapack_potri.cc \
        scalapack_api/scalapack_potrs.cc \
        scalapack_api/scalapack_symm.cc \
        scalapack_api/scalapack_syr2k.cc \
        scalapack_api/scalapack_syrk.cc \
        scalapack_api/scalapack_trmm.cc \
        scalapack_api/scalapack_trsm.cc \
        # End. Add alphabetically.

scalapack_api_obj = $(addsuffix .o, $(basename $(scalapack_api_src)))

dep += $(addsuffix .d, $(basename $(scalapack_api_src)))

SCALAPACK_API_LDFLAGS += -L./lib
SCALAPACK_API_LIBS    += -lslate ${SCALAPACK_LIBRARIES}

scalapack_api: $(scalapack_api)

scalapack_api/clean:
	rm -f $(scalapack_api) $(scalapack_api_obj)

$(scalapack_api_a): $(scalapack_api_obj) $(libslate)
	-rm $@
	ar cr $@ $(scalapack_api_obj)
	ranlib $@

ifneq (${SCALAPACK_LIBRARIES},none)
    $(scalapack_api_so): $(scalapack_api_obj) $(libslate)
		$(LD) $(SCALAPACK_API_LDFLAGS) $(LDFLAGS) $(scalapack_api_obj) \
			$(SCALAPACK_API_LIBS) $(LIBS) -shared $(install_name) -o $@
else
    $(scalapack_api_so):
		@echo "Error: building $@ requires ScaLAPACK library, currently set to ${SCALAPACK_LIBRARIES}."
		false
endif

#-------------------------------------------------------------------------------
# lapack_api library
lapack_api_a  = lib/libslate_lapack_api.a
lapack_api_so = lib/libslate_lapack_api.so
lapack_api    = lib/libslate_lapack_api.$(lib_ext)

lapack_api_src += \
        lapack_api/lapack_gels.cc \
        lapack_api/lapack_gemm.cc \
        lapack_api/lapack_gesv.cc \
        lapack_api/lapack_gesvMixed.cc \
        lapack_api/lapack_getrf.cc \
        lapack_api/lapack_getri.cc \
        lapack_api/lapack_getrs.cc \
        lapack_api/lapack_hemm.cc \
        lapack_api/lapack_her2k.cc \
        lapack_api/lapack_herk.cc \
        lapack_api/lapack_lange.cc \
        lapack_api/lapack_lanhe.cc \
        lapack_api/lapack_lansy.cc \
        lapack_api/lapack_lantr.cc \
        lapack_api/lapack_posv.cc \
        lapack_api/lapack_potrf.cc \
        lapack_api/lapack_potri.cc \
        lapack_api/lapack_symm.cc \
        lapack_api/lapack_syr2k.cc \
        lapack_api/lapack_syrk.cc \
        lapack_api/lapack_trmm.cc \
        lapack_api/lapack_trsm.cc \
        # End. Add alphabetically.

lapack_api_obj = $(addsuffix .o, $(basename $(lapack_api_src)))

dep += $(addsuffix .d, $(basename $(lapack_api_src)))

LAPACK_API_LDFLAGS += -L./lib
LAPACK_API_LIBS    += -lslate

lapack_api: $(lapack_api)

lapack_api/clean:
	rm -f $(lapack_api) $(lapack_api_obj)

$(lapack_api_a): $(lapack_api_obj) $(libslate)
	-rm $@
	ar cr $@ $(lapack_api_obj)
	ranlib $@

$(lapack_api_so): $(lapack_api_obj) $(libslate)
	$(LD) $(LAPACK_API_LDFLAGS) $(LDFLAGS) $(lapack_api_obj) \
		$(LAPACK_API_LIBS) $(LIBS) -shared $(install_name) -o $@

#-------------------------------------------------------------------------------
# HIP sources converted from CUDA sources.

# if_md5_outdated applies the given build rule ($1) only if the md5 sums
# of the target's dependency ($<) doesn't match that stored in the
# target's dep file ($@.dep). If the target ($@) is already up-to-date
# based on md5 sums, its timestamp is updated so make will recognize it
# as up-to-date. Otherwise, the target is built and its dep file
# updated. Instead of depending on the src file, the target depends on
# the md5 file of the src file. This can be adapted for multiple dependencies.
# Example usage:
#
# %: %.c.md5
#     ${call if_md5_outdated,\
#            gcc -o $@ ${basename $<}}
#
define if_md5_outdated
    if [ -e $@ ] && diff $< $@.dep > /dev/null 2>&1; then \
        echo "  make: '$@' is up-to-date based on md5sum."; \
        echo "  touch $@"; \
                touch $@; \
    else \
        echo "  make: '$@' is out-of-date based on md5sum."; \
        echo "  ${strip $1}"; \
        $1; \
        cp $< $@.dep; \
    fi
endef

# From GNU manual: Commas ... cannot appear in an argument as written.
# The[y] can be put into the argument value by variable substitution.
comma := ,

# Convert CUDA => HIP code.
# Explicitly mention ${hip_src}, ${hip_hdr}, ${md5_files}
# to prevent them from being intermediate files,
# so they are _always_ generated and never removed.
# Perl updates includes and removes excess spaces that fail style hook.
${hip_src}: src/hip/%.hip.cc: src/cuda/%.cu.md5 | src/hip
	@${call if_md5_outdated, \
	        ${hipify} ${basename $<} > $@; \
	        perl -pi -e 's/\.cuh/.hip.hh/g; s/ +(${comma}|;|$$)/$$1/g;' $@}

${hip_hdr}: src/hip/%.hip.hh: src/cuda/%.cuh.md5 | src/hip
	@${call if_md5_outdated, \
	        ${hipify} ${basename $<} > $@; \
	        perl -pi -e 's/\.cuh/.hip.hh/g; s/ +(${comma}|;|$$)/$$1/g;' $@}

hipify: ${hip_src} ${hip_hdr}

md5_files := ${addsuffix .md5, ${cuda_src} ${cuda_hdr}}

${md5_files}: %.md5: %
	${md5sum} $< > $@

src/hip:
	mkdir -p $@

#-------------------------------------------------------------------------------
# general rules

lib: $(libslate)

clean: test/clean unit_test/clean scalapack_api/clean lapack_api/clean include/clean
	rm -f $(libslate_a) $(libslate_so) $(libslate_obj) $(dep)
	rm -f trace_*.svg

distclean: clean
	rm -f src/c_api/matrix.cc
	rm -f src/c_api/wrappers_precisions.cc
	rm -f src/c_api/util.cc
	rm -f include/slate/c_api/wrappers.h
	rm -f include/slate/c_api/matrix.h
	rm -f include/slate/c_api/util.hh
	rm -f src/fortran/slate_module.f90
	rm -f ${md5_files}
	cd testsweeper && $(MAKE) distclean
	cd blaspp      && $(MAKE) distclean
	cd lapackpp    && $(MAKE) distclean

# Install git hooks
hooks = .git/hooks/pre-commit .git/hooks/pre-push

hooks: ${hooks}

.git/hooks/%: tools/hooks/%
	@if [ -e .git/hooks ]; then \
		echo cp $< $@ ; \
		cp $< $@ ; \
	fi

%.hip.o: %.hip.cc | $(hip_hdr)
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.f
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.f90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# preprocess source
# test/%.i depend on testsweeper; for simplicity just add it here.
%.i: %.cc
	$(CXX) $(CXXFLAGS) -I./testsweeper -E $< -o $@

# precompile header to check for errors
# test/%.gch depend on testsweeper; for simplicity just add it here.
%.gch: %.hh
	$(CXX) $(CXXFLAGS) -I./testsweeper -c $< -o $@

-include $(dep)

#-------------------------------------------------------------------------------
# Extra dependencies to force TestSweeper, BLAS++, LAPACK++ to be compiled before SLATE.

$(libslate_obj):      | $(libblaspp) $(liblapackpp)
$(tester_obj):        | $(libblaspp) $(liblapackpp)
$(unit_test_obj):     | $(libblaspp) $(liblapackpp)
$(unit_obj):          | $(libblaspp) $(liblapackpp)
$(lapack_api_obj):    | $(libblaspp) $(liblapackpp)
$(scalapack_api_obj): | $(libblaspp) $(liblapackpp)

#-------------------------------------------------------------------------------
# debugging
echo:
	@echo "---------- Options"
	@echo "mpi           = '$(mpi)'"
	@echo "blas          = '$(blas)'"
	@echo "blas_int      = '$(blas_int)'"
	@echo "blas_threaded = '$(blas_threaded)'"
	@echo "blas_fortran  = '$(blas_fortran)'"
	@echo "mkl_blacs     = '$(mkl_blacs)'"
	@echo "openmp        = '$(openmp)'"
	@echo "static        = '$(static)'"
	@echo "gpu_backend   = '${gpu_backend}'"
	@echo "prefix        = '${prefix}'"
	@echo "c_api         = '${c_api}'"
	@echo "fortran_api   = '${fortran_api}'"
	@echo "SCALAPACK_LIBRARIES = '${SCALAPACK_LIBRARIES}'"
	@echo
	@echo "---------- Internal variables"
	@echo "ostype        = '$(ostype)'"
	@echo "macos         = '$(macos)'"
	@echo "id            = '$(id)'"
	@echo "last_id       = '$(last_id)'"
	@echo
	@echo "---------- Dependencies"
	@echo "libblaspp     = $(libblaspp)"
	@echo "liblapackpp   = $(liblapackpp)"
	@echo "testsweeper   = $(testsweeper)"
	@echo
	@echo "---------- Libraries"
	@echo "libslate_a    = $(libslate_a)"
	@echo "libslate_so   = $(libslate_so)"
	@echo "libslate      = $(libslate)"
	@echo
	@echo "---------- Files"
	@echo "libslate_src  = $(libslate_src)"
	@echo
	@echo "libslate_obj  = $(libslate_obj)"
	@echo
	@echo "tester_src    = $(tester_src)"
	@echo
	@echo "tester_obj    = $(tester_obj)"
	@echo
	@echo "tester        = $(tester)"
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
	@echo "---------- C++ compiler"
	@echo "CXX           = $(CXX)"
	@echo "CXXFLAGS      = $(CXXFLAGS)"
	@echo
	@echo "---------- CUDA options"
	@echo "cuda          = '$(cuda)'"
	@echo "cuda_arch     = $(cuda_arch)"
	@echo "cuda_arch_    = $(cuda_arch_)"
	@echo "NVCC          = $(NVCC)"
	@echo "NVCC_which    = $(NVCC_which)"
	@echo "CUDA_DIR      = $(CUDA_DIR)"
	@echo "NVCCFLAGS     = $(NVCCFLAGS)"
	@echo "sms           = $(sms)"
	@echo "nv_sm         = $(nv_sm)"
	@echo "nv_compute    = $(nv_compute)"
	@echo "nwords        = $(nwords)"
	@echo "nwords_1      = $(nwords_1)"
	@echo "nv_compute_last = $(nv_compute_last)"
	@echo
	@echo "---------- HIP options"
	@echo "hip           = '$(hip)'"
	@echo "hip_arch      = '$(hip_arch)'"
	@echo "hip_arch_     = '$(hip_arch_)'"
	@echo "gfx           = $(gfx)"
	@echo "HIPCC         = $(HIPCC)"
	@echo "HIPCC_which   = $(HIPCC_which)"
	@echo "ROCM_DIR      = $(ROCM_DIR)"
	@echo "HIPCCFLAGS    = $(HIPCCFLAGS)"
	@echo "amdgpu_targets = $(amdgpu_targets)"
	@echo "hipify        = ${hipify}"
	@echo "cuda_src      = ${cuda_src}"
	@echo "cuda_hdr      = ${cuda_hdr}"
	@echo "hip_src       = ${hip_src}"
	@echo "hip_hdr       = ${hip_hdr}"
	@echo "md5_files     = $(md5_files)"
	@echo
	@echo "---------- Fortran compiler"
	@echo "FC            = $(FC)"
	@echo "FCFLAGS       = $(FCFLAGS)"
	@echo "have_fortran  = $(have_fortran)"
	@echo
	@echo "---------- Link flags"
	@echo "LD            = $(LD)"
	@echo "LDFLAGS       = $(LDFLAGS)"
	@echo "LIBS          = $(LIBS)"
	@echo
	@echo "TEST_LDFLAGS  = $(TEST_LDFLAGS)"
	@echo "TEST_LIBS     = $(TEST_LIBS)"
	@echo
	@echo "UNIT_LDFLAGS  = $(UNIT_LDFLAGS)"
	@echo "UNIT_LIBS     = $(UNIT_LIBS)"
