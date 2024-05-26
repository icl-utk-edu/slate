# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
# Define functions.

# Get parent directory, stripping trailing /.
dir_strip = ${patsubst %/,%,${dir ${1}}}

#-------------------------------------------------------------------------------
# Set defaults
# Do all ?= before strip!
prefix          ?= /opt/slate

blas_int        ?= int
openmp          ?= 1
c_api           ?= 0
fortran_api     ?= 0

# Strip whitespace.
blas            := ${strip ${blas}}

# MKL doesn't oversubscribe within OpenMP tasks, so it's safe and
# desirable to use multi-threaded BLAS.
ifeq (${blas},mkl)
    blas_threaded ?= 1
else
    blas_threaded ?= 0
endif

NVCC            ?= nvcc
HIPCC           ?= hipcc
hipify          ?= hipify-perl
md5sum          ?= tools/md5sum.pl

gpu_backend     ?= auto

python          ?= python3

# Strip whitespace from variables, in case make.inc had trailing spaces.
mpi             := ${strip ${mpi}}
blas_int        := ${strip ${blas_int}}
blas_threaded   := ${strip ${blas_threaded}}
blas_fortran    := ${strip ${blas_fortran}}
mkl_blacs       := ${strip ${mkl_blacs}}
openmp          := ${strip ${openmp}}
static          := ${strip ${static}}
gpu_backend     := ${strip ${gpu_backend}}
cuda_arch       := ${strip ${cuda_arch}}
hip_arch        := ${strip ${hip_arch}}
prefix          := ${strip ${prefix}}
c_api           := ${strip ${c_api}}
fortran_api     := ${strip ${fortran_api}}

abs_prefix      := ${abspath ${prefix}}

#-------------------------------------------------------------------------------
# Export variables to sub-make for testsweeper, BLAS++, LAPACK++.
export CXX blas blas_int blas_threaded openmp static gpu_backend

CXXFLAGS   += -O3 -std=c++17 -Wall -Wshadow -pedantic -MMD
NVCCFLAGS  += -O3 -std=c++11 --compiler-options '-Wall -Wno-unused-function'
HIPCCFLAGS += -std=c++14 -DTCE_HIP -fno-gpu-rdc

force: ;

# Auto-detect CUDA, HIP, SYCL.
ifneq (,${filter-out auto cuda hip sycl none, ${gpu_backend}})
    ${error ERROR: gpu_backend = ${gpu_backend} is unknown}
endif

cuda = 0
ifneq (,${filter auto cuda, ${gpu_backend}})
    NVCC_which := ${shell which ${NVCC} 2>/dev/null}
    ifneq (${NVCC_which},)
        cuda = 1
        ifeq (${CUDA_PATH},)
            ifneq (${CUDA_HOME},)
                CUDA_PATH = ${CUDA_HOME}
            else
                CUDA_PATH = ${call dir_strip, ${call dir_strip, ${NVCC_which}}}
            endif
        endif
    else ifeq (${gpu_backend},cuda)
        ${error ERROR: gpu_backend = ${gpu_backend}, but NVCC = ${NVCC} not found}
    endif
endif

hip = 0
ifneq (${cuda},1)
    ifneq (,${filter auto hip, ${gpu_backend}})
        HIPCC_which = ${shell which ${HIPCC} 2>/dev/null}
        ifneq (${HIPCC_which},)
            hip = 1
            ROCM_PATH ?= ${call dir_strip, ${call dir_strip, ${HIPCC_which}}}
        else ifeq (${gpu_backend},hip)
            ${error ERROR: gpu_backend = ${gpu_backend}, but HIPCC = ${HIPCC} not found}
        endif
    endif
endif

omptarget = 0
ifneq (${cuda},1)
ifneq (${hip},1)
    ifeq (${gpu_backend},sycl)
        # enable the omptarget offload kernels in SLATE for oneMKL-SYCL devices
        ${info Note: enabling omp-target-offload kernels}
        omptarget = 1

        # -Wno-unused-command-line-argument avoids
        # icpx warning: -Wl,-rpath,...: 'linker' input unused.
        #
        # -Wno-c99-extensions avoids
        # icpx warning: '_Complex' is a C99 extension.
        #
        # -Wno-pass-failed avoids (on src/omptarget/device_transpose.cc)
        # icpx warning: loop not vectorized.
        #
        CXXFLAGS += -fsycl -fp-model=precise -Wno-unused-command-line-argument \
                    -Wno-c99-extensions -Wno-pass-failed
        LIBS += -lsycl
    endif
endif
endif

# Default LD=ld won't work; use CXX. Can override in make.inc or environment.
ifeq (${origin LD},default)
    LD = ${CXX}
endif

# Use abi-compliance-checker to compare the ABI (application binary
# interface) of 2 releases. Changing the ABI does not necessarily change
# the API (application programming interface). Rearranging a struct or
# changing a by-value argument from int64 to int doesn't change the
# API--no source code changes are required, just a recompile.
#
# if structs or routines are changed or removed:
#     bump major version and reset minor, revision = 0;
# else if structs or routines are added:
#     bump minor version and reset revision = 0;
# else (e.g., bug fixes):
#     bump revision
#
# soversion is major ABI version.
abi_version = 1.0.0
soversion = ${word 1, ${subst ., ,${abi_version}}}

#-------------------------------------------------------------------------------
ldflags_shared = -shared

# auto-detect OS
# $OSTYPE may not be exported from the shell, so echo it
ostype := ${shell echo $${OSTYPE}}
ifneq (,${findstring darwin, ${ostype}})
    # MacOS is darwin
    macos = 1
    # MacOS needs shared library's path set, and shared library version.
    ldflags_shared += -install_name @rpath/${notdir $@} \
                      -current_version ${abi_version} \
                      -compatibility_version ${soversion}
    so = dylib
    so2 = .dylib
    # on macOS, .dylib comes after version: libfoo.4.dylib
else
    # Linux needs shared library's soname.
    ldflags_shared += -Wl,-soname,${notdir ${lib_soname}}
    so = so
    so1 = .so
    # on Linux, .so comes before version: libfoo.so.4
endif

# Check if Fortran compiler exists.
# Note that 'make' sets ${FC} to f77 by default.
have_fortran := ${shell which ${FC} 2>/dev/null}
ifeq (${have_fortran},)
    fortran_api = 0
endif

# Fortran API depends on C API.
ifneq (${c_api},1)
    fortran_api = 0
endif

#-------------------------------------------------------------------------------
# if shared
ifneq (${static},1)
    CXXFLAGS   += -fPIC
    LDFLAGS    += -fPIC
    FCFLAGS    += -fPIC
    NVCCFLAGS  += --compiler-options '-fPIC'
    HIPCCFLAGS += -fPIC
    lib_ext = ${so}
else
    lib_ext = a
endif

#-------------------------------------------------------------------------------
# if OpenMP
ifeq (${openmp},1)
    ifeq (${gpu_backend},sycl)
        # Intel icpx options for OpenMP offload.
        CXXFLAGS += -fiopenmp -fopenmp-targets=spir64
        LDFLAGS  += -fiopenmp -fopenmp-targets=spir64
    else
        # Most other compilers recognize this.
        CXXFLAGS += -fopenmp
        LDFLAGS  += -fopenmp
    endif
else
    slate_src += src/stubs/openmp_stubs.cc
endif

#-------------------------------------------------------------------------------
# if MPI
ifneq (,${filter mpi%,${CXX}})
    # CXX = mpicxx, mpic++, ...
    # Generic MPI via compiler wrapper. No flags to set.
else ifeq (${mpi},cray)
    # Cray MPI via compiler wrapper. No flags to set.
else ifeq (${mpi},1)
    # Generic MPI.
    LIBS  += -lmpi
else ifeq (${mpi},spectrum)
    # IBM Spectrum MPI
    LIBS  += -lmpi_ibm
else
    FLAGS += -DSLATE_NO_MPI
    slate_src += src/stubs/mpi_stubs.cc
    fortran_api = 0
endif

#-------------------------------------------------------------------------------
# BLAS and LAPACK
# todo: really should get these libraries from BLAS++ and LAPACK++.
# If using shared libraries, and Fortran files that directly call BLAS are
# removed, BLAS++ would pull in the BLAS library for us.

ifeq (${blas},mkl)
    # Intel MKL
    # Auto-detect whether to use Intel or GNU conventions.
    # Won't detect if CXX = mpicxx.
    ifeq (${CXX},icpc)
        blas_fortran = ifort
    endif
    # BLAS is Intel MKL and SLATE is using the SYCL backend
    # Use ifort, threaded-blas and mkl_intel_thread
    ifeq (${gpu_backend},sycl)
        blas_fortran = ifort
        blas_threaded = 1
    endif
    ifeq (${macos},1)
        # MKL on MacOS (version 20180001) has only Intel Fortran version
        blas_fortran = ifort
    endif
    ifeq (${blas_fortran},ifort)
        # use Intel Fortran conventions
        ifeq (${blas_int},int64)
            LIBS += -lmkl_intel_ilp64
        else
            LIBS += -lmkl_intel_lp64
        endif

        # if threaded, use Intel OpenMP (iomp5)
        ifeq (${blas_threaded},1)
            LIBS += -lmkl_intel_thread
        else
            LIBS += -lmkl_sequential
        endif
    else
        # use GNU Fortran conventions
        ifeq (${blas_int},int64)
            LIBS += -lmkl_gf_ilp64
        else
            LIBS += -lmkl_gf_lp64
        endif

        # if threaded, use GNU OpenMP (gomp)
        ifeq (${blas_threaded},1)
            LIBS += -lmkl_gnu_thread
        else
            LIBS += -lmkl_sequential
        endif
    endif

    LIBS += -lmkl_core -lpthread -lm -ldl

    # MKL on MacOS doesn't include ScaLAPACK; use default.
    # For others, link with appropriate version of ScaLAPACK and BLACS.
    ifneq (${macos},1)
        ifeq (${mkl_blacs},openmpi)
            ifeq (${blas_int},int64)
                SCALAPACK_LIBRARIES ?= -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64
            else
                SCALAPACK_LIBRARIES ?= -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64
            endif
        else
            ifeq (${blas_int},int64)
                SCALAPACK_LIBRARIES ?= -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64
            else
                SCALAPACK_LIBRARIES ?= -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
            endif
        endif
    endif
else ifeq (${blas},essl)
    # IBM ESSL
    # todo threaded, int64
    # hmm... likely LAPACK won't be int64 even if ESSL is.
    LIBS += -lessl -llapack
else ifeq (${blas},openblas)
    # OpenBLAS
    LIBS += -lopenblas
else ifeq (${blas},libsci)
    # Cray LibSci
    # no LIBS to add
    SCALAPACK_LIBRARIES ?=
else
    ${error ERROR: unknown `blas=${blas}`. Set blas to one of mkl, essl, \
            openbblas, libsci, accelerate}
endif

# If not set by user or above, set default.
SCALAPACK_LIBRARIES ?= -lscalapack

#-------------------------------------------------------------------------------
# if CUDA
ifeq (${cuda},1)
    # Generate flags for which CUDA architectures to build.
    # cuda_arch_ is a local copy to modify.
    cuda_arch_ = ${cuda_arch}
    ifneq (,${findstring kepler, ${cuda_arch_}})
        cuda_arch_ += sm_30
    endif
    ifneq (,${findstring maxwell, ${cuda_arch_}})
        cuda_arch_ += sm_50
    endif
    ifneq (,${findstring pascal, ${cuda_arch_}})
        cuda_arch_ += sm_60
    endif
    ifneq (,${findstring volta, ${cuda_arch_}})
        cuda_arch_ += sm_70
    endif
    ifneq (,${findstring turing, ${cuda_arch_}})
        cuda_arch_ += sm_75
    endif
    ifneq (,${findstring ampere, ${cuda_arch_}})
        cuda_arch_ += sm_80
    endif
    ifneq (,${findstring hopper, ${cuda_arch_}})
        cuda_arch_ += sm_90
    endif

    # Extract CUDA sm architectures.
    sms = ${sort ${patsubst sm_%, %, ${filter sm_%, ${cuda_arch_}}}}

    # Generate nvcc gencode options for all sm_XY in cuda_arch_.
    # code=sm_XX is binary, code=compute_XX is PTX
    nv_sm      = ${foreach sm, ${sms},-gencode arch=compute_${sm},code=sm_${sm}}
    nv_compute = ${foreach sm, ${sms},-gencode arch=compute_${sm},code=compute_${sm}}

    ifeq (${sms},)
        # Error if cuda_arch is not empty and sms is empty.
        ifneq (${cuda_arch},)
            ${error ERROR: unknown `cuda_arch=${cuda_arch}`. Set cuda_arch \
                    to one or more of kepler, maxwell, pascal, volta, \
                    turing, ampere, hopper, \
                    or valid sm_XY from nvcc -h}
        endif
    else
        # Get last option (last 2 words) of nv_compute.
        nwords := ${words ${nv_compute}}
        nwords_1 := ${shell expr ${nwords} - 1}
        nv_compute_last := ${wordlist ${nwords_1}, ${nwords}, ${nv_compute}}
    endif

    # Use all sm_XX (binary), and the last compute_XX (PTX) for forward compatibility.
    NVCCFLAGS += ${nv_sm} ${nv_compute_last}

    libdir := ${CUDA_PATH}/lib64
    ifeq (${wildcard ${libdir}},)
        libdir := ${CUDA_PATH}/lib
    endif
    FLAGS += -I${CUDA_PATH}/include
    LIBS  += -L${libdir} -Wl,-rpath,${libdir} -lcusolver -lcublas -lcudart
endif

#-------------------------------------------------------------------------------
# if HIP
ifeq (${hip},1)
    # Generate flags for which HIP architectures to build.
    # hip_arch_ is a local copy to modify.
    hip_arch_ = ${hip_arch}
    ifneq (,${findstring mi25, ${hip_arch_}})
        hip_arch_ += gfx900
    endif
    ifneq (,${findstring mi50, ${hip_arch_}})
        hip_arch_ += gfx906
    endif
    ifneq (,${findstring mi100, ${hip_arch_}})
        hip_arch_ += gfx908
    endif
    ifneq (,${findstring mi200, ${hip_arch_}})
        hip_arch_ += gfx90a
    endif

    # Extract AMD gfx architectures.
    gfx = ${sort ${filter gfx%, ${hip_arch_}}}
    ifeq (${gfx},)
        # Error if hip_arch is not empty and gfx is empty.
        ifneq (${hip_arch},)
            ${error ERROR: unknown `hip_arch=${hip_arch}`. Set hip_arch \
                    to one or more of mi25, mi50, mi100, or valid gfxXYZ. \
                    See https://llvm.org/docs/AMDGPUUsage.html}
        endif
    endif

    # Generate hipcc target options for all gfx in hip_arch_.
    offload_arch = ${foreach arch, ${gfx},--offload-arch=${arch}}
    HIPCCFLAGS += ${offload_arch}
    FLAGS += -I${ROCM_PATH}/include -D__HIP_PLATFORM_AMD__
    LIBS  += -L${ROCM_PATH}/lib -Wl,-rpath,${ROCM_PATH}/lib -lrocsolver -lrocblas -lamdhip64

    # ROCm 4.0 has errors in its headers that produce excessive warnings.
    CXXFLAGS := ${filter-out -pedantic, ${CXXFLAGS}}
    CXXFLAGS += -Wno-unused-result
endif

#-------------------------------------------------------------------------------
# Files

# types and classes
slate_src += \
        src/auxiliary/Debug.cc \
        src/auxiliary/Trace.cc \
        src/core/Memory.cc \
        src/core/enums.cc \
        src/core/types.cc \
        src/version.cc \
        # End. Add alphabetically.

# internal
slate_src += \
        src/internal/internal_comm.cc \
        src/internal/internal_util.cc \
        # End. Add alphabetically.

# Most unit testers don't need the whole library, only the above subset.
ifneq (${only_unit},1)
    slate_src += \
        src/internal/internal_copyhb2st.cc \
        src/internal/internal_copytb2bd.cc \
        src/internal/internal_gbnorm.cc \
        src/internal/internal_geadd.cc \
        src/internal/internal_gebr.cc \
        src/internal/internal_gecopy.cc \
        src/internal/internal_gerbt.cc \
        src/internal/internal_rbt_generate.cc \
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
        src/internal/internal_reduce_info.cc \
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

#-------------------------------------------------------------------------------
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

hip_src := ${patsubst src/cuda/%.cu,src/hip/%.hip.cc,${cuda_src}}
hip_hdr := ${patsubst src/cuda/%.cuh,src/hip/%.hip.hh,${cuda_hdr}}

# OpenMP implementations of device kernels
omptarget_src := \
        src/omptarget/device_geadd.cc \
        src/omptarget/device_gecopy.cc \
        src/omptarget/device_genorm.cc \
        src/omptarget/device_gescale.cc \
        src/omptarget/device_gescale_row_col.cc \
        src/omptarget/device_geset.cc \
        src/omptarget/device_henorm.cc \
        src/omptarget/device_synorm.cc \
        src/omptarget/device_transpose.cc \
        src/omptarget/device_trnorm.cc \
        src/omptarget/device_tzadd.cc \
        src/omptarget/device_tzcopy.cc \
        src/omptarget/device_tzscale.cc \
        src/omptarget/device_tzset.cc \
        # End. Add alphabetically.

ifeq (${cuda},1)
    slate_src += ${cuda_src}
else ifeq (${hip},1)
    slate_src += ${hip_src}
else
    # Used for both OpenMP offload (${omptarget} == 1) and as stubs for
    # CPU-only build.
    slate_src += ${omptarget_src}
endif

#-------------------------------------------------------------------------------
# driver
ifneq (${only_unit},1)
    slate_src += \
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
        src/gerbt.cc \
        src/gels.cc \
        src/gels_cholqr.cc \
        src/gels_qr.cc \
        src/gemm.cc \
        src/gemmA.cc \
        src/gemmC.cc \
        src/geqrf.cc \
        src/gesv.cc \
        src/gesv_mixed.cc \
        src/gesv_mixed_gmres.cc \
        src/gesv_nopiv.cc \
        src/gesv_rbt.cc \
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
        src/pocondest.cc \
        src/posv.cc \
        src/posv_mixed.cc \
        src/posv_mixed_gmres.cc \
        src/potrf.cc \
        src/potri.cc \
        src/potrs.cc \
        src/print.cc \
        src/redistribute.cc \
        src/scale.cc \
        src/scale_row_col.cc \
        src/set.cc \
        src/set_lambdas.cc \
        src/stedc.cc \
        src/stedc_deflate.cc \
        src/stedc_merge.cc \
        src/stedc_secular.cc \
        src/stedc_solve.cc \
        src/stedc_sort.cc \
        src/stedc_z_vector.cc \
        src/steqr2.cc \
        src/sterf.cc \
        src/svd.cc \
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
        src/unmbr_ge2tb.cc \
        src/unmqr.cc \
        src/unmtr_hb2st.cc \
        src/unmtr_he2hb.cc \
        src/work/work_trmm.cc \
        src/work/work_trsm.cc \
        src/work/work_trsmA.cc \
        # End. Add alphabetically.
endif

ifneq (${have_fortran},)
    slate_src += \
        src/ssteqr2.f \
        src/dsteqr2.f \
        src/csteqr2.f \
        src/zsteqr2.f \
        # End. Add alphabetically, by base name after precision.
else
    ${error ERROR: Fortran compiler FC='${FC}' not found. Set FC to a \
            Fortran compiler (mpif90, gfortran, ifort, xlf, ftn, ...}. \
            We hope to eventually remove this requirement.)
endif

# C API
ifeq (${c_api},1)
    slate_src += \
        src/c_api/matrix.cc \
        src/c_api/util.cc \
        src/c_api/wrappers.cc \
        src/c_api/wrappers_precisions.cc \
        # End. Add alphabetically.
endif

# Fortran module
ifeq (${fortran_api},1)
    slate_src += \
        src/fortran/slate_module.f90 \
        # End. Add alphabetically.
endif

# main tester
tester_src += \
        test/matrix_params.cc \
        test/matrix_utils.cc \
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
        test/test_pocondest.cc \
        test/test_posv.cc \
        test/test_potri.cc \
        test/test_scale.cc \
        test/test_scale_row_col.cc \
        test/test_set.cc \
        test/test_stedc.cc \
        test/test_stedc_deflate.cc \
        test/test_stedc_secular.cc \
        test/test_stedc_sort.cc \
        test/test_stedc_z_vector.cc \
        test/test_steqr2.cc \
        test/test_sterf.cc \
        test/test_svd.cc \
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

# Compile fixes for ScaLAPACK routines if Fortran compiler ${FC} exists.
ifneq (${SCALAPACK_LIBRARIES},none)
ifneq (${have_fortran},)
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
    unit_test/test_func.cc \
    unit_test/test_geadd.cc \
    unit_test/test_gecopy.cc \
    unit_test/test_gescale.cc \
    unit_test/test_geset.cc \
    unit_test/test_internal_blas.cc \
    unit_test/test_norm.cc \
    unit_test/test_util.cc \
    # End. Add alphabetically.

ifeq (${c_api},1)
    unit_src += \
        unit_test/test_c_api.cc
endif

ifneq (${only_unit},1)
    unit_src += \
        unit_test/test_lq.cc \
        unit_test/test_qr.cc \
        # End. Add alphabetically.
endif

# unit test framework
unit_test_obj = \
        unit_test/unit_test.o

slate_obj  = ${addsuffix .o, ${basename ${slate_src}}}
tester_obj = ${addsuffix .o, ${basename ${tester_src}}}
unit_obj   = ${addsuffix .o, ${basename ${unit_src}}}
dep        = ${addsuffix .d, ${basename ${slate_src} \
                                        ${tester_src} ${unit_src} \
                                        ${unit_test_obj}}}

tester    = test/tester
unit_test = ${basename ${unit_src}}

pkg = lib/pkgconfig/slate.pc

# For `tester --debug`, lldb may need test.o compiled with -O0 (after -O3)
# to see variable `i`.
test/test.o: CXXFLAGS += -O0

#-------------------------------------------------------------------------------
# Get Mercurial id, and make version.o depend on it via .id file.

ifneq (${wildcard .git},)
    id := ${shell git rev-parse --short HEAD}
    src/version.o: CXXFLAGS += -DSLATE_ID='"${id}"'
endif

last_id := ${shell [ -e .id ] && cat .id || echo 'NA'}
ifneq (${id},${last_id})
    .id: force
endif

.id:
	echo ${id} > .id

src/version.o: .id

#-------------------------------------------------------------------------------
# SLATE specific flags and libraries
# FLAGS accumulates definitions, include dirs, etc. for both CXX and NVCC.
# FLAGS += -I.
FLAGS += -I./blaspp/include
FLAGS += -I./lapackpp/include
FLAGS += -I./include
FLAGS += -I./src

CXXFLAGS   += ${FLAGS}
NVCCFLAGS  += ${FLAGS}
HIPCCFLAGS += ${FLAGS}

# libraries to create libslate.so
LDFLAGS  += -L./blaspp/lib
LDFLAGS  += -L./lapackpp/lib
LIBS     := -lblaspp -llapackpp ${LIBS}

# additional flags and libraries for testers
${tester_obj}:    CXXFLAGS += -I./testsweeper
${unit_obj}:      CXXFLAGS += -I./testsweeper
${unit_test_obj}: CXXFLAGS += -I./testsweeper

TEST_LDFLAGS  = ${LDFLAGS} -L./lib -Wl,-rpath,${abspath ./lib}
TEST_LDFLAGS += -L./testsweeper -Wl,-rpath,${abspath ./testsweeper}
TEST_LDFLAGS += -Wl,-rpath,${abspath ./blaspp/lib}
TEST_LDFLAGS += -Wl,-rpath,${abspath ./lapackpp/lib}
TEST_LIBS     = -lslate -lslate_matgen -ltestsweeper ${LIBS}
ifneq (${SCALAPACK_LIBRARIES},none)
    TEST_LIBS += ${SCALAPACK_LIBRARIES}
    CXXFLAGS  += -DSLATE_HAVE_SCALAPACK
endif

UNIT_LDFLAGS  = ${LDFLAGS} -L./lib -Wl,-rpath,${abspath ./lib}
UNIT_LDFLAGS += -L./testsweeper -Wl,-rpath,${abspath ./testsweeper}
UNIT_LDFLAGS += -Wl,-rpath,${abspath ./blaspp/lib}
UNIT_LDFLAGS += -Wl,-rpath,${abspath ./lapackpp/lib}
UNIT_LIBS     = -lslate -ltestsweeper ${LIBS}

#-------------------------------------------------------------------------------
# Rules
.DELETE_ON_ERROR:
.SUFFIXES:
.PHONY: all docs hooks lib test tester unit_test clean distclean testsweeper blaspp lapackpp
.DEFAULT_GOAL := all

all: lib unit_test hooks

ifneq (${only_unit},1)
    all: tester lapack_api matgen
    install: lapack_api matgen
    ifneq (${SCALAPACK_LIBRARIES},none)
        all: scalapack_api
        install: scalapack_api
    endif
endif

install: lib ${pkg}
	cd blaspp   && ${MAKE} install prefix=${abs_prefix}
	@echo
	cd lapackpp && ${MAKE} install prefix=${abs_prefix}
	@echo
	mkdir -p ${DESTDIR}${abs_prefix}/include/slate/internal
	mkdir -p ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}
	mkdir -p ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/pkgconfig
	cp include/slate/*.hh          ${DESTDIR}${abs_prefix}/include/slate/
	cp include/slate/internal/*.hh ${DESTDIR}${abs_prefix}/include/slate/internal/
	cp -av lib/lib*                ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/
	cp ${pkg}                      ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/pkgconfig/
	if [ ${c_api} -eq 1 ]; then \
		mkdir -p ${DESTDIR}${abs_prefix}/include/slate/c_api; \
		cp include/slate/c_api/*.h ${DESTDIR}${abs_prefix}/include/slate/c_api; \
	fi
	if [ ${fortran_api} -eq 1 ]; then \
		cp slate.mod               ${DESTDIR}${abs_prefix}/include/; \
	fi

uninstall:
	cd blaspp   && ${MAKE} uninstall prefix=${abs_prefix}
	@echo
	cd lapackpp && ${MAKE} uninstall prefix=${abs_prefix}
	@echo
	${RM}    ${DESTDIR}${abs_prefix}/include/slate.mod
	${RM} -r ${DESTDIR}${abs_prefix}/include/slate
	${RM}    ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/libslate*
	${RM}    ${DESTDIR}${abs_prefix}/lib${LIB_SUFFIX}/pkgconfig/slate.pc

docs:
	doxygen docs/doxygen/doxyfile.conf
	@echo "------------------------------------------------------------"
	@echo "Errors:"
	perl -pe 's@^/.*?slate/@@' docs/doxygen/errors.txt

#-------------------------------------------------------------------------------
# C API
ifeq (${c_api},1)
    include/slate/c_api/wrappers.h: src/c_api/wrappers.cc
		${python} tools/c_api/generate_wrappers.py $< $@ \
			src/c_api/wrappers_precisions.cc

    include/slate/c_api/matrix.h: include/slate/Tile.hh include/slate/types.hh
		${python} tools/c_api/generate_matrix.py $^ $@ src/c_api/matrix.cc

    src/c_api/util.hh: include/slate/c_api/types.h
		${python} tools/c_api/generate_util.py $< $@ src/c_api/util.cc

    src/c_api/wrappers_precisions.cc: include/slate/c_api/wrappers.h src/c_api/util.hh
    src/c_api/matrix.cc: include/slate/c_api/matrix.h src/c_api/util.hh
    src/c_api/util.cc: src/c_api/util.hh
    src/c_api/wrappers.o: include/slate/c_api/wrappers.h src/c_api/util.hh

    generate: include/slate/c_api/wrappers.h
    generate: include/slate/c_api/matrix.h
    generate: src/c_api/util.hh
endif

#-------------------------------------------------------------------------------
# Fortran module
ifeq (${fortran_api},1)
    src/fortran/slate_module.f90: include/slate/c_api/wrappers.h \
                                  include/slate/c_api/types.h \
                                  include/slate/c_api/matrix.h
		${python} tools/fortran/generate_fortran_module.py $^ --output $@

    generate: src/fortran/slate_module.f90
endif

#-------------------------------------------------------------------------------
# testsweeper library
testsweeper_src = ${wildcard testsweeper/*.hh testsweeper/*.cc}

testsweeper = testsweeper/libtestsweeper.${lib_ext}

${testsweeper}: ${testsweeper_src}
	cd testsweeper && ${MAKE} lib

testsweeper: ${testsweeper}

#-------------------------------------------------------------------------------
# BLAS++ library
blaspp_src = ${wildcard blaspp/include/blas/*.h \
                        blaspp/include/blas/*.hh \
                        blaspp/src/*.cc}

blaspp = blaspp/lib/blaspp.${lib_ext}

# dependency on testsweeper serializes compiles
${blaspp}: ${blaspp_src} | ${testsweeper}
	cd blaspp && ${MAKE} lib

blaspp: ${blaspp}

#-------------------------------------------------------------------------------
# LAPACK++ library
lapackpp_src = ${wildcard lapackpp/include/lapack/*.h \
                          lapackpp/include/lapack/*.hh \
                          lapackpp/src/*.cc}

lapackpp = lapackpp/lib/lapackpp.${lib_ext}

# dependency on testsweeper, BLAS++ serializes compiles
${lapackpp}: ${lapackpp_src} | ${testsweeper} ${blaspp}
	cd lapackpp && ${MAKE} lib

lapackpp: ${lapackpp}

#-------------------------------------------------------------------------------
# Generic rule for shared libraries.
# For libfoo.so version 4.5.6, this creates libfoo.so.4.5.6 and symlinks
# libfoo.so.4 -> libfoo.so.4.5.6
# libfoo.so   -> libfoo.so.4
#
# Needs [private] variables set (shown with example values):
# LDFLAGS     = -L/path/to/lib
# LIBS        = -lmylib
# lib_obj     = src/foo.o src/bar.o
# lib_so_abi  = libfoo.so.4.5.6
# lib_soname  = libfoo.so.4
# abi_version = 4.5.6
# soversion   = 4
%.${lib_ext}:
	mkdir -p lib
	${LD} ${LDFLAGS} ${ldflags_shared} ${LIBS} ${lib_obj} -o ${lib_so_abi}
	ln -fs ${notdir ${lib_so_abi}} ${lib_soname}
	ln -fs ${notdir ${lib_soname}} $@

# Generic rule for static libraries, creates libfoo.a.
# The library should depend only on its objects.
%.a:
	mkdir -p lib
	${RM} $@
	${AR} cr $@ $^
	${RANLIB} $@

#-------------------------------------------------------------------------------
# SLATE library
# so     is like libfoo.so       or libfoo.dylib
# so_abi is like libfoo.so.4.5.6 or libfoo.4.5.6.dylib
# soname is like libfoo.so.4     or libfoo.4.dylib
slate_name   = lib/libslate
slate_a      = ${slate_name}.a
slate_so     = ${slate_name}.${so}
slate        = ${slate_name}.${lib_ext}
slate_so_abi = ${slate_name}${so1}.${abi_version}${so2}
slate_soname = ${slate_name}${so1}.${soversion}${so2}

${slate_so}: ${slate_obj}
${slate_so}: private lib_obj    = ${slate_obj}
${slate_so}: private lib_so_abi = ${slate_so_abi}
${slate_so}: private lib_soname = ${slate_soname}

${slate_a}: ${slate_obj}

src: ${slate}

src/clean:
	${RM} ${slate_a} ${slate_so} ${slate_so_abi} ${slate_soname} ${slate_obj}

#-------------------------------------------------------------------------------
# headers
# precompile headers to verify self-sufficiency
headers     = ${wildcard include/slate/*.hh \
                         include/slate/internal/*.hh \
                         test/*.hh \
                         include/slate/c_api/*.h \
                         include/slate/c_api/*.hh}

headers_gch = ${addsuffix .gch, ${basename ${headers}}}

headers: ${headers_gch}

# sub-directory rules
include: headers

include/clean:
	${RM} include/*/*.gch test/*.gch

#-------------------------------------------------------------------------------
# matgen library
matgen_name   = lib/libslate_matgen
matgen_a      = ${matgen_name}.a
matgen_so     = ${matgen_name}.${so}
matgen        = ${matgen_name}.${lib_ext}
matgen_so_abi = ${matgen_name}${so1}.${abi_version}${so2}
matgen_soname = ${matgen_name}${so1}.${soversion}${so2}

matgen_src += \
        matgen/generate_matrix_ge.cc \
        matgen/generate_matrix_he_and_tz.cc \
        matgen/generate_matrix_utils.cc \
        matgen/random.cc \
        # End. Add alphabetically.

matgen_obj = ${addsuffix .o, ${basename ${matgen_src}}}
dep       += ${addsuffix .d, ${basename ${matgen_src}}}

# See generic rule for shared libraries.
${matgen_so}: ${matgen_obj} ${slate_so}
${matgen_so}: private LDFLAGS += -L./lib
${matgen_so}: private LIBS    := -lslate ${LIBS}
${matgen_so}: private lib_obj    = ${matgen_obj}
${matgen_so}: private lib_so_abi = ${matgen_so_abi}
${matgen_so}: private lib_soname = ${matgen_soname}

${matgen_a}: ${matgen_obj}

matgen: ${matgen}

matgen/clean:
	${RM} ${matgen_name}* ${matgen_obj}

#-------------------------------------------------------------------------------
# main tester
# Note 'test' is sub-directory rule; 'tester' is CMake-compatible rule.
test: ${tester}
tester: ${tester}

test/clean:
	rm -f ${tester} ${tester_obj}

${tester}: ${tester_obj} ${slate} ${matgen} ${testsweeper}
	${LD} ${TEST_LDFLAGS} ${TEST_LIBS} ${tester_obj} -o $@

test/check: check
unit_test/check: check

check: test unit_test
	cd test; ${python} run_tests.py --quick gesv posv gels heev svd
	cd unit_test; ${python} run_tests.py

#-------------------------------------------------------------------------------
# unit testers
unit_test: ${unit_test}

unit_test/clean:
	rm -f ${unit_test} ${unit_obj} ${unit_test_obj}

${unit_test}: %: %.o ${unit_test_obj} ${slate}
	${LD} ${UNIT_LDFLAGS} ${UNIT_LIBS} $< ${unit_test_obj} -o $@

#-------------------------------------------------------------------------------
# scalapack_api library
scalapack_api_name   = lib/libslate_scalapack_api
scalapack_api_a      = ${scalapack_api_name}.a
scalapack_api_so     = ${scalapack_api_name}.${so}
scalapack_api        = ${scalapack_api_name}.${lib_ext}
scalapack_api_so_abi = ${scalapack_api_name}${so1}.${abi_version}${so2}
scalapack_api_soname = ${scalapack_api_name}${so1}.${soversion}${so2}

scalapack_api_src += \
        scalapack_api/scalapack_gecon.cc \
        scalapack_api/scalapack_gels.cc \
        scalapack_api/scalapack_gemm.cc \
        scalapack_api/scalapack_gesv.cc \
        scalapack_api/scalapack_gesv_mixed.cc \
        scalapack_api/scalapack_gesvd.cc \
        scalapack_api/scalapack_getrf.cc \
        scalapack_api/scalapack_getrs.cc \
        scalapack_api/scalapack_heev.cc \
        scalapack_api/scalapack_heevd.cc \
        scalapack_api/scalapack_hemm.cc \
        scalapack_api/scalapack_her2k.cc \
        scalapack_api/scalapack_herk.cc \
        scalapack_api/scalapack_lange.cc \
        scalapack_api/scalapack_lanhe.cc \
        scalapack_api/scalapack_lansy.cc \
        scalapack_api/scalapack_lantr.cc \
        scalapack_api/scalapack_pocon.cc \
        scalapack_api/scalapack_posv.cc \
        scalapack_api/scalapack_potrf.cc \
        scalapack_api/scalapack_potri.cc \
        scalapack_api/scalapack_potrs.cc \
        scalapack_api/scalapack_symm.cc \
        scalapack_api/scalapack_syr2k.cc \
        scalapack_api/scalapack_syrk.cc \
        scalapack_api/scalapack_trcon.cc \
        scalapack_api/scalapack_trmm.cc \
        scalapack_api/scalapack_trsm.cc \
        # End. Add alphabetically.

scalapack_api_obj = ${addsuffix .o, ${basename ${scalapack_api_src}}}
dep              += ${addsuffix .d, ${basename ${scalapack_api_src}}}

# See generic rule for shared libraries.
${scalapack_api_so}: ${scalapack_api_obj} ${slate_so}
${scalapack_api_so}: private LDFLAGS += -L./lib
${scalapack_api_so}: private LIBS    := -lslate ${SCALAPACK_LIBRARIES} ${LIBS}
${scalapack_api_so}: private lib_obj    = ${scalapack_api_obj}
${scalapack_api_so}: private lib_so_abi = ${scalapack_api_so_abi}
${scalapack_api_so}: private lib_soname = ${scalapack_api_soname}

ifeq (${SCALAPACK_LIBRARIES},none)
    ${scalapack_api_so_abi}:
		@echo "Error: building $@ requires ScaLAPACK library, but currently SCALAPACK_LIBRARIES=${SCALAPACK_LIBRARIES}."
		false
endif

${scalapack_api_a}: ${scalapack_api_obj}

scalapack_api: ${scalapack_api}

scalapack_api/clean:
	rm -f ${scalapack_api_name}* ${scalapack_api_obj}

#-------------------------------------------------------------------------------
# lapack_api library
lapack_api_name   = lib/libslate_lapack_api
lapack_api_a      = ${lapack_api_name}.a
lapack_api_so     = ${lapack_api_name}.${so}
lapack_api        = ${lapack_api_name}.${lib_ext}
lapack_api_so_abi = ${lapack_api_name}${so1}.${abi_version}${so2}
lapack_api_soname = ${lapack_api_name}${so1}.${soversion}${so2}

lapack_api_src += \
        lapack_api/lapack_gecon.cc \
        lapack_api/lapack_gels.cc \
        lapack_api/lapack_gemm.cc \
        lapack_api/lapack_gesv.cc \
        lapack_api/lapack_gesv_mixed.cc \
        lapack_api/lapack_gesvd.cc \
        lapack_api/lapack_getrf.cc \
        lapack_api/lapack_getri.cc \
        lapack_api/lapack_getrs.cc \
        lapack_api/lapack_heev.cc \
        lapack_api/lapack_heevd.cc \
        lapack_api/lapack_hemm.cc \
        lapack_api/lapack_her2k.cc \
        lapack_api/lapack_herk.cc \
        lapack_api/lapack_lange.cc \
        lapack_api/lapack_lanhe.cc \
        lapack_api/lapack_lansy.cc \
        lapack_api/lapack_lantr.cc \
        lapack_api/lapack_pocon.cc \
        lapack_api/lapack_posv.cc \
        lapack_api/lapack_potrf.cc \
        lapack_api/lapack_potri.cc \
        lapack_api/lapack_symm.cc \
        lapack_api/lapack_syr2k.cc \
        lapack_api/lapack_syrk.cc \
        lapack_api/lapack_trcon.cc \
        lapack_api/lapack_trmm.cc \
        lapack_api/lapack_trsm.cc \
        # End. Add alphabetically.

lapack_api_obj = ${addsuffix .o, ${basename ${lapack_api_src}}}
dep           += ${addsuffix .d, ${basename ${lapack_api_src}}}

# See generic rule for shared libraries.
${lapack_api_so}: ${lapack_api_obj} ${slate_so}
${lapack_api_so}: private LDFLAGS += -L./lib
${lapack_api_so}: private LIBS    := -lslate ${LIBS}
${lapack_api_so}: private lib_obj    = ${lapack_api_obj}
${lapack_api_so}: private lib_so_abi = ${lapack_api_so_abi}
${lapack_api_so}: private lib_soname = ${lapack_api_soname}

${lapack_api_a}: ${lapack_api_obj}

lapack_api: ${lapack_api}

lapack_api/clean:
	rm -f ${lapack_api_name}* ${lapack_api_obj}

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
${hip_hdr}: src/hip/%.hip.hh: src/cuda/%.cuh.md5 | src/hip

${hip_src} ${hip_hdr}:
	@${call if_md5_outdated, \
	        ${hipify} ${basename $<} > $@; \
	        ./tools/slate-hipify.pl $@ }

hipify: ${hip_src} ${hip_hdr}

md5_files := ${addsuffix .md5, ${cuda_src} ${cuda_hdr}}

${md5_files}: %.md5: %
	${md5sum} $< > $@

src/hip:
	mkdir -p $@

#-------------------------------------------------------------------------------
# pkgconfig
# Keep -std=c++11 in CXXFLAGS. Keep -fopenmp in LDFLAGS.
CXXFLAGS_clean = ${filter-out -O% -W% -pedantic -D% -I./% -MMD -fPIC, ${CXXFLAGS}}
CPPFLAGS_clean = ${filter-out -O% -W% -pedantic -D% -I./% -MMD -fPIC, ${CPPFLAGS}}
LDFLAGS_clean  = ${filter-out -fPIC -L./%, ${LDFLAGS}}

.PHONY: ${pkg}
${pkg}:
	perl -pe "s'#VERSION'2023.11.05'; \
	          s'#PREFIX'${abs_prefix}'; \
	          s'#CXX\b'${CXX}'; \
	          s'#CXXFLAGS'${CXXFLAGS_clean}'; \
	          s'#CPPFLAGS'${CPPFLAGS_clean}'; \
	          s'#LDFLAGS'${LDFLAGS_clean}'; \
	          s'#LIBS'${LIBS}'; \
	          s'#SCALAPACK_LIBRARIES'${SCALAPACK_LIBRARIES}'; \
	          s'#C_API'${c_api}'; \
	          s'#FORTRAN_API'${fortran_api}';" \
	          $@.in > $@

#-------------------------------------------------------------------------------
# general rules

lib: ${slate} ${matgen}

clean: test/clean unit_test/clean scalapack_api/clean lapack_api/clean include/clean
	rm -f lib/lib* ${dep}
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
	cd testsweeper && ${MAKE} distclean
	cd blaspp      && ${MAKE} distclean
	cd lapackpp    && ${MAKE} distclean

# Install git hooks
hooks = .git/hooks/pre-commit .git/hooks/pre-push

hooks: ${hooks}

.git/hooks/%: tools/hooks/%
	@if [ -e .git/hooks ]; then \
		echo cp $< $@ ; \
		cp $< $@ ; \
	fi

#-------------------------------------------------------------------------------
# Compile object files

# .hip.cc rule before .cc rule.
%.hip.o: %.hip.cc | ${hip_hdr}
	${HIPCC} ${HIPCCFLAGS} -c $< -o $@

%.o: %.cc
	${CXX} ${CXXFLAGS} -c $< -o $@

%.o: %.f
	${FC} ${FCFLAGS} -c $< -o $@

%.o: %.f90
	${FC} ${FCFLAGS} -c $< -o $@

%.o: %.cu
	${NVCC} ${NVCCFLAGS} -c $< -o $@

#-------------------------------------------------------------------------------
# Preprocess source

# test/%.i depend on testsweeper; for simplicity just add it here.
%.i: %.cc
	${CXX} ${CXXFLAGS} -I./testsweeper -E $< -o $@

#-------------------------------------------------------------------------------
# Precompile header to check for errors

# test/%.gch depend on testsweeper; for simplicity just add it here.
%.gch: %.hh
	${CXX} ${CXXFLAGS} -I./testsweeper -c $< -o $@

-include ${dep}

#-------------------------------------------------------------------------------
# Extra dependencies to force TestSweeper, BLAS++, LAPACK++ to be compiled before SLATE.

${slate_obj}:         | ${blaspp} ${lapackpp}
${matgen_obj}:        | ${blaspp} ${lapackpp}
${tester_obj}:        | ${blaspp} ${lapackpp}
${unit_test_obj}:     | ${blaspp} ${lapackpp}
${unit_obj}:          | ${blaspp} ${lapackpp}
${lapack_api_obj}:    | ${blaspp} ${lapackpp}
${scalapack_api_obj}: | ${blaspp} ${lapackpp}

#-------------------------------------------------------------------------------
# debugging
echo:
	@echo "---------- Options"
	@echo "mpi           = '${mpi}'"
	@echo "blas          = '${blas}'"
	@echo "blas_int      = '${blas_int}'"
	@echo "blas_threaded = '${blas_threaded}'"
	@echo "blas_fortran  = '${blas_fortran}'"
	@echo "mkl_blacs     = '${mkl_blacs}'"
	@echo "openmp        = '${openmp}'"
	@echo "static        = '${static}'"
	@echo "gpu_backend   = '${gpu_backend}'"
	@echo "prefix        = '${prefix}'"
	@echo "abs_prefix    = '${abs_prefix}'"
	@echo "c_api         = '${c_api}'"
	@echo "fortran_api   = '${fortran_api}'"
	@echo "SCALAPACK_LIBRARIES = '${SCALAPACK_LIBRARIES}'"
	@echo
	@echo "---------- Internal variables"
	@echo "ostype        = '${ostype}'"
	@echo "macos         = '${macos}'"
	@echo "id            = '${id}'"
	@echo "last_id       = '${last_id}'"
	@echo "abi_version   = '${abi_version}'"
	@echo "soversion     = '${soversion}'"
	@echo
	@echo "---------- Dependencies"
	@echo "blaspp        = ${blaspp}"
	@echo "lapackpp      = ${lapackpp}"
	@echo "testsweeper   = ${testsweeper}"
	@echo
	@echo "---------- Libraries"
	@echo "slate_a       = ${slate_a}"
	@echo "slate_so      = ${slate_so}"
	@echo "slate         = ${slate}"
	@echo "slate_so_abi  = ${slate_so_abi}"
	@echo "slate_soname  = ${slate_soname}"
	@echo
	@echo "pkg           = ${pkg}"
	@echo
	@echo "matgen_a      = ${matgen_a}"
	@echo "matgen_so     = ${matgen_so}"
	@echo "matgen        = ${matgen}"
	@echo "matgen_so_abi = ${matgen_so_abi}"
	@echo "matgen_soname = ${matgen_soname}"
	@echo
	@echo "scalapack_api_a      = ${scalapack_api_a}"
	@echo "scalapack_api_so     = ${scalapack_api_so}"
	@echo "scalapack_api        = ${scalapack_api}"
	@echo "scalapack_api_so_abi = ${scalapack_api_so_abi}"
	@echo "scalapack_api_soname = ${scalapack_api_soname}"
	@echo
	@echo "lapack_api_a      = ${lapack_api_a}"
	@echo "lapack_api_so     = ${lapack_api_so}"
	@echo "lapack_api        = ${lapack_api}"
	@echo "lapack_api_so_abi = ${lapack_api_so_abi}"
	@echo "lapack_api_soname = ${lapack_api_soname}"
	@echo
	@echo "---------- Files"
	@echo "slate_src     = ${slate_src}"
	@echo
	@echo "slate_obj     = ${slate_obj}"
	@echo
	@echo "matgen_src    = ${matgen_src}"
	@echo
	@echo "matgen_obj    = ${matgen_obj}"
	@echo
	@echo "tester_src    = ${tester_src}"
	@echo
	@echo "tester_obj    = ${tester_obj}"
	@echo
	@echo "tester        = ${tester}"
	@echo
	@echo "unit_src      = ${unit_src}"
	@echo
	@echo "unit_obj      = ${unit_obj}"
	@echo
	@echo "unit_test_obj = ${unit_test_obj}"
	@echo
	@echo "unit_test     = ${unit_test}"
	@echo
	@echo "dep           = ${dep}"
	@echo
	@echo "---------- C++ compiler"
	@echo "CXX           = ${CXX}"
	@echo "CXXFLAGS      = ${CXXFLAGS}"
	@echo "CXXFLAGS_clean= ${CXXFLAGS_clean}"
	@echo "CPPFLAGS      = ${CPPFLAGS}"
	@echo "CPPFLAGS_clean= ${CPPFLAGS_clean}"
	@echo
	@echo "---------- CUDA options"
	@echo "cuda          = '${cuda}'"
	@echo "cuda_arch     = ${cuda_arch}"
	@echo "cuda_arch_    = ${cuda_arch_}"
	@echo "NVCC          = ${NVCC}"
	@echo "NVCC_which    = ${NVCC_which}"
	@echo "CUDA_PATH     = ${CUDA_PATH}"
	@echo "NVCCFLAGS     = ${NVCCFLAGS}"
	@echo "sms           = ${sms}"
	@echo "nv_sm         = ${nv_sm}"
	@echo "nv_compute    = ${nv_compute}"
	@echo "nwords        = ${nwords}"
	@echo "nwords_1      = ${nwords_1}"
	@echo "nv_compute_last = ${nv_compute_last}"
	@echo
	@echo "---------- HIP options"
	@echo "hip           = '${hip}'"
	@echo "hip_arch      = '${hip_arch}'"
	@echo "hip_arch_     = '${hip_arch_}'"
	@echo "gfx           = ${gfx}"
	@echo "HIPCC         = ${HIPCC}"
	@echo "HIPCC_which   = ${HIPCC_which}"
	@echo "ROCM_PATH     = ${ROCM_PATH}"
	@echo "HIPCCFLAGS    = ${HIPCCFLAGS}"
	@echo "offload_arch  = ${offload_arch}"
	@echo "hipify        = ${hipify}"
	@echo "cuda_src      = ${cuda_src}"
	@echo "cuda_hdr      = ${cuda_hdr}"
	@echo "hip_src       = ${hip_src}"
	@echo "hip_hdr       = ${hip_hdr}"
	@echo "md5_files     = ${md5_files}"
	@echo
	@echo "---------- SYCL options"
	@echo "sycl          = '${sycl}'"
	@echo
	@echo "---------- OMP target-offload kernel options"
	@echo "omptarget     = '${omptarget}'"
	@echo "omptarget_src = ${omptarget_src}"
	@echo
	@echo "---------- Fortran compiler"
	@echo "FC            = ${FC}"
	@echo "FCFLAGS       = ${FCFLAGS}"
	@echo "have_fortran  = ${have_fortran}"
	@echo
	@echo "---------- Link flags"
	@echo "LD            = ${LD}"
	@echo "LDFLAGS       = ${LDFLAGS}"
	@echo "LIBS          = ${LIBS}"
	@echo "ldflags_shared = ${ldflags_shared}"
	@echo
	@echo "TEST_LDFLAGS  = ${TEST_LDFLAGS}"
	@echo "TEST_LIBS     = ${TEST_LIBS}"
	@echo
	@echo "UNIT_LDFLAGS  = ${UNIT_LDFLAGS}"
	@echo "UNIT_LIBS     = ${UNIT_LIBS}"
