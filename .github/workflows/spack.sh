#!/bin/bash -e

bstage=$1
device=$2

set +x
trap 'echo "# $BASH_COMMAND"' DEBUG

source /etc/profile

COMPILER=gcc@11
MPI=intel-oneapi-mpi
BLAS=intel-oneapi-mkl

# Get the system-install ROCM version
ROCM_VER=$(ls /opt | grep rocm- | sed s/rocm-//)
# Get the system-install CUDA version
CUDA_VER=$(cd /usr/local && ls -d cuda-*.* | sed "s/cuda-//")

if [ "${bstage}" = "deps" ]; then
  git clone https://github.com/spack/spack
  cp .github/workflows/spack_packages_yaml spack/etc/spack/packages.yaml
  # Use $ROCM_VER in the spack package config
  sed -i "s/ROCMVER/$ROCM_VER/" spack/etc/spack/packages.yaml
  # Use $CUDA_VER in the spack package config
  sed -i "s/CUDAVER/$CUDA_VER/" spack/etc/spack/packages.yaml
fi

export HOME=$(pwd)
source spack/share/spack/setup-env.sh
module load $COMPILER
spack compiler find --scope=site

if [ "${device}" = "cpu" ]; then
  SPEC=""
elif [ "${device}" = "gpu_nvidia" ]; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | sed -e 's/\.//')
  SPEC="+cuda cuda_arch=$ARCH"
elif [ "${device}" = "gpu_amd" ]; then
  TARGET=$(rocminfo | grep Name | grep gfx | head -1 | awk '{print $2}')
  SPEC="+rocm amdgpu_target=$TARGET ^hip@$ROCM_VER"
else
  SPEC="+sycl"
  COMPILER=oneapi
  module load intel-oneapi-compilers
  spack compiler find --scope=site
fi
SPEC="slate@master $SPEC %$COMPILER ^$MPI ^$BLAS"
echo SPEC=$SPEC

if [ "${bstage}" = "deps" ]; then
  # Change the stage directory so we can find it later
  spack config --scope=site add config:build_stage:`pwd`/spack-stage
  spack spec $SPEC
  spack install --only=dependencies --fail-fast $SPEC
elif [ "${bstage}" = "build" ]; then
  spack dev-build -i $SPEC
elif [ "${bstage}" = "test" ]; then
  spack uninstall -y slate
  spack dev-build --test=root $SPEC || cat spack-stage/github/spack-stage-slate-master-*/install-time-test-log.txt
elif [ "${bstage}" = "smoke" ]; then
  spack test run slate
fi

