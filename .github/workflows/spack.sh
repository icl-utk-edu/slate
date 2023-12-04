#!/bin/bash -e

bstage=$1
device=$2

set +x
trap 'echo "# $BASH_COMMAND"' DEBUG

source /etc/profile

if [ "${bstage}" = "build" ]; then
  git submodule update --init
  rm -rf spack || true
  git clone -b gragghia/slate_sycl https://github.com/G-Ragghianti/spack
  cp .github/workflows/spack_packages.yaml spack/etc/spack/packages.yaml
  spack config add upstreams:spack-instance-1:install_tree:/spack/
fi

source spack/share/spack/setup-env.sh
export HOME=$(pwd)

module load gcc@10.4.0
if [ "${device}" = "gpu_nvidia" ]; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | sed -e 's/\.//')
  SPEC="+cuda cuda_arch=$ARCH %gcc"
elif [ "${device}" = "gpu_amd" ]; then
  TARGET=$(rocminfo | grep Name | grep gfx | head -1 | awk '{print $2}')
  SPEC="+rocm amdgpu_target=$TARGET %gcc"
else
  SPEC="+sycl %oneapi"
  module load intel-oneapi-compilers
fi

if [ "${bstage}" = "test" ]; then
  TEST="--test=root"
  spack uninstall -a slate || true
fi

spack compiler find
spack test remove slate
spack spec slate@master $SPEC
spack dev-build $TEST slate@master $SPEC

if [ "${bstage}" = "smoke" ]; then
  spack test run slate
fi
