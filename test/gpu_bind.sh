#!/bin/bash
#
# Run job with $CUDA_VISIBLE_DEVICES xor $ROCR_VISIBLE_DEVICES
# set to the local MPI rank mod num devices.
# ROCm recognizes both $CUDA_VISIBLE_DEVICES and $ROCR_VISIBLE_DEVICES,
# and they can interact to hide all GPUs, hence the xor.

#-------------------------------------------------------------------------------
if [ -n "${MPI_LOCALRANKID+x}" ]; then
    # IBM MPI, Intel MPI
    rank_var=MPI_LOCALRANKID
    local_rank=${MPI_LOCALRANKID}

elif [ -n "${OMPI_COMM_WORLD_LOCAL_RANK+x}" ]; then
    # Open MPI
    rank_var=OMPI_COMM_WORLD_LOCAL_RANK
    local_rank=${OMPI_COMM_WORLD_LOCAL_RANK}

elif [ -n "${PMI_RANK+x}" ]; then
    # Intel MPI (older? unclear when MPI_LOCALRANKID was added.)
    # PMI_RANK might work only for single node jobs. Need PMI_RANK % ranks_per_node.
    # Cf. https://community.intel.com/t5/Intel-oneAPI-HPC-Toolkit/Environment-variables-defined-by-intel-mpirun/td-p/1096703
    rank_var=PMI_RANK
    local_rank=${PMI_RANK}

else
    # Unknown MPI.
    rank_var=unknown
    local_rank=0
fi

#-------------------------------------------------------------------------------
# Ignore GPUs that are already have processes.
# Use cached ${idle_gpus} if it was already set in setup_env.sh
if [ -z "${idle_gpus+x}" ]; then
    echo "${local_rank} ./idle_gpus.py"
    export idle_gpus=$(./idle_gpus.py)
fi

# Arrays can't be exported, effectively.
idle_gpus_array=(${idle_gpus})  # convert to array
gpu_kind=${idle_gpus_array[0]}  # element 0 is gpu_kind: cuda or rocm
idle_gpus_array=(${idle_gpus_array[@]:1})  # slice elements 1:end

#-------------------------------------------------------------------------------
# Get max( 1, ndev ), so local_rank % ndev doesn't divide by 0.
ndev=${#idle_gpus_array[*]}  # length
ndev=$(( ${ndev} < 1 ? 1 : ${ndev} ))

dev=$(( ${local_rank} % ${ndev} ))
visible_devices=${idle_gpus_array[ $dev ]}
echo "local_rank ${local_rank}, gpu_kind ${gpu_kind}, idle_gpus_array ${idle_gpus_array[*]}, ndev ${ndev}, dev ${dev}, visible_devices ${visible_devices}, rank_var ${rank_var}"

if [ "${gpu_kind}" == "cuda" ]; then
    export CUDA_VISIBLE_DEVICES=${visible_devices}

elif [ "${gpu_kind}" == "rocm" ]; then
    export ROCR_VISIBLE_DEVICES=${visible_devices}
fi

# Run program.
$@
