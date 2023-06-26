Usage
================================================================================

Environment variables that affect execution.
--------------------------------------------------------------------------------

* `OMP_NUM_THREADS`

    Sets the number of OpenMP threads per MPI rank.

* `CUDA_VISIBLE_DEVICES` (for CUDA)
* `ROCR_VISIBLE_DEVICES` (for HIP/ROCm)

    Sets which GPUs are visible to the program. For example, to make
    only GPU 0 visible:

        export CUDA_VISIBLE_DEVICES=0

* `SLATE_GPU_AWARE_MPI`

    Setting to `1` enables use of GPU-aware MPI within SLATE.
    If the MPI library is not actually GPU-aware, this will cause segfaults.


Example run
--------------------------------------------------------------------------------

Uses 4 MPI ranks, GPU-aware MPI enabled, with 10 OpenMP threads and 1 GPU
per MPI rank.  (Output abbreviated.)

```
slate/test> export OMP_NUM_THREADS=10
slate/test> export CUDA_VISIBLE_DEVICES=0
slate/test> export SLATE_GPU_AWARE_MPI=1
slate/test> mpirun -np 4 ./tester --dim 1234 --dim 1000 --origin d --target d gemm
% SLATE version 2022.07.00, id 04de6a43
% input: ./tester --dim 1234 --dim 1000 --origin d --target d gemm
% 2023-06-26 15:41:22, 4 MPI ranks, GPU-aware MPI, 10 OpenMP threads, 1 GPU devices per MPI rank

type  origin  target      m       n       k    nb    p    q      error   time (s)  status
   d     dev     dev   1234    1234    1234   384    2    2   2.55e-16     0.0529  pass
   d     dev     dev  10000   10000   10000   384    2    2   2.05e-16      0.712  pass
```
