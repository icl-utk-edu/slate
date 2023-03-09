
#include <algorithm>
#include <complex>
#include <vector>
#include <cuComplex.h>

#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.cuh"

#include <iostream>


namespace slate {
namespace device {

// templated casting from C++ types to cuda types
template<typename scalar_t>
static scalar_t to_cutype(scalar_t z) {
    return z;
}
static cuFloatComplex to_cutype(std::complex<float> z) {
    return make_cuFloatComplex(z.real(), z.imag());
}
static cuFloatComplex* to_cutype(std::complex<float>* z) {
    return (cuFloatComplex*)z;
}
static cuFloatComplex** to_cutype(std::complex<float>** z) {
    return (cuFloatComplex**)z;
}
static cuDoubleComplex to_cutype(std::complex<double> z) {
    return make_cuDoubleComplex(z.real(), z.imag());
}
static cuDoubleComplex* to_cutype(std::complex<double>* z) {
    return (cuDoubleComplex*)z;
}
static cuDoubleComplex** to_cutype(std::complex<double>** z) {
    return (cuDoubleComplex**)z;
}

// Type safe conj routine
static __device__ double nl_conj(double z) {
    return z;
}
static __device__ float nl_conj(float z) {
    return z;
}
static __device__ cuDoubleComplex nl_conj(cuDoubleComplex z) {
    return cuConj(z);
}
static __device__ cuFloatComplex nl_conj(cuFloatComplex z) {
    return cuConjf(z);
}

template<typename scalar_t>
__device__ scalar_t as_scalar(double r) {
    return scalar_t(r);
}
template<>
__device__ cuDoubleComplex as_scalar<cuDoubleComplex>(double r) {
    return make_cuDoubleComplex(r, 0.0);
}
template<>
__device__ cuFloatComplex as_scalar<cuFloatComplex>(double r) {
    return make_cuFloatComplex(r, 0.0);
}

#include "magma_gemm_template_device.cuh"


template <typename scalar_t>
void batch_gemm(
    blas::Layout layout,
    blas::Op     transA,
    blas::Op     transB,
    int64_t      mb,
    int64_t      nb,
    int64_t      kb,
    scalar_t alpha,
    std::vector<scalar_t*>& Aarray, int64_t Ai, int64_t Aj, int64_t ldda,
    std::vector<scalar_t*>& Barray, int64_t Bi, int64_t Bj, int64_t lddb,
    scalar_t beta,
    std::vector<scalar_t*>& Carray, int64_t Ci, int64_t Cj, int64_t lddc,
    const size_t batch,
    blas::Queue &queue )
{
    std::vector<blas::Op> transA_v {transA};
    std::vector<blas::Op> transB_v {transB};
    std::vector<int64_t>  m_v {mb};
    std::vector<int64_t>  n_v {nb};
    std::vector<int64_t>  k_v {kb};
    std::vector<scalar_t> alpha_v {alpha};
    std::vector<scalar_t> beta_v  {beta };
    std::vector<int64_t>  info (0);

    std::vector<scalar_t*> Aarray_v(batch);
    std::vector<int64_t>   ldda_v {ldda};
    std::vector<scalar_t*> Barray_v(batch);
    std::vector<int64_t>   lddb_v {lddb};
    std::vector<scalar_t*> Carray_v(batch);
    std::vector<int64_t>   lddc_v {lddc};

    if (layout == blas::Layout::ColMajor) {
        for (size_t i = 0; i < batch; ++i) {
            Aarray_v[i] = Aarray[i] + Ai + Aj*ldda;
            Barray_v[i] = Barray[i] + Bi + Bj*lddb;
            Carray_v[i] = Carray[i] + Ci + Cj*lddc;
        }
    }
    else {
        for (size_t i = 0; i < batch; ++i) {
            Aarray_v[i] = Aarray[i] + Ai*ldda + Aj;
            Barray_v[i] = Barray[i] + Bi*lddb + Bj;
            Carray_v[i] = Carray[i] + Ci*lddc + Cj;
        }
    }

    blas::batch::gemm(
        layout, transA_v, transB_v, m_v, n_v, k_v,
        alpha_v, Aarray_v, ldda_v,
                 Barray_v, lddb_v,
        beta_v,  Carray_v, lddc_v,
        batch, info, queue);
}
template <typename scalar_t>
void batch_trsm(
    blas::Layout layout,
    blas::Side   side,
    blas::Uplo   uplo,
    blas::Op     trans,
    blas::Diag   diag,
    int64_t      mb,
    int64_t      nb,
    scalar_t alpha,
    std::vector<scalar_t*>& Aarray, int64_t Ai, int64_t Aj, int64_t ldda,
    std::vector<scalar_t*>& Barray, int64_t Bi, int64_t Bj, int64_t lddb,
    const size_t batch,
    blas::Queue &queue )
{
    std::vector<blas::Side> side_v {side};
    std::vector<blas::Uplo> uplo_v {uplo};
    std::vector<blas::Op>  trans_v {trans};
    std::vector<blas::Diag> diag_v {diag};
    std::vector<int64_t>  m_v {mb};
    std::vector<int64_t>  n_v {nb};
    std::vector<scalar_t> alpha_v {alpha};
    std::vector<int64_t>  info (0);

    std::vector<scalar_t*> Aarray_v(batch);
    std::vector<int64_t>   ldda_v {ldda};
    std::vector<scalar_t*> Barray_v(batch);
    std::vector<int64_t>   lddb_v {lddb};

    if (layout == blas::Layout::ColMajor) {
        for (size_t i = 0; i < batch; ++i) {
            Aarray_v[i] = Aarray[i] + Ai + Aj*ldda;
            Barray_v[i] = Barray[i] + Bi + Bj*lddb;
        }
    }
    else {
        for (size_t i = 0; i < batch; ++i) {
            Aarray_v[i] = Aarray[i] + Ai*ldda + Aj;
            Barray_v[i] = Barray[i] + Bi*lddb + Bj;
        }
    }

    blas::batch::trsm(
        layout, side_v, uplo_v, trans_v, diag_v, m_v, n_v,
        alpha_v, Aarray_v, ldda_v,
                 Barray_v, lddb_v,
        batch, info, queue);
}

template <typename scalar_t>
__device__ void tb_gemm_magma(bool transA, bool transB,
                              int mb, int nb, int kb,
                              scalar_t alpha, scalar_t* __restrict__ dA, int ldda,
                                              scalar_t* __restrict__ dB, int lddb,
                                              scalar_t* __restrict__ dC, int lddc) {

    if (mb == 0 || nb == 0) {
        return;
    }

    if (transA && !transB) {
        // based on config 207
        constexpr int DIM_X = 16;
        constexpr int DIM_Y = 16;
        constexpr int BLK_M = 32; //48;
        constexpr int BLK_N = 32;
        constexpr int BLK_K = 16;
        constexpr int DIM_XA = 16;
        constexpr int DIM_YA = 16;
        constexpr int DIM_XB = 16;
        constexpr int DIM_YB = 16;

        for (int ii = 0; ii < mb; ii += BLK_M) {
            for (int jj = 0; jj < nb; jj += BLK_N) {
                __syncthreads();
                gemm_template_device_tn<scalar_t, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), 0, 0>(
                    min(BLK_M, mb-ii), min(BLK_N, nb-jj), kb,
                    dA +      ii*ldda, ldda,
                    dB +      jj*lddb, lddb,
                    dC + ii + jj*lddc, lddc,
                    alpha, as_scalar<scalar_t>(0.0));
            }
        }
    } else if (!transA && transB) {
        // based on config 160
        constexpr int DIM_X = 16;
        constexpr int DIM_Y = 16; //8;
        constexpr int BLK_M = 32;
        constexpr int BLK_N = 32;
        constexpr int BLK_K = 16; //8;
        constexpr int DIM_XA = 16;
        constexpr int DIM_YA = 16; //8;
        constexpr int DIM_XB = 16;
        constexpr int DIM_YB = 16; //8;

        for (int ii = 0; ii < mb; ii += BLK_M) {
            for (int jj = 0; jj < nb; jj += BLK_N) {
                gemm_template_device_nt<scalar_t, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), 0, 0>(
                    min(BLK_M, mb-ii), min(BLK_N, nb-jj), kb,
                    dA + ii,           ldda,
                    dB + jj,           lddb,
                    dC + ii + jj*lddc, lddc,
                    alpha, as_scalar<scalar_t>(0.0));
                __syncthreads();
            }
        }
    }
}

// Compute a gemm within the threadblock
// beta=0
template <typename scalar_t>
__device__ void tb_gemm(bool transA, bool transB,
                        int mb, int nb, int kb,
                        scalar_t alpha, scalar_t* __restrict__ dA, int64_t ldda,
                                        scalar_t* __restrict__ dB, int64_t lddb,
                                        scalar_t* __restrict__ dC, int64_t lddc)
{

    // Due to linking, dynamic shared memory can't be declared as scalar_t
    extern __shared__ char shared_ptr[];
    //__shared__ scalar_t shared_ptr [32*33];

    const scalar_t zero = as_scalar<scalar_t>(0.0);

    const int offseti = threadIdx.x;
    const int stridei = 32; //blockDim.x;
    const int offsetj = threadIdx.y;
    const int stridej = blockDim.y;

    if (transA) {
        if (transB) {
            for (int i = offseti; i < mb; i += stridei) {
                for (int j = offsetj; j < nb; j += stridej) {
                    scalar_t sum = zero;
                    for (int k = 0; k < kb; k += 1) {
                        sum += nl_conj(dA[k + i*ldda]) * nl_conj(dB[j + k*lddb]);
                    }
                    dC[i + j*lddc] = alpha*sum;
                }
            }
        } else {
            scalar_t* sA = (scalar_t*)shared_ptr;
            const int ldsa = 33;
            scalar_t* sB = sA + ldsa*32;
            const int ldsb = 33;

            for (int ii = 0; ii < mb; ii += 32) {
                int iib = min(32, mb-ii);

                if (offseti < iib) {
                    for (int j = offsetj; j < nb; j += stridej) {
                        dC[ii+offseti + j*lddc] = zero;
                    }
                }

                for (int kk = 0; kk < kb; kk += 32) {
                    int kkb = min(32, kb-kk);

                    {
                        for (int j = offsetj; j < nb; j += stridej) {
                            if (offseti < kkb && j < iib) {
                                sA[offseti + j*ldsa] = dA[kk+offseti + (ii+j)*ldda];
                            } else {
                                sA[offseti + j*ldsa] = zero;
                            }
                            if (offseti < kkb && j < nb) {
                                sB[offseti + j*ldsb] = dB[kk+offseti + j*lddb];
                            } else {
                                sB[offseti + j*ldsb] = zero;
                            }
                        }
                    }
                    __syncthreads();
                    int i = offseti;
                    if (i < iib) {
                        for (int j = offsetj; j < nb; j += stridej) {
                            scalar_t sum = zero;
                            //scalar_t* sAki = sA + i*ldsa;
                            //scalar_t* dBkj = dB + kk + j*lddb;
                            #pragma unroll 16
                            for (int k = 0; k < 32; k += 1) {
                                //scalar_t rBkj = k < kkb ? dBkj[0] : zero;
                                //sum += nl_conj(sAki[0]) * rBkj;
                                //sum += nl_conj(sAki[0]) * dBkj[0];
                                //sAki++;
                                //dBkj++;

                                //scalar_t rBkj = k < kkb ? dB[(kk+k) + j*lddb] : zero;
                                //scalar_t rBkj = k < kkb ? dBkj[k] : zero;
                                //sum += nl_conj(sA[    k  +     i *ldsa]) * rBkj;

                                sum += nl_conj(sA[k + i*ldsa]) * sB[k + j*ldsb];

                              //sum += nl_conj(sA[    k  +     i *ldsa]) * dB[(kk+k) + j*lddb];
                              //sum += nl_conj(dA[(kk+k) + (ii+i)*ldda]) * dB[(kk+k) + j*lddb];
                            }
                            dC[ii+i + j*lddc] += alpha*sum;
                        }
                    }
                    __syncthreads();
                }
            }

            // copy into shared memory to get coalessed access
            // N.B. Loop index names mismatch offseti, offsetj, stridei, stridej
            /*for (int k = offseti; k < kb; k += stridei) {
                for (int i = offsetj; i < mb; i += stridej) {
                    sA[k + i*ldsa] = dA[k + i*ldda];
                }
            }
            __syncthreads();
            for (int i = offseti; i < mb; i += stridei) {
                for (int j = offsetj; j < nb; j += stridej) {
                    scalar_t sum = zero;
                    scalar_t* sAki = sA + i*ldsa;
                    scalar_t* dBkj = dB + j*lddb;
                    for (int k = 0; k < kb; k += 1) {
                        sum += nl_conj(sAki[0]) * dBkj[0];
                        sAki++;
                        dBkj++;
                        //sum += nl_conj(sA[k + i*ldsa]) * dB[k + j*lddb];
                        //sum += nl_conj(dA[k + i*ldda]) * dB[k + j*lddb];
                    }
                    dC[i + j*lddc] = alpha*sum;
                }
            }*/

            /*for (int i = offseti; i < mb; i += stridei) {
                for (int j = offsetj; j < nb; j += stridej) {
                    scalar_t sum = zero;
                    scalar_t* dAki = dA + i*ldda;
                    scalar_t* dBkj = dB + j*lddb;
                    for (int k = 0; k < kb; k += 1) {
                        sum += nl_conj(dAki[0]) * dBkj[0];
                        dAki++;
                        dBkj++;
                        //sum += nl_conj(dA[k + i*ldda]) * dB[k + j*lddb];
                    }
                    dC[i + j*lddc] = alpha*sum;
                }
            }*/
        }
    } else {
        if (transB) {
            for (int j = offsetj; j < nb; j += stridej) {
                for (int i = offseti; i < mb; i += stridei) {
                    scalar_t sum = zero;
                    scalar_t* dAik = dA + i;
                    scalar_t* dBjk = dB + j;
                    for (int k = 0; k < kb; k += 1) {
                        //sum += dA[i + k*ldda] * nl_conj(dB[j + k*lddb]);
                        sum += dAik[0] * nl_conj(dBjk[0]);
                        dAik += ldda;
                        dBjk += lddb;
                    }
                    dC[i + j*lddc] = alpha*sum;
                }
            }
        } else {
            for (int i = offseti; i < mb; i += stridei) {
                for (int j = offsetj; j < nb; j += stridej) {
                    scalar_t sum = zero;
                    for (int k = 0; k < kb; k += 1) {
                        sum += dA[i + k*ldda] * dB[k + j*lddb];
                    }
                    dC[i + j*lddc] = alpha*sum;
                }
            }
        }
    }
}

template <typename scalar_t>
__device__ void tb_copy(
    int mb,
    int nb,
    scalar_t* __restrict__ dA, int64_t ldda,
    scalar_t* __restrict__ dB, int64_t lddb)
{
    int offseti = threadIdx.x;
    int stridei = blockDim.x;
    int offsetj = threadIdx.y;
    int stridej = blockDim.y;

    for (int i = offseti; i < mb; i += stridei) {
        for (int j = offsetj; j < nb; j += stridej) {
            dB[i + j*lddb] = dA[i + j*ldda];
        }
    }
}

template <typename scalar_t>
__device__ void tb_scale_copy(
    bool isLeft,
    int mb,
    int nb,
    blas::real_type<scalar_t>* dS,
    scalar_t* __restrict__ dA, int64_t ldda,
    scalar_t* __restrict__ dB, int64_t lddb)
{
    int offseti = threadIdx.x;
    int stridei = blockDim.x;
    int offsetj = threadIdx.y;
    int stridej = blockDim.y;

    if (isLeft) {
        for (int i = offseti; i < mb; i += stridei) {
            blas::real_type<scalar_t> S_i = dS[i];
            blas::real_type<scalar_t> inv_S_i = 1/S_i;
            for (int j = offsetj; j < nb; j += stridej) {
                dB[i + j*lddb] = dA[i + j*ldda] * inv_S_i;
                //dB[i + j*lddb] = dA[i + j*ldda] / S_i;
            }
        }
    } else {
        for (int j = offsetj; j < nb; j += stridej) {
            blas::real_type<scalar_t> S_j = dS[j];
            blas::real_type<scalar_t> inv_S_j = 1/S_j;
            for (int i = offseti; i < mb; i += stridei) {
                dB[i + j*lddb] = dA[i + j*ldda] * inv_S_j;
                //dB[i + j*lddb] = dA[i + j*ldda] / S_j;
            }
        }
    }
}

template <bool isUpper, bool isLeft, typename scalar_t>
__global__ void __launch_bounds__(256,3) batch_diag_kernel(
                    int mb, int nb,
                    scalar_t alpha,
                    scalar_t** dUarray,  int64_t Ui,            int64_t lddu,
                    scalar_t** dVTarray, int64_t VTi,           int64_t lddvt,
                    blas::real_type<scalar_t>** dSarray,
                    scalar_t** dBarray, int64_t Bi, int64_t Bj, int64_t lddb,
                    scalar_t** dWarray, int64_t lddw)
{
    using real_t = blas::real_type<scalar_t>;

    int batch = blockIdx.x;

    scalar_t* B_local = dBarray[batch] + Bi + Bj*lddb;
    scalar_t* W_local = dWarray[batch];

    if (!isUpper && !isLeft) { // lower right

        int step = (mb-1)/gridDim.y + 1;
        B_local += step * blockIdx.y;
        W_local += step * blockIdx.y;
        int mb_local = min(step, mb - step*blockIdx.y);

        scalar_t* U_local = dUarray[batch] + Ui + Ui*lddu;
        tb_gemm_magma(false, true, mb_local, nb, nb,
                alpha, B_local, lddb,
                       U_local, lddu,
                       W_local, lddw);
        //__syncthreads();
        tb_copy(mb_local, nb, W_local, lddw, B_local, lddb);

    } else if (!isUpper && isLeft) { // lower left

        //[nb-96, nb-97)
        int step = (nb-1)/gridDim.y + 1;
        B_local += step * blockIdx.y * lddb;
        W_local += step * blockIdx.y * lddw;
        int nb_local = min(step, nb - step*blockIdx.y);

        scalar_t* U_local = dUarray[batch] + Ui + Ui*lddu;
        tb_copy(mb, nb_local, B_local, lddb, W_local, lddw);
        //__syncthreads();
        tb_gemm_magma(true, false, mb, nb_local, mb,
                alpha, U_local, lddu,
                       W_local, lddw,
                       B_local, lddb);

    } else if (isUpper && !isLeft) { // upper right

        int step = (mb-1)/gridDim.y + 1;
        B_local += step * blockIdx.y;
        W_local += step * blockIdx.y;
        int mb_local = min(step, mb - step*blockIdx.y);

        scalar_t* VT_local = dVTarray[batch] + VTi + VTi*lddvt;
        real_t* S_local = dSarray[batch] + VTi;
        tb_gemm_magma(false, true, mb_local, nb, nb,
                alpha,  B_local, lddb,
                       VT_local, lddvt,
                        W_local, lddw);
        //__syncthreads();
        tb_scale_copy(false, mb_local, nb,
                    S_local,
                    W_local, lddw,
                    B_local, lddb);

    } else if (isUpper && isLeft) { // upper left

        int step = (nb-1)/gridDim.y + 1;
        B_local += step * blockIdx.y * lddb;
        W_local += step * blockIdx.y * lddw;
        int nb_local = min(step, nb - step*blockIdx.y);

        scalar_t* VT_local = dVTarray[batch] + VTi + VTi*lddvt;
        real_t* S_local = dSarray[batch] + VTi;
        tb_scale_copy(true, mb, nb_local,
                    S_local,
                    B_local, lddb,
                    W_local, lddw);
        //__syncthreads();
        tb_gemm_magma(true, false, mb, nb_local, mb,
                alpha, VT_local, lddvt,
                        W_local, lddw,
                        B_local, lddb);
    }
}

template <blas::Side side, blas::Uplo uplo, typename scalar_t>
void batch_trsm_addmod_diag(
    blas::Layout layout,
    //blas::Side   side,
    //blas::Uplo   uplo,
    int64_t      mb,
    int64_t      nb,
    scalar_t     alpha,
    std::vector<scalar_t*>& Uarray, int64_t Ui, int64_t Uj, int64_t lddu,
    std::vector<scalar_t*>& VTarray, int64_t VTi, int64_t VTj, int64_t lddvt,
    std::vector<blas::real_type<scalar_t>*>& Sarray, int64_t Si,
    std::vector<scalar_t*>& Barray, int64_t Bi, int64_t Bj, int64_t lddb,
    std::vector<scalar_t*>& Warray, int64_t lddw,
    const size_t batch,
    blas::Queue &queue )
{
    using real_t = blas::real_type<scalar_t>;

    assert(layout == blas::Layout::ColMajor);
    assert(blas::MaxBatchChunk >= batch);
    constexpr int isUpper = uplo == blas::Uplo::Upper;
    constexpr int isLeft  = side == blas::Side::Left;

    assert(Ui == Uj && VTi == VTj && VTi == Si);

    queue.work_ensure_size<void*>( 2*batch );

    scalar_t** dBarray = (scalar_t**)queue.work();
    scalar_t** dWarray = dBarray + batch;
    // need either U or (VT and S), can use the same dev ptr array
    scalar_t** dUarray = dWarray + batch;
    scalar_t** dVTarray = dUarray;
    real_t**   dSarray = (real_t**)(dUarray + batch);

    blas::device_setvector<scalar_t*>(batch, Barray.data(), 1, dBarray, 1, queue);
    blas::device_setvector<scalar_t*>(batch, Warray.data(), 1, dWarray, 1, queue);
    if (isUpper) {
        blas::device_setvector<scalar_t*>(batch, VTarray.data(), 1, dVTarray, 1, queue);
        blas::device_setvector<real_t*>  (batch, Sarray.data(), 1, dSarray, 1, queue);
    } else {
        blas::device_setvector<scalar_t*>(batch, Uarray.data(), 1, dUarray, 1, queue);
    }

    //int dimA = isLeft ? mb : nb;
    int nrhs = isLeft ? nb : mb;

    dim3 grid_dim;
    grid_dim.x = batch;
    grid_dim.y = (nrhs-1)/32 + 1;

    dim3 block_dim;
    block_dim.x = 16;
    block_dim.y = 16; //isLeft ? 16 : 8;

    int shmem_size = 0; //(isLeft ? 2*32*33 : 0)*sizeof(scalar_t);

    //printf("setting up CUDA call for subproblem at %ld of size %dx%d with %db type and %d batch entries\n", Ai, int(mb), int(nb), int(sizeof(scalar_t)), int(batch));

    batch_diag_kernel<isUpper, isLeft><<< grid_dim, block_dim, shmem_size, queue.stream() >>>(
                    mb, nb,
                    to_cutype(alpha),
                    to_cutype(dUarray),   Ui,   lddu,
                    to_cutype(dVTarray), VTi,   lddvt,
                    dSarray,
                    to_cutype(dBarray), Bi, Bj, lddb,
                    to_cutype(dWarray),         lddw);


    //printf("Kernel launch args:\n    grid size=(%d, %d, %d)\n    tb size=(%d, %d, %d)\n    shmem_size=%db\n", grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z, shmem_size);

    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess)
    //    printf("Error in kernel launch: %s\n", cudaGetErrorString(err));

    //cudaStreamSynchronize(queue.stream());
    //err = cudaGetLastError();
    //if (err != cudaSuccess)
    //    printf("Error in kernel execut: %s\n", cudaGetErrorString(err));

}

template <bool isLeft, typename scalar_t>
__global__ void __launch_bounds__(512,4) batch_trsm_scale_copy_kernel(
                    int64_t mb,
                    int64_t nb,
                    blas::real_type<scalar_t>** dSarray, int64_t Si,
                    scalar_t** dAarray, int64_t Ai, int64_t Aj, int64_t ldda,
                    scalar_t** dBarray, int64_t Bi, int64_t Bj, int64_t lddb)
{
    using real_t = blas::real_type<scalar_t>;

    int batch = blockIdx.x;
    int blkidx_m = blockIdx.y;
    int blksiz_m =  gridDim.y;
    int blkidx_n = blockIdx.z;
    int blksiz_n =  gridDim.z;

    int64_t step_m = (mb-1)/blksiz_m + 1;
    int64_t m_offset = step_m*blkidx_m;
    int64_t mb_local = min(step_m, mb - m_offset);

    int64_t step_n = (nb-1)/blksiz_n + 1;
    int64_t n_offset = step_n*blkidx_n;
    int64_t nb_local = min(step_n, nb - n_offset);


    real_t*   S_local = dSarray[batch] + Si + (isLeft ? m_offset : n_offset);
    scalar_t* A_local = dAarray[batch] + (Ai+m_offset) + (Aj+n_offset)*ldda;
    scalar_t* B_local = dBarray[batch] + (Bi+m_offset) + (Bj+n_offset)*lddb;

    tb_scale_copy(isLeft, mb_local, nb_local, S_local, A_local, ldda, B_local, lddb);
}

template <typename scalar_t>
void batch_scale_copy(
    blas::Layout layout,
    blas::Side side,
    int64_t      mb,
    int64_t      nb,
    std::vector<blas::real_type<scalar_t>*>& Sarray, int64_t Si,
    std::vector<scalar_t*>& Aarray, int64_t Ai, int64_t Aj, int64_t ldda,
    std::vector<scalar_t*>& Barray, int64_t Bi, int64_t Bj, int64_t lddb,
    const size_t batch,
    blas::Queue &queue )
{
    using real_t = blas::real_type<scalar_t>;

    assert(layout == blas::Layout::ColMajor);
    assert(blas::MaxBatchChunk >= batch);
    queue.work_ensure_size<void*>( 2*batch );

    scalar_t** dAarray = (scalar_t**)queue.work();
    scalar_t** dBarray = dAarray + batch;
    real_t**   dSarray = (real_t**)(dBarray + batch);

    blas::device_setvector<real_t*  >(batch, Sarray.data(), 1, dSarray, 1, queue);
    blas::device_setvector<scalar_t*>(batch, Aarray.data(), 1, dAarray, 1, queue);
    blas::device_setvector<scalar_t*>(batch, Barray.data(), 1, dBarray, 1, queue);

    dim3 grid_dim;
    grid_dim.x = batch;
    grid_dim.y = (mb-1)/64 + 1;
    grid_dim.z = (nb-1)/64 + 1;

    dim3 block_dim;
    block_dim.x = 32;
    block_dim.y = 16;

    if (side == blas::Side::Left) {
        batch_trsm_scale_copy_kernel<true><<< grid_dim, block_dim, 0, queue.stream()>>>(
                mb, nb,
                to_cutype(dSarray), Si,
                to_cutype(dAarray), Ai, Aj, ldda,
                to_cutype(dBarray), Bi, Bj, lddb);
    } else {
        batch_trsm_scale_copy_kernel<false><<< grid_dim, block_dim, 0, queue.stream()>>>(
                mb, nb,
                to_cutype(dSarray), Si,
                to_cutype(dAarray), Ai, Aj, ldda,
                to_cutype(dBarray), Bi, Bj, lddb);
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(512,4) batch_trsm_copy_kernel(
                    int64_t mb,
                    int64_t nb,
                    scalar_t** dAarray, int64_t Ai, int64_t Aj, int64_t ldda,
                    scalar_t** dBarray, int64_t Bi, int64_t Bj, int64_t lddb)
{
    using real_t = blas::real_type<scalar_t>;

    int batch = blockIdx.x;
    int blkidx_m = blockIdx.y;
    int blksiz_m =  gridDim.y;
    int blkidx_n = blockIdx.z;
    int blksiz_n =  gridDim.z;

    int64_t step_m = (mb-1)/blksiz_m + 1;
    int64_t m_offset = step_m*blkidx_m;
    int64_t mb_local = min(step_m, mb - m_offset);

    int64_t step_n = (nb-1)/blksiz_n + 1;
    int64_t n_offset = step_n*blkidx_n;
    int64_t nb_local = min(step_n, nb - n_offset);


    scalar_t* A_local = dAarray[batch] + (Ai+m_offset) + (Aj+n_offset)*ldda;
    scalar_t* B_local = dBarray[batch] + (Bi+m_offset) + (Bj+n_offset)*lddb;

    tb_copy(mb_local, nb_local, A_local, ldda, B_local, lddb);
}

template <typename scalar_t>
void batch_copy(
    blas::Layout layout,
    int64_t      mb,
    int64_t      nb,
    std::vector<scalar_t*>& Aarray, int64_t Ai, int64_t Aj, int64_t ldda,
    std::vector<scalar_t*>& Barray, int64_t Bi, int64_t Bj, int64_t lddb,
    const size_t batch,
    blas::Queue &queue )
{
    assert(layout == blas::Layout::ColMajor);
    assert(blas::MaxBatchChunk >= batch);
    queue.work_ensure_size<void*>( 2*batch );

    scalar_t** dAarray = (scalar_t**)queue.work();
    scalar_t** dBarray = dAarray + batch;

    blas::device_setvector<scalar_t*>(batch, Aarray.data(), 1, dAarray, 1, queue);
    blas::device_setvector<scalar_t*>(batch, Barray.data(), 1, dBarray, 1, queue);

    dim3 grid_dim;
    grid_dim.x = batch;
    grid_dim.y = (mb-1)/64 + 1;
    grid_dim.z = (nb-1)/64 + 1;

    dim3 block_dim;
    block_dim.x = 32;
    block_dim.y = 16;

    batch_trsm_copy_kernel<<< grid_dim, block_dim, 0, queue.stream()>>>(
            mb, nb,
            to_cutype(dAarray), Ai, Aj, ldda,
            to_cutype(dBarray), Bi, Bj, lddb);
}


template <BlockFactor factorType, typename scalar_t>
void batch_trsm_addmod_rec(
    blas::Layout layout,
    blas::Side   side,
    blas::Uplo   uplo,
    int64_t      mb,
    int64_t      nb,
    int64_t      ib,
    scalar_t     alpha,
    std::vector<scalar_t*>&  Aarray, int64_t  Ai, int64_t  Aj, int64_t ldda,
    std::vector<scalar_t*>&  Uarray, int64_t  Ui, int64_t  Uj, int64_t lddu,
    std::vector<scalar_t*>& VTarray, int64_t VTi, int64_t VTj, int64_t lddvt,
    std::vector<blas::real_type<scalar_t>*>& Sarray, int64_t Si,
    std::vector<scalar_t*>&  Barray, int64_t  Bi, int64_t  Bj, int64_t lddb,
    std::vector<scalar_t*>&  Warray,
    const size_t batch,
    blas::Queue &queue )
{
    scalar_t one  = 1.0;
    scalar_t zero = 0.0;

    constexpr int64_t cublas_threshold = 80;

    bool isUpper = uplo == blas::Uplo::Upper;
    bool isLeft  = side == blas::Side::Left;

    int64_t lddw = (layout == blas::Layout::ColMajor) ? mb : nb;

    blas::Op trans_op = std::is_same<scalar_t, blas::real_type<scalar_t>>::value ? blas::Op::Trans : blas::Op::ConjTrans;

    if (isUpper && isLeft) {
        if (mb <= ib) {
            // halt recursion
            if constexpr (factorType == BlockFactor::SVD) {
                if (ib < cublas_threshold) {
                    batch_trsm_addmod_diag<blas::Side::Left, blas::Uplo::Upper>(
                                layout, mb, nb, alpha,
                                Uarray,  Ui,  Uj, lddu,
                               VTarray, VTi, VTj, lddvt,
                                Sarray,  Si,
                                Barray,  Bi,  Bj, lddb,
                                Warray,           lddw,
                                batch, queue);
                } else {
                    batch_scale_copy(layout, side, mb, nb,
                            Sarray, Si,
                            Barray, Bi, Bj, lddb,
                            Warray,  0,  0, lddw,
                            batch, queue);
                    batch_gemm(layout, trans_op, blas::Op::NoTrans, mb, nb, mb,
                            alpha, VTarray, VTi, VTj, lddvt,
                                    Warray,   0,   0, lddw,
                            zero,   Barray,  Bi,  Bj, lddb,
                            batch, queue);
                }
            }
            else if constexpr (factorType == BlockFactor::QLP
                          || factorType == BlockFactor::QRCP) {
                auto uplo = factorType == BlockFactor::QLP ? blas::Uplo::Lower : blas::Uplo::Upper;
                batch_trsm(layout, blas::Side::Left, uplo, blas::Op::NoTrans, blas::Diag::NonUnit, mb, nb,
                        one, Aarray, Ai, Aj, ldda,
                             Barray, Bi, Bj, lddb,
                        batch, queue);
                batch_copy(layout, mb, nb,
                        Barray, Bi, Bj, lddb,
                        Warray,  0,  0, lddw,
                        batch, queue);
                batch_gemm(layout, trans_op, blas::Op::NoTrans, mb, nb, mb,
                        alpha, VTarray, VTi, VTj, lddvt,
                                Warray,   0,   0, lddw,
                        zero,   Barray,  Bi,  Bj, lddb,
                        batch, queue);
            }
            else if constexpr (factorType == BlockFactor::QR) {
                batch_trsm(layout, blas::Side::Left, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, mb, nb,
                        one, Aarray, Ai, Aj, ldda,
                             Barray, Bi, Bj, lddb,
                        batch, queue);
            }
            else {
                static_assert(factorType == BlockFactor::SVD, "Unsupported block factor");
            }
        }
        else {
            // recurse
            int64_t m1 = (((mb-1)/ib+1)/2) * ib; // half the tiles, rounded down
            int64_t m2 = mb-m1;

            batch_trsm_addmod_rec<factorType>(layout, side, uplo, m2, nb, ib, alpha,
                    Aarray,  Ai+m1,  Aj+m1, ldda,
                    Uarray,  Ui+m1,  Uj+m1, lddu,
                   VTarray, VTi+m1, VTj+m1, lddvt,
                    Sarray,  Si+m1,
                    Barray,  Bi+m1,  Bj,    lddb,
                    Warray, batch, queue);

            batch_gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, m1, nb, m2,
                    -one,  Aarray, Ai,    Aj+m1, ldda,
                           Barray, Bi+m1, Bj,    lddb,
                    alpha, Barray, Bi,    Bj,    lddb,
                    batch, queue); 

            batch_trsm_addmod_rec<factorType>(layout, side, uplo, m1, nb, ib, one,
                    Aarray,  Ai,  Aj, ldda,
                    Uarray,  Ui,  Uj, lddu,
                   VTarray, VTi, VTj, lddvt,
                    Sarray,  Si,
                    Barray,  Bi,  Bj, lddb,
                    Warray, batch, queue);
        }
    }
    else if (isUpper && !isLeft) {
        if (nb <= ib) {
            // halt recursion
            if constexpr (factorType == BlockFactor::SVD) {
                if (ib < cublas_threshold) {
                    batch_trsm_addmod_diag<blas::Side::Right, blas::Uplo::Upper>(
                                layout, mb, nb, alpha,
                                Uarray,  Ui,  Uj, lddu,
                               VTarray, VTi, VTj, lddvt,
                                Sarray,  Si,
                                Barray,  Bi,  Bj, lddb,
                                Warray,           lddw,
                                batch, queue);
                } else {
                    batch_gemm(layout, blas::Op::NoTrans, trans_op, mb, nb, nb,
                            alpha,  Barray,  Bi,  Bj, lddb,
                                   VTarray, VTi, VTj, lddvt,
                            zero,   Warray,   0,   0, lddw,
                            batch, queue);
                    batch_scale_copy(layout, side, mb, nb,
                            Sarray, Si,
                            Warray,  0,  0, lddw,
                            Barray, Bi, Bj, lddb,
                            batch, queue);
                }
            }
            else if constexpr (factorType == BlockFactor::QLP
                          || factorType == BlockFactor::QRCP) {
                batch_gemm(layout, blas::Op::NoTrans, trans_op, mb, nb, nb,
                        alpha,  Barray,  Bi,  Bj, lddb,
                               VTarray, VTi, VTj, lddvt,
                        zero,   Warray,   0,   0, lddw,
                        batch, queue);
                batch_copy(layout, mb, nb,
                        Warray,  0,  0, lddw,
                        Barray, Bi, Bj, lddb,
                        batch, queue);
                auto uplo = factorType == BlockFactor::QLP ? blas::Uplo::Lower : blas::Uplo::Upper;
                batch_trsm(layout, blas::Side::Right, uplo, blas::Op::NoTrans, blas::Diag::NonUnit, mb, nb,
                        one, Aarray, Ai, Aj, ldda,
                             Barray, Bi, Bj, lddb,
                        batch, queue);
            }
            else if constexpr (factorType == BlockFactor::QR) {
                batch_trsm(layout, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, mb, nb,
                        one, Aarray, Ai, Aj, ldda,
                             Barray, Bi, Bj, lddb,
                        batch, queue);
            }
            else {
                static_assert(factorType == BlockFactor::SVD, "Unsupported block factor");
            }
        }
        else {
            // recurse
            int64_t n1 = (((nb-1)/ib)/2+1) * ib; // half the tiles, rounded up
            int64_t n2 = nb-n1;

            batch_trsm_addmod_rec<factorType>(layout, side, uplo, mb, n1, ib, alpha,
                    Aarray, Ai, Aj, ldda,
                    Uarray, Ui, Uj, lddu,
                   VTarray, VTi, VTj, lddvt,
                    Sarray, Si,
                    Barray, Bi, Bj,    lddb,
                    Warray, batch, queue);

            batch_gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, mb, n2, n1,
                    -one,  Barray, Bi, Bj,    lddb,
                           Aarray, Ai, Aj+n1, ldda,
                    alpha, Barray, Bi, Bj+n1, lddb,
                    batch, queue); 

            batch_trsm_addmod_rec<factorType>(layout, side, uplo, mb, n2, ib, one,
                    Aarray,  Ai+n1,  Aj+n1, ldda,
                    Uarray,  Ui+n1,  Uj+n1, lddu,
                   VTarray, VTi+n1, VTj+n1, lddvt,
                    Sarray,  Si+n1,
                    Barray,  Bi,     Bj+n1, lddb,
                    Warray, batch, queue);
        }
    }
    else if (!isUpper && isLeft) {
        if (mb <= ib) {
            // halt recursion
            if constexpr (factorType == BlockFactor::SVD
                          || factorType == BlockFactor::QLP
                          || factorType == BlockFactor::QRCP
                          || factorType == BlockFactor::QR) {
                if (ib < cublas_threshold) {
                    batch_trsm_addmod_diag<blas::Side::Left, blas::Uplo::Lower>(
                                layout, /*side, uplo,*/ mb, nb, alpha,
                                Uarray,  Ui,  Uj, lddu,
                               VTarray, VTi, VTj, lddvt,
                                Sarray,  Si,
                                Barray,  Bi,  Bj, lddb,
                                Warray, lddw,
                                batch, queue);
                } else {
                    batch_copy(layout, mb, nb,
                            Barray, Bi, Bj, lddb,
                            Warray,  0,  0, lddw,
                            batch, queue);
                    batch_gemm(layout, trans_op, blas::Op::NoTrans, mb, nb, mb,
                            alpha, Uarray, Ui, Uj, lddu,
                                   Warray,  0,  0, lddw,
                            zero,  Barray, Bi, Bj, lddb,
                            batch, queue);
                }
            }
            else {
                static_assert(factorType == BlockFactor::SVD, "Unsupported block factor");
            }
        }
        else {
            // recurse
            int64_t m1 = (((mb-1)/ib)/2+1) * ib; // half the tiles, rounded up
            int64_t m2 = mb-m1;

            batch_trsm_addmod_rec<factorType>(layout, side, uplo, m1, nb, ib, alpha,
                    Aarray,  Ai,  Aj, ldda,
                    Uarray,  Ui,  Uj, lddu,
                   VTarray, VTi, VTj, lddvt,
                    Sarray,  Si,
                    Barray,  Bi,  Bj, lddb,
                    Warray, batch, queue);

            batch_gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, m2, nb, m1,
                    -one,  Aarray, Ai+m1, Aj, ldda,
                           Barray, Bi,    Bj, lddb,
                    alpha, Barray, Bi+m1, Bj, lddb,
                    batch, queue); 

            batch_trsm_addmod_rec<factorType>(layout, side, uplo, m2, nb, ib, one,
                     Aarray,  Ai+m1,  Aj+m1, ldda,
                     Uarray,  Ui+m1,  Uj+m1, lddu,
                    VTarray, VTi+m1, VTj+m1, lddvt,
                     Sarray,  Si+m1,
                     Barray,  Bi+m1,  Bj,    lddb,
                     Warray, batch, queue);
        }
    }
    else if (!isUpper && !isLeft) {
        if (nb <= ib) {
            // halt recursion
            if constexpr (factorType == BlockFactor::SVD
                          || factorType == BlockFactor::QLP
                          || factorType == BlockFactor::QRCP
                          || factorType == BlockFactor::QR) {
                if (ib < cublas_threshold) {
                    batch_trsm_addmod_diag<blas::Side::Right, blas::Uplo::Lower>(
                                layout, /*side, uplo,*/ mb, nb, alpha,
                                Uarray,  Ui,  Uj, lddu,
                               VTarray, VTi, VTj, lddvt,
                                Sarray,  Si,
                                Barray,  Bi,  Bj, lddb,
                                Warray, lddw,
                                batch, queue);
                } else {
                    batch_gemm(layout, blas::Op::NoTrans, trans_op, mb, nb, nb,
                            alpha, Barray, Bi, Bj, lddb,
                                   Uarray, Ui, Uj, ldda,
                            zero,  Warray,  0,  0, lddw,
                            batch, queue);
                    batch_copy(layout, mb, nb,
                            Warray,  0,  0, lddw,
                            Barray, Bi, Bj, lddb,
                            batch, queue);
                }
            }
            else {
                static_assert(factorType == BlockFactor::SVD, "Unsupported block factor");
            }
        }
        else {
            // recurse
            int64_t n1 = (((nb-1)/ib+1)/2) * ib; // half the tiles, rounded down
            int64_t n2 = nb-n1;

            batch_trsm_addmod_rec<factorType>(layout, side, uplo, mb, n2, ib, alpha,
                    Aarray,  Ai+n1,  Aj+n1, ldda,
                    Uarray,  Ui+n1,  Uj+n1, lddu,
                   VTarray, VTi+n1, VTj+n1, lddvt,
                    Sarray,  Si+n1,
                    Barray,  Bi,     Bj+n1, lddb,
                    Warray, batch, queue);

            batch_gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, mb, n1, n2,
                    -one,  Barray, Bi,    Bj+n1, lddb,
                           Aarray, Ai+n1, Aj,    ldda,
                    alpha, Barray, Bi,    Bj,    lddb,
                    batch, queue); 

            batch_trsm_addmod_rec<factorType>(layout, side, uplo, mb, n1, ib, one,
                    Aarray,  Ai,  Aj, ldda,
                    Uarray,  Ui,  Uj, lddu,
                   VTarray, VTi, VTj, lddvt,
                    Sarray,  Si,
                    Barray,  Bi,  Bj, lddb,
                    Warray, batch, queue);
        }
    }
}



template <typename scalar_t>
void batch_trsm_addmod(
    BlockFactor factorType,
    blas::Layout layout,
    blas::Side   side,
    blas::Uplo   uplo,
    int64_t      mb,
    int64_t      nb,
    int64_t      ib,
    scalar_t     alpha,
    std::vector<scalar_t*>   Aarray, int64_t ldda,
    std::vector<scalar_t*>   Uarray, int64_t lddu,
    std::vector<scalar_t*>  VTarray, int64_t lddvt,
    std::vector<blas::real_type<scalar_t>*>   Sarray,
    std::vector<scalar_t*>   Barray, int64_t lddb,
    std::vector<scalar_t*>   Warray,
    const size_t batch,
    blas::Queue &queue )
{
    // TODO could assume A, U, S are shared between all thread blocks

    if (factorType == BlockFactor::SVD) {
        batch_trsm_addmod_rec<BlockFactor::SVD>(layout, side, uplo, mb, nb, ib, alpha,
                Aarray, 0, 0, ldda,
                Uarray, 0, 0, lddu,
               VTarray, 0, 0, lddvt,
                Sarray, 0,
                Barray, 0, 0, lddb,
                Warray, batch, queue);
    }
    else if (factorType == BlockFactor::QLP) {
        batch_trsm_addmod_rec<BlockFactor::QLP>(layout, side, uplo, mb, nb, ib, alpha,
                Aarray, 0, 0, ldda,
                Uarray, 0, 0, lddu,
               VTarray, 0, 0, lddvt,
                Sarray, 0,
                Barray, 0, 0, lddb,
                Warray, batch, queue);
    }
    else if (factorType == BlockFactor::QRCP) {
        batch_trsm_addmod_rec<BlockFactor::QRCP>(layout, side, uplo, mb, nb, ib, alpha,
                Aarray, 0, 0, ldda,
                Uarray, 0, 0, lddu,
               VTarray, 0, 0, lddvt,
                Sarray, 0,
                Barray, 0, 0, lddb,
                Warray, batch, queue);
    }
    else if (factorType == BlockFactor::QR) {
        batch_trsm_addmod_rec<BlockFactor::QR>(layout, side, uplo, mb, nb, ib, alpha,
                Aarray, 0, 0, ldda,
                Uarray, 0, 0, lddu,
               VTarray, 0, 0, lddvt,
                Sarray, 0,
                Barray, 0, 0, lddb,
                Warray, batch, queue);
    }
    else {
        slate_not_implemented("Only SVD is supported on device");
    }
}

// Explicit instantiation
template
void batch_trsm_addmod(
    BlockFactor factorType,
    blas::Layout layout,
    blas::Side   side,
    blas::Uplo   uplo,
    int64_t      mb,
    int64_t      nb,
    int64_t      ib,
    float        alpha,
    std::vector<float*>      Aarray, int64_t ldda,
    std::vector<float*>      Uarray, int64_t lddu,
    std::vector<float*>     VTarray, int64_t lddvt,
    std::vector<float*>      Sarray,
    std::vector<float*>      Barray, int64_t lddb,
    std::vector<float*>      Warray,
    const size_t batch,
    blas::Queue &queue );

template
void batch_trsm_addmod(
    BlockFactor factorType,
    blas::Layout layout,
    blas::Side   side,
    blas::Uplo   uplo,
    int64_t      mb,
    int64_t      nb,
    int64_t      ib,
    double       alpha,
    std::vector<double*>     Aarray, int64_t ldda,
    std::vector<double*>     Uarray, int64_t lddu,
    std::vector<double*>    VTarray, int64_t lddvt,
    std::vector<double*>     Sarray,
    std::vector<double*>     Barray, int64_t lddb,
    std::vector<double*>     Warray,
    const size_t batch,
    blas::Queue &queue );

template
void batch_trsm_addmod(
    BlockFactor factorType,
    blas::Layout layout,
    blas::Side   side,
    blas::Uplo   uplo,
    int64_t      mb,
    int64_t      nb,
    int64_t      ib,
    std::complex<float>   alpha,
    std::vector<std::complex<float>*>  Aarray, int64_t ldda,
    std::vector<std::complex<float>*>  Uarray, int64_t lddu,
    std::vector<std::complex<float>*> VTarray, int64_t lddvt,
    std::vector<float*>                Sarray,
    std::vector<std::complex<float>*>  Barray, int64_t lddb,
    std::vector<std::complex<float>*>  Warray,
    const size_t batch,
    blas::Queue &queue );

template
void batch_trsm_addmod(
    BlockFactor factorType,
    blas::Layout layout,
    blas::Side   side,
    blas::Uplo   uplo,
    int64_t      mb,
    int64_t      nb,
    int64_t      ib,
    std::complex<double>   alpha,
    std::vector<std::complex<double>*>  Aarray, int64_t ldda,
    std::vector<std::complex<double>*>  Uarray, int64_t lddu,
    std::vector<std::complex<double>*> VTarray, int64_t lddvt,
    std::vector<double*>                Sarray,
    std::vector<std::complex<double>*>  Barray, int64_t lddb,
    std::vector<std::complex<double>*>  Warray,
    const size_t batch,
    blas::Queue &queue );

} // namespace device
} // namespace slate
