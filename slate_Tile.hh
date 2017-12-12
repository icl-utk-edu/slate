
#ifndef SLATE_TILE_HH
#define SLATE_TILE_HH

#include "slate_Memory.hh"

#include "blas.hh"
#include "lapack.hh"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#ifdef SLATE_WITH_CUDA
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
#endif

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#ifdef SLATE_WITH_OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

extern "C" void trace_cpu_start();
extern "C" void trace_cpu_stop(const char *color);

namespace slate {

//------------------------------------------------------------------------------
template <typename FloatType>
class Tile {
public:
    Tile(const Tile<FloatType> *src_tile, int dst_device_num)
    {
        *this = *src_tile;
        device_num_ = dst_device_num;
        allocate();
    }
    Tile(int64_t mb, int64_t nb, Memory *memory)
        : mb_(mb), nb_(nb), memory_(memory), device_num_(host_num_), life_(0)
    {
        allocate();
    }
    ~Tile() {
        deallocate();
    }

    //-------------------------------------------------
    virtual void copyTo(FloatType *a, int64_t lda) = 0;
    virtual void copyFrom(FloatType *a, int64_t lda) = 0;

    virtual Tile<FloatType>* copyToHost(cudaStream_t stream) = 0;
    virtual Tile<FloatType>* copyToDevice(
        int device_num, cudaStream_t stream) = 0;

    //--------------------------------------------------------------
    Tile<FloatType>* copyDataToHost(const Tile<FloatType> *dst_tile,
                                    cudaStream_t stream)
    {
        trace_cpu_start();
        cudaError_t error;
        error = cudaSetDevice(device_num_);
        assert(error == cudaSuccess);

        error = cudaMemcpyAsync(dst_tile->data_, data_, size(),
                                cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);

        error = cudaStreamSynchronize(stream);
        assert(error == cudaSuccess);
        trace_cpu_stop("Gray");
    }
    Tile<FloatType>* copyDataToDevice(const Tile<FloatType> *dst_tile,
                                      cudaStream_t stream)
    {
        trace_cpu_start();
        cudaError_t error;
        error = cudaSetDevice(dst_tile->device_num_);
        assert(error == cudaSuccess);

        error = cudaMemcpyAsync(dst_tile->data_, data_, size(),
                                cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);

        error = cudaStreamSynchronize(stream);
        assert(error == cudaSuccess);
        trace_cpu_stop("LightGray");
    }

    //----------------------------------------------------------
    static void gemm(blas::Op transa, blas::Op transb,
                     FloatType alpha, Tile<FloatType> *a,
                                      Tile<FloatType> *b,
                     FloatType beta,  Tile<FloatType> *c)
    {
        trace_cpu_start();
        blas::gemm(blas::Layout::ColMajor, transa, transb,
                   c->mb_, c->nb_, a->nb_,
                   alpha, a->data_, a->mb_,
                          b->data_, b->mb_,
                   beta,  c->data_, c->mb_);
        trace_cpu_stop("MediumAquamarine");
        a->tick();
        b->tick();
    }
    static void potrf(blas::Uplo uplo, Tile<FloatType> *a)
    {
        trace_cpu_start();
        lapack::potrf(blas::Layout::ColMajor, uplo, a->nb_, a->data_, a->nb_);
        trace_cpu_stop("RosyBrown");
    }
    static void syrk(blas::Uplo uplo, blas::Op trans,
                     FloatType alpha, Tile<FloatType> *a,
                     FloatType beta,  Tile<FloatType> *c)
    {
        trace_cpu_start();
        blas::syrk(blas::Layout::ColMajor, uplo, trans,
                   c->nb_, a->nb_,
                   alpha, a->data_, a->mb_,
                   beta,  c->data_, c->mb_);
        trace_cpu_stop("CornflowerBlue");
        a->tick();
        a->tick();
    }
    static void trsm(blas::Side side, blas::Uplo uplo,
                     blas::Op transa, blas::Diag diag,
                     FloatType alpha, Tile<FloatType> *a,
                                      Tile<FloatType> *b)
    {
        trace_cpu_start();
        blas::trsm(blas::Layout::ColMajor, side, uplo, transa, diag,
                   b->mb_, b->nb_,
                   alpha, a->data_, a->mb_,
                          b->data_, b->mb_);
        trace_cpu_stop("MediumPurple");
        a->tick();
    }

    void tick()
    {
        if (!local_)
            #pragma omp critical(slate_tile)
            {
                if (--life_ == 0)
                    deallocate();
            }
    }

    int64_t mb_;
    int64_t nb_;

    FloatType *data_;

    bool local_ = true;
    int64_t life_;

    MPI_Request bcast_request_;
    MPI_Group bcast_group_;
    MPI_Comm bcast_comm_;

protected:
    size_t size() {
        return sizeof(FloatType)*mb_*nb_;
    }
    void allocate()
    {
        trace_cpu_start();
        data_ = (FloatType*)memory_->alloc(device_num_);
        trace_cpu_stop("Orchid");
    }
    void deallocate()
    {
        trace_cpu_start();
        memory_->free(data_, device_num_);
        data_ = nullptr;
        trace_cpu_stop("Crimson");
    }

    void print()
    {
        for (int64_t m = 0; m < mb_; ++m) {
            for (int64_t n = 0; n < nb_; ++n) {
                printf("%8.2lf", data_[mb_*n+m]);
            }
            printf("\n");
        }
        printf("\n");
    }

    static int host_num_;
    int device_num_;

    Memory *memory_;
};

} // namespace slate

#endif // SLATE_TILE_HH
