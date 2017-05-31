
#ifndef SLATE_TILE_HH
#define SLATE_TILE_HH

#include "Ccblas.hh"
#include "Cclapack.hh"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>

extern "C" void trace_cpu_start();
extern "C" void trace_cpu_stop(const char *color);

namespace Slate {

//------------------------------------------------------------------------------
template<class FloatType>
class Tile {
public:
    int64_t mb_;
    int64_t nb_;

    FloatType *data_;
    FloatType *packed_a_;
    FloatType *packed_b_;

    int64_t packed_a_life_;
    int64_t packed_b_life_;

    //------------------------------------------------------
    void allocate(int64_t mb, int64_t nb)
    {
        data_ = (FloatType*)malloc(sizeof(FloatType)*mb*nb);
        assert(data_ != nullptr);        
    }
    void copyTo(FloatType *a, int64_t lda)
    {
        for (int64_t n = 0; n < nb_; ++n)
            memcpy(&data_[n*mb_], &a[n*lda], sizeof(FloatType)*mb_);
    }
    void copyFrom(FloatType *a, int64_t lda)
    {
        for (int64_t n = 0; n < nb_; ++n)
            memcpy(&a[n*lda], &data_[n*mb_], sizeof(FloatType)*mb_);
    }
    void pack_a(int64_t life) {
        // trace_cpu_start();
        // packed_a_ = cblas_dgemm_alloc(CblasAMatrix, mb_, nb_, mb_);
        // cblas_dgemm_pack(CblasColMajor, CblasAMatrix, CblasNoTrans,
        //                  mb_, nb_, mb_, -1.0, data_, mb_, packed_a_);
        // packed_a_life_ = life;
        // trace_cpu_stop("Black");
    }
    void pack_b(int64_t life) {
        // trace_cpu_start();
        // packed_b_ = cblas_dgemm_alloc(CblasBMatrix, mb_, nb_, mb_);
        // cblas_dgemm_pack(CblasColMajor, CblasBMatrix, CblasTrans,
        //                  mb_, nb_, mb_, 1.0, data_, mb_, packed_b_);
        // packed_b_life_ = life;
        // trace_cpu_stop("Black");
    }

    Tile(int64_t mb, int64_t nb) : mb_(mb), nb_(nb) {
        allocate(mb, nb);
    }
    Tile(int64_t mb, int64_t nb, FloatType *a, int64_t lda) : mb_(mb), nb_(nb)
    {
        allocate(mb, nb);
        copyTo(a, lda);
    }
    ~Tile() {
        free(data_);
    }

    //------------------------------------------------------
    void gemm(Ccblas::Tran transa, Ccblas::Tran transb, FloatType alpha,
              Tile<FloatType> *a, Tile<FloatType> *b, FloatType beta)
    {
        trace_cpu_start();
        Ccblas::gemm(Ccblas::Order::ColMajor, transa, transb,
                     mb_, nb_, a->nb_, alpha, a->data_, a->mb_,
                     b->data_, b->mb_, beta, data_, mb_);
        // cblas_dgemm_compute(CblasColMajor, CblasPacked, CblasPacked,
        //     mb_, nb_, a->nb_, a->packed_a_, a->mb_, b->packed_b_, b->mb_,
        //     beta, data_, mb_);   

        // #pragma omp critical
        // {
        //     --a->packed_a_life_;
        //     if (a->packed_a_life_ == 0)
        //         cblas_dgemm_free(a->packed_a_);
        // }
        // #pragma omp critical
        // {
        //     --b->packed_b_life_;
        //     if (b->packed_b_life_ == 0)
        //         cblas_dgemm_free(b->packed_b_);
        // }
        trace_cpu_stop("MediumAquamarine");
    }
    void potrf(Ccblas::Uplo uplo)
    {
        trace_cpu_start();
        Cclapack::potrf(Ccblas::Order::ColMajor, uplo, nb_, data_, nb_);
        trace_cpu_stop("RosyBrown");
    }
    void syrk(Ccblas::Uplo uplo, Ccblas::Tran trans,
              FloatType alpha, Tile<FloatType> *a, FloatType beta)
    {
        trace_cpu_start();
        Ccblas::syrk(Ccblas::Order::ColMajor, uplo, trans,
                     nb_, a->nb_, alpha, a->data_, a->mb_, beta, data_, mb_);
        trace_cpu_stop("CornflowerBlue");
    }
    void trsm(Ccblas::Side side, Ccblas::Uplo uplo, Ccblas::Tran transa,
              Ccblas::Diag diag, FloatType alpha, Tile<FloatType> *a)
    {
        trace_cpu_start();
        Ccblas::trsm(Ccblas::Order::ColMajor, side, uplo, transa, diag,
                     mb_, nb_, alpha, a->data_, mb_, data_, mb_);
        trace_cpu_stop("MediumPurple");
    }

    //------------------------------------------------------
    void print()
    {
        for (int64_t m = 0; m < mb_; ++m) {
            for (int64_t n = 0; n < nb_; ++n) {
                printf("%8.2lf", data_[(size_t)mb_*n+m]);
            }
            printf("\n");
        }
        printf("\n");
    }
};

} // namespace Slate

#endif // SLATE_TILE_HH
