
#ifndef SLATE_MATRIX_HH
#define SLATE_MATRIX_HH

#include "slate_Tile.hh"

#include <functional>
#include <map>
#include <utility>

#include <mpi.h>
#include <omp.h>

namespace slate {

//------------------------------------------------------------------------------
template<class FloatType>
class Matrix {
public:
    int64_t it_; ///< first row of tiles
    int64_t jt_; ///< first column of tiles
    int64_t mt_; ///< number of tile rows
    int64_t nt_; ///< number of tile columns

    // TODO: replace by unordered_map
    std::map<std::pair<int64_t, int64_t>, Tile<FloatType>*> *tiles_;

    Matrix(int64_t m, int64_t n, double *a, int64_t lda,
           int64_t mb, int64_t nb);

    Matrix(int64_t m, int64_t n, double *a, int64_t lda,
           int64_t mb, int64_t nb, MPI_Comm mpi_comm, int64_t p, int64_t q);

    Matrix(const Matrix &a, int64_t it, int64_t jt, int64_t mt, int64_t nt);

    void copyTo(int64_t m, int64_t n, FloatType *a, int64_t lda,
                int64_t mb, int64_t nb);

    void copyFrom(int64_t m, int64_t n, FloatType *a, int64_t lda,
                  int64_t mb, int64_t nb); 

    Tile<FloatType>* &operator()(int64_t m, int64_t n) {
        return (*tiles_)[std::pair<int64_t, int64_t>(it_+m, jt_+n)];
    }
    Tile<FloatType>* &operator()(int64_t m, int64_t n) const {
        return (*tiles_)[std::pair<int64_t, int64_t>(it_+m, jt_+n)];
    }

    void trsm(blas::Side side, blas::Uplo uplo,
              blas::Op trans, blas::Diag diag,
              FloatType alpha, const Matrix &a);

    void potrf(blas::Uplo uplo, int64_t lookahead=1);

private:
    MPI_Comm mpi_comm_;
    int64_t mpi_size_;
    int64_t mpi_rank_;
    std::function <int64_t (int64_t m, int64_t n)> tileLocation;

    bool tileIsLocal(int64_t m, int64_t n) {
        return tileLocation(m, n) == mpi_rank_;
    }

    void syrkTask(blas::Uplo uplo, blas::Op trans,
                  FloatType alpha, const Matrix &a, FloatType beta);

    void syrkNest(blas::Uplo uplo, blas::Op trans,
                  FloatType alpha, const Matrix &a, FloatType beta);

    void syrkBatch(blas::Uplo uplo, blas::Op trans,
                   FloatType alpha, const Matrix &a, FloatType beta);
};

//------------------------------------------------------------------------------
template<class FloatType>
Matrix<FloatType>::Matrix(int64_t m, int64_t n, double *a, int64_t lda,
                          int64_t mb, int64_t nb)
{
    tiles_ = new std::map<std::pair<int64_t, int64_t>, Tile<FloatType>*>;
    it_ = 0;
    jt_ = 0;
    mt_ = m % mb == 0 ? m/mb : m/mb+1;
    nt_ = n % nb == 0 ? n/nb : n/nb+1;

    tileLocation = [] (int64_t m, int64_t n) { return 0; };

    copyTo(m, n, a, lda, mb, nb);
}

//------------------------------------------------------------------------------
template<class FloatType>
Matrix<FloatType>::Matrix(int64_t m, int64_t n, double *a, int64_t lda,
                          int64_t mb, int64_t nb,
                          MPI_Comm mpi_comm, int64_t p, int64_t q)
{
    tiles_ = new std::map<std::pair<int64_t, int64_t>, Tile<FloatType>*>;
    it_ = 0;
    jt_ = 0;
    mt_ = m % mb == 0 ? m/mb : m/mb+1;
    nt_ = n % nb == 0 ? n/nb : n/nb+1;

    mpi_comm_ = mpi_comm;
    int rank;
    int size;
    assert(MPI_Comm_rank(mpi_comm_, &rank) == MPI_SUCCESS);
    assert(MPI_Comm_size(mpi_comm_, &size) == MPI_SUCCESS);
    mpi_rank_ = rank;
    mpi_size_ = size;

    tileLocation = [=] (int64_t m, int64_t n) {
        return ((it_+m)%p) + ((jt_+n)%q) * p;
    };

    copyTo(m, n, a, lda, mb, nb);
}

//------------------------------------------------------------------------------
template<class FloatType>
Matrix<FloatType>::Matrix(const Matrix &a, int64_t it, int64_t jt,
                          int64_t mt, int64_t nt)
{
    assert(it+mt <= a.mt_);
    assert(jt+nt <= a.nt_);
    *this = a;
    it_ += it;
    jt_ += jt;
    mt_ = mt;
    nt_ = nt;
}

//------------------------------------------------------------------------------
template<class FloatType>
void Matrix<FloatType>::copyTo(int64_t m, int64_t n, FloatType *a,
                               int64_t lda, int64_t mb, int64_t nb)
{
    for (int64_t i = 0; i < m; i += mb)
        for (int64_t j = 0; j < n; j += nb)
            if (j <= i)
                (*this)(i/mb, j/nb) =
                    new Tile<FloatType>(std::min(mb, m-i), std::min(nb, n-j),
                                        &a[(size_t)lda*j+i], lda);
}

//------------------------------------------------------------------------------
template<class FloatType>
void Matrix<FloatType>::copyFrom(int64_t m, int64_t n, FloatType *a,
                                 int64_t lda, int64_t mb, int64_t nb)
{
    for (int64_t i = 0; i < m; i += mb)
        for (int64_t j = 0; j < n; j += nb)
            if (j <= i)
                (*this)(i/mb, j/nb)->copyFrom(&a[(size_t)lda*j+i], lda);
}

//------------------------------------------------------------------------------
template<class FloatType>
void Matrix<FloatType>::syrkTask(blas::Uplo uplo, blas::Op trans,
                                 FloatType alpha, const Matrix &a,
                                 FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;

    // Lower, NoTrans
    for (int64_t n = 0; n < nt_; ++n) {

        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);

        for (int64_t m = n+1; m < mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                #pragma omp task
                c(m, n)->gemm(trans, Op::Trans,
                              alpha, a(m, k), a(n, k), k == 0 ? beta : 1.0);
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<class FloatType>
void Matrix<FloatType>::syrkNest(blas::Uplo uplo, blas::Op trans,
                                 FloatType alpha, const Matrix &a,
                                 FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;

    for (int64_t n = 0; n < nt_; ++n) {
        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
    }

//  #pragma omp parallel for collapse(3) schedule(dynamic, 1) num_threads(60)
    #pragma omp parallel for collapse(3) schedule(dynamic, 1)
    for (int64_t n = 0; n < nt_; ++n) {
        for (int64_t m = 0; m < mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (m >= n+1)
                    c(m, n)->gemm(trans, Op::Trans,
                                  alpha, a(m, k), a(n, k), k == 0 ? beta : 1.0);
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<class FloatType>
void Matrix<FloatType>::syrkBatch(blas::Uplo uplo, blas::Op trans,
                                  FloatType alpha, const Matrix &a,
                                  FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;

    // Lower, NoTrans
    for (int64_t n = 0; n < nt_; ++n) {
        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
    }

    CBLAS_TRANSPOSE transa_array[1];
    CBLAS_TRANSPOSE transb_array[1];
    int m_array[1];
    int n_array[1];
    int k_array[1];
    double alpha_array[1];
    const double **a_array;
    int lda_array[1];
    const double **b_array;
    int ldb_array[1];
    double beta_array[1];
    double **c_array;
    int ldc_array[1];

    int nb = c(0, 0)->nb_;
    transa_array[0] = CblasNoTrans;
    transb_array[0] = CblasTrans;
    m_array[0] = nb;
    n_array[0] = nb;
    k_array[0] = nb;
    alpha_array[0] = alpha;
    lda_array[0] = nb;
    ldb_array[0] = nb;
    beta_array[0] = beta;
    ldc_array[0] = nb;
    int group_size = (nt_*(nt_-1))/2;

    a_array = (const double**)malloc(sizeof(double*)*group_size);
    b_array = (const double**)malloc(sizeof(double*)*group_size);
    c_array = (double**)malloc(sizeof(double*)*group_size);
    assert(a_array != nullptr);
    assert(b_array != nullptr);
    assert(c_array != nullptr);

    int i;
    for (int64_t n = 0; n < nt_; ++n) {
        for (int64_t m = n+1; m < mt_; ++m) {
            for (int64_t k = 0; k < a.nt_; ++k) {
                a_array[i] = a(m, k)->data_;
                b_array[i] = a(n, k)->data_;
                c_array[i] = c(m, n)->data_;
                ++i;
            }
        }
    }
    trace_cpu_start();
//  mkl_set_num_threads_local(60);
    cblas_dgemm_batch(CblasColMajor, transa_array, transb_array,
                      m_array, n_array, k_array, alpha_array,
                      a_array, lda_array, b_array, ldb_array, beta_array,
                      c_array, ldc_array, 1, &group_size);
    mkl_set_num_threads_local(1);
    trace_cpu_stop("DarkGreen");

    free(a_array);
    free(b_array);
    free(c_array);

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<class FloatType>
void Matrix<FloatType>::trsm(blas::Side side, blas::Uplo uplo,
                             blas::Op trans, blas::Diag diag,
                             FloatType alpha, const Matrix &a)
{
    using namespace blas;

    Matrix<FloatType> b = *this;

    // Right, Lower, Trans
    for (int64_t k = 0; k < nt_; ++k) {

        for (int64_t m = 0; m < mt_; ++m) {
            #pragma omp task
            b(m, k)->trsm(side, uplo, trans, diag, 1.0, a(k, k)); 

            for (int64_t n = k+1; n < nt_; ++n)
                #pragma omp task
                b(m, n)->gemm(Op::NoTrans, trans,
                              -1.0/alpha, b(m, k), a(n, k), 1.0);
        }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<class FloatType>
void Matrix<FloatType>::potrf(blas::Uplo uplo, int64_t lookahead)
{
    using namespace blas;

    Matrix<FloatType> a = *this;
    uint8_t *column;

//  #pragma omp parallel num_threads(8)
    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < nt_; ++k) {
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            a(k, k)->potrf(uplo);

            for (int64_t m = k+1; m < nt_; ++m) {
                #pragma omp task priority(1)
                {
                    a(m, k)->trsm(Side::Right, Uplo::Lower,
                                  Op::Trans, Diag::NonUnit,
                                  1.0, a(k, k));

                    if (m-k-1 > 0)
                        a(m, k)->packA(m-k-1);

                    if (nt_-m-1 > 0)
                        a(m, k)->packB(nt_-m-1);
                }
            }
            #pragma omp taskwait
        }
        for (int64_t n = k+1; n < k+1+lookahead && n < nt_; ++n) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[n]) priority(1)
            {
                #pragma omp task priority(1)
                a(n, n)->syrk(Uplo::Lower, Op::NoTrans,
                              -1.0, a(n, k), 1.0);

                for (int64_t m = n+1; m < nt_; ++m) {
                    #pragma omp task priority(1)
                    a(m, n)->gemm(Op::NoTrans, Op::Trans,
                                  -1.0, a(m, k), a(n, k), 1.0);
                }
                #pragma omp taskwait
            }
        }
        if (k+1+lookahead < nt_)
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[nt_-1])
            Matrix(a, k+1+lookahead, k+1+lookahead,
                   nt_-1-k-lookahead, nt_-1-k-lookahead).syrkNest(
                Uplo::Lower, Op::NoTrans,
                -1.0, Matrix(a, k+1+lookahead, k, nt_-1-k-lookahead, 1), 1.0);
    }
}

} // namespace slate

#endif // SLATE_MATRIX_HH
