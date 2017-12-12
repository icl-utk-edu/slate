
#include "slate_Matrix.hh"

namespace slate {

namespace internal {

//------------------------------------------------------------------------------
template <typename FloatType, Target target>
void potrf(TargetType<target>,
           blas::Uplo uplo, Matrix<FloatType> a, int64_t lookahead)
{
    using namespace blas;

    uint8_t *column;

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < a.nt_; ++k) {
        // panel
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            if (a.tileIsLocal(k, k)) {
                Tile<FloatType>::potrf(uplo, a(k, k));
            }

            if (k < a.nt_-1)
                a.tileSend(k, k, {k+1, a.nt_-1, k, k});

            for (int64_t m = k+1; m < a.nt_; ++m) {

                #pragma omp task priority(1)
                if (a.tileIsLocal(m, k)) {
                    Tile<FloatType>::trsm(Side::Right, Uplo::Lower,
                                          Op::Trans, Diag::NonUnit,
                                          1.0, a(k, k),
                                               a(m, k));
                }
            }
            #pragma omp taskwait

            for (int64_t m = k+1; m < a.nt_; ++m)
                a.tileSend(m, k, {m, m, k+1, m},
                                 {m, a.nt_-1, m, m});
        }
        // lookahead column(s)
        for (int64_t n = k+1; n < k+1+lookahead && n < a.nt_; ++n) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[n]) priority(1)
            {
                #pragma omp task
                if (a.tileIsLocal(n, n)) {
                    Tile<FloatType>::syrk(Uplo::Lower, Op::NoTrans,
                                          -1.0, a(n, k),
                                           1.0, a(n, n));
                }

                for (int64_t m = n+1; m < a.nt_; ++m) {
                    #pragma omp task
                    if (a.tileIsLocal(m, n)) {
                        Tile<FloatType>::gemm(Op::NoTrans, Op::Trans,
                                              -1.0, a(m, k),
                                                    a(n, k),
                                               1.0, a(m, n));
                    }
                }
                #pragma omp taskwait
            }
        }
        // trailing submatrix
        if (k+1+lookahead < a.nt_) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[a.nt_-1])
            {
                Matrix<FloatType> syrk_a =
                    a(k+1+lookahead, a.nt_-1, k, k);

                Matrix<FloatType> syrk_b =
                    a(k+1+lookahead, a.nt_-1, k+1+lookahead, a.nt_-1);

                syrk_b.template syrk<target>(Uplo::Lower, Op::NoTrans,-1.0, syrk_a, 1.0);
            }
        }
    }

    a.checkLife();
    a.printLife();
}

//------------------------------------------------------------------------------
template <typename FloatType>
void potrf(TargetType<Target::Devices>,
           blas::Uplo uplo, Matrix<FloatType> a, int64_t lookahead)
{
    using namespace blas;

    uint8_t *column;

    for (int device = 0; device < a.num_devices_; ++device)
        a.memory_->addDeviceBlocks(device, a.getMaxDeviceTiles(device));

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < a.nt_; ++k) {
        // panel
        #pragma omp task depend(inout:column[k])
        {
            if (a.tileIsLocal(k, k)) {
                Tile<FloatType>::potrf(uplo, a(k, k));
            }

            if (k < a.nt_-1)
                a.tileSend(k, k, {k+1, a.nt_-1, k, k});

            for (int64_t m = k+1; m < a.nt_; ++m) {

                #pragma omp task
                if (a.tileIsLocal(m, k)) {
                    a.tileMoveToHost(m, k, a.tileDevice(m, k));
                    Tile<FloatType>::trsm(Side::Right, Uplo::Lower,
                                          Op::Trans, Diag::NonUnit,
                                          1.0, a(k, k),
                                               a(m, k));
                }
            }
            #pragma omp taskwait

            for (int64_t m = k+1; m < a.nt_; ++m)
                a.template tileSend<Target::Devices>(m, k, {m, m, k+1, m},
                                                           {m, a.nt_-1, m, m});
        }
        // trailing submatrix
        if (k+1+lookahead < a.nt_) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[a.nt_-1])
            {
                Matrix<FloatType> syrk_a =
                    a(k+1+lookahead, a.nt_-1, k, k);

                Matrix<FloatType> syrk_b =
                    a(k+1+lookahead, a.nt_-1, k+1+lookahead, a.nt_-1);

                syrk_b.template syrk<Target::Devices>(
                    Uplo::Lower, Op::NoTrans,-1.0, syrk_a, 1.0);
            }
        }
        // lookahead column(s)
        for (int64_t n = k+1; n < k+1+lookahead && n < a.nt_; ++n) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[n])
            {
                #pragma omp task
                if (a.tileIsLocal(n, n)) {
                    Tile<FloatType>::syrk(Uplo::Lower, Op::NoTrans,
                                          -1.0, a(n, k),
                                           1.0, a(n, n));
                }

                for (int64_t m = n+1; m < a.nt_; ++m) {
                    #pragma omp task
                    if (a.tileIsLocal(m, n)) {
                        a.tileMoveToHost(m, n, a.tileDevice(m, n));
                        Tile<FloatType>::gemm(Op::NoTrans, Op::Trans,
                                              -1.0, a(m, k),
                                                    a(n, k),
                                               1.0, a(m, n));
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    for (int device = 0; device < a.num_devices_; ++device)
        a.memory_->clearDeviceBlocks(device);

    a.checkLife();
    a.printLife();
}

//------------------------------------------------------------------------------
// Precision and target templated function for implementing complex logic.
//
template <typename FloatType, Target target>
void potrf(blas::Uplo uplo, Matrix<FloatType> a, int64_t lookahead)
{
    potrf(TargetType<target>(), uplo, a, lookahead);
}

} // namespace internal

//------------------------------------------------------------------------------
// Target-templated, precision-overloaded functions for the user.
//
template <Target target>
void potrf(blas::Uplo uplo, Matrix<double> a, int64_t lookahead)
{
    internal::potrf<double, target>(uplo, a, lookahead);
}

//------------------------------------------------------------------------------
template
void potrf<Target::HostTask>(
    blas::Uplo uplo, Matrix<double> a, int64_t lookahead);

template
void potrf<Target::HostNest>(
    blas::Uplo uplo, Matrix<double> a, int64_t lookahead);

template
void potrf<Target::HostBatch>(
    blas::Uplo uplo, Matrix<double> a, int64_t lookahead);

template
void potrf<Target::Devices>(
    blas::Uplo uplo, Matrix<double> a, int64_t lookahead);

} // namespace slate
