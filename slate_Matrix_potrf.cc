
#include "slate_Matrix.hh"

namespace slate {

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::potrf(blas::Uplo uplo, int64_t lookahead)
{
    using namespace blas;

    Matrix<FloatType> a = *this;
    uint8_t *column;    

    for (int device = 0; device < num_devices_; ++device)
        a.memory_->addDeviceBlocks(device, getMaxDeviceTiles(device));

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < nt_; ++k) {
        // panel
        #pragma omp task depend(inout:column[k])
        {
            if (tileIsLocal(k, k)) {
                a(k, k)->potrf(uplo);
            }

            if (k < nt_-1)
                tileIbcast(k, k, {k+1, nt_-1, k, k});

            for (int64_t m = k+1; m < nt_; ++m) {

                #pragma omp task
                if (tileIsLocal(m, k)) {
                    tileWait(k, k);
                    a.tileMoveToHost(m, k, tileDevice(m, k));
                    a(m, k)->trsm(Side::Right, Uplo::Lower,
                                  Op::Trans, Diag::NonUnit,
                                  1.0, a(k, k));
                }
            }
            #pragma omp taskwait

            for (int64_t m = k+1; m < nt_; ++m)
                tileIbcast(m, k, {m, m, k+1, m},
                                 {m, nt_-1, m, m});
        }
        // trailing submatrix
        if (k+1+lookahead < nt_) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[nt_-1])
            Matrix(a, k+1+lookahead, k+1+lookahead,
                   nt_-1-k-lookahead, nt_-1-k-lookahead).syrkAcc(
                Uplo::Lower, Op::NoTrans,
                -1.0, Matrix(a, k+1+lookahead, k, nt_-1-k-lookahead, 1), 1.0);
        }
        // lookahead column(s)
        for (int64_t n = k+1; n < k+1+lookahead && n < nt_; ++n) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[n])
            {
                #pragma omp task
                if (tileIsLocal(n, n)) {
                    tileWait(n, k);
                    a(n, n)->syrk(Uplo::Lower, Op::NoTrans,
                                  -1.0, a(n, k), 1.0);
                }

                for (int64_t m = n+1; m < nt_; ++m) {
                    #pragma omp task
                    if (tileIsLocal(m, n)) {
                        tileWait(m, k);
                        tileWait(n, k);
                        a.tileMoveToHost(m, n, tileDevice(m, n));
                        a(m, n)->gemm(Op::NoTrans, Op::Trans,
                                      -1.0, a(m, k), a(n, k), 1.0);
                    }
                }
                #pragma omp taskwait
            }
        }
    }

    for (int device = 0; device < num_devices_; ++device)
        a.memory_->clearDeviceBlocks(device);

    a.checkLife();
    a.printLife();
}

template
void Matrix<double>::potrf(blas::Uplo uplo, int64_t lookahead);

} // namespace slate
