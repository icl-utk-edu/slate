// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

////// TODO
#include "../test/print_matrix.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Multiplies the general m-by-n matrix C by Q from `slate::hb2st` as
/// follows:
///
/// op              |  side = Left  |  side = Right (not supported)
/// --------------- | ------------- | --------------
/// op = NoTrans    |  $Q C  $      |  $C Q  $
/// op = ConjTrans  |  $Q^H C$      |  $C Q^H$
///
/// where $Q$ is a unitary matrix defined as the product of k
/// elementary reflectors
/// \[
///     Q = H(1) H(2) . . . H(k)
/// \]
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///     - Side::Left:  apply $Q$ or $Q^H$ from the left;
///     - Side::Right: apply $Q$ or $Q^H$ from the right (not supported).
///
/// @param[in] op
///     - Op::NoTrans    apply $Q$;
///     - Op::ConjTrans: apply $Q^H$;
///     - Op::Trans:     apply $Q^T$ (only if real).
///       In the real case, Op::Trans is equivalent to Op::ConjTrans.
///       In the complex case, Op::Trans is not allowed.
///
/// @param[in] A
///     Details of the factorization of the original matrix $A$,
///     as returned by `slate::he2hb`.
///
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, $C$ is overwritten by $Q C$, $Q^H C$, $C Q$, or $C Q^H$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup heev_computational
///
template <typename scalar_t>
void unmtr_hb2st(
    Side side, Op op,
    Matrix<scalar_t>& V,
    Matrix<scalar_t>& C,
    const std::map<Option, Value>& opts)
{
    bool verbose = false;
    slate_assert(side == Side::Left);

    const scalar_t zero = 0, one = 1;

    int64_t mb = V.tileMb(0); // == 2 nb
    int64_t nb = V.tileNb(0);
    assert( mb == 2*nb );

    int64_t mt = C.mt();
    int64_t nt = C.nt();
    assert( mt*(mt + 1)/2 == V.nt() );

    // Slice off 1st row of V.
    int64_t vm = V.m();
    int64_t vn = V.n();
    auto V_ = V.slice( 1, vm-1, 0, vn-1 );
    vm -= 1;
    if (verbose) {
        printf( "mb %lld, nb %lld, mt %lld, nt %lld, vm_ %lld, vn %lld\n",
                mb, nb, mt, nt, vm, vn );
        print_matrix( "V_", V_, 6, 2 );
    }

    // Workspaces: T, VT = V T, VC = V^H C, tau is diag of each V tile.
    // (I - V T V^H) C = C - (V T) V^H C = C - VT VC.
    // todo: need only mt or ceildiv( mt, 2 ) tiles. O(n*nb) instead of O(n^2/2) data.
    Matrix<scalar_t> T_matrix( nb, vn, nb, 1, 1, V_.mpiComm() );
    Matrix<scalar_t> VT_matrix  = V_.emptyLike();
    Matrix<scalar_t> VC_matrix = V_.emptyLike( nb, nb );
    T_matrix.insertLocalTiles();
    VT_matrix.insertLocalTiles();
    VC_matrix.insertLocalTiles();
    std::vector<scalar_t> tau_vector(mt*nb);

    // OpenMP needs pointer types, but vectors are exception safe.
    // Add one phantom row at bottom to ease specifying dependencies.
    std::vector< uint8_t > row_vector(mt+1);
    uint8_t* row = row_vector.data();

    for (int64_t j = mt-1; j >= 0; --j) {
        for (int64_t i = j; i < mt; ++i) {
            #pragma omp task depend(inout:row[i]) \
                             depend(inout:row[i+1])
            {
                int64_t mb0 = C.tileMb(i) - 1;
                int64_t mb1 = i+1 < mt ? C.tileMb(i+1) : 0;
                int64_t vm_ = mb0 + mb1;
                int64_t vnb = std::min( nb, vm_ );
                assert(vm_ <= vm);

                // Index of block of V, using lower triangular packed indexing.
                int64_t q = i - j + j*mt - j*(j-1)/2;
                if (verbose) {
                    printf( "i %2lld, j %2lld, q %4lld\n", i, j, q );
                }

                auto Vq = V_(0, q);
                scalar_t* Vq_data = Vq.data();
                int64_t ldv = Vq.stride();

                // todo: index these by i or i/2 instead of q? see above.
                auto  T =  T_matrix(0, q);
                auto VT = VT_matrix(0, q);
                auto VC = VC_matrix(0, q);

                if (verbose) {
                    printf( "Vq (%lld-by-%lld) %lld-by-%lld, T (%lld-by-%lld) %lld-by-%lld, VT (%lld-by-%lld) %lld-by-%lld, VC (%lld-by-%lld) %lld-by-%lld (max nb)\n",
                            Vq.mb(), Vq.nb(), vm_, vnb,
                             T.mb(),  T.nb(), vnb, vnb,
                            VT.mb(), VT.nb(), vm_, vnb,
                            VC.mb(), VC.nb(), vnb, VC.nb() );
                }

                // Copy tau, which is stored on diag(Vq), and set diag(Vq) = 1.
                // diag(Vq) is restored later.
                scalar_t* tau = &tau_vector[ i*nb ];
                for (int64_t ii = 0; ii < vnb; ++ii) {
                    tau[ii] = Vq_data[ii + ii*ldv];
                    Vq_data[ii + ii*ldv] = 1;
                }

                // Form T from Vq and tau.
                T.set(zero, zero);
                lapack::larft(Direction::Forward, lapack::StoreV::Columnwise,
                              vm_, vnb,
                              Vq.data(), Vq.stride(), tau,
                              T.data(), T.stride());

                // Form VT = V * T. Assumes 0's stored in lower T.
                // vm_-by-vnb = (vm_-by-vnb) (vnb-by-vnb)
                blas::gemm(Layout::ColMajor,
                           Op::NoTrans, Op::NoTrans,
                           vm_, vnb, vnb,
                           one,  Vq.data(), Vq.stride(),
                                  T.data(),  T.stride(),
                           zero, VT.data(), VT.stride());

                // Vq = [ Vq0 ],  VT = [ VT0 ],  [ Ci     ] = [ C0 ],
                //      [ Vq1 ]        [ VT1 ]   [ C{i+1} ] = [ C1 ]
                // Vq and VT are (mb0 + mb1)-by-vnb = vm_-by-vnb,
                // C0 is mb0-by-cnb,
                // C1 is mb1-by-cnb.
                for (int64_t k = 0; k < nt; ++k) {
                    if (C.tileIsLocal(i, k)) {
                        auto C0 = C(i, k);
                        int64_t cnb = C0.nb();
                        assert( cnb <= VC.nb() );
                        assert( C0.mb()-1 == mb0 );  // After 1st row sliced off.

                        // VC = Vq0^H C0
                        // vnb-by-cnb = (mb0-by-vnb)^H (mb0-by-cnb)
                        // Slice off 1st row of C0.
                        blas::gemm(Layout::ColMajor,
                                   Op::ConjTrans, Op::NoTrans,
                                   vnb, cnb, mb0,
                                   one,  Vq.data(),   Vq.stride(),
                                         C0.data()+1, C0.stride(),
                                   zero, VC.data(),   VC.stride());

                        // VC += Vq1^H C1
                        // vnb-by-cnb += (mb1-by-vnb)^H (mb1-by-cnb)
                        Tile<scalar_t> C1;
                        if (i+1 < mt) {
                            assert(C.tileIsLocal(i+1, k));
                            scalar_t* Vq1data = &Vq.data()[ mb0 ];
                            C1 = C(i+1, k);
                            blas::gemm(Layout::ColMajor,
                                       Op::ConjTrans, Op::NoTrans,
                                       vnb, cnb, mb1,
                                       one, Vq1data,   Vq.stride(),
                                            C1.data(), C1.stride(),
                                       one, VC.data(), VC.stride());
                        }
                        if (verbose) {
                            printf( "k %lld, C0 (%lld-by-%lld) %lld-by-%lld, C1 (%lld-by-%lld) %lld-by-%lld\n",
                                    k, C0.mb(), C0.nb(), mb0, cnb,
                                       C1.mb(), C1.nb(), mb1, cnb );
                        }

                        // C0 -= (V0 T) VC
                        // mb0-by-cnb -= (mb0-by-vnb) (vnb-by-cnb)
                        // Slice off 1st row of C0.
                        blas::gemm(Layout::ColMajor,
                                   Op::NoTrans, Op::NoTrans,
                                   mb0, cnb, vnb,
                                   -one, VT.data(),   VT.stride(),
                                         VC.data(),   VC.stride(),
                                   one,  C0.data()+1, C0.stride());

                        // C1 -= (V1 T) VC
                        // mb1-by-cnb -= (mb1-by-vnb) (vnb-by-cnb)
                        if (i+1 < mt) {
                            scalar_t* VT1data = &VT.data()[ mb0 ];
                            blas::gemm(Layout::ColMajor,
                                       Op::NoTrans, Op::NoTrans,
                                       mb1, cnb, vnb,
                                       -one, VT1data,   VT.stride(),
                                             VC.data(), VC.stride(),
                                       one,  C1.data(), C1.stride());
                        }
                    }
                }

                // Restore diag(Vq) = tau.
                for (int64_t ii = 0; ii < vnb; ++ii) {
                    Vq_data[ii + ii*ldv] = tau[ii];
                }

                if (verbose) {
                    printf( "\n" );
                }
            }
        }
    }
    if (verbose) {
        print_matrix( "T",   T_matrix, 6, 2 );
        print_matrix( "VT", VT_matrix, 6, 2 );
        print_matrix( "VC", VC_matrix, 6, 2 );
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmtr_hb2st<float>(
    Side side, Op op,
    Matrix<float>& V,
    Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<double>(
    Side side, Op op,
    Matrix<double>& V,
    Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st< std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& V,
    Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st< std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& V,
    Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

} // namespace slate
