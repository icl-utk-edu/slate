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
/// @param[in] V
///     Householder vectors as returned by `slate::hb2st`.
///
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, $C$ is overwritten by $Q C$, $Q^H C$, $C Q$, or $C Q^H$.
///     C must be distributed 1D block column (cyclic or non-cyclic).
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

    // Local workspaces: T, VT = V T, VC = V^H C, tau is diag of each V tile.
    // (I - V T V^H) C = C - (V T) V^H C = C - VT VC.
    // todo: don't need distribution; these are local to each rank.
    int64_t mt_2 = ceildiv(mt, int64_t(2));
    Matrix<scalar_t>  T_matrix( mt_2*nb, nb, nb, nb, 1, 1, V_.mpiComm() );
    Matrix<scalar_t> VT_matrix( mt_2*vm, nb, vm, nb, 1, 1, V_.mpiComm() );
    Matrix<scalar_t> VC_matrix( mt_2*nb, nb, nb, nb, 1, 1, V_.mpiComm() );
    for (int64_t i = 0; i < mt_2; ++i) {
         T_matrix.tileInsertWorkspace(i, 0);
        VT_matrix.tileInsertWorkspace(i, 0);
        VC_matrix.tileInsertWorkspace(i, 0);
    }
    std::vector<scalar_t> tau_vector(mt_2*nb);

    // OpenMP needs pointer types, but vectors are exception safe.
    // Add one phantom row at bottom to ease specifying dependencies.
    std::vector< uint8_t > row_vector(mt+1);
    uint8_t* row = row_vector.data();
    std::vector< uint8_t > col_vector(nt+1);
    uint8_t* col = col_vector.data();
    row[0] = col[0] = 0;

    // Early exit if this rank has no data in C.
    // This lets later code assume every rank gets tiles in V, etc.
    std::set<int> ranks;
    auto Crow = C.sub(0, 0, 0, nt-1);
    Crow.getRanks(&ranks);
    if (ranks.find( C.mpiRank() ) == ranks.end())
        return;


    std::cout << __func__ << std::endl;
    for (int64_t j = mt-1; j >= 0; --j) {
        for (int64_t i = j; i < mt; ++i) {
            //#pragma omp task depend(inout:row[i]) depend(inout:row[i+1])
            //#pragma omp task 
            {
                slate::trace::Block trace_block(std::string("t j:"+std::to_string(j)+" i:"+std::to_string(i)).c_str());
                int64_t mb0 = C.tileMb(i) - 1;
                int64_t mb1 = i+1 < mt ? C.tileMb(i+1) : 0;
                int64_t vm_ = mb0 + mb1;
                int64_t vnb = std::min( nb, vm_ );
                assert(vm_ <= vm);

                // Index of block of V, using lower triangular packed indexing.
                int64_t r = i - j + j*mt - j*(j-1)/2;

                // Send V(0, r) across ranks owning row C(i, :).
                // Send from V to be contiguous, instead of V_.
                // todo make async; put in different task.
                V.tileBcast(0, r, C.sub(i, i, 0, nt-1), Layout::ColMajor);

                auto Vr = V_(0, r);
                scalar_t* Vr_data = Vr.data();
                int64_t ldv = Vr.stride();

                auto  T =  T_matrix(i/2, 0);
                auto VT = VT_matrix(i/2, 0);
                auto VC = VC_matrix(i/2, 0);
                
                std::cout << "i:" << i << " j:" << j 
                    << " Vr:0-" << r 
                    << " T:" << i/2 << "-0" 
                    << " VT:" << i/2 << "-0"
                    << std::endl;

                // Copy tau, which is stored on diag(Vr), and set diag(Vr) = 1.
                // diag(Vr) is restored later.
                scalar_t* tau = &tau_vector[ (i/2)*nb ];
                for (int64_t ii = 0; ii < vnb; ++ii) {
                    tau[ii] = Vr_data[ii + ii*ldv];
                    Vr_data[ii + ii*ldv] = 1;
                }

                // Form T from Vr and tau.
                T.set(zero, zero);
                lapack::larft(Direction::Forward, lapack::StoreV::Columnwise,
                              vm_, vnb,
                              Vr.data(), Vr.stride(), tau,
                              T.data(), T.stride());

                // Form VT = V * T. Assumes 0's stored in lower T.
                // vm_-by-vnb = (vm_-by-vnb) (vnb-by-vnb)
                blas::gemm(Layout::ColMajor,
                           Op::NoTrans, Op::NoTrans,
                           vm_, vnb, vnb,
                           one,  Vr.data(), Vr.stride(),
                                  T.data(),  T.stride(),
                           zero, VT.data(), VT.stride());

                // Vr = [ Vr0 ],  VT = [ VT0 ],  [ Ci     ] = [ C0 ],
                //      [ Vr1 ]        [ VT1 ]   [ C{i+1} ] = [ C1 ]
                // Vr and VT are (mb0 + mb1)-by-vnb = vm_-by-vnb,
                // C0 is mb0-by-cnb,
                // C1 is mb1-by-cnb.
                for (int64_t k = 0; k < nt; ++k) {
                    if (C.tileIsLocal(i, k)) {
                        auto C0 = C(i, k);
                        int64_t cnb = C0.nb();
                        assert( cnb <= VC.nb() );
                        assert( C0.mb()-1 == mb0 );  // After 1st row sliced off.

                        // VC = Vr0^H C0
                        // vnb-by-cnb = (mb0-by-vnb)^H (mb0-by-cnb)
                        // Slice off 1st row of C0.
                        blas::gemm(Layout::ColMajor,
                                   Op::ConjTrans, Op::NoTrans,
                                   vnb, cnb, mb0,
                                   one,  Vr.data(),   Vr.stride(),
                                         C0.data()+1, C0.stride(),
                                   zero, VC.data(),   VC.stride());

                        // VC += Vr1^H C1
                        // vnb-by-cnb += (mb1-by-vnb)^H (mb1-by-cnb)
                        Tile<scalar_t> C1;
                        if (i+1 < mt) {
                            assert(C.tileIsLocal(i+1, k));
                            scalar_t* Vr1data = &Vr.data()[ mb0 ];
                            C1 = C(i+1, k);
                            blas::gemm(Layout::ColMajor,
                                       Op::ConjTrans, Op::NoTrans,
                                       vnb, cnb, mb1,
                                       one, Vr1data,   Vr.stride(),
                                            C1.data(), C1.stride(),
                                       one, VC.data(), VC.stride());
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
                        V.tileTick(0, r);
                    }
                }

                // Restore diag(Vr) = tau.
                if (V_.tileIsLocal(0, r)) {
                    for (int64_t ii = 0; ii < vnb; ++ii) {
                        Vr_data[ii + ii*ldv] = tau[ii];
                    }
                }
            }
        }
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
