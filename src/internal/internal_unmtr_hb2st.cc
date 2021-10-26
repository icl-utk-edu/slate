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
namespace internal {

//------------------------------------------------------------------------------
/// @ingroup heev_internal
///
template <Target target, typename scalar_t>
void unmtr_hb2st(
    Side side, Op op,
    Matrix<scalar_t>& V,
    Matrix<scalar_t>& C,
    const std::map<Option, Value>& opts) 
{
    unmtr_hb2st(internal::TargetType<target>(),
                side, op, V, C, opts);
}

//------------------------------------------------------------------------------
/// Host OpenMP task implementation.
/// @ingroup heev_internal
///
/*
template <typename scalar_t>
void unmtr_hb2st(internal::TargetType<Target::HostTask>,
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
    
    // Early exit if this rank has no data in C.
    // This lets later code assume every rank gets tiles in V, etc.
    std::set<int> ranks;
    auto Crow = C.sub(0, 0, 0, nt-1);
    Crow.getRanks(&ranks);
    if (ranks.find( C.mpiRank() ) == ranks.end())
        return;
    
    for (int64_t j2 = mt-1; j2 > -mt; --j2) {
        double n_ = 1.0 * j2 / 2.0;
        for (int64_t j = 0; j < mt; ++j) {
            int64_t i = (j - n_) * 2;
            if (i < mt && i >= j) {
                #pragma omp task firstprivate(i, j)
                {
                    slate::trace::Block trace_block(std::string(""+std::to_string(j)+","+std::to_string(i)).c_str());
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
                    V.tileBcast(0, r, C.sub(i, i, 0, nt-1), Layout::ColMajor, j);

                    auto Vr = V_(0, r);
                    scalar_t* Vr_data = Vr.data();
                    int64_t ldv = Vr.stride();

                    auto  T =  T_matrix(i/2, 0);
                    auto VT = VT_matrix(i/2, 0);
                    auto VC = VC_matrix(i/2, 0);

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
        #pragma omp taskwait
    }
}
*/
//------------------------------------------------------------------------------
/// GPU device batched implementation.
/// @ingroup heev_internal
///
template <Target target, typename scalar_t>
//template <typename scalar_t>
void unmtr_hb2st(//internal::TargetType<Target::Devices>,
                 internal::TargetType<target>,
                 Side side, Op op,
                 Matrix<scalar_t>& V,
                 Matrix<scalar_t>& C,
                 const std::map<Option, Value>& opts)
{
    //Target target = Target::Devices;
    //Target target = Target::HostTask;
    trace::Block trace_block("unmtr_hb2st");
    
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
    Matrix<scalar_t>  T( mt_2*nb, nb, nb, nb, 1, 1, V_.mpiComm() );
    Matrix<scalar_t> VT( mt_2*vm, nb, vm, nb, 1, 1, V_.mpiComm() );
    {slate::trace::Block trace_block(std::string("TVTalloc").c_str());
    for (int64_t i = 0; i < mt_2; ++i) {
         T.tileInsertWorkspace(i, 0);
        VT.tileInsertWorkspace(i, 0);
        if (target == Target::Devices) {
            T.tileModified(i, 0);
            VT.tileModified(i, 0);
        }
    }}
    // Number of column blocks of VC
    // Allocate temporary tile on each device
    int64_t vc_nt;
    if (target == Target::Devices) {
        vc_nt = C.num_devices();
    } else {
        vc_nt = 1;
    }

    Matrix<scalar_t> VC( mt_2*nb, vc_nt*nb, nb, nb, 1, 1, V_.mpiComm() );
    {slate::trace::Block trace_block(std::string("VCalloc").c_str());
    for (int64_t i = 0; i < mt_2; ++i) {
        for (int64_t j = 0; j < vc_nt; ++j) {
            if (target == Target::Devices) {
                int device = VC.tileDevice(i, j);
                VC.tileInsertWorkspace(i, j, device);
                VC.tileModified(i, j);
            } else {
                VC.tileInsertWorkspace(i, j);
            }
        }
    }}

    std::vector<scalar_t> tau_vector(mt_2*nb);
    
    // Early exit if this rank has no data in C.
    // This lets later code assume every rank gets tiles in V, etc.
    std::set<int> ranks;
    auto Crow = C.sub(0, 0, 0, nt-1);
    Crow.getRanks(&ranks);
    if (ranks.find( C.mpiRank() ) == ranks.end())
        return;
    
    // TODO loops in src folder
    for (int64_t j2 = mt-1; j2 > -mt; --j2) { // outer loop
        double n_ = 1.0 * j2 / 2.0;
        for (int64_t j = 0; j < mt; ++j) {
            int64_t i = (j - n_) * 2;
            if (i < mt && i >= j) {
                #pragma omp task firstprivate(i, j)
                {
                    // TODO task in internal folder

                    slate::trace::Block trace_block(std::string(""+std::to_string(j)+","+std::to_string(i)).c_str());
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
                    {slate::trace::Block trace_block(std::string("BC").c_str());
                    V.tileBcast(0, r, C.sub(i, i, 0, nt-1), Layout::ColMajor, j);}

                    auto Vr = V_(0, r);
                    scalar_t* Vr_data = Vr.data();
                    int64_t ldv = Vr.stride();

                    // Copy tau, which is stored on diag(Vr), and set diag(Vr) = 1.
                    // diag(Vr) is restored later.
                    scalar_t* tau = &tau_vector[ (i/2)*nb ];
                    {slate::trace::Block trace_block(std::string("t").c_str());
                    for (int64_t ii = 0; ii < vnb; ++ii) {
                        tau[ii] = Vr_data[ii + ii*ldv];
                        Vr_data[ii + ii*ldv] = 1;
                    }}

                    #pragma omp taskgroup
                    {
                        #pragma omp task
                        {
                            int device = C.tileDevice(i, 0);
                            #pragma omp taskgroup
                            {
                                #pragma omp task
                                {
                                    // Form T from Vr and tau.
                                    if (target == Target::Devices) {
                                        {slate::trace::Block trace_block(std::string("T").c_str());
                                        T.tileGetForWriting(i/2, 0, LayoutConvert::None);}
                                    }
                                    {slate::trace::Block trace_block(std::string("larft").c_str());
                                    T(i/2, 0).set(zero, zero);
                                    lapack::larft(Direction::Forward, lapack::StoreV::Columnwise,
                                                  vm_, vnb,
                                                  Vr.data(), Vr.stride(), tau,
                                                  T(i/2, 0).data(), T(i/2, 0).stride());}
                                    if (target == Target::Devices) {
                                        slate::trace::Block trace_block(std::string("1T").c_str());
                                        T.tileGetForReading(i/2, 0, device, LayoutConvert::None);
                                    }
                                }
                                if (target == Target::Devices) {
                                    #pragma omp task default(none) firstprivate(r, device) shared(V_)
                                    {
                                        slate::trace::Block trace_block(std::string("1V").c_str());
                                        V_.tileGetForReading(0, r, device, LayoutConvert::None);
                                    }
                                    #pragma omp task default(none) firstprivate(i, device) shared(VT)
                                    {
                                        slate::trace::Block trace_block(std::string("1VT").c_str());
                                        // VT is only written so use tileAcquire
                                        VT.tileAcquire(i/2, 0, device, Layout::ColMajor);
                                        VT.tileModified(i/2, 0, device, true);
                                    }
                                }
                            }
                            // Form VT = V * T. Assumes 0's stored in lower T.
                            // vm_-by-vnb = (vm_-by-vnb) (vnb-by-vnb)
                            {
                                slate::trace::Block trace_block(std::string("1gemm").c_str());

                                if (target == Target::Devices) {
                                    int device = C.tileDevice(i, 0);
                                    blas::Queue* queue = C.compute_queue(device, omp_get_thread_num()/*queue_index*/);   
                                    blas::gemm(Layout::ColMajor,
                                               Op::NoTrans, Op::NoTrans,
                                               vm_, vnb, vnb,
                                               one,  
                                               V_(0, r, device).data(), 
                                               V_(0, r, device).stride(),
                                               T(i/2, 0, device).data(),  
                                               T(i/2, 0, device).stride(),
                                               zero, 
                                               VT(i/2, 0, device).data(), 
                                               VT(i/2, 0, device).stride(), 
                                               *queue);
                                    {slate::trace::Block trace_block(std::string("1s").c_str());
                                    queue->sync();}
                                }
                                else {
                                    //TODO Tile tmp = V_.slice(1,r)
                                    //gemm(tmp, C(i+1, k), VC(i/2,0))
                                    blas::gemm(Layout::ColMajor,
                                               Op::NoTrans, Op::NoTrans,
                                               vm_, vnb, vnb,
                                               one,  
                                               V_(0, r).data(), 
                                               V_(0, r).stride(),
                                               T(i/2, 0).data(),  
                                               T(i/2, 0).stride(),
                                               zero, 
                                               VT(i/2, 0).data(), 
                                               VT(i/2, 0).stride());
                                }
                            }
                            if (target == Target::Devices) {
                                #pragma omp taskgroup
                                {
                                    for (int d = 0; d < C.num_devices(); ++d) {
                                        // prefetch VT on all devices C -= VT VC operation
                                        #pragma omp task default(none) firstprivate(d, i) shared(VT)
                                        {
                                            slate::trace::Block trace_block(std::string("p"+std::to_string(d)+"VT").c_str());
                                            VT.tileGetForReading(i/2, 0, d, LayoutConvert::None);
                                        }
                                    }
                                }
                            }
                        }
                        if (target == Target::Devices) {
                            // prefetch VT, V, and C  on all devices C -= VT VC operation
                            for (int d = 0; d < C.num_devices(); ++d) {
                                #pragma omp task default(none) firstprivate(d, r) shared(V_)
                                {
                                    slate::trace::Block trace_block(std::string("p"+std::to_string(d)+"V").c_str());
                                    V_.tileGetForReading(0, r, d, LayoutConvert::None);
                                }
                            }
                            for (int64_t k = 0; k < nt; ++k) {
                                if (C.tileIsLocal(i, k)) {
                                    int device = C.tileDevice(i, k);
                                    #pragma omp task default(none) firstprivate(i, k, device) shared(C)
                                    {
                                        slate::trace::Block trace_block(std::string("p"+std::to_string(device)+"Ci").c_str());
                                        C.tileGetForReading(i, k, device, LayoutConvert::None);
                                    }
                                    if (i+1 < mt) {
                                        #pragma omp task default(none) firstprivate(i, k, device) shared(C)
                                        {
                                            slate::trace::Block trace_block(std::string("p"+std::to_string(device)+"Ci1").c_str());
                                            // Device of C(i+1, k) is equal to C(i, k) since 1D column 
                                            // cyclic distribution is used.
                                            C.tileGetForReading(i+1, k, device, LayoutConvert::None);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Vr = [ Vr0 ],  VT = [ VT0 ],  [ Ci     ] = [ C0 ],
                    //      [ Vr1 ]        [ VT1 ]   [ C{i+1} ] = [ C1 ]
                    // Vr and VT are (mb0 + mb1)-by-vnb = vm_-by-vnb,
                    // C0 is mb0-by-cnb,
                    // C1 is mb1-by-cnb.
                    for (int64_t k = 0; k < nt; ++k) {
                        if (C.tileIsLocal(i, k)) {
                            slate::trace::Block trace_block(std::string(std::to_string(k)).c_str());

                            auto C0 = C(i, k);
                            int64_t cnb = C0.nb();
                            if (target != Target::Devices) {
                                assert( cnb <= VC(i/2, 0).nb() );
                            }
                            assert( C0.mb()-1 == mb0 );  // After 1st row sliced off.
                            int device = C.tileDevice(i, k);

                            // VC = Vr0^H C0
                            // vnb-by-cnb = (mb0-by-vnb)^H (mb0-by-cnb)
                            // Slice off 1st row of C0.
                            { 
                                slate::trace::Block trace_block(std::string("2gemm").c_str());

                                // C0
                                if (target == Target::Devices) {
                                    //int device;
                                    //device = VC.tileDevice(i/2, 0);
                                    {slate::trace::Block trace_block(std::string("2V").c_str());
                                    V_.tileGetForReading(0, r, device, LayoutConvert::None);}
                                    //printf("2VCi: %d %ld,%ld,%ld %d\n", omp_get_thread_num(), i, j, k, device);
                                    {slate::trace::Block trace_block(std::string("2C0").c_str());
                                    C.tileGetForReading(i, k, device, LayoutConvert::None);}
                                    {slate::trace::Block trace_block(std::string("2VC").c_str());
                                    VC.tileAcquire(i/2, device, device, Layout::ColMajor);
                                    VC.tileModified(i/2, device, device, true);}
                                    blas::Queue* queue = C.compute_queue(device, omp_get_thread_num()/*queue_index*/);
                                    blas::gemm(Layout::ColMajor,
                                               Op::ConjTrans, Op::NoTrans,
                                               vnb, cnb, mb0,
                                               one, 
                                               V_(0, r, device).data(), 
                                               V_(0, r, device).stride(),
                                               &C(i, k, device).data()[ 1 ], 
                                               C(i, k, device).stride(),
                                               zero, 
                                               VC(i/2, device, device).data(), 
                                               VC(i/2, device, device).stride(), 
                                               *queue);
                                    {slate::trace::Block trace_block(std::string("2s").c_str());
                                    queue->sync();}
                                }
                                else {
                                    blas::gemm(Layout::ColMajor,
                                               Op::ConjTrans, Op::NoTrans,
                                               vnb, cnb, mb0,
                                               one, 
                                               V_(0, r).data(), 
                                               V_(0, r).stride(),
                                               &C(i, k).data()[ 1 ], 
                                               C(i, k).stride(),
                                               zero, 
                                               VC(i/2, 0).data(), 
                                               VC(i/2, 0).stride()); 
                                }
                            }

                            // VC += Vr1^H C1
                            // vnb-by-cnb += (mb1-by-vnb)^H (mb1-by-cnb)
                            Tile<scalar_t> C1;
                            if (i+1 < mt) {
                                assert(C.tileIsLocal(i+1, k));
                                C1 = C(i+1, k);
                                {
                                    slate::trace::Block trace_block(std::string("3gemm").c_str());
                                
                                    if (target == Target::Devices) {
                                        //int device;
                                        //device = VC.tileDevice(i/2, 0);
                                        //printf("3VCi1: %d %ld,%ld,%ld %d\n", omp_get_thread_num(), i, j, k, device);
                                        {slate::trace::Block trace_block(std::string("3C1").c_str());
                                        C.tileGetForReading(i+1, k, device, LayoutConvert::None);}
                                        blas::Queue* queue = C.compute_queue(device, omp_get_thread_num()/*queue_index*/);   
                                        blas::gemm(Layout::ColMajor,
                                                   Op::ConjTrans, Op::NoTrans,
                                                   vnb, cnb, mb1,
                                                   one, 
                                                   &(V_(0, r, device).data()[ mb0 ]),   
                                                   V_(0, r, device).stride(),
                                                   C(i+1, k, device).data(), 
                                                   C(i+1, k, device).stride(),
                                                   one, 
                                                   VC(i/2, device, device).data(), 
                                                   VC(i/2, device, device).stride(), 
                                                   *queue);
                                        queue->sync(); 
                                    }
                                    else {
                                        blas::gemm(Layout::ColMajor,
                                                   Op::ConjTrans, Op::NoTrans,
                                                   vnb, cnb, mb1,
                                                   one, 
                                                   &(V_(0, r).data()[ mb0 ]),   
                                                   V_(0, r).stride(),
                                                   C(i+1, k).data(), 
                                                   C(i+1, k).stride(),
                                                   one, 
                                                   VC(i/2, 0).data(), 
                                                   VC(i/2, 0).stride()); 
                                    }
                                }
                            }
                            #pragma omp taskgroup
                            {
                            // C0 -= (V0 T) VC
                            // mb0-by-cnb -= (mb0-by-vnb) (vnb-by-cnb)
                            // Slice off 1st row of C0.
                            #pragma omp task
                            {
                                slate::trace::Block trace_block(std::string("4gemm").c_str());
                                if (target == Target::Devices) {
                                    //int device;
                                    //device = C.tileDevice(i, k);
                                    //printf("4Ci,k: %d %ld,%ld,%ld %d\n", omp_get_thread_num(), i, j, k, device);
                                    {slate::trace::Block trace_block(std::string("4VT").c_str());
                                    VT.tileGetForReading(i/2, 0, device, LayoutConvert::None);}
                                    {slate::trace::Block trace_block(std::string("4VC").c_str());
                                    VC.tileGetForReading(i/2, device, device, LayoutConvert::None);}
                                    {slate::trace::Block trace_block(std::string("4C0").c_str());
                                    C.tileGetForWriting(i, k, device, LayoutConvert::None);}
                                    blas::Queue* queue = C.compute_queue(device, omp_get_thread_num()/*queue_index*/);   
                                    blas::gemm(Layout::ColMajor,
                                               Op::NoTrans, Op::NoTrans,
                                               mb0, cnb, vnb,
                                               -one, 
                                               VT(i/2, 0, device).data(),   
                                               VT(i/2, 0, device).stride(),
                                               VC(i/2, device, device).data(),   
                                               VC(i/2, device, device).stride(),
                                               one,  
                                               &C(i, k, device).data()[ 1 ], 
                                               C(i, k, device).stride(), 
                                               *queue);
                                    {slate::trace::Block trace_block(std::string("4s").c_str());
                                    queue->sync();}
                                }
                                else {
                                    blas::gemm(Layout::ColMajor,
                                               Op::NoTrans, Op::NoTrans,
                                               mb0, cnb, vnb,
                                               -one, 
                                               VT(i/2, 0).data(),   
                                               VT(i/2, 0).stride(),
                                               VC(i/2, 0).data(),   
                                               VC(i/2, 0).stride(),
                                               one,  
                                               &C(i, k).data()[ 1 ], 
                                               C(i, k).stride()); 
                                }
                            }

                            // C1 -= (V1 T) VC
                            // mb1-by-cnb -= (mb1-by-vnb) (vnb-by-cnb)
                            if (i+1 < mt) {
                                #pragma omp task
                                {
                                    slate::trace::Block trace_block(std::string("5gemm").c_str());

                                    if (target == Target::Devices) {
                                        //int device;
                                        //device = C.tileDevice(i+1, k);
                                        //printf("5Ci1,k: %d %ld,%ld,%ld %d\n", omp_get_thread_num(), i, j, k, device);
                                        {slate::trace::Block trace_block(std::string("5VT").c_str());
                                        VT.tileGetForReading(i/2, 0, device, LayoutConvert::None);}
                                        {slate::trace::Block trace_block(std::string("5VC").c_str());
                                        VC.tileGetForReading(i/2, device, device, LayoutConvert::None);}
                                        {slate::trace::Block trace_block(std::string("5C1").c_str());
                                        C.tileGetForWriting(i+1, k, device, LayoutConvert::None);}
                                        blas::Queue* queue = C.compute_queue(device, omp_get_thread_num()/*queue_index*/);   
                                        blas::gemm(Layout::ColMajor,
                                                   Op::NoTrans, Op::NoTrans,
                                                   mb1, cnb, vnb,
                                                   -one, 
                                                   &VT(i/2, 0, device).data()[ mb0 ],
                                                   VT(i/2, 0, device).stride(),
                                                   VC(i/2, device, device).data(), 
                                                   VC(i/2, device, device).stride(),
                                                   one,  
                                                   C(i+1, k, device).data(), 
                                                   C(i+1, k, device).stride(), 
                                                   *queue);
                                        {slate::trace::Block trace_block(std::string("5s").c_str());
                                        queue->sync();}
                                    } else {
                                        blas::gemm(Layout::ColMajor,
                                                   Op::NoTrans, Op::NoTrans,
                                                   mb1, cnb, vnb,
                                                   -one, 
                                                   &VT(i/2, 0).data()[ mb0 ],
                                                   VT(i/2, 0).stride(),
                                                   VC(i/2, 0).data(), 
                                                   VC(i/2, 0).stride(),
                                                   one,  
                                                   C(i+1, k).data(), 
                                                   C(i+1, k).stride());                                    
                                    }
                                }
                            }
                            } //#pragma omp taskwait
                            V.tileTick(0, r);
                        } // if C(i, k) is local
                    } // inner for loop

                    // Restore diag(Vr) = tau.
                    if (V_.tileIsLocal(0, r)) {
                        for (int64_t ii = 0; ii < vnb; ++ii) {
                            Vr_data[ii + ii*ldv] = tau[ii];
                        }
                    }
                }
            }
        }
        #pragma omp taskwait
    } // outer loop
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmtr_hb2st<Target::HostTask, float>(
    Side side, Op op,
    Matrix<float>& V,
    Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<Target::HostTask, double>(
    Side side, Op op,
    Matrix<double>& V,
    Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<Target::HostTask, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& V,
    Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<Target::HostTask, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& V,
    Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<Target::Devices, float>(
    Side side, Op op,
    Matrix<float>& V,
    Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<Target::Devices, double>(
    Side side, Op op,
    Matrix<double>& V,
    Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<Target::Devices, std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& V,
    Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void unmtr_hb2st<Target::Devices, std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& V,
    Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);
} // namespace internal
} // namespace slate
