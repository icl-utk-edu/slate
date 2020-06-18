//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate/slate.hh"
// #include "aux/Debug.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

#include <atomic>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::bdsqr from internal::specialization::bdsqr
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// computes the singular values and, optionally, the right and/or
/// left singular vectors from the singular value decomposition (SVD) of
/// a real (upper or lower) bidiagonal matrix.
/// Generic implementation for any target.
/// @ingroup bdsqr_specialization
///
// ATTENTION: only singular values computed for now, no singular vectors.
// only host computation supported for now
//
template <Target target, typename scalar_t>
void bdsqr(slate::internal::TargetType<target>,
           lapack::Job jobu, lapack::Job jobvt,
           std::vector< blas::real_type<scalar_t> >& D,
           std::vector< blas::real_type<scalar_t> >& E,
           Matrix<scalar_t>& U,
           Matrix<scalar_t>& VT)
{
    trace::Block trace_block("slate::bdsqr");

    using blas::max;

    int64_t m, n, nb, mb;
    int64_t min_mn = D.size();
    //assert(m >= n);

    int mpi_size;

    scalar_t zero = 0.0, one = 1.0;

    // Find the total number of processors.
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    int myrow, mycol;
    int izero = 0;

    int64_t nru  = 0;
    int64_t ncvt = 0;

    int64_t ldu = 1;
    int64_t ldvt = 1;

    std::vector<scalar_t> u1d(1);
    std::vector<scalar_t> vt1d(1);
    scalar_t dummy[1];

    bool wantu  = (jobu  == Job::Vec ||
                   jobu  == Job::AllVec ||
                   jobu  == Job::SomeVec );
    bool wantvt = (jobvt == Job::Vec ||
                   jobvt == Job::AllVec ||
                   jobvt == Job::SomeVec );

    // Compute the local number of the eigenvectors.
    // Build the 1-dim distributed U and VT
    slate::Matrix<scalar_t> U1d;
    slate::Matrix<scalar_t> VT1d;
    if (wantu) {
        m = U.m();
        mb = U.tileMb(0);
        nb = U.tileNb(0);
        myrow = U.mpiRank();
        nru  = numberLocalRowOrCol(m, mb, myrow, izero, mpi_size);
        ldu = max( 1, nru );
        u1d.resize(ldu*min_mn);
        U1d = slate::Matrix<scalar_t>::fromScaLAPACK(
              m, min_mn, &u1d[0], ldu, nb, mpi_size, 1, MPI_COMM_WORLD);
        set(zero, one, U1d);
    }
    if (wantvt) {
        n = VT.n();
        nb = VT.tileNb(0);
        mycol = VT.mpiRank();
        ncvt = numberLocalRowOrCol(n, nb, mycol, izero, mpi_size);
        ldvt = max( 1, min_mn );
        vt1d.resize(ldvt*ncvt);
        VT1d = slate::Matrix<scalar_t>::fromScaLAPACK(
               min_mn, n, &vt1d[0], ldvt, nb, 1, mpi_size, MPI_COMM_WORLD);
        set(zero, one, VT1d);
    }

    // Call the SVD
    lapack::bdsqr(Uplo::Upper, min_mn, ncvt, nru, 0,
                  &D[0], &E[0], &vt1d[0], min_mn, &u1d[0], ldu, dummy, 1);

    // Redistribute the 1-dim distributed U and VT into 2-dim matrices
    if (wantu) {
        U.redistribute(U1d);
    }
    if (wantvt) {
        VT.redistribute(VT1d);
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup bdsqr_specialization
///
template <Target target, typename scalar_t>
void bdsqr(lapack::Job jobu, lapack::Job jobvt,
           std::vector< blas::real_type<scalar_t> >& D,
           std::vector< blas::real_type<scalar_t> >& E,
           Matrix<scalar_t>& U,
           Matrix<scalar_t>& VT,
           Options const& opts)
{
    internal::specialization::bdsqr<target, scalar_t>(internal::TargetType<target>(),
                                    jobu, jobvt, D, E, U, VT);
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
void bdsqr(lapack::Job jobu, lapack::Job jobvt,
           std::vector< blas::real_type<scalar_t> >& D,
           std::vector< blas::real_type<scalar_t> >& E,
           Matrix<scalar_t>& U,
           Matrix<scalar_t>& VT,
           Options const& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            bdsqr<Target::HostTask, scalar_t>(jobu, jobvt, D, E, U, VT, opts);
            break;
        case Target::HostNest:
            bdsqr<Target::HostNest, scalar_t>(jobu, jobvt, D, E, U, VT, opts);
            break;
        case Target::HostBatch:
            bdsqr<Target::HostBatch, scalar_t>(jobu, jobvt, D, E, U, VT, opts);
            break;
        case Target::Devices:
            bdsqr<Target::Devices, scalar_t>(jobu, jobvt, D, E, U, VT, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void bdsqr<float>(
    lapack::Job jobu,
    lapack::Job jobvt,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix<float>& U,
    Matrix<float>& VT,
    Options const& opts);

template
void bdsqr<double>(
    lapack::Job jobu,
    lapack::Job jobvt,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix<double>& U,
    Matrix<double>& VT,
    Options const& opts);

template
void bdsqr< std::complex<float> >(
    lapack::Job jobu,
    lapack::Job jobvt,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix< std::complex<float> >& U,
    Matrix< std::complex<float> >& VT,
    Options const& opts);

template
void bdsqr< std::complex<double> >(
    lapack::Job jobu,
    lapack::Job jobvt,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix< std::complex<double> >& U,
    Matrix< std::complex<double> >& VT,
    Options const& opts);

} // namespace slate
