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
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"
#include "slate_steqr2.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::steqr2 from internal::specialization::steqr2
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// computes all eigenvalues/eigenvectors of a symmetric tridiagonal matrix
/// using the Pal-Walker-Kahan variant of the QL or QR algorithm.
/// Generic implementation for any target.
/// @ingroup steqr2_specialization
///
// ATTENTION: only host computation supported for now
//
template <Target target, typename scalar_t>
void steqr2(slate::internal::TargetType<target>,
           Job jobz,
           std::vector< blas::real_type<scalar_t> >& D,
           std::vector< blas::real_type<scalar_t> >& E,
           Matrix<scalar_t>& Z)
{
    trace::Block trace_block("lapack::steqr2");

    using blas::max;

    int64_t nb;
    int64_t n = D.size();

    int mpi_size;
    int64_t info = 0;
    int64_t nrc, ldc;

    int myrow;
    int izero = 0;
    scalar_t zero = 0.0, one = 1.0;

    bool wantz = (jobz == Job::Vec);

    // Find the total number of processors.
    slate_mpi_call(
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    nrc = 0;
    ldc = 1;
    std::vector<scalar_t> Q(1);
    std::vector< blas::real_type<scalar_t> > work(max( 1, 2*n-2 ));

    // Compute the local number of the eigenvectors.
    // Build the matrix Z using 1-dim grid.
    slate::Matrix<scalar_t> Z1d;
    if (wantz) {
        n = Z.n();
        nb = Z.tileNb(0);
        myrow = Z.mpiRank();
        nrc = numberLocalRowOrCol(n, nb, myrow, izero, mpi_size);
        ldc = max( 1, nrc );
        Q.resize(nrc*n);
        Z1d = slate::Matrix<scalar_t>::fromScaLAPACK(
              n, n, &Q[0], nrc, nb, mpi_size, 1, MPI_COMM_WORLD);
        set(zero, one, Z1d);
    }

    // Call the eigensolver.
    slate_steqr2( jobz, n, &D[0], &E[0], &Q[0], ldc, nrc, &work[0], &info);

    // Redstribute the 1-dim eigenvector matrix into 2-dim matrix.
    if (wantz) {
        Z.redistribute(Z1d);
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup bdsqr_specialization
///
template <Target target, typename scalar_t>
void steqr2(lapack::Job job,
           std::vector< blas::real_type<scalar_t> >& D,
           std::vector< blas::real_type<scalar_t> >& E,
           Matrix<scalar_t>& Z,
           Options const& opts)
{
    internal::specialization::steqr2<target, scalar_t>(
                                    internal::TargetType<target>(),
                                    job, D, E, Z);
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
void steqr2(lapack::Job job,
           std::vector< blas::real_type<scalar_t> >& D,
           std::vector< blas::real_type<scalar_t> >& E,
           Matrix<scalar_t>& Z,
           Options const& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    // only HostTask implementation is provided, since it calls LAPACK only
    switch (target) {
        case Target::Host:
        case Target::HostTask:
        case Target::HostNest:
        case Target::HostBatch:
        case Target::Devices:
            steqr2<Target::HostTask, scalar_t>(job, D, E, Z, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void steqr2<float>(
    Job job,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix<float>& Z,
    Options const& opts);

template
void steqr2<double>(
    Job job,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix<double>& Z,
    Options const& opts);

template
void steqr2< std::complex<float> >(
    Job job,
    std::vector<float>& D,
    std::vector<float>& E,
    Matrix< std::complex<float> >& Z,
    Options const& opts);

template
void steqr2< std::complex<double> >(
    Job job,
    std::vector<double>& D,
    std::vector<double>& E,
    Matrix< std::complex<double> >& Z,
    Options const& opts);

} // namespace slate
