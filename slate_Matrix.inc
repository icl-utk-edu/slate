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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

//-----------------------------------------
// gemm()
template <Target target = Target::HostTask>
static void gemm(blas::Op opa, blas::Op opb,
                 scalar_t alpha, Matrix &&a,
                                  Matrix &&b,
                 scalar_t beta,  Matrix &&c,
                 int priority = 0);

static void gemm(internal::TargetType<Target::HostTask>,
                 blas::Op opa, blas::Op opb,
                 scalar_t alpha, Matrix &a,
                                  Matrix &b,
                 scalar_t beta,  Matrix &c,
                 int priority = 0);

//-----------------------------------------
// potrf()
template <Target target = Target::HostTask>
static void potrf(blas::Uplo uplo, Matrix &&a, int priority = 0);

static void potrf(internal::TargetType<Target::HostTask>,
                  blas::Uplo uplo, Matrix &a, int priority = 0);

//-----------------------------------------
// syrk()
template <Target target = Target::HostTask>
static void syrk(blas::Uplo uplo, blas::Op op,
                 scalar_t alpha, Matrix &&a,
                 scalar_t beta,  Matrix &&c,
                 int priority = 0);

static void syrk(internal::TargetType<Target::HostTask>,
                 blas::Uplo uplo, blas::Op op,
                 scalar_t alpha, Matrix &a,
                 scalar_t beta,  Matrix &c,
                 int priority = 0);

static void syrk(internal::TargetType<Target::HostNest>,
                 blas::Uplo uplo, blas::Op op,
                 scalar_t alpha, Matrix &a,
                 scalar_t beta,  Matrix &c,
                 int priority = 0);

static void syrk(internal::TargetType<Target::HostBatch>,
                 blas::Uplo uplo, blas::Op op,
                 scalar_t alpha, Matrix &a,
                 scalar_t beta,  Matrix &c,
                 int priority = 0);

static void syrk(internal::TargetType<Target::Devices>,
                 blas::Uplo uplo, blas::Op op,
                 scalar_t alpha, Matrix &a,
                 scalar_t beta,  Matrix &c,
                 int priority = 0);

//-----------------------------------------
// trsm()
template <Target target = Target::HostTask>
static void trsm(blas::Side side, blas::Uplo uplo,
                 blas::Op op, blas::Diag diag,
                 scalar_t alpha, Matrix &&a,
                                  Matrix &&b,
                 int priority = 0);

static void trsm(internal::TargetType<Target::HostTask>,
                 blas::Side side, blas::Uplo uplo,
                 blas::Op op, blas::Diag diag,
                 scalar_t alpha, Matrix &a,
                                  Matrix &b,
                 int priority = 0);
