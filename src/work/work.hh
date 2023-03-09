// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_WORK_HH
#define SLATE_WORK_HH

#include "slate/Matrix.hh"
#include "slate/TriangularMatrix.hh"

namespace slate {

//------------------------------------------------------------------------------
/// @namespace slate::work
/// Namespace used for SLATE actual work implementation.
/// It is intended that application code would not call any internal SLATE
/// functions.
namespace work {

//-----------------------------------------
// trmm()
template <Target target=Target::HostTask, typename scalar_t>
void trmm(Side side, scalar_t alpha, TriangularMatrix<scalar_t> A,
                                               Matrix<scalar_t> B,
          uint8_t* bcast, uint8_t* gemm, int64_t lookahead=1);

//-----------------------------------------
// trsm()
template <Target target=Target::HostTask, typename scalar_t>
void trsm(Side side, scalar_t alpha, TriangularMatrix<scalar_t> A,
                                               Matrix<scalar_t> B,
          uint8_t* row, Options const& opts);

//-----------------------------------------
// trsmA()
template <Target target=Target::HostTask, typename scalar_t>
void trsmA(Side side, scalar_t alpha, TriangularMatrix<scalar_t> A,
                                                Matrix<scalar_t> B,
          uint8_t* row, Options const& opts);

} // namespace work
} // namespace slate

#endif // SLATE_WORK_HH
