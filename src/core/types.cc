// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/types.hh"

namespace slate {

//------------------------------------------------------------------------------
MPI_Datatype mpi_type<float >::value = MPI_FLOAT;
MPI_Datatype mpi_type<double>::value = MPI_DOUBLE;
MPI_Datatype mpi_type< std::complex<float>  >::value = MPI_C_COMPLEX;
MPI_Datatype mpi_type< std::complex<double> >::value = MPI_C_DOUBLE_COMPLEX;

MPI_Datatype mpi_type< max_loc_type<float>  >::value = MPI_FLOAT_INT;
MPI_Datatype mpi_type< max_loc_type<double> >::value = MPI_DOUBLE_INT;

} // namespace slate
