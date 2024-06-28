// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_MPI_HH
#define SLATE_MPI_HH

#include "slate/Exception.hh"

#include <mpi.h>

namespace slate {

//------------------------------------------------------------------------------
/// Exception class for slate_mpi_call().
class MpiException : public Exception {
public:
    MpiException(const char* call,
                 int code,
                 const char* func,
                 const char* file,
                 int line)
        : Exception()
    {
        char string[MPI_MAX_ERROR_STRING] = "unknown error";
        int resultlen;
        MPI_Error_string(code, string, &resultlen);

        what(std::string("SLATE MPI ERROR: ")
             + call + " failed: " + string
             + " (" + std::to_string(code) + ")",
             func, file, line);
    }
};

/// Throws an MpiException if the MPI call fails.
/// Example:
///
///     try {
///         slate_mpi_call( MPI_Barrier( MPI_COMM_WORLD ) );
///     }
///     catch (MpiException& e) {
///         ...
///     }
///
#define slate_mpi_call(call) \
    do { \
        int slate_mpi_call_ = call; \
        if (slate_mpi_call_ != MPI_SUCCESS) \
            throw slate::MpiException( \
                #call, slate_mpi_call_, __func__, __FILE__, __LINE__); \
    } while(0)

} // namespace slate

#endif // SLATE_MPI_HH
