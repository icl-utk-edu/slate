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

#ifndef SLATE_EXCEPTION_HH
#define SLATE_EXCEPTION_HH

#include <string>
#include <exception>

#include "slate/internal/cuda.hh"
#include "slate/internal/cublas.hh"
#include "slate/internal/mpi.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Base class for SLATE exceptions.
class Exception : public std::exception {
public:
    Exception()
        : std::exception()
    {}

    /// Sets the what() message to msg.
    Exception(std::string const& msg)
        : std::exception(),
          msg_(msg)
    {}

    /// Sets the what() message to msg with func, file, line appended.
    Exception(std::string const& msg,
              const char* func, const char* file, int line)
        : std::exception(),
          msg_(msg + " in " + func + " at " + file + ":" + std::to_string(line))
    {}

    /// @return message describing the execption.
    virtual char const* what() const noexcept override
    {
        return msg_.c_str();
    }

protected:
    /// Sets the what() message to msg with func, file, line appended.
    void what(std::string const& msg,
              const char* func, const char* file, int line)
    {
        msg_ = msg + " in " + func + " at " + file + ":" + std::to_string(line);
    }

    std::string msg_;
};

/// Throws Exception with given message.
#define slate_error(msg) \
    do { \
        throw slate::Exception(msg, __func__, __FILE__, __LINE__); \
    } while(0)

//------------------------------------------------------------------------------
class NotImplemented : public Exception {
public:
    NotImplemented(const char* msg,
                   const char* func,
                   const char* file,
                   int line)
        : Exception(std::string("SLATE ERROR: Not yet implemented: ") + msg,
                    func, file, line)
    {}
};

/// Throws NotImplemented exception with given message.
#define slate_not_implemented(msg) \
    do { \
        throw slate::NotImplemented(msg, __func__, __FILE__, __LINE__); \
    } while(0)

//------------------------------------------------------------------------------
/// Exception class for slate_error_if().
class TrueConditionException : public Exception {
public:
    TrueConditionException(const char* cond,
                           const char* func,
                           const char* file,
                           int line)
        : Exception(std::string("SLATE ERROR: Error condition '")
                    + cond + "' occured",
                    func, file, line)
    {}
};

/// Throws TrueConditionException if cond is true.
#define slate_error_if(cond) \
    do { \
        if ((cond)) \
            throw slate::TrueConditionException( \
                #cond, __func__, __FILE__, __LINE__); \
    } while(0)

//------------------------------------------------------------------------------
/// Exception class for slate_assert().
class FalseConditionException : public Exception {
public:
    FalseConditionException(const char* cond,
                            const char* func,
                            const char* file,
                            int line)
        : Exception(std::string("SLATE ERROR: Error check '")
                    + cond + "' failed",
                    func, file, line)
    {}
};

/// Throws FalseConditionException if cond is false.
#define slate_assert(cond) \
    do { \
        if (! (cond)) \
            throw slate::FalseConditionException( \
                #cond, __func__, __FILE__, __LINE__); \
    } while(0)

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

//------------------------------------------------------------------------------
/// Exception class for slate_cuda_call().
class CudaException : public Exception {
public:
    CudaException(const char* call,
                  cudaError_t code,
                  const char* func,
                  const char* file,
                  int line)
        : Exception()
    {
        const char* name = cudaGetErrorName(code);
        const char* string = cudaGetErrorString(code);

        what(std::string("SLATE CUDA ERROR: ")
             + call + " failed: " + string
             + " (" + name + "=" + std::to_string(code) + ")",
             func, file, line);
    }
};

/// Throws a CudaException if the CUDA call fails.
/// Example:
///
///     try {
///         slate_cuda_call( cudaSetDevice( device ) );
///     }
///     catch (CudaException& e) {
///         ...
///     }
///
#define slate_cuda_call(call) \
    do { \
        cudaError_t slate_cuda_call_ = call; \
        if (slate_cuda_call_ != cudaSuccess) \
            throw slate::CudaException( \
                #call, slate_cuda_call_, __func__, __FILE__, __LINE__); \
    } while(0)

//------------------------------------------------------------------------------
const char* getCublasErrorName(cublasStatus_t status);

//------------------------------------------------------------------------------
/// Exception class for slate_cublas_call().
class CublasException : public Exception {
public:
    CublasException(const char* call,
                    cublasStatus_t code,
                    const char* func,
                    const char* file,
                    int line)
        : Exception()
    {
        const char* name = getCublasErrorName(code);

        what(std::string("SLATE CUBLAS ERROR: ")
             + call + " failed: " + name
             + " (" + std::to_string(code) + ")",
             func, file, line);
    }
};

/// Throws a CublasException if the CUBLAS call fails.
/// Example:
///
///     try {
///         slate_cublas_call( cublasCreate( &handle ) );
///     }
///     catch (CublasException& e) {
///         ...
///     }
///
#define slate_cublas_call(call) \
    do { \
        cublasStatus_t slate_cublas_call_ = call; \
        if (slate_cublas_call_ != CUBLAS_STATUS_SUCCESS) \
            throw slate::CublasException( \
                #call, slate_cublas_call_, __func__, __FILE__, __LINE__); \
    } while(0)

} // namespace slate

#endif // SLATE_EXCEPTION_HH
