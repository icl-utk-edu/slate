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

#ifndef SLATE_EXCEPTION_HH
#define SLATE_EXCEPTION_HH

namespace slate {

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
enum class Error
{
    Argument, ///< invalid argument
    Cuda,     ///< CUDA error
    Mpi,      ///< MPI error
};

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
class Exception : public std::exception
{
public:
    Exception(Error error, const char* file, const char* func, int line)
        : error_(error), file_(file), func_(func), line_(line)
    {}

protected:
    Error error_;
    std::string file_;
    std::string func_;
    int line_;
};

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
class TrueConditionException : public Exception
{
public:
    TrueConditionException(const char* cond,
                           Error error,
                           const char* file,
                           const char* func,
                           int line)
        : Exception(error, file, func, line),
          cond_(cond)
    {
        msg_ = "SLATE ERROR: Error condition '" + cond_ +
               "' occured, in file '" + file_ +
               "', function '" + func_ +
               "', line " + std::to_string(line_) + ".";
    }

    virtual char const* what() const noexcept
    {
        return msg_.c_str();
    }

protected:
    std::string cond_;
    std::string msg_;
};

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
class FalseConditionException : public Exception
{
public:
    FalseConditionException(const char* cond,
                            Error error,
                            const char* file,
                            const char* func,
                            int line)
        : Exception(error, file, func, line),
          cond_(cond)
    {
        msg_ = "SLATE ERROR: Error check '" + cond_ +
               "' failed, in file '" + file_ +
               "', function '" + func_ +
               "', line " + std::to_string(line_) + ".";
    }

    virtual char const* what() const noexcept
    {
        return msg_.c_str();
    }

protected:
    std::string cond_;
    std::string msg_;
};

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
class MpiException : public Exception
{
public:
    MpiException(const char* call,
                 int code,
                 const char* file,
                 const char* func,
                 int line)
        : Exception(Error::Mpi, file, func, line),
          call_(call),
          code_(code)
    {
        int resultlen;
        int retval = MPI_Error_string(code_, string_, &resultlen);

        msg_ = "SLATE ERROR: The MPI call '" + call_ +
               "' failed, in file '" + file_ +
               "', function '" + func_ +
               "', line " + std::to_string(line_) +
               ", returning error code " + std::to_string(code_) +
               ", with the corresponding error string:\n" + string_;
    }

    virtual char const* what() const noexcept
    {
        return msg_.c_str();
    }

protected:
    int code_;
    char* string_;
    std::string call_;
    std::string msg_;
};

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
class CudaException : public Exception
{
public:
    CudaException(const char* call,
                  cudaError_t code,
                  const char* file,
                  const char* func,
                  int line)
        : Exception(Error::Cuda, file, func, line),
          call_(call),
          code_(code)
    {
        name_ = cudaGetErrorName(code_);
        string_ = cudaGetErrorString(code_);

        msg_ = "SLATE ERROR: The CUDA call '" + call_ +
               "' failed, in file '" + file_ +
               "', function '" + func_ +
               "', line " + std::to_string(line_) +
               ", returning error " + name_ +
               ", with the corresponding error string:\n" + string_;
    }

    virtual char const* what() const noexcept
    {
        return msg_.c_str();
    }

protected:
    cudaError_t code_;
    const char* name_;
    const char* string_;
    std::string call_;
    std::string msg_;
};

} // namespace slate

#endif // SLATE_EXCEPTION_HH
