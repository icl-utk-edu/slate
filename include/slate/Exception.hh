// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_EXCEPTION_HH
#define SLATE_EXCEPTION_HH

#include <string>
#include <exception>

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

} // namespace slate

#endif // SLATE_EXCEPTION_HH
