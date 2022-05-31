// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_METHOD_HH
#define SLATE_METHOD_HH

namespace slate {

typedef int Method;

// This defines default values that MUST be considered in the inner namespaces
constexpr char baseMethodError_str[]  = "error";
constexpr char baseMethodAuto_str[]   = "auto";

const Method baseMethodError = -1;
const Method baseMethodAuto  = 0;

//------------------------------------------------------------------------------
/// Select the right algorithm to perform the trsm
namespace MethodTrsm {

    constexpr char TrsmA_str[] = "A";
    constexpr char TrsmB_str[] = "B";
    const Method Error  = baseMethodError;
    const Method Auto   = baseMethodAuto;
    const Method TrsmA  = 1;  ///< Select trsmA algorithm
    const Method TrsmB  = 2;  ///< Select trsmB algorithm

    template <typename TA, typename TB>
    inline Method select_algo(TA& A, TB& B) {
        return (B.nt() < 2 ? TrsmA : TrsmB);
    }

    inline Method str2methodTrsm(const char* method)
    {
        std::string method_ = method;
        std::transform(
            method_.begin(), method_.end(), method_.begin(), ::tolower );

        if (method_ == "auto")
            return Auto;
        else if (method_ == "a" || method_ == "trsma")
            return TrsmA;
        else if (method_ == "b" || method_ == "trsmb")
            return TrsmB;
        else
            throw slate::Exception("unknown trsm method");
    }

    inline const char* methodTrsm2str(Method method)
    {
        switch (method) {
            case Auto:  return baseMethodAuto_str;
            case TrsmA: return TrsmA_str;
            case TrsmB: return TrsmB_str;
            default:    return baseMethodError_str;
        }
    }

} // namespace MethodTrsm

//------------------------------------------------------------------------------
/// Select the right algorithm to perform the gemm
namespace MethodGemm {

    constexpr char GemmA_str[] = "A";
    constexpr char GemmC_str[] = "C";
    const Method Error  = baseMethodError;
    const Method Auto   = baseMethodAuto;
    const Method GemmA  = 1;  ///< Select gemmA algorithm
    const Method GemmC  = 2;  ///< Select gemmC algorithm

    template <typename TA, typename TB>
    inline Method select_algo(TA& A, TB& B) {
        return (B.nt() < 2 ? GemmA : GemmC);
    }

    inline Method str2methodGemm(const char* method)
    {
        std::string method_ = method;
        std::transform(
            method_.begin(), method_.end(), method_.begin(), ::tolower );

        if (method_ == "auto")
            return Auto;
        else if (method_ == "a" || method_ == "gemma")
            return GemmA;
        else if (method_ == "c" || method_ == "gemmc")
            return GemmC;
        else
            throw slate::Exception("unknown gemm method");
    }

    inline const char* methodGemm2str(Method method)
    {
        switch (method) {
            case Auto:  return baseMethodAuto_str;
            case GemmA: return GemmA_str;
            case GemmC: return GemmC_str;
            default:    return baseMethodError_str;
        }
    }

} // namespace MethodGemm

//------------------------------------------------------------------------------
/// Select the right algorithm to perform the hemm
namespace MethodHemm {

    constexpr char HemmA_str[] = "A";
    constexpr char HemmC_str[] = "C";
    const Method Error  = baseMethodError;
    const Method Auto   = baseMethodAuto;
    const Method HemmA  = 1;  ///< Select hemmA algorithm
    const Method HemmC  = 2;  ///< Select hemmC algorithm

    template <typename TA, typename TB>
    inline Method select_algo(TA& A, TB& B) {
        return (B.nt() < 2 ? HemmA : HemmC);
    }

    inline Method str2methodHemm(const char* method)
    {
        std::string method_ = method;
        std::transform(
            method_.begin(), method_.end(), method_.begin(), ::tolower );

        if (method_ == "auto")
            return Auto;
        else if (method_ == "a" || method_ == "hemma")
            return HemmA;
        else if (method_ == "c" || method_ == "hemmc")
            return HemmC;
        else
            throw slate::Exception("unknown hemm method");
    }

    inline const char* methodHemm2str(Method method)
    {
        switch (method) {
            case Auto:  return baseMethodAuto_str;
            case HemmA: return HemmA_str;
            case HemmC: return HemmC_str;
            default:    return baseMethodError_str;
        }
    }

} // namespace MethodHemm

} // namespace slate

#endif // SLATE_METHOD_HH
