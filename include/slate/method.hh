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
    inline Method select_algo(TA& A, TB& B, Options const& opts) {
        // TODO replace the default value by a unique value located elsewhere
        Target target = get_option( opts, Option::Target, Target::HostTask );

        Method method = (B.nt() < 2 ? TrsmA : TrsmB);

        // XXX For now, when target == device, we fallback to trsmB on device
        if (target == Target::Devices && method == TrsmA)
            method = TrsmB;

        return method;
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
    inline Method select_algo(TA& A, TB& B, Options& opts) {
        // TODO replace the default value by a unique value located elsewhere
        Target target = get_option( opts, Option::Target, Target::HostTask );

        Method method = (B.nt() < 2 ? GemmA : GemmC);

        if (target == Target::Devices && method == GemmA && A.num_devices() > 1)
            opts[ Option::Target ] = Target::HostTask;

        return method;
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
    inline Method select_algo(TA& A, TB& B, Options const& opts) {
        // TODO replace the default value by a unique value located elsewhere
        Target target = get_option( opts, Option::Target, Target::HostTask );

        Method method = (B.nt() < 2 ? HemmA : HemmC);

        // XXX For now, when target == device, we fallback to HemmC on device
        if (target == Target::Devices && method == HemmA)
            method = HemmC;

        return method;
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

//------------------------------------------------------------------------------
/// Select the right algorithm to perform the AH * A in CholQR
namespace MethodCholQR {
    static constexpr char HerkC_str[] = "herkC";
    static constexpr char GemmA_str[] = "gemmA";
    static constexpr char GemmC_str[] = "gemmC";
    static const Method Error = baseMethodError; ///< Error flag
    static const Method Auto  = baseMethodAuto;  ///< Let the algorithm decide
    static const Method HerkC = 1;  ///< Select herkC algorithm
    static const Method GemmA = 2;  ///< Select gemmA algorithm
    static const Method GemmC = 3;  ///< Select gemmC algorithm

    template <typename TA, typename TB>
    inline Method select_algo(TA& A, TB& B, Options const& opts) {

        Target target = get_option( opts, Option::Target, Target::HostTask );

        Method method = (target == Target::Devices ? HerkC : GemmA);

        return method;
    }

    inline Method str2methodCholQR(const char* method)
    {
        std::string method_ = method;
        std::transform(
            method_.begin(), method_.end(), method_.begin(), ::tolower );

        if (method_ == "auto")
            return Auto;
        else if (method_ == "herkc")
            return HerkC;
        else if (method_ == "gemmc")
            return GemmC;
        else if (method_ == "gemma")
            return GemmA;
        else
            throw slate::Exception("unknown cholQR method");
    }

    inline const char* methodCholQR2str(Method method)
    {
        switch (method) {
            case Auto:   return baseMethodAuto_str;
            case HerkC:  return HerkC_str;
            case GemmA:  return GemmA_str;
            case GemmC:  return GemmC_str;
            default:     return baseMethodError_str;
        }
    }

} // namespace MethodCholQR

//------------------------------------------------------------------------------
/// Select the right algorithm to solve least squares problems
namespace MethodGels {
    static constexpr char Cholqr_str[]  = "cholqr";
    static constexpr char Geqrf_str[]   = "qr";
    static const Method Error   = baseMethodError; ///< Error flag
    static const Method Auto    = baseMethodAuto;  ///< Let the algorithm decide
    static const Method Cholqr  = 1;  ///< Select cholqr algorithm
    static const Method Geqrf   = 2;  ///< Select geqrf algorithm

    template <typename TA, typename TB>
    inline Method select_algo(TA& A, TB& B, Options const& opts) {
        return Geqrf;
    }

    inline Method str2methodGels(const char* method)
    {
        std::string method_ = method;
        std::transform(
            method_.begin(), method_.end(), method_.begin(), ::tolower );

        if (method_ == "auto")
            return Auto;
        else if (method_ == "qr")
            return Geqrf;
        else if (method_ == "cholqr")
            return Cholqr;
        else
            throw slate::Exception("unknown gels method");
    }

    inline const char* methodGels2str(Method method)
    {
        switch (method) {
            case Auto:   return baseMethodAuto_str;
            case Geqrf:  return Geqrf_str;
            case Cholqr: return Cholqr_str;
            default:     return baseMethodError_str;
        }
    }

} // namespace MethodGels

//------------------------------------------------------------------------------
/// Select the LU factorization algorithm.
namespace MethodLU {

    static constexpr char PartialPiv_str[] = "PPLU";
    static constexpr char CALU_str[]       = "CALU";
    static constexpr char NoPiv_str[]      = "NoPiv";
    static const Method Error = baseMethodError; ///< Error flag
    static const Method PartialPiv = 1;  ///< Select partial pivoting LU
    static const Method CALU       = 2;  ///< Select communication avoiding LU
    static const Method NoPiv      = 3;  ///< Select no pivoting LU

    inline Method str2methodLU( const char* method )
    {
        std::string method_ = method;
        std::transform(
            method_.begin(), method_.end(), method_.begin(), ::tolower );

        if (method_ == "pplu" || method_ == "partialpiv")
            return PartialPiv;
        else if (method_ == "calu")
            return CALU;
        else if (method_ == "nopiv")
            return NoPiv;
        else
            throw slate::Exception("unknown LU method");
    }

    inline const char* methodLU2str( Method method )
    {
        switch (method) {
            case PartialPiv: return PartialPiv_str;
            case CALU:       return CALU_str;
            case NoPiv:      return NoPiv_str;
            default:         return baseMethodError_str;
        }
    }

} // namespace MethodLU

} // namespace slate

#endif // SLATE_METHOD_HH
