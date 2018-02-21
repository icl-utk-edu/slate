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

#ifndef SLATE_TRACE_HH
#define SLATE_TRACE_HH

#include <stdio.h>
#include <vector>

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#ifdef _OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

namespace slate {
namespace trace {

///-----------------------------------------------------------------------------
/// \class
/// \brief X 11 color names (https://en.wikipedia.org/wiki/X11_color_names)
///
enum class Color {
    Pink                 = 0xFFC0CB,
    LightPink            = 0xFFB6C1,
    HotPink              = 0xFF69B4,
    DeepPink             = 0xFF1493,
    PaleVioletRed        = 0xDB7093,
    MediumVioletRed      = 0xC71585,
    LightSalmon          = 0xFFA07A,
    Salmon               = 0xFA8072,
    DarkSalmon           = 0xE9967A,
    LightCoral           = 0xF08080,
    IndianRed            = 0xCD5C5C,
    Crimson              = 0xDC143C,
    FireBrick            = 0xB22222,
    DarkRed              = 0x8B0000,
    Red                  = 0xFF0000,
    OrangeRed            = 0xFF4500,
    Tomato               = 0xFF6347,
    Coral                = 0xFF7F50,
    DarkOrange           = 0xFF8C00,
    Orange               = 0xFFA500,
    Yellow               = 0xFFFF00,
    LightYellow          = 0xFFFFE0,
    LemonChiffon         = 0xFFFACD,
    LightGoldenrodYellow = 0xFAFAD2,
    PapayaWhip           = 0xFFEFD5,
    Moccasin             = 0xFFE4B5,
    PeachPuff            = 0xFFDAB9,
    PaleGoldenrod        = 0xEEE8AA,
    Khaki                = 0xF0E68C,
    DarkKhaki            = 0xBDB76B,
    Gold                 = 0xFFD700,
    Cornsilk             = 0xFFF8DC,
    BlanchedAlmond       = 0xFFEBCD,
    Bisque               = 0xFFE4C4,
    NavajoWhite          = 0xFFDEAD,
    Wheat                = 0xF5DEB3,
    BurlyWood            = 0xDEB887,
    Tan                  = 0xD2B48C,
    RosyBrown            = 0xBC8F8F,
    SandyBrown           = 0xF4A460,
    Goldenrod            = 0xDAA520,
    DarkGoldenrod        = 0xB8860B,
    Peru                 = 0xCD853F,
    Chocolate            = 0xD2691E,
    SaddleBrown          = 0x8B4513,
    Sienna               = 0xA0522D,
    Brown                = 0xA52A2A,
    Maroon               = 0x800000,
    DarkOliveGreen       = 0x556B2F,
    Olive                = 0x808000,
    OliveDrab            = 0x6B8E23,
    YellowGreen          = 0x9ACD32,
    LimeGreen            = 0x32CD32,
    Lime                 = 0x00FF00,
    LawnGreen            = 0x7CFC00,
    Chartreuse           = 0x7FFF00,
    GreenYellow          = 0xADFF2F,
    SpringGreen          = 0x00FF7F,
    MediumSpringGreen    = 0x00FA9A,
    LightGreen           = 0x90EE90,
    PaleGreen            = 0x98FB98,
    DarkSeaGreen         = 0x8FBC8F,
    MediumAquamarine     = 0x66CDAA,
    MediumSeaGreen       = 0x3CB371,
    SeaGreen             = 0x2E8B57,
    ForestGreen          = 0x228B22,
    Green                = 0x008000,
    DarkGreen            = 0x006400,
    Aqua                 = 0x00FFFF,
    Cyan                 = 0x00FFFF,
    LightCyan            = 0xE0FFFF,
    PaleTurquoise        = 0xAFEEEE,
    Aquamarine           = 0x7FFFD4,
    Turquoise            = 0x40E0D0,
    MediumTurquoise      = 0x48D1CC,
    DarkTurquoise        = 0x00CED1,
    LightSeaGreen        = 0x20B2AA,
    CadetBlue            = 0x5F9EA0,
    DarkCyan             = 0x008B8B,
    Teal                 = 0x008080,
    LightSteelBlue       = 0xB0C4DE,
    PowderBlue           = 0xB0E0E6,
    LightBlue            = 0xADD8E6,
    SkyBlue              = 0x87CEEB,
    LightSkyBlue         = 0x87CEFA,
    DeepSkyBlue          = 0x00BFFF,
    DodgerBlue           = 0x1E90FF,
    CornflowerBlue       = 0x6495ED,
    SteelBlue            = 0x4682B4,
    RoyalBlue            = 0x4169E1,
    Blue                 = 0x0000FF,
    MediumBlue           = 0x0000CD,
    DarkBlue             = 0x00008B,
    Navy                 = 0x000080,
    MidnightBlue         = 0x191970,
    Lavender             = 0xE6E6FA,
    Thistle              = 0xD8BFD8,
    Plum                 = 0xDDA0DD,
    Violet               = 0xEE82EE,
    Orchid               = 0xDA70D6,
    Fuchsia              = 0xFF00FF,
    Magenta              = 0xFF00FF,
    MediumOrchid         = 0xBA55D3,
    MediumPurple         = 0x9370DB,
    BlueViolet           = 0x8A2BE2,
    DarkViolet           = 0x9400D3,
    DarkOrchid           = 0x9932CC,
    DarkMagenta          = 0x8B008B,
    Purple               = 0x800080,
    Indigo               = 0x4B0082,
    DarkSlateBlue        = 0x483D8B,
    SlateBlue            = 0x6A5ACD,
    MediumSlateBlue      = 0x7B68EE,
    White                = 0xFFFFFF,
    Snow                 = 0xFFFAFA,
    Honeydew             = 0xF0FFF0,
    MintCream            = 0xF5FFFA,
    Azure                = 0xF0FFFF,
    AliceBlue            = 0xF0F8FF,
    GhostWhite           = 0xF8F8FF,
    WhiteSmoke           = 0xF5F5F5,
    Seashell             = 0xFFF5EE,
    Beige                = 0xF5F5DC,
    OldLace              = 0xFDF5E6,
    FloralWhite          = 0xFFFAF0,
    Ivory                = 0xFFFFF0,
    AntiqueWhite         = 0xFAEBD7,
    Linen                = 0xFAF0E6,
    LavenderBlush        = 0xFFF0F5,
    MistyRose            = 0xFFE4E1,
    Gainsboro            = 0xDCDCDC,
    LightGray            = 0xD3D3D3,
    Silver               = 0xC0C0C0,
    DarkGray             = 0xA9A9A9,
    Gray                 = 0x808080,
    DimGray              = 0x696969,
    LightSlateGray       = 0x778899,
    SlateGray            = 0x708090,
    DarkSlateGray        = 0x2F4F4F,
    Black                = 0x000000
};

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
class Event {
public:
    friend class Trace;

    Event() {};
    Event(Color color)
        : start_(omp_get_wtime()),
          color_(color) {}

    void stop() { stop_ = omp_get_wtime(); }

private:
    double start_;
    double stop_;
    Color color_;
};

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
class Trace {
public:
    static void on() { tracing_ = true; }
    static void off() { tracing_ = false; }

    static void insert(Event event);
    static void finish();

private:
    static double getTimeSpan();
    static void printThreads(int mpi_rank, int mpi_size,
                             double timespan, FILE *trace_file);
    static void sendThreads();
    static void recvThreads(int rank);

    static const int width_ = 2390;
    static const int height_ = 1000;

    static bool tracing_;
    static int num_threads_;
    static std::vector<std::vector<Event>> events_;
};

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
class Block {
public:
    Block(Color color)
        : event_(color) {}

    ~Block() { Trace::insert(event_); }
private:
    Event event_;
};

} // namespace trace
} // namespace slate

#endif // SLATE_TRACE_HH
