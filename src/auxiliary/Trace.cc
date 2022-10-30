// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/Trace.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <string>

namespace slate {
namespace trace {

//------------------------------------------------------------------------------
/// X11 color names (https://en.wikipedia.org/wiki/X11_color_names)
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

//------------------------------------------------------------------------------
const int max_nest          = 4;
const int font_size         = 18;
const int max_rank_chars    = 6;
const int ylabel_width      = font_size * max_rank_chars;

// Used by Block. If this is a static member of Block, some compilers
// have a link error.
static int s_nest = 0;
#pragma omp threadprivate( s_nest )

int Trace::width_  = 0;
int Trace::height_ = 0;

double Trace::vscale_ = 50;
double Trace::hscale_ = 100;

bool Trace::tracing_ = false;
int Trace::num_threads_ = omp_get_max_threads();

std::string comment_;

std::vector<std::vector<Event>> Trace::events_ =
    std::vector<std::vector<Event>>(omp_get_max_threads());

std::map<std::string, Color> function_color_ = {

    {"blas::add",   Color::LightSkyBlue},
    {"blas::gemm",  Color::MediumAquamarine},
    {"blas::hemm",  Color::MediumAquamarine},
    {"blas::her2k", Color::MediumAquamarine},
    {"blas::herk",  Color::MediumAquamarine},
    {"blas::symm",  Color::CornflowerBlue},
    {"blas::syr2k", Color::CornflowerBlue},
    {"blas::syrk",  Color::CornflowerBlue},
    {"blas::trmm",  Color::MediumOrchid},
    {"blas::trsm",  Color::MediumPurple},
    {"blas::scale", Color::Goldenrod},

    {"cblas_gemm_batch",  Color::DarkGreen},
    {"blas::batch::gemm", Color::PaleGreen},

    {"blas::device_malloc",        Color::HotPink},
    {"blas::device_malloc_pinned", Color::DeepPink},
    {"blas::device_memcpy",        Color::LightGray},
    {"blas::device_memcpy2D",      Color::LightGray},
    {"blas::device_free",          Color::LightSalmon},
    {"blas::device_free_pinned",   Color::Salmon},

    {"internal::gebr1",  Color::Moccasin},
    {"internal::gebr2",  Color::LightBlue},
    {"internal::gebr3",  Color::LightPink},
    {"internal::swap",  Color::Thistle},

    {"internal::hebr1",  Color::Moccasin},
    {"internal::hebr2",  Color::LightBlue},
    {"internal::hebr3",  Color::LightBlue},

    {"internal::geqrf",  Color::RosyBrown},
    {"internal::ttqrt",  Color::DeepSkyBlue},
    {"internal::unmqr",  Color::BurlyWood},
    {"internal::ttmqr",  Color::DarkGreen},

    {"lapack::geqrf",  Color::RosyBrown},
    {"lapack::getrf",  Color::RosyBrown},
    {"lapack::lange",  Color::LightBlue},
    {"lapack::lauum",  Color::DodgerBlue},
    {"lapack::potrf",  Color::BurlyWood},
    {"lapack::tpmqrt", Color::LightSkyBlue},
    {"lapack::tpqrt",  Color::DeepSkyBlue},
    {"lapack::trtri",  Color::DodgerBlue},
    {"lapack::larft",  Color::LightBlue},

    {"Memory::alloc", Color::Aqua},
    {"Memory::free",  Color::Aquamarine},

    {"MPI_Reduce",            Color::Purple},
    {"MPI_Allreduce",         Color::Purple},
    {"MPI_Barrier",           Color::SlateGray},
    {"MPI_Bcast",             Color::Purple},
    {"MPI_Comm_create_group", Color::DarkRed},
    {"MPI_Recv",              Color::Crimson},
    {"MPI_Send",              Color::LightCoral},

    {"slate::device::genorm",    Color::LightSkyBlue},
    {"slate::device::transpose", Color::SkyBlue},
    {"slate::convert_layout",    Color::DeepSkyBlue},
    {"slate::bdsqr",             Color::DeepSkyBlue},
    {"slate::gatherAll",         Color::RosyBrown},

    {"task::panel",     Color::LightSkyBlue},
    {"task::lookahead", Color::Orange},
    {"task::trailing",  Color::Yellow},
    {"task::bcast",     Color::LightPink},
};

//------------------------------------------------------------------------------
/// Create a block, which marks the beginning of an event in the trace.
///
Block::Block( const char* name, int64_t index )
    : event_( name, index, s_nest++ )
{}

//------------------------------------------------------------------------------
/// Destroy a block, which marks the end of an event in the trace.
///
Block::~Block()
{
    s_nest--;
    Trace::insert( event_ );
}

//------------------------------------------------------------------------------
///
void Trace::insert(Event event)
{
    if (tracing_) {
        event.stop();
        events_[omp_get_thread_num()].push_back(event);
    }
}

//------------------------------------------------------------------------------
void Trace::comment(std::string const& str)
{
    comment_ += str;
}

//------------------------------------------------------------------------------
/// Returns string that is the same as name, but with non-alphanumeric or -
/// characters replaced with _, to be suitable as a CSS class name.
///
std::string cleanName(std::string const& name)
{
    std::string name_cleaned = name;
    for (size_t i = 0; i < name_cleaned.size(); ++i) {
        char ch = name_cleaned[i];
        if (! (isalnum(ch) || ch == '_' || ch == '-'))
            name_cleaned[i] = '_';
    }
    return name_cleaned;
}

//------------------------------------------------------------------------------
/// Returns an rgb color darkened by factor, i.e.,
/// r * factor, g * factor, b * factor.
///
unsigned int darken(unsigned int rgb, double factor)
{
    unsigned int r = (rgb & 0xff0000) >> 16;
    unsigned int g = (rgb & 0x00ff00) >>  8;
    unsigned int b = (rgb & 0x0000ff);
    r *= factor;
    g *= factor;
    b *= factor;
    rgb = (r << 16) | (g << 8) | b;
    return rgb;
}

//------------------------------------------------------------------------------
// Used in printf, takes arguments:
// hscale, vscale,
// x, y, width-viewBox, height-viewBox,
// width, height,
// legend-font, tick-font
//
const char* header =
"<?xml version=\"1.0\" standalone=\"no\"?>\n"
"<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
"    \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
"\n"
"<!-- hscale %.2f, vscale %.2f -->\n"
"\n"
"<svg version=\"1.1\" baseProfile=\"full\"\n"
"     xmlns=\"http://www.w3.org/2000/svg\"\n"
"     xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\"\n"
"     viewBox=\"%d %d %d %d\"\n"
"     width=\"%d\" height=\"%d\" preserveAspectRatio=\"none\">\n"
"\n"
"<style type=\"text/css\">\n"
"text.legend { font-family: monospace; font-size: %dpx; }\n"
"text.tick   { font-family: monospace; font-size: %dpx; }\n"
"line { stroke-width: 2; stroke: #666666; stroke-dasharray: 10,10; }\n"
"rect { stroke-width: 0; }\n";

//------------------------------------------------------------------------------
// Used in printf, takes arguments:
// color1, color2.
//
const char* gradient =
"<linearGradient id=\"grad_%s\" x1=\"0%%\" y1=\"0%%\" x2=\"100%%\" y2=\"0%%\">\n"
"  <stop offset=\"0%%\"   stop-color=\"#%06x\"/>\n"
"  <stop offset=\"100%%\" stop-color=\"#%06x\"/>\n"
"</linearGradient>\n";

//------------------------------------------------------------------------------
///
void Trace::finish()
{
    // Find rank and size.
    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // Start the trace file.
    FILE* trace_file = nullptr;
    std::string file_name("trace_" + std::to_string(time(nullptr)) + ".svg");

    // Find the global timespan.
    double timespan = getTimeSpan();

    // Compute width and vertical scaling factor.
    width_ = hscale_ * timespan;
    height_ = vscale_ * (mpi_size * (num_threads_ + 1) - 1);

    // Print header.
    if (mpi_rank == 0) {
        trace_file = fopen(file_name.c_str(), "w");
        assert(trace_file != nullptr);

        // viewBox includes tick marks and legend, plus a margin.
        // Hopefully chars*legend_font_size_ is over estimate of string width.
        int max_legend_width = 24; // chars
        int w = width_ + 3*legend_space_ + max_legend_width*legend_font_size_;

        // Height h needs to include tick marks and comment lines.
        double line_height = tick_font_size_ * 1.2;
        int lines = std::count(comment_.begin(), comment_.end(), '\n') + 1;
        int h = height_ + 2*tick_height_ + line_height*(lines + 2);

        // Height h also needs to be large enough for all legend entries.
        // Build the set of labels.
        // todo: replicated from printLegend().
        std::set<std::string> legend_set;
        for (auto& thread : events_)
            for (auto& event : thread)
                legend_set.insert(event.name_);
        h = std::max(h, int(legend_set.size() * 2 * legend_space_));

        fprintf(trace_file, header,
                hscale_, vscale_,
                -margin_, -margin_, w + margin_, h + margin_,
                w + 2*margin_, h + 2*margin_,
                legend_font_size_, tick_font_size_);

        // Print CSS entries.
        for (auto& key_value : function_color_) {
            auto name = cleanName(key_value.first);
            fprintf(trace_file, ".%-32s { fill: url(#grad_%s) }\n",
                    name.c_str(), name.c_str());
        }
        fprintf(trace_file, "</style>\n\n");

        // Print gradients for CSS entries.
        fprintf(trace_file, "<defs>\n");
        for (auto& key_value : function_color_) {
            auto name = cleanName(key_value.first);
            fprintf(trace_file, gradient, name.c_str(),
                    (unsigned int)key_value.second,
                    darken((unsigned int)key_value.second, 0.8));
        }
        fprintf(trace_file, "</defs>\n\n");
    }

    // Print the events.
    if (mpi_rank == 0) {
        printProcEvents(0, mpi_size, timespan, trace_file);
        for (int rank = 1; rank < mpi_size; ++rank) {
            recvProcEvents(rank);
            printProcEvents(rank, mpi_size, timespan, trace_file);
        }
    }
    else
        sendProcEvents();

    // Finish the trace file.
    if (mpi_rank == 0) {

        printTicks(timespan, trace_file);
        printComment(trace_file);
        printLegend(trace_file);

        fprintf(trace_file, "\n</svg>\n");
        fclose(trace_file);
        fprintf(stderr, "trace file: %s\n", file_name.c_str());
    }

    // Clear events.
    for (auto& thread : events_)
        thread.clear();
}

//------------------------------------------------------------------------------
///
double Trace::getTimeSpan()
{
    double min_time = std::numeric_limits<double>::max();
    double max_time = std::numeric_limits<double>::min();

    for (auto& thread : events_) {
        for (auto& event : thread) {

            if (event.stop_ < min_time)
                min_time = event.stop_;

            if (event.stop_ > max_time)
                max_time = event.stop_;
        }
    }

    double timespan;
    double temp = max_time - min_time;
    MPI_Reduce(&temp, &timespan, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return timespan;
}

//------------------------------------------------------------------------------
///
void Trace::printProcEvents(int mpi_rank, int mpi_size,
                            double timespan, FILE* trace_file)
{
    double y = mpi_rank * (num_threads_ + 1) * vscale_;
    double height = 0.9 * vscale_ / max_nest;
    using llong = long long;

    fprintf(trace_file, "\n<!-- data -->\n");
    for (auto& thread : events_) {
        for (int nest = 0; nest < max_nest; ++nest) {
            double h = std::max( max_nest - nest, 1 ) * height;
            for (auto& event : thread) {
                if (event.nest_ == nest) {

                    double x = (event.start_ - events_[0][0].stop_) * hscale_;
                    double width = (event.stop_ - event.start_) * hscale_;

                    fprintf(trace_file,
                            "<rect x=\"%.4f\" y=\"%.0f\" "
                            "width=\"%.4f\" height=\"%.0f\" "
                            "class=\"%s\" "
                            "inkscape:label=\"%s %lld\"/>\n",
                            x, y,
                            width, h,
                            cleanName(event.name_).c_str(),
                            event.name_, llong( event.index_ ));
                }
            }
        }
        y += vscale_;
    }
}

//------------------------------------------------------------------------------
///
void Trace::printTicks(double timespan, FILE* trace_file)
{
    // Tick spacing is a nice (1 or 5) multiple of power of 10,
    // spaced 200-1000 pixels.
    double pwr = floor(log10(500. / hscale_));
    double tick = pow(10.0, pwr);
    if (tick * hscale_ < 200)
        tick *= 5;
    // Force at least 2 ticks in timespan.
    if (tick >= timespan) {
        pwr = floor(log10(timespan));
        tick = pow(10.0, pwr);
    }
    int decimal_places = pwr < 0 ? (int)(-pwr) : 0;

    fprintf(trace_file, "\n<!-- ticks -->\n");
    for (double time = 0; time < timespan; time += tick) {
        fprintf(trace_file,
                "<line x1=\"%.4f\" x2=\"%.4f\" y1=\"%.4f\" y2=\"%.4f\"/>\n"
                "<text class=\"tick\" x=\"%.4f\" y=\"%.4f\">%.*lf</text>\n\n",
                hscale_ * time,
                hscale_ * time,
                0.0,  //(double)height_,  // start at top or bottom?
                (double)height_ + tick_height_,
                hscale_ * time,
                (double)height_ + tick_height_ + tick_font_size_,
                decimal_places, time);
    }
}

//------------------------------------------------------------------------------
///
void Trace::printComment(FILE* trace_file)
{
    // SVG (until v2) doesn't wrap text, so print each line in <text> tag.
    std::string line;
    double line_height = tick_font_size_ * 1.2;
    double y = height_ + 2*tick_height_ + 2*line_height;
    size_t begin = 0;
    size_t end = comment_.find('\n');
    while (end != std::string::npos) {
        fprintf(trace_file,
                "<text x=\"0\" y=\"%.4f\" class=\"tick\">%s</text>\n",
                y, comment_.substr(begin, end - begin).c_str());
        begin = end+1;
        end = comment_.find('\n', begin);
        y += line_height;
    }
    fprintf(trace_file,
            "<text x=\"0\" y=\"%.4f\" class=\"tick\">%s</text>\n",
            y, comment_.substr(begin).c_str());
}

//------------------------------------------------------------------------------
///
void Trace::printLegend(FILE* trace_file)
{
    std::set<std::string> legend_set;

    // Build the set of labels.
    for (auto& thread : events_)
        for (auto& event : thread)
            legend_set.insert(event.name_);

    // Convert the set to a vector.
    std::vector<std::string> legend_vec(legend_set.begin(), legend_set.end());

    // Sort the vector alphabetically.
    std::sort(legend_vec.begin(), legend_vec.end());

    // Print the labels.
    fprintf(trace_file, "\n<!-- legend -->\n");
    double y_pos = 0.0;
    for (auto& label : legend_vec) {
        fprintf(trace_file,
                "<rect x=\"%.4f\" y=\"%.4f\" width=\"%.4f\" height=\"%.4f\" "
                "class=\"%s\"/>\n"
                "<text x=\"%.4f\" y=\"%.4f\" class=\"legend\">%s</text>\n\n",
                (double)width_ + legend_space_,
                y_pos,
                (double)legend_space_,
                (double)legend_space_,
                cleanName(label).c_str(),
                (double)width_ + legend_space_ * 3.0,
                y_pos + legend_space_,
                label.c_str());

        y_pos += legend_space_ * 2.0;
    }
}

//------------------------------------------------------------------------------
///
void Trace::sendProcEvents()
{
    for (int thread = 0; thread < num_threads_; ++thread) {

        // Send the number of events.
        long int num_events = events_[thread].size();
        MPI_Send(&num_events, 1, MPI_LONG,
                 0, 0, MPI_COMM_WORLD);

        // Send the events.
        MPI_Send(&events_[thread][0], sizeof(Event)*num_events, MPI_BYTE,
                 0, 0, MPI_COMM_WORLD);
    }
}

//------------------------------------------------------------------------------
///
void Trace::recvProcEvents(int rank)
{
    for (int thread = 0; thread < num_threads_; ++thread) {

        // Receive the number of events.
        long int num_events;
        MPI_Recv(&num_events, 1, MPI_LONG,
                 rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Resize the vector and receive the events.
        events_[thread].resize(num_events);
        MPI_Recv(&events_[thread][0], sizeof(Event)*num_events, MPI_BYTE,
                 rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

} // namespace trace
} // namespace slate
