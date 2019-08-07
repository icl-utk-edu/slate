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

#ifndef SLATE_TRACE_HH
#define SLATE_TRACE_HH

#include <map>
#include <set>
#include <vector>
#include <cstring>

#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {
namespace trace {

//------------------------------------------------------------------------------
///
class Event {
public:
    friend class Trace;

    Event()
    {}

    Event(const char* name)
        : start_(omp_get_wtime())
    {
        strncpy(name_, name, 30);
        name_[30]='\0';
    }

    void stop() { stop_ = omp_get_wtime(); }

private:
    char name_[31];
    double start_;
    double stop_;
};

//------------------------------------------------------------------------------
///
class Trace {
public:
    friend class Block;

    static void on() { tracing_ = true; }
    static void off() { tracing_ = false; }

    static void insert(Event event);
    static void finish();
    static void comment(std::string const& str);

    // Vertical scale: pixel height of each thread.
    static double thread_height() { return vscale_; }
    static void   thread_height(double s) { vscale_ = s; }

    // Horizontal scale, in pixels per second.
    static double pixels_per_second() { return hscale_; }
    static void   pixels_per_second(double s) { hscale_ = s; }

private:
    static double getTimeSpan();
    static void printProcEvents(int mpi_rank, int mpi_size,
                                double timespan, FILE* trace_file);
    static void printTicks(double timespan, FILE* trace_file);
    static void printLegend(FILE* trace_file);
    static void printComment(FILE* trace_file);
    static void sendProcEvents();
    static void recvProcEvents(int rank);

    static int width_;
    static int height_;

    static const int tick_height_ = 32;
    static const int tick_font_size_ = 24;

    static const int legend_space_ = 28;
    static const int legend_font_size_ = 24;

    static const int margin_ = 28;

    static double vscale_;
    static double hscale_;

    static bool tracing_;
    static int num_threads_;

    static std::vector<std::vector<Event>> events_;
};

//------------------------------------------------------------------------------
///
class Block {
public:
    Block(const char* name)
        : event_(name)
    {}

    ~Block() { Trace::insert(event_); }
private:
    Event event_;
};

} // namespace trace
} // namespace slate

#endif // SLATE_TRACE_HH
