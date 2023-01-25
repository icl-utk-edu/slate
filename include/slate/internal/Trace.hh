// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TRACE_HH
#define SLATE_TRACE_HH

#include <map>
#include <set>
#include <vector>
#include <string>

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

    Event(const char* name, int index, int nest)
        : start_(omp_get_wtime()),
          index_( index ),
          nest_(nest)
    {
        // todo: do with C++ instead of cstring?
        strncpy(name_, name, 30);
        name_[30]='\0';
    }

    void stop() { stop_ = omp_get_wtime(); }

private:
    char name_[31];
    double start_;
    double stop_;
    int64_t index_;
    int nest_;
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
    Block( const char* name, int64_t index=0 );
    ~Block();

private:
    Event event_;
};

} // namespace trace
} // namespace slate

#endif // SLATE_TRACE_HH
