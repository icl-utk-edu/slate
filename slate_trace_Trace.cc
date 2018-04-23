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

#include "slate_trace_Trace.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <string>

namespace slate {
namespace trace {

double Trace::vscale_;
double Trace::hscale_;

bool Trace::tracing_ = false;
int Trace::num_threads_ = omp_get_max_threads();

std::vector<std::vector<Event>> Trace::events_ =
    std::vector<std::vector<Event>>(omp_get_max_threads());

std::map<std::string, Color> Trace::function_color_ = {

    {"blas::gemm", Color::MediumAquamarine},
    {"blas::syrk", Color::CornflowerBlue},
    {"blas::trmm", Color::MediumOrchid},
    {"blas::trsm", Color::MediumPurple},

    {"cblas_dgemm_batch",  Color::DarkGreen},
    {"cublasDgemmBatched", Color::PaleGreen},

    {"cudaMemcpy2DAsync", Color::LightGray},
    {"cudaMemcpyAsync",   Color::LightGray},

    {"lapack::potrf", Color::RosyBrown},

    {"Memory::alloc", Color::Aqua},
    {"Memory::free",  Color::Aquamarine},

    {"MPI_Barrier", Color::Black},
    {"MPI_Bcast",   Color::Crimson}
};

///-----------------------------------------------------------------------------
/// \brief
///
void Trace::insert(Event event)
{
    if (tracing_) {
        event.stop();
        events_[omp_get_thread_num()].push_back(event);
    }
}

///-----------------------------------------------------------------------------
/// \brief
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

    if (mpi_rank == 0) {
        trace_file = fopen(file_name.c_str(), "w");
        assert(trace_file != nullptr);
        fprintf(trace_file, "<svg viewBox=\"0 0 %d %d\">\n", width_, height_);
    }

    // Find the global timespan.
    double timespan = getTimeSpan();

    // Compute scaling factors.
    hscale_ = width_ / timespan;
    vscale_ = height_ / (mpi_size * num_threads_);

    // Print the events.
    if (mpi_rank == 0) {
        printProcEvents(0, mpi_size, timespan, trace_file);
        for (int rank = 1; rank < mpi_size; ++rank) {
            recvProcEvents(rank);
            printProcEvents(rank, mpi_size, timespan, trace_file);
        }
    }
    else {
        sendProcEvents();
    }

    // Finish the trace file.
    if (mpi_rank == 0) {

        printTicks(timespan, trace_file);
        printLegend(trace_file);

        fprintf(trace_file, "</svg>\n");
        fclose(trace_file);
        fprintf(stderr, "trace file: %s\n", file_name.c_str());
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
double Trace::getTimeSpan()
{
    double min_time = std::numeric_limits<double>::max();
    double max_time = std::numeric_limits<double>::min();

    for (auto thread : events_)
        for (auto event : thread) {

            if (event.stop_ < min_time)
                min_time = event.stop_;

            if (event.stop_ > max_time)
                max_time = event.stop_;
        }

    double timespan;
    double temp = max_time - min_time;
    MPI_Reduce(&temp, &timespan, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return timespan;
}

///-----------------------------------------------------------------------------
/// \brief
///
void Trace::printProcEvents(int mpi_rank, int mpi_size,
                            double timespan, FILE* trace_file)
{
    double y = mpi_rank * num_threads_ * vscale_;
    double height = 0.9 * vscale_;
    double stroke_width = vscale_ / 500.0;

    for (auto thread : events_) {
        for (auto event : thread) {

            double x = (event.start_ - events_[0][0].stop_) * hscale_;
            double width = (event.stop_ - event.start_) * hscale_;

            fprintf(trace_file,
                "<rect x=\"%lf\" y=\"%lf\" "
                "width=\"%lf\" height=\"%lf\" "
                "fill=\"#%06x\" "
                "stroke=\"#000000\" stroke-width=\"%lf\" "
                "inkscape:label=\"%s\"/>\n",
                x, y,
                width, height,
                (unsigned int)function_color_[event.name_],
                stroke_width,
                event.name_);
        }
        y += vscale_;
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
void Trace::printTicks(double timespan, FILE* trace_file)
{
    // Tick spacing is power of 10, with at most 20 tick marks.
    double pwr = ceil(log10(timespan/20.0));
    double tick = pow(10.0, pwr);
    int decimal_places = pwr < 0 ? (int)(-pwr) : 0;

    for (double time = 0; time < timespan; time += tick) {
        fprintf(trace_file,
            "<line x1=\"%lf\" x2=\"%lf\" y1=\"%lf\" y2=\"%lf\" "
            "stroke=\"#000000\" stroke-width=\"%d\"/>\n"
            "<text x=\"%lf\" y=\"%lf\" "
            "font-family=\"monospace\" font-size=\"%d\">%.*lf</text>\n",
            hscale_ * time,
            hscale_ * time,
            (double)height_,
            (double)height_ + tick_height_,
            tick_stroke_,
            hscale_ * time,
            (double)height_ + tick_height_ * 2.0,
            tick_font_size_, decimal_places, time);
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
void Trace::printLegend(FILE* trace_file)
{
    std::set<std::string> legend_set;

    // Build the set of labels.
    for (auto thread : events_)
        for (auto event : thread)
            legend_set.insert(event.name_);

    // Convert the set to a vector.
    std::vector<std::string> legend_vec(legend_set.begin(), legend_set.end());

    // Sort the vector alphabetically.
    std::sort(legend_vec.begin(), legend_vec.end());

    // Print the labels.
    double y_pos = 0.0;
    for (auto label : legend_vec) {
        fprintf(trace_file,
            "<rect x=\"%lf\" y=\"%lf\" width=\"%lf\" height=\"%lf\" "
            "fill=\"#%06x\" stroke=\"#000000\" stroke-width=\"%d\"/>\n"
            "<text x=\"%lf\" y=\"%lf\" "
            "font-family=\"monospace\" font-size=\"%d\">%s</text>\n",
            (double)width_ + legend_space_,
            y_pos,
            (double)legend_space_,
            (double)legend_space_,
            (unsigned int)function_color_[label],
            legend_stroke_,
            (double)width_ + legend_space_ * 3.0,
            y_pos + legend_space_,
            legend_font_size_, label.c_str());

            y_pos += legend_space_ * 2.0;
    }
}

///-----------------------------------------------------------------------------
/// \brief
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

///-----------------------------------------------------------------------------
/// \brief
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
        MPI_Recv(&events_[thread][0],sizeof(Event)*num_events, MPI_BYTE,
                 rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

} // namespace trace
} // namespace slate
