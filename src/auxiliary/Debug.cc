// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "auxiliary/Debug.hh"

namespace slate {

bool Debug::debug_ = true;

//------------------------------------------------------------------------------
/// Prints a summary of differences between matrices A and B.
/// Uses "." for small relative error, "#" for large relative error in an entry.
/// For each tile, checks only the four corner 2x2 blocks, marked by letters:
///
///     [ a e - - i m ]  output as:  -----------
///     [ b f - - j n ]              | a e i m |
///     [ - - - - - - ]              | b f j n |
///     [ - - - - - - ]              | c g k o |
///     [ c g - - k o ]              | d h l p |
///     [ d h - - l p ]              -----------
///
template <typename scalar_t>
void Debug::diffLapackMatrices(
    int64_t m, int64_t n,
    scalar_t const* A, int64_t lda,
    scalar_t const* B, int64_t ldb,
    int64_t mb, int64_t nb)
{
    using real_t = blas::real_type<scalar_t>;
    const real_t eps = std::numeric_limits<real_t>::epsilon();

    if (! debug_) return;
    for (int64_t i = 0; i < m; ++i) {

        if (i%mb == 2)
            i += mb-4;

        for (int64_t j = 0; j < n; ++j) {

            if (j%nb == 2)
                j += nb-4;

            real_t error = std::abs(A[(size_t)lda*j+i] - B[(size_t)ldb*j+i])
                         / std::abs(A[(size_t)lda*j+i]);
            printf("%c", error < 100*eps ? '.' : '#');

            if ((j+1)%nb == 0)
                printf("|");
        }
        printf("\n");

        if ((i+1)%mb == 0) {
            for (int64_t j = 0; j < (n/nb)*5; ++j)
                printf("-");
            printf("\n");
        }
    }
    printf("\n");
}

//------------------------------------------------------------------------------
/// Prints information about tiles that have non-zero life.
template <typename scalar_t>
void Debug::checkTilesLives( BaseMatrix<scalar_t> const& A )
{
    if (! debug_) return;
    // i, j are global indices
    for (auto iter = A.storage_->tiles_.begin();
              iter != A.storage_->tiles_.end(); ++iter) {
        int64_t i = std::get<0>(iter->first);
        int64_t j = std::get<1>(iter->first);

        if (! A.tileIsLocal(i, j)) {
            if (iter->second->lives() != 0 ||
                iter->second->numInstances() != 0) {

                std::cout << "RANK "  << std::setw(3) << A.mpi_rank_
                          << " TILE " << std::setw(3) << std::get<0>(iter->first)
                          << " "      << std::setw(3) << std::get<1>(iter->first)
                          << " LIFE " << std::setw(3)
                          << iter->second->lives();
                for (int d = HostNum; d < A.num_devices(); ++d) {
                    if (iter->second->existsOn(d)) {
                        std::cout << " DEV "  << d
                                  << " data " << iter->second->at(d).tile()->data() << "\n";
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Checks if existing tile instances in matrix have the same layout as the
/// matrix.
///
template <typename scalar_t>
bool Debug::checkTilesLayout( BaseMatrix<scalar_t> const& A )
{
    if (! debug_) return true;

    // i, j are tile indices
    // if (A.mpi_rank_ == 0)
    {
        auto index = A.globalIndex(0, 0);
        auto tmp_tile = A.storage_->tiles_.find(index);
        auto tile_end = A.storage_->tiles_.end();

        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = 0; j < A.nt(); ++j) {
                index = A.globalIndex(i, j);
                tmp_tile = A.storage_->tiles_.find(index);
                if (tmp_tile != tile_end
                    && tmp_tile->second->at( HostNum ).valid()
                    && tmp_tile->second->at( HostNum ).tile()->layout() != A.layout()) {
                    return false;
                }
            }
        }
    }
    return true;
}

//------------------------------------------------------------------------------
char to_char( MOSI mosi )
{
    switch (mosi) {
        case MOSI::Modified: return 'm'; break;
        case MOSI::Shared:   return 's'; break;
        case MOSI::Invalid:  return 'i'; break;
        default:             return '?'; break;
    }
}

//------------------------------------------------------------------------------
/// Prints map of all tiles with requested fields. The `printTiles`
/// macro adds the matrix name, function, file, and line. Example usage:
///
///     Debug::printTiles( A, Field_Kind | Field_MOSI )
///
/// For each tile (i, j), prints instances, first Host, then device 0, 1, etc.,
/// then prints life, which applies to all instances, if desired.
/// Each instance is a collection of letters as described below.
///
/// For all fields:
///  - '.' if tile doesn't exist
///
/// For Field_Kind:
///  - 'u' if user-owned  origin
///  - 'o' if slate-owned origin (local non-workspace)
///  - 'w' if slate-owned workspace
///
/// For Field_MOSI:
///  - 'm' if modified, 'M' if also on hold
///  - 's' if shared,   'S' if also on hold
///  - 'i' if invalid,  'I' if also on hold
///
/// For Field_Layout:
///  - 'c' if ColMajor
///  - 'r' if RowMajor
///
/// For Field_Buffer:
///  - 'b' if regular buffer
///  - 'e' if extended
///
template <typename scalar_t>
void Debug::printTiles_(
    BaseMatrix<scalar_t> const& A, const char* name, int fields,
    const char* func, const char* file, int line )
{
    if (! debug_) return;

    const int tag_0 = 0;

    bool do_kind   = (fields & Fields::Field_Kind  ) != 0;
    bool do_mosi   = (fields & Fields::Field_MOSI  ) != 0;
    bool do_layout = (fields & Fields::Field_Layout) != 0;
    bool do_buffer = (fields & Fields::Field_Buffer) != 0;
    bool do_life   = (fields & Fields::Field_Life  ) != 0;
    bool multi     = do_kind + do_mosi + do_layout + do_buffer + do_life > 1;

    // Padding between columns. When there are multiple fields,
    // one space is put between instances of the same tile (i, j),
    // so add more space between tiles.
    const char* pad = multi ? "    " : "  ";

    char buf[ 80 ];
    std::string msg = std::string( "\n" ) + pad
                    + "% rank " + std::to_string( A.mpiRank() ) + "\n";

    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            msg += pad;
            int life = 0;
            for (int device = HostNum; device < A.num_devices_; ++device) {
                // Space between tiles if multiple fields.
                if (multi && device > HostNum)
                    msg += ' ';

                auto iter = A.storage_->find( A.globalIndex( i, j, device ) );
                if (iter != A.storage_->end()) {
                    auto tile = iter->second->at( device ).tile();
                    if (do_kind) {
                        msg += tile->origin()
                                ? (tile->allocated() ? 'o' : 'u')
                                : 'w';
                    }
                    if (do_mosi) {
                        char ch = to_char( iter->second->at( device ).getState() );
                        if (iter->second->at( device ).stateOn( MOSI::OnHold ))
                            ch = toupper( ch );
                        msg += ch;
                    }
                    if (do_layout) {
                        msg += tile->layout() == Layout::ColMajor ? 'c' : 'r';
                    }
                    if (do_buffer) {
                        msg += tile->extended() ? 'e' : 'b';
                    }
                    life = A.tileLife( i, j );
                }
                else {
                    if (do_kind)   { msg += '.'; }
                    if (do_mosi)   { msg += '.'; }
                    if (do_layout) { msg += '.'; }
                    if (do_buffer) { msg += '.'; }
                }
            }
            if (do_life) {
                snprintf( buf, sizeof(buf), " %-3d", life );
                msg += buf;
            }
        }
        msg += "\n";
    }

    // Flush output, to attempt to avoid mixing with output on other ranks.
    fflush( nullptr );
    slate_mpi_call(
        MPI_Barrier( A.mpiComm() ) );

    if (A.mpiRank() == 0) {
        // Print header.
        printf( "%s = [ %% %lld x %lld tiles, in %s at %s:%d\n"
                "%s%% Fields:",
                name, llong( A.mt() ), llong( A.nt() ),
                func, file, line, pad );
        // Delim is comma after 1st field.
        const char* delim = "";
        if (do_kind) {
            printf( "%s kind( user, origin, workspace )", delim );
            delim = ",";
        }
        if (do_mosi) {
            printf( "%s MOSI( modified, shared, invalid; upper=hold )", delim );
            delim = ",";
        }
        if (do_layout) {
            printf( "%s layout( col, row )", delim );
            delim = ",";
        }
        if (do_buffer) {
            printf( "%s buffer( extended )", delim );
            delim = ",";
        }
        if (do_life) {
            printf( "%s life", delim );
        }

        // Print rank 0 data.
        printf( "%s", msg.c_str() );

        int mpi_size;
        slate_mpi_call(
            MPI_Comm_size( A.mpiComm(), &mpi_size ) );

        // Print data from other ranks.
        for (int src = 1; src < mpi_size; ++src) {
            // Recv size, then string, and print.
            int size;
            slate_mpi_call(
                MPI_Recv( &size, 1, MPI_INT, src, tag_0, A.mpiComm(),
                          MPI_STATUS_IGNORE ) );
            msg.resize( size );
            slate_mpi_call(
                MPI_Recv( &msg[0], size, MPI_CHAR, src, tag_0, A.mpiComm(),
                          MPI_STATUS_IGNORE ) );
            printf( "%s", msg.data() );
        }
        printf( "];\n" );
    }
    else {
        // Send size, then string to rank 0. No need to include null byte.
        int size = msg.size();
        MPI_Send( &size, 1, MPI_INT, 0, tag_0, A.mpiComm() );
        MPI_Send( msg.c_str(), size, MPI_CHAR, 0, tag_0, A.mpiComm() );
    }
}

//------------------------------------------------------------------------------
/// Prints the number of free blocks for each device.
void Debug::printNumFreeMemBlocks(Memory const& m)
{
    if (! debug_) return;
    printf("\n");
    for (auto iter = m.free_blocks_.begin(); iter != m.free_blocks_.end(); ++iter) {
        printf("\tdevice: %d\tfree blocks: %lu\n", iter->first,
               (unsigned long) iter->second.size());
    }
}

//------------------------------------------------------------------------------
/// Checks whether blocks were leaked or freed too many times, for host.
void Debug::checkHostMemoryLeaks(Memory const& m)
{
    using llu = long long unsigned;
    if (! debug_) return;
    if (m.free_blocks_.at( HostNum ).size() < m.capacity_.at( HostNum )) {
        fprintf(stderr,
                "Error: memory leak: freed %llu of %llu blocks on host\n",
                (llu) m.free_blocks_.at( HostNum ).size(),
                (llu) m.capacity_.at( HostNum ));
    }
    else if (m.free_blocks_.at( HostNum ).size() > m.capacity_.at( HostNum )) {
        fprintf(stderr,
                "Error: freed too many: %llu of %llu blocks on host\n",
                (llu) m.free_blocks_.at( HostNum ).size(),
                (llu) m.capacity_.at( HostNum ));
    }
}

//------------------------------------------------------------------------------
/// Checks whether blocks were leaked or freed too many times, for device.
void Debug::checkDeviceMemoryLeaks(Memory const& m, int device)
{
    using llu = long long unsigned;
    if (! debug_) return;
    if (m.free_blocks_.at(device).size() < m.capacity_.at(device)) {
        fprintf(stderr,
                "Error: memory leak: freed %llu of %llu blocks on device %d\n",
                (llu) m.free_blocks_.at(device).size(),
                (llu) m.capacity_.at(device), device);
    }
    else if (m.free_blocks_.at(device).size() > m.capacity_.at(device)) {
        fprintf(stderr,
                "Error: freed too many: %llu of %llu blocks on device %d\n",
                (llu) m.free_blocks_.at(device).size(),
                (llu) m.capacity_.at(device), device);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.

//------------------------------------------------------------------------------
template
void Debug::diffLapackMatrices(
    int64_t m, int64_t n,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    int64_t mb, int64_t nb );

template
void Debug::diffLapackMatrices(
    int64_t m, int64_t n,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    int64_t mb, int64_t nb );

template
void Debug::diffLapackMatrices(
    int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    int64_t mb, int64_t nb );

template
void Debug::diffLapackMatrices(
    int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    int64_t mb, int64_t nb);

//------------------------------------------------------------------------------
template
void Debug::checkTilesLives( BaseMatrix<float> const& A );

template
void Debug::checkTilesLives( BaseMatrix<double> const& A );

template
void Debug::checkTilesLives( BaseMatrix< std::complex<float> > const& A );

template
void Debug::checkTilesLives( BaseMatrix< std::complex<double> > const& A );

//------------------------------------------------------------------------------
template
bool Debug::checkTilesLayout( BaseMatrix< float > const& A );

template
bool Debug::checkTilesLayout( BaseMatrix< double > const& A );

template
bool Debug::checkTilesLayout( BaseMatrix< std::complex<float> > const& A );

template
bool Debug::checkTilesLayout( BaseMatrix< std::complex<double> > const& A );

//------------------------------------------------------------------------------
template
void Debug::printTiles_(
    BaseMatrix<float> const& A, const char* name, int fields,
    const char* func, const char* file, int line );

template
void Debug::printTiles_(
    BaseMatrix<double> const& A, const char* name, int fields,
    const char* func, const char* file, int line );

template
void Debug::printTiles_(
    BaseMatrix< std::complex<float> > const& A, const char* name, int fields,
    const char* func, const char* file, int line );

template
void Debug::printTiles_(
    BaseMatrix< std::complex<double> > const& A, const char* name, int fields,
    const char* func, const char* file, int line );

} // namespace slate
