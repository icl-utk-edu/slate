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
/// Print lives of all tiles on all MPI ranks. Ranks send output to rank 0
/// to print in a safe manner.
/// Uses
///  - "." if tile doesn't exist,
///  - "o" if it is origin (local non-workspace)
///  - life if it is workspace
///
template <typename scalar_t>
void Debug::printTilesLives(BaseMatrix<scalar_t> const& A)
{
    if (! debug_) return;

    // i, j are tile indices
    std::string msg;
    char buf[ 8192 ];
    int len = sizeof( buf );

    auto index = A.globalIndex(0, 0);
    auto tmp_tile = A.storage_->find(index);
    auto tile_end = A.storage_->end();

    BaseMatrix<scalar_t>& A_ = const_cast<BaseMatrix<scalar_t>&>( A );

    for (int64_t i = 0; i < A.mt(); ++i) {
        snprintf( buf, len, "%02d [%4lld]: ", A.mpiRank(), llong( i ) );
        msg += buf;
        for (int64_t j = 0; j < A.nt(); ++j) {
            index = A.globalIndex(i, j);
            tmp_tile = A.storage_->find(index);
            if (tmp_tile == tile_end)
                snprintf( buf, len, "   ." );
            else {
                auto T = A_(i, j);
                if (T.workspace())
                    snprintf( buf, len, " %3lld", llong( A.tileLife(i, j) ) );
                else
                    snprintf( buf, len, "   o" );
            }
            msg += buf;
        }
        msg += "\n";
    }

    if (A.mpiRank() == 0) {
        // Print msg on rank 0.
        printf( "%02d: %s\n%s\n", 0, __func__, msg.c_str() );

        // Recv and print msg from other ranks.
        int mpi_size;
        MPI_Comm_size( A.mpiComm(), &mpi_size );
        for (int rank = 1; rank < mpi_size; ++rank ) {
            MPI_Recv( &len, 1, MPI_INT, rank, 0, A.mpiComm(), MPI_STATUS_IGNORE );
            msg.resize( len );
            MPI_Recv( &msg[0], len, MPI_CHAR, rank, 0, A.mpiComm(), MPI_STATUS_IGNORE );
            printf( "%02d: %s\n%s\n", rank, __func__, msg.c_str() );
        }
    }
    else {
        // Other ranks send msg to rank 0.
        len = msg.size();
        MPI_Send( &len, 1, MPI_INT, 0, 0, A.mpiComm() );
        MPI_Send( msg.data(), len, MPI_CHAR, 0, 0, A.mpiComm() );
    }
}

//------------------------------------------------------------------------------
/// Prints map of all tiles.
/// Uses
///  - "." if tile doesn't exist,
///  - "o" if it is origin (local non-workspace)
///  - "w" if it is workspace.
///
template <typename scalar_t>
void Debug::printTilesMaps(BaseMatrix<scalar_t> const& A)
{
    if (! debug_) return;

    // i, j are tile indices
    printf("host\n");
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            auto it = A.storage_->find( A.globalIndex( i, j, HostNum ) );
            if (it != A.storage_->end()) {
                auto tile = it->second->at( HostNum ).tile();
                if (tile->origin())
                    printf("o");
                else
                    printf("w");
            }
            else
                printf(".");
        }
        printf("\n");
    }
    for (int device = 0; device < A.num_devices_; ++device) {
        printf("device %d\n", device);
        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = 0; j < A.nt(); ++j) {
                auto it = A.storage_->find(A.globalIndex(i, j, device));
                if (it != A.storage_->end()) {
                    auto tile = it->second->at(device).tile();
                    if (tile->origin())
                        printf("o");
                    else
                        printf("x");
                }
                else
                    printf(".");
            }
            printf("\n");
        }
    }
}

//------------------------------------------------------------------------------
/// Prints map of all tiles with MOSI state.
/// Uses
///  - "." if tile doesn't exist,
///  - "o" if it is origin (local non-workspace)
///  - "w" if it is workspace.
///
/// Prints 2 chars for MOSI. First char:
///  - "m" if modified
///  - "s" if shared
///  - "i" if invalid
/// Second char:
///  - "h" if on hold
///  - "_" otherwise
///
/// Prints additional char for Layout:
///  - "|" if ColMajor
///  - "-" if RowMajor
///
/// Prints additional char for extended buffer:
///  - "u" if user data
///  - "e" if extended
///  - " " otherwise
///
template <typename scalar_t>
void Debug::printTilesMOSI(BaseMatrix<scalar_t> const& A, const char* name,
                           const char* func, const char* file, int line)
{
    if (! debug_) return;
    // i, j are tile indices
    printf("%s on host, rank %d, %s, %s, %d\n", name, A.mpiRank(), func, file, line);
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            auto it = A.storage_->find( A.globalIndex( i, j, HostNum ) );
            if (it != A.storage_->end()) {
                auto tile = it->second->at( HostNum ).tile();
                if (tile->origin())
                    printf("o");
                else
                    printf("w");

                auto mosi = it->second->at( HostNum ).getState();
                switch (mosi) {
                    case MOSI::Modified:  printf("m"); break;
                    case MOSI::Shared:    printf("s"); break;
                    case MOSI::Invalid:   printf("i"); break;
                    case MOSI::OnHold: break;  // below
                }
                if (it->second->at( HostNum ).stateOn( MOSI::OnHold ))
                    printf("h");
                else
                    printf("_");
                if (tile->layout() == Layout::ColMajor)
                    printf("|");
                else
                    printf("-");
                if (tile->extended()) {
                    if (tile->userData() == tile->data())
                        printf("u");
                    else
                        printf("e");
                }
                else
                    printf(" ");
                printf(" ");
            }
            else
                printf(".     ");
        }
        printf("\n");
    }
    for (int device = 0; device < A.num_devices_; ++device) {
        printf("%s on device %d, rank %d, %s, %s, %d\n", name, device, A.mpiRank(), func, file, line);
        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = 0; j < A.nt(); ++j) {
                auto it = A.storage_->find(A.globalIndex(i, j, device));
                if (it != A.storage_->end()) {
                    auto tile = it->second->at(device).tile();
                    if (tile->origin())
                        printf("o");
                    else
                        printf("w");

                    auto mosi = it->second->at(device).getState();
                    switch (mosi) {
                        case MOSI::Modified:  printf("m"); break;
                        case MOSI::Shared:    printf("s"); break;
                        case MOSI::Invalid:   printf("i"); break;
                        case MOSI::OnHold: break;  // below
                    }
                    if (it->second->at(device).stateOn(MOSI::OnHold))
                        printf("h");
                    else
                        printf("_");
                    if (tile->layout() == Layout::ColMajor)
                        printf("|");
                    else
                        printf("-");
                    if (tile->extended())
                        printf("e");
                    else
                        printf(" ");
                    printf(" ");
                }
                else
                    printf(".     ");
            }
            printf("\n");
        }
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
template
void Debug::diffLapackMatrices(int64_t m, int64_t n,
                               float const* A, int64_t lda,
                               float const* B, int64_t ldb,
                               int64_t mb, int64_t nb);
template
void Debug::checkTilesLives(BaseMatrix<float> const& A);

template
bool Debug::checkTilesLayout(BaseMatrix< float > const& A);

template
void Debug::printTilesLives(BaseMatrix<float> const& A);

template
void Debug::printTilesMaps(BaseMatrix<float> const& A);

template
void Debug::printTilesMOSI(BaseMatrix<float> const& A, const char* name,
                           const char* func, const char* file, int line);

//------------------------------------------------------------------------------
template
void Debug::diffLapackMatrices(int64_t m, int64_t n,
                               double const* A, int64_t lda,
                               double const* B, int64_t ldb,
                               int64_t mb, int64_t nb);
template
void Debug::checkTilesLives(BaseMatrix<double> const& A);

template
bool Debug::checkTilesLayout(BaseMatrix< double > const& A);

template
void Debug::printTilesLives(BaseMatrix<double> const& A);

template
void Debug::printTilesMaps(BaseMatrix<double> const& A);

template
void Debug::printTilesMOSI(BaseMatrix<double> const& A, const char* name,
                           const char* func, const char* file, int line);

//------------------------------------------------------------------------------
template
void Debug::diffLapackMatrices(int64_t m, int64_t n,
                               std::complex<float> const* A, int64_t lda,
                               std::complex<float> const* B, int64_t ldb,
                               int64_t mb, int64_t nb);
template
void Debug::checkTilesLives(BaseMatrix< std::complex<float> > const& A);

template
bool Debug::checkTilesLayout(BaseMatrix< std::complex<float> > const& A);

template
void Debug::printTilesLives(BaseMatrix< std::complex<float> > const& A);

template
void Debug::printTilesMaps(BaseMatrix< std::complex<float> > const& A);

template
void Debug::printTilesMOSI(BaseMatrix< std::complex<float> > const& A, const char* name,
                           const char* func, const char* file, int line);

//------------------------------------------------------------------------------
template
void Debug::diffLapackMatrices(int64_t m, int64_t n,
                               std::complex<double> const* A, int64_t lda,
                               std::complex<double> const* B, int64_t ldb,
                               int64_t mb, int64_t nb);
template
void Debug::checkTilesLives(BaseMatrix< std::complex<double> > const& A);

template
bool Debug::checkTilesLayout(BaseMatrix< std::complex<double> > const& A);

template
void Debug::printTilesLives(BaseMatrix< std::complex<double> > const& A);

template
void Debug::printTilesMaps(BaseMatrix< std::complex<double> > const& A);

template
void Debug::printTilesMOSI(BaseMatrix< std::complex<double> > const& A, const char* name,
                           const char* func, const char* file, int line);


} // namespace slate
