// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "aux/Debug.hh"

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
void Debug::diffLapackMatrices(int64_t m, int64_t n,
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
void Debug::checkTilesLives(BaseMatrix<scalar_t> const& A)
{
    if (! debug_) return;
    // i, j are global indices
    for (auto it = A.storage_->tiles_.begin();
             it != A.storage_->tiles_.end(); ++it) {
        int64_t i = std::get<0>(it->first);
        int64_t j = std::get<1>(it->first);

        if (! A.tileIsLocal(i, j)) {
            if (it->second->lives() != 0 ||
                it->second->numInstances() != 0) {

                std::cout << "RANK "  << std::setw(3) << A.mpi_rank_
                          << " TILE " << std::setw(3) << std::get<0>(it->first)
                          << " "      << std::setw(3) << std::get<1>(it->first)
                          << " LIFE " << std::setw(3)
                          << it->second->lives();
                for (int d = A.hostNum(); d < A.num_devices(); ++d) {
                    if (it->second->existsOn(d)) {
                        std::cout << " DEV "  << d
                                  << " data " << it->second->at(d).tile()->data() << "\n";
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
bool Debug::checkTilesLayout(BaseMatrix<scalar_t> const& A)
{
    if (! debug_) return true;
    // i, j are tile indices
    // if (A.mpi_rank_ == 0)
    {
        auto index = A.globalIndex(0, 0);
        auto tmp_tile = A.storage_->tiles_.find(index);
        auto tile_end = A.storage_->tiles_.end();

        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = 0; j < A.nt(); j++) {
                index = A.globalIndex(i, j);
                tmp_tile = A.storage_->tiles_.find(index);
                if (tmp_tile != tile_end &&
                    tmp_tile->second->at(A.host_num_).valid() &&
                    tmp_tile->second->at(A.host_num_).tile()->layout() != A.layout()) {
                    return false;
                }
            }
        }
    }
    return true;
}

//------------------------------------------------------------------------------
/// On MPI rank 0 only,
/// print lives of all tiles, with "." if tile doesn't exist.
///
template <typename scalar_t>
void Debug::printTilesLives(BaseMatrix<scalar_t> const& A)
{
    if (! debug_) return;
    // i, j are tile indices
    if (A.mpi_rank_ == 0) {
        auto index = A.globalIndex(0, 0);
        auto tmp_tile = A.storage_->find(index);
        auto tile_end = A.storage_->end();

        for (int64_t i = 0; i < A.mt(); ++i) {
            for (int64_t j = 0; j < A.nt(); j++) {
                index = A.globalIndex(i, j);
                tmp_tile = A.storage_->find(index);
                if (tmp_tile == tile_end)
                    printf("  .");
                else
                    printf("%3lld", (long long) A.tileLife(i, j));
            }
            printf("\n");
        }
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
            auto it = A.storage_->find(A.globalIndex(i, j, A.host_num_));
            if (it != A.storage_->end()) {
                auto tile = it->second->at(A.host_num_).tile();
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
///  - " " otherwise
///
/// Prints aditional char for Layout:
///  - "|" if ColMajor
///  - "-" if RowMajor
///
/// Prints aditional char for extended buffer:
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
            auto it = A.storage_->find(A.globalIndex(i, j, A.host_num_));
            if (it != A.storage_->end()) {
                auto tile = it->second->at(A.host_num_).tile();
                if (tile->origin())
                    printf("o");
                else
                    printf("w");

                auto mosi = it->second->at(A.host_num_).getState();
                switch (mosi) {
                    case MOSI::Modified:  printf("m"); break;
                    case MOSI::Shared:    printf("s"); break;
                    case MOSI::Invalid:   printf("i"); break;
                    case MOSI::OnHold: break;  // below
                }
                if (it->second->at(A.host_num_).stateOn(MOSI::OnHold))
                    printf("h");
                else
                    printf(" ");
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
                        printf("x");

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
                        printf(" ");
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
    for (auto it = m.free_blocks_.begin(); it != m.free_blocks_.end(); ++it) {
        printf("\tdevice: %d\tfree blocks: %lu\n", it->first,
               (unsigned long) it->second.size());
    }
}

//------------------------------------------------------------------------------
/// Checks whether blocks were leaked or freed too many times, for host.
void Debug::checkHostMemoryLeaks(Memory const& m)
{
    using llu = long long unsigned;
    if (! debug_) return;
    if (m.free_blocks_.at(m.host_num_).size() < m.capacity_.at(m.host_num_)) {
        fprintf(stderr,
                "Error: memory leak: freed %llu of %llu blocks on host\n",
                (llu) m.free_blocks_.at(m.host_num_).size(),
                (llu) m.capacity_.at(m.host_num_));
    }
    else if (m.free_blocks_.at(m.host_num_).size() > m.capacity_.at(m.host_num_)) {
        fprintf(stderr,
                "Error: freed too many: %llu of %llu blocks on host\n",
                (llu) m.free_blocks_.at(m.host_num_).size(),
                (llu) m.capacity_.at(m.host_num_));
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
