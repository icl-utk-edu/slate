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

#include "slate_Debug.hh"

namespace slate {

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Debug::diffLapackMatrices(int64_t m, int64_t n,
                               scalar_t *a, int64_t lda,
                               scalar_t *b, int64_t ldb,
                               int64_t mb, int64_t nb)
{
    for (int64_t i = 0; i < m; ++i) {

        if (i%mb == 2)
            i += mb-4;

        for (int64_t j = 0; j < n; ++j) {

            if (j%nb == 2)
                j += nb-4;

            scalar_t error = a[(size_t)lda*j+i] - b[(size_t)lda*j+i];
            printf("%c", error < 0.00000000000001 ? '.' : '#');

            if ((j+1)%nb == 0)
                printf("|");
        }
        printf("\n");

        if ((i+1)%mb == 0) {
            for (int64_t j = 0; j < (n/nb)*5; ++j) {
                printf("-");
            }
            printf("\n");        
        }
    }
    printf("\n");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Debug::checkTilesLives(Matrix<scalar_t> &a)
{
    for (auto it = a.tiles_->begin(); it != a.tiles_->end(); ++it) {

        int64_t i = std::get<0>(it->first);
        int64_t j = std::get<1>(it->first);       

        if (!a.tileIsLocal(i, j))
            if ((*a.lives_)[{i, j}] != 0 || it->second->data_ != nullptr)

                std::cout << "P" << a.mpi_rank_
                          << " TILE " << std::get<0>(it->first)
                          << " " << std::get<1>(it->first)
                          << " LIFE " << (*a.lives_)[{i, j}]
                          << " data_ " << it->second->data_ 
                          << " DEV " << std::get<2>(it->first) << std::endl;
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Debug::printTilesLives(Matrix<scalar_t> &a)
{
    if (a.mpi_rank_ == 0) {
        for (int64_t i = 0; i < a.mt_; ++i) {
            for (int64_t j = 0; j < a.nt_; j++) {
                if (a.tiles_->find({i, j, a.host_num_}) == a.tiles_->end())
                    printf("  .");
                else
                    printf("%3lld", (long long) (*a.lives_)[{a.it_+i, a.jt_+j}]);
            }
            printf("\n");
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename scalar_t>
void Debug::printTilesMaps(Matrix<scalar_t> &a)
{
    for (int64_t i = 0; i < a.mt_; ++i) {
        for (int64_t j = 0; j <= i && j < a.nt_; ++j) {
            auto it = a.tiles_->find({i, j, a.host_num_});
            if (it != a.tiles_->end()) {
                auto tile = it->second;
                if (tile->origin_ == true)
                    printf("o");
                else
                    printf("x");
            }
            else {
                printf(".");
            }
        }
        printf("\n");
    }
    for (int device = 0; device < a.num_devices_; ++device) {
        for (int64_t i = 0; i < a.mt_; ++i) {
            for (int64_t j = 0; j <= i && j < a.nt_; ++j) {
                auto it = a.tiles_->find({i, j, device});
                if (it != a.tiles_->end()) {
                    auto tile = it->second;
                    if (tile->origin_ == true)
                        printf("o");
                    else
                        printf("x");
                }
                else {
                    printf(".");
                }
            }
            printf("\n");
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief
///
void Debug::printNumFreeMemBlocks(Memory &m)
{
    printf("\n");
    for (auto it = m.free_blocks_.begin(); it != m.free_blocks_.end(); ++it)
        printf("\tdevice: %d\tfree blocks: %lu\n", it->first,
               (unsigned long) it->second.size());
}

//------------------------------------------------------------------------------
template
void Debug::diffLapackMatrices(int64_t m, int64_t n,
                               double *a, int64_t lda,
                               double *b, int64_t ldb,
                               int64_t mb, int64_t nb);
template
void Debug::checkTilesLives(Matrix<double> &a);

template
void Debug::printTilesLives(Matrix<double> &a);

template
void Debug::printTilesMaps(Matrix<double> &a);

} // namespace slate
