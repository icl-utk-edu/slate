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

#include "test.hh"
#include "slate_Memory.hh"

// -----------------------------------------------------------------------------
void test_memory( int nb )
{
    Test name( __func__ );
    int device = 0;

    slate::Memory mem( sizeof(double) * nb * nb );

    const int cnt0 = 10;
    const int cnt  = 12;

    std::cout << "add blocks\n";
    mem.addHostBlocks( cnt0 );
    assert( mem.available( g_host_num ) == size_t( cnt0 ));
    assert( mem.capacity ( g_host_num ) == size_t( cnt0 ));

    if (g_num_devices > 0) {
        mem.addDeviceBlocks( device, cnt0 );
        assert( mem.available( device ) == size_t( cnt0 ));
        assert( mem.capacity ( device ) == size_t( cnt0 ));
    }

    double* hx[ cnt ];
    double* dx[ cnt ];

    std::cout << "alloc\n";
    for (int k = 0; k < cnt; ++k) {
        hx[k] = (double*) mem.alloc( g_host_num );
        assert( mem.available( g_host_num ) == size_t( std::max( 0, cnt0 - (k+1) )));
        assert( mem.capacity ( g_host_num ) == size_t( std::max( cnt0, k+1 )));

        if (g_num_devices > 0) {
            dx[k] = (double*) mem.alloc( device );
            assert( mem.available( device ) == size_t( std::max( 0, cnt0 - (k+1) )));
            assert( mem.capacity ( device ) == size_t( std::max( cnt0, k+1 )));
        }
    }

    // test memory to make sure it's legit
    std::cout << "test memory\n";
    for (size_t k = 0; k < cnt; ++k) {
        for (size_t j = 0; j < nb*nb; ++j)
            hx[k][j] = (k + 1)*1000 + j;

        cudaMemcpy( dx[k], hx[k], sizeof(double) * nb * nb,
                    cudaMemcpyDeviceToHost );
    }

    std::cout << "free some\n";
    int some = cnt/2;
    for (int k = 0; k < some; ++k) {
        mem.free( hx[k], g_host_num );
        assert( mem.available( g_host_num ) == size_t( k+1 ) );

        if (g_num_devices > 0) {
            mem.free( dx[k], device );
            assert( mem.available( device ) == size_t( k+1 ) );
        }
    }

    std::cout << "alloc some again\n";
    for (int k = 0; k < some; ++k) {
        hx[k] = (double*) mem.alloc( g_host_num );
        assert( mem.available( g_host_num ) == size_t( some - (k+1) ) );

        if (g_num_devices > 0) {
            dx[k] = (double*) mem.alloc( device );
            assert( mem.available( device ) == size_t( some - (k+1) ) );
        }
    }

    std::cout << "free again\n";
    for (int k = 0; k < cnt; ++k) {
        mem.free( hx[k], g_host_num );
        assert( mem.available( g_host_num ) == size_t( k+1 ) );

        if (g_num_devices > 0) {
            mem.free( dx[k], device );
            assert( mem.available( device ) == size_t( k+1 ) );
        }
    }

    std::cout << "clear\n";
    mem.clearHostBlocks();
    if (g_num_devices > 0) {
        mem.clearDeviceBlocks( device );
    }

    assert( mem.available( g_host_num ) == 0 );
    assert( mem.capacity ( g_host_num ) == 0 );

    if (g_num_devices > 0) {
        assert( mem.available( device ) == 0 );
        assert( mem.capacity ( device ) == 0 );
    }
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    g_mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank( g_mpi_comm, &g_mpi_rank );
    MPI_Comm_size( g_mpi_comm, &g_mpi_size );

    int nb = 10;
    if (argc > 1) nb = atoi( argv[1] );

    test_memory( nb );

    MPI_Finalize();
    return 0;
}
