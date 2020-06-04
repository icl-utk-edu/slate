#ifndef SLATE_GRID_UTILS_HH
#define SLATE_GRID_UTILS_HH

#include <stdint.h>

//------------------------------------------------------------------------------
// Similar to ScaLAPACK numroc (number of rows or columns).
inline int64_t localRowsCols(int64_t n, int64_t nb, int iproc, int mpi_size)
{
    int64_t nblocks = n / nb;
    int64_t num = (nblocks / mpi_size) * nb;
    int64_t extra_blocks = nblocks % mpi_size;
    if (iproc < extra_blocks) {
        // extra full blocks
        num += nb;
    }
    else if (iproc == extra_blocks) {
        // last partial block
        num += n % nb;
    }
    return num;
}

//------------------------------------------------------------------------------
// Similar to BLACS gridinfo
// (local row ID and column ID in 2D block cyclic distribution).
inline int64_t whoismyrow(int mpi_rank, int64_t p)
{
    return (mpi_rank % p);
}

inline int64_t whoismycol(int mpi_rank, int64_t p)
{
    return (mpi_rank / p);
}

#endif // SLATE_GRID_UTILS_HH
