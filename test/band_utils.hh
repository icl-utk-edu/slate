#ifndef SLATE_BAND_UTILS_HH
#define SLATE_BAND_UTILS_HH

#include "slate/BandMatrix.hh"

//------------------------------------------------------------------------------
// Returns local index for element i, or the next element after i if this rank
// doesn't contain i.
//
// i    is global index
// nb   is block size
// p    is # processes
// rank is current processor's rank
//
inline int64_t global2local(int64_t i, int64_t nb, int p, int rank)
{
    // rank containing i
    int irank  = (i / nb) % p;

    // local block containing i or next local block after i
    int64_t iblock = (i / nb) / p;
    if (rank < irank)
        iblock += 1;

    // local index of i or element after i
    int64_t ilocal = iblock*nb;
    if (rank == irank)
        ilocal += i % nb;
    return ilocal;
}

//------------------------------------------------------------------------------
// Returns local index for row i, or the next row after i if this rank
// doesn't contain i.
//
// i    is global row index
// mb   is height of block rows
// p    is # rows in 2D process grid
// q    is # cols in 2D process grid
// rank is current process's rank
//
inline int64_t global2local_2Drow(int64_t i, int64_t nb, int p, int q, int rank)
{
    int myrow = rank % p;
    return global2local(i, nb, p, myrow);
}

//------------------------------------------------------------------------------
// Returns local index for column j, or the next column after j if this rank
// doesn't contain j.
//
// j    is global col index
// nb   is width of block cols
// p    is # rows in 2D process grid
// q    is # cols in 2D process grid
// rank is current process's rank
//
inline int64_t global2local_2Dcol(int64_t j, int64_t nb, int p, int q, int rank)
{
    int mycol = rank / p;
    return global2local(j, nb, q, mycol);
}

//------------------------------------------------------------------------------
// Initializes a SLATE band matrix.
// Inserts tiles that overlap the band and initializes them to random values
// inside the band, zero outside the band.
// idist and iseed are passed to LAPACK's larnv.
template <typename scalar_t>
void initializeRandom(
    int64_t idist, int64_t iseed[4],
    slate::BandMatrix<scalar_t>& A)
{
    int64_t kl = A.lowerBandwidth();
    int64_t ku = A.upperBandwidth();

    // initially, assume fixed nb
    int64_t nb = A.tileNb(0);
    int64_t klt = slate::ceildiv(kl, nb);
    int64_t kut = slate::ceildiv(ku, nb);
    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j) && i >= j - kut && i <= j + klt) {
                A.tileInsert(i, j);
                auto T = A(i, j);
                lapack::larnv(2, iseed, T.size(), T.data());
                for (int64_t tj = 0; tj < T.nb(); ++tj) {
                    for (int64_t ti = 0; ti < T.mb(); ++ti) {
                        int64_t j_i = (jj + tj) - (ii + ti);
                        if (-kl > j_i || j_i > ku) {
                            T.at(ti, tj) = 0;
                        }
                    }
                }
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }
}

//------------------------------------------------------------------------------
// Given a full ScaLAPACK matrix, sets data outside the band to zero.
// This is primarily useful to test SLATE's band routines (like gbmm, tbsm)
// that have no equivalent in ScaLAPACK.
// todo: use ScaLAPACK descriptor
template <typename scalar_t>
void zeroOutsideBand(
    scalar_t* A, int64_t m, int64_t n, int64_t kl, int64_t ku,
    int64_t mb, int64_t nb,
    int64_t myrow, int64_t mycol, int64_t p, int64_t q,
    int64_t lldA)
{
    using blas::max;
    using blas::min;

    slate_assert(mb == nb);  // todo: allow different?

    // Zero out data outside bandwidth in A for ScaLAPACK.
    // Since ScaLAPACK lacks gbmm and gbmv, we will use gemm.
    for (int64_t jj = 0; jj < n; ++jj) {
        int pcol = (jj / nb) % q;
        if (pcol == mycol) {
            int64_t jlocal = global2local(jj, nb, q, mycol);

            // set rows [0 : k - ku - 1] = 0
            int64_t i0 = max(global2local(0,       nb, p, myrow), 0);
            int64_t i1 = min(global2local(jj - ku, nb, p, myrow), m);

            // set rows [k + kl + 1 : m - 1] = 0
            int64_t i2 = max(global2local(jj + kl + 1, nb, p, myrow), 0);
            int64_t i3 = min(global2local(m,           nb, p, myrow), m);

            for (int64_t i = i0; i < i1; ++i) {
                A[ i + jlocal*lldA ] = 0;
            }
            for (int64_t i = i2; i < i3; ++i) {
                A[ i + jlocal*lldA ] = 0;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Constructs a SLATE band matrix from a full ScaLAPACK matrix.
// This is primarily useful to test SLATE's band routines (like gbmm, tbsm)
// that have no equivalent in ScaLAPACK.
template <typename scalar_t>
slate::BandMatrix<scalar_t> BandFromScaLAPACK(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    scalar_t* Adata, int64_t lldA, int64_t nb,
    int p, int q, MPI_Comm comm)
{
    auto A = slate::BandMatrix<scalar_t>(
                 m, n, kl, ku, nb, p, q, comm);
    int64_t klt = slate::ceildiv(kl, nb);
    int64_t kut = slate::ceildiv(ku, nb);
    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t jb = A.tileNb(j);
        // Using Scalapack indxg2l
        int64_t jj_local = nb*(jj / (nb*q)) + (jj % nb);
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            int64_t ib = A.tileMb(i);
            if (A.tileIsLocal(i, j) && i >= j - kut && i <= j + klt) {
                // Using Scalapack indxg2l
                int64_t ii_local = nb*(ii / (nb*p)) + (ii % nb);
                A.tileInsert(i, j, A.hostNum(),
                             &Adata[ ii_local + jj_local*lldA ], lldA);
            }
            ii += ib;
        }
        jj += jb;
    }
    return A;
}

#include "slate/HermitianBandMatrix.hh"

//------------------------------------------------------------------------------
// Given a full ScaLAPACK matrix, sets data outside the hermitian band to zero.
// This is primarily useful to test SLATE's band routines (like gbmm, tbsm)
// that have no equivalent in ScaLAPACK.
// todo: use ScaLAPACK descriptor
template <typename scalar_t>
void zeroOutsideBand(
    slate::Uplo uplo,
    scalar_t* A, int64_t n, int64_t kd,
    int64_t nb, int64_t myrow, int64_t mycol,
    int64_t p, int64_t q, int64_t lldA)
{
    using blas::max;
    using blas::min;

    // Zero out data outside bandwidth in A for ScaLAPACK.
    // Since ScaLAPACK lacks hbmm and hbmv, we will use gemm.
    for (int64_t jj = 0; jj < n; ++jj) {
        int pcol = (jj / nb) % q;
        if (pcol == mycol) {
            int64_t jlocal = global2local(jj, nb, q, mycol);

            if (uplo == slate::Uplo::Upper) {
                // set rows [0 : k - kd - 1] = 0
                int64_t i0 = max(global2local(0,       nb, p, myrow), 0);
                int64_t i1 = min(global2local(jj - kd, nb, p, myrow), n);

                for (int64_t i = i0; i < i1; ++i) {
                    A[ i + jlocal*lldA ] = 0;
                }

                int64_t i2 = max(global2local(jj + 0 + 1, nb, p, myrow), 0);
                int64_t i3 = min(global2local(n,          nb, p, myrow), n);

                for (int64_t i = i2; i < i3; ++i) {
                    A[ i + jlocal*lldA ] = 0;
                }

            }
            else if (uplo == slate::Uplo::Lower) {
                int64_t i0 = max(global2local(0,       nb, p, myrow), 0);
                int64_t i1 = min(global2local(jj - 0,  nb, p, myrow), n);

                for (int64_t i = i0; i < i1; ++i) {
                    A[ i + jlocal*lldA ] = 0;
                }

                // set rows [k + kd + 1 : m - 1] = 0
                int64_t i2 = max(global2local(jj + kd + 1, nb, p, myrow), 0);
                int64_t i3 = min(global2local(n,           nb, p, myrow), n);

                for (int64_t i = i2; i < i3; ++i) {
                    A[ i + jlocal*lldA ] = 0;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Constructs a SLATE hermitian band matrix from a full ScaLAPACK matrix.
// This is primarily useful to test SLATE's band routines (like hbmm, hermitian tbsm)
// that have no equivalent in ScaLAPACK.
template <typename scalar_t>
slate::HermitianBandMatrix<scalar_t> HermitianBandFromScaLAPACK(
    slate::Uplo uplo,
    int64_t n, int64_t kd,
    scalar_t* Adata, int64_t lldA, int64_t nb,
    int p, int q, MPI_Comm comm)
{
    auto A = slate::HermitianBandMatrix<scalar_t>(
              uplo, n, kd, nb, p, q, MPI_COMM_WORLD);

    int64_t kdt = slate::ceildiv(kd, nb);

    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t jb = A.tileNb(j);
        // Using Scalapack indxg2l
        int64_t jj_local = nb*(jj / (nb*q)) + (jj % nb);
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            int64_t ib = A.tileMb(i);
            if (A.tileIsLocal(i, j)) {
                if ((A.uplo() == slate::Uplo::Lower && i <= j + kdt && j <= i) ||
                    (A.uplo() == slate::Uplo::Upper && i >= j - kdt && j >= i)) {
                    // Using Scalapack indxg2l
                    int64_t ii_local = nb*(ii / (nb*p)) + (ii % nb);
                    A.tileInsert(i, j, A.hostNum(),
                                 &Adata[ ii_local + jj_local*lldA ], lldA);
                }
            }
            ii += ib;
        }
        jj += jb;
    }

    return A;
}

#endif // SLATE_BAND_UTILS_HH
