
#include "slate/types.hh"
#include "internal/internal.hh"
#include "internal/Tile_gerbt.hh"


namespace slate {

namespace internal {

template<class scalar_t>
void subtile_fetch_send(Matrix<scalar_t>& A,
                        int64_t i, int64_t j,
                        int64_t ioffset, int64_t joffset,
                        int64_t mb, int64_t nb,
                        int dst,
                        int64_t tag)
{
    trace::Block trace_block("MPI_Send");

    assert(tag >= 0);

    if (A.tileIsLocal(i, j) && dst != A.mpiRank()) {
        A.tileGetForReading(i, j, LayoutConvert::ColMajor);
        auto Aij = A(i, j);
        auto mpi_comm = A.mpiComm();
        MPI_Request request;

        // If no stride.
        if (mb == Aij.stride()) {
            // Use simple send.
            int count = mb*nb;
            auto data = Aij.data() + joffset*mb;

            slate_mpi_call(
                MPI_Isend(data, count, mpi_type<scalar_t>::value, dst, tag,
                         mpi_comm, &request));
        } else {
            // Otherwise, use strided send.
            int count = nb;
            int blocklength = mb;
            int stride = Aij.stride();
            auto data = Aij.data() + ioffset + joffset*stride;
            MPI_Datatype newtype;

            slate_mpi_call(
                MPI_Type_vector(count, blocklength, stride,
                                mpi_type<scalar_t>::value, &newtype));

            slate_mpi_call(MPI_Type_commit(&newtype));
            slate_mpi_call(MPI_Isend(data, 1, newtype, dst, tag, mpi_comm, &request));
            slate_mpi_call(MPI_Type_free(&newtype));
        }

        // Communication pattern ensures the buffer is not re-written
        // before the data is fully sent
        MPI_Request_free(&request);
    }
}

template<class scalar_t>
void subtile_fetch_recv(Matrix<scalar_t>& A,
                        int64_t i, int64_t j,
                        int64_t ioffset, int64_t joffset,
                        int64_t mb, int64_t nb,
                        int64_t tag,
                        scalar_t* data, int64_t stride)
{
    trace::Block trace_block("MPI_Recv");

    assert(tag >= 0);

    if (!A.tileIsLocal(i, j)) {
        auto mpi_comm = A.mpiComm();
        int src = A.tileRank(i, j);

        // If no stride.
        if (mb == stride) {
            // Use simple send.
            int count = mb*nb;

            slate_mpi_call(
                MPI_Recv(data, count, mpi_type<scalar_t>::value, src, tag,
                         mpi_comm, MPI_STATUS_IGNORE));
        } else {
            // Otherwise, use strided send.
            int count = nb;
            int blocklength = mb;
            MPI_Datatype newtype;

            slate_mpi_call(
                MPI_Type_vector(count, blocklength, stride,
                                mpi_type<scalar_t>::value, &newtype));

            slate_mpi_call(MPI_Type_commit(&newtype));
            slate_mpi_call(MPI_Recv(data, 1, newtype, src, tag,
                                    mpi_comm, MPI_STATUS_IGNORE));
            slate_mpi_call(MPI_Type_free(&newtype));
        }
    } else {
        A.tileGetForReading(i, j, LayoutConvert::ColMajor);
        auto Aij = A(i, j);

        for (int64_t jj = 0; jj < nb; ++jj) {
            for (int64_t ii = 0; ii < mb; ++ii) {
                data[ii + jj*stride] = Aij.at(ii+ioffset, jj+joffset);
            }
        }
    }
}

template<class scalar_t>
void subtile_return_send(Matrix<scalar_t>& A,
                         int64_t i, int64_t j,
                         int64_t ioffset, int64_t joffset,
                         int64_t mb, int64_t nb,
                         int64_t tag,
                         scalar_t* data, int64_t stride)
{
    trace::Block trace_block("MPI_Send");

    assert(tag >= 0);

    if (!A.tileIsLocal(i, j)) {
        auto mpi_comm = A.mpiComm();
        int dst = A.tileRank(i, j);

        // If no stride.
        if (mb == stride) {
            // Use simple send.
            int count = mb*nb;

            slate_mpi_call(
                MPI_Send(data, count, mpi_type<scalar_t>::value, dst, tag,
                         mpi_comm));
        } else {
            // Otherwise, use strided send.
            int count = nb;
            int blocklength = mb;
            MPI_Datatype newtype;

            slate_mpi_call(
                MPI_Type_vector(count, blocklength, stride,
                                mpi_type<scalar_t>::value, &newtype));

            slate_mpi_call(MPI_Type_commit(&newtype));
            slate_mpi_call(MPI_Send(data, 1, newtype, dst, tag, mpi_comm));
            slate_mpi_call(MPI_Type_free(&newtype));
        }
    } else {
        A.tileGetForWriting(i, j, LayoutConvert::ColMajor);
        auto Aij = A(i, j);

        for (int64_t jj = 0; jj < nb; ++jj) {
            for (int64_t ii = 0; ii < mb; ++ii) {
                Aij.at(ii+ioffset, jj+joffset) = data[ii + jj*stride];
            }
        }
    }
}

template<class scalar_t>
void subtile_return_recv(Matrix<scalar_t>& A,
                         int64_t i, int64_t j,
                         int64_t ioffset, int64_t joffset,
                         int64_t mb, int64_t nb,
                         int src,
                         int64_t tag)
{
    trace::Block trace_block("MPI_Send");

    assert(tag >= 0);

    if (A.tileIsLocal(i, j) && src != A.mpiRank()) {
        A.tileGetForWriting(i, j, LayoutConvert::ColMajor);
        auto Aij = A(i, j);
        auto mpi_comm = A.mpiComm();

        // If no stride.
        if (mb == Aij.stride()) {
            // Use simple send.
            int count = mb*nb;
            auto data = Aij.data() + joffset*mb;

            slate_mpi_call(
                MPI_Recv(data, count, mpi_type<scalar_t>::value, src, tag,
                         mpi_comm, MPI_STATUS_IGNORE));
        } else {
            // Otherwise, use strided send.
            int count = nb;
            int blocklength = mb;
            int stride = Aij.stride();
            auto data = Aij.data() + ioffset + joffset*stride;
            MPI_Datatype newtype;

            slate_mpi_call(
                MPI_Type_vector(count, blocklength, stride,
                                mpi_type<scalar_t>::value, &newtype));

            slate_mpi_call(MPI_Type_commit(&newtype));
            slate_mpi_call(MPI_Recv(data, 1, newtype, src, tag,
                                    mpi_comm, MPI_STATUS_IGNORE));
            slate_mpi_call(MPI_Type_free(&newtype));
        }
    }
}


template<class scalar_t>
void tile_iterator(Matrix<scalar_t> A1, Matrix<scalar_t> A2,
                   const int64_t ii1, int64_t ii2, const int64_t jj1, int64_t jj2,
                   int64_t ioffset2, int64_t joffset2,
                   std::function<void(int64_t, int64_t, int64_t, int64_t)> i_iter,
                   std::function<void(int64_t, int64_t, int64_t, int64_t)> j_iter,
                   std::function<void(int64_t, int64_t, int64_t,
                                      int64_t, int64_t, int64_t, int64_t, int64_t)> ij_iter)
{
    const int64_t mb1 = A1.tileMb(ii1);
    const int64_t nb1 = A1.tileNb(jj1);

    int64_t mb2 = A2.tileMb(ii2);
    int64_t m_step_total = 0;

    while (m_step_total < mb1) {
        const int64_t m_step = std::min(mb1-m_step_total, mb2 - ioffset2);

        i_iter(ii2, m_step_total, ioffset2, m_step);

        int64_t joffset2_inner = joffset2;
        int64_t jj2_inner = jj2;
        int64_t nb2 = A2.tileNb(jj2);
        int64_t n_step_total = 0;
        while (n_step_total < nb1) {
            const int64_t n_step = std::min(nb1-n_step_total, nb2 - joffset2_inner);

            ij_iter(ii2, jj2_inner, m_step_total, ioffset2, n_step_total, joffset2_inner, m_step, n_step);

            n_step_total += n_step;
            joffset2_inner += n_step;
            if (joffset2_inner >= nb2) {
                joffset2_inner = 0;
                ++jj2_inner;
                nb2 = A2.tileNb(jj2_inner);
            }
        }

        m_step_total += m_step;
        ioffset2 += m_step;
        if (ioffset2 >= mb2) {
            ioffset2 = 0;
            ++ii2;
            mb2 = A2.tileMb(ii2);
        }
    }

    int64_t nb2 = A2.tileNb(jj2);
    int64_t n_step_total = 0;
    while (n_step_total < nb1) {
        const int64_t n_step = std::min(nb1-n_step_total, nb2 - joffset2);

        j_iter(jj2, n_step_total, joffset2, n_step);

        n_step_total += n_step;
        joffset2 += n_step;
        if (joffset2 >= nb2) {
            joffset2 = 0;
            ++jj2;
            nb2 = A2.tileNb(jj2);
        }
    }
}


template<class scalar_t>
void tile_iterator(Matrix<scalar_t> A1, Matrix<scalar_t> A2,
                   const int64_t ii1, int64_t ii2,
                   int64_t ioffset2,
                   std::function<void(int64_t, int64_t, int64_t, int64_t)> i_iter)
{

    const int64_t mb1 = A1.tileMb(ii1);

    int64_t mb2 = A2.tileMb(ii2);
    int64_t total_m_step = 0;
    while (total_m_step < mb1) {
        const int64_t m_step = std::min(mb1-total_m_step, mb2-ioffset2);

        i_iter(ii2, total_m_step, ioffset2, m_step);

        total_m_step += m_step;
        ioffset2 += m_step;
        if (ioffset2 >= mb2) {
            ioffset2 = 0;
            ++ii2;
            mb2 = A2.tileMb(ii2);
        }
    }
}

template<class scalar_t>
void tile_step_i(Matrix<scalar_t>& A1, Matrix<scalar_t>& A2,
                 const int64_t ii1, int64_t& ii2,
                 int64_t& ioffset2)
{
    const int64_t mb1 = A1.tileMb(ii1);
    int64_t mb2 = A2.tileMb(ii2);
    int64_t step = 0;
    while (step + mb2 - ioffset2 < mb1) {
        step += mb2;
        ++ii2;
        mb2 = A2.tileMb(ii2);
    }
    ioffset2 = mb1 + ioffset2 - step;
}

template<class scalar_t>
void tile_step_j(Matrix<scalar_t>& A1, Matrix<scalar_t>& A2,
                 const int64_t jj1, int64_t& jj2,
                 int64_t& joffset2)
{
    const int64_t nb1 = A1.tileNb(jj1);
    int64_t nb2 = A2.tileNb(jj2);
    int64_t step = 0;
    while (step + nb2 - joffset2 < nb1) {
        step += nb2;
        ++jj2;
        nb2 = A2.tileNb(jj2);
    }
    joffset2 = nb1 + joffset2 - step;
}


template<typename scalar_t>
void gerbt(Matrix<scalar_t> A11,
           Matrix<scalar_t> A12,
           Matrix<scalar_t> A21,
           Matrix<scalar_t> A22,
           Matrix<scalar_t> U1,
           Matrix<scalar_t> U2,
           Matrix<scalar_t> V1,
           Matrix<scalar_t> V2)
{
    // Assuming U and V have same structure as the A's where they're multiplied

    const int64_t m = A11.m();
    slate_assert(m == A12.m());
    slate_assert(m == A21.m());
    slate_assert(m == A22.m());
    const int64_t n = A11.n();
    slate_assert(n == A12.n());
    slate_assert(n == A21.n());
    slate_assert(n == A22.n());

    const int64_t mt1 = A11.mt();
    const int64_t nt1 = A11.nt();

    // Send remote parts
    #pragma omp task firstprivate(A11, A12, A21, A22, U1, U2, V1, V2) \
                     priority(1)
    {
        int64_t ii2 = 0, ioffset2 = 0;

        for (int64_t ii1 = 0; ii1 < mt1; ++ii1) {
            const int64_t mb1 = A11.tileMb(ii1);

            int64_t jj2 = 0, joffset2 = 0;

            for (int64_t jj1 = 0; jj1 < nt1; ++jj1) {
                const int64_t nb1 = A11.tileNb(jj1);

                const int tag_increment = mt1*nt1;
                int tag = jj1 + ii1*nt1;

                const int compute_rank = A11.tileRank(ii1, jj1);

                subtile_fetch_send(U1,
                                   ii1, 0, 0, 0, mb1, 1,
                                   compute_rank, tag);
                tag += tag_increment;
                subtile_fetch_send(V1,
                                   jj1, 0, 0, 0, nb1, 1,
                                   compute_rank, tag);
                tag += tag_increment;

                tile_iterator(
                    A11, A22,
                    ii1, ii2, jj1, jj2, ioffset2, joffset2,
                              // i_iter
                    [&] (int64_t ii2, int64_t ioffset1, int64_t ioffset2, int64_t m_step) mutable {
                        subtile_fetch_send(A21,
                                           ii2, jj1, ioffset2, 0, m_step, nb1,
                                           compute_rank, tag);
                        tag += tag_increment;
                        subtile_fetch_send(U2,
                                           ii2, 0, ioffset2, 0, m_step, 1,
                                           compute_rank, tag);
                        tag += tag_increment;
                    },
                    [&] (int64_t jj2, int64_t joffset1, int64_t joffset2, int64_t n_step) mutable {
                        subtile_fetch_send(A12,
                                           ii1, jj2, 0, joffset2, mb1, n_step,
                                           compute_rank, tag);
                        tag += tag_increment;
                        subtile_fetch_send(V2,
                                           jj2, 0, joffset2, 0, n_step, 1,
                                           compute_rank, tag);
                        tag += tag_increment;
                    },
                    [&] (int64_t ii2, int64_t jj2,
                         int64_t ioffset1, int64_t ioffset2, int64_t joffset1, int64_t joffset2,
                         int64_t m_step, int64_t n_step) mutable {
                        subtile_fetch_send(A22,
                                           ii2, jj2, ioffset2, joffset2, m_step, n_step,
                                           compute_rank, tag);
                        tag += tag_increment;
                    });
                tile_step_j(A11, A22, jj1, jj2, joffset2);
            }

            tile_step_i(A11, A22, ii1, ii2, ioffset2);
        }
    }



    int64_t ii2 = 0, ioffset2 = 0;

    for (int64_t ii1 = 0; ii1 < mt1; ++ii1) {
        const int64_t mb1 = A11.tileMb(ii1);

        int64_t jj2 = 0, joffset2 = 0;

        for (int64_t jj1 = 0; jj1 < nt1; ++jj1) {
            const int64_t nb1 = A11.tileNb(jj1);

            if (A11.tileIsLocal(ii1, jj1)) {
                #pragma omp task firstprivate(A11, A12, A21, A22, U1, U2, V1, V2, \
                                              ii1, ii2, jj1, jj2, \
                                              ioffset2, joffset2, mb1, nb1)
                {
                    Tile<scalar_t> T11 = A11(ii1, jj1);
                    std::vector<scalar_t> T12_vector(mb1*nb1);
                    Tile<scalar_t> T12 (mb1, nb1, T12_vector.data(), mb1,
                                        HostNum, TileKind::Workspace, Layout::ColMajor);
                    std::vector<scalar_t> T21_vector(mb1*nb1);
                    Tile<scalar_t> T21 (mb1, nb1, T21_vector.data(), mb1,
                                        HostNum, TileKind::Workspace, Layout::ColMajor);
                    std::vector<scalar_t> T22_vector(mb1*nb1);
                    Tile<scalar_t> T22 (mb1, nb1, T22_vector.data(), mb1,
                                        HostNum, TileKind::Workspace, Layout::ColMajor);
                    std::vector<scalar_t> TU1_vector(mb1);
                    Tile<scalar_t> TU1 (mb1, 1, TU1_vector.data(), mb1,
                                        HostNum, TileKind::Workspace, Layout::ColMajor);
                    std::vector<scalar_t> TU2_vector(mb1);
                    Tile<scalar_t> TU2 (mb1, 1, TU2_vector.data(), mb1,
                                        HostNum, TileKind::Workspace, Layout::ColMajor);
                    std::vector<scalar_t> TV1_vector(nb1);
                    Tile<scalar_t> TV1 (nb1, 1, TV1_vector.data(), nb1,
                                        HostNum, TileKind::Workspace, Layout::ColMajor);
                    std::vector<scalar_t> TV2_vector(nb1);
                    Tile<scalar_t> TV2 (nb1, 1, TV2_vector.data(), nb1,
                                        HostNum, TileKind::Workspace, Layout::ColMajor);

                    const int tag_increment = mt1*nt1;
                    int tag = jj1 + ii1*nt1;

                    subtile_fetch_recv(U1,
                                       ii1, 0, 0, 0, mb1, 1,
                                       tag,
                                       TU1.data(), mb1);
                    tag += tag_increment;
                    subtile_fetch_recv(V1,
                                       jj1, 0, 0, 0, nb1, 1,
                                       tag,
                                       TV1.data(), nb1);
                    tag += tag_increment;

                    tile_iterator(
                        A11, A22,
                        ii1, ii2, jj1, jj2, ioffset2, joffset2,
                        // i_iter
                        [&] (int64_t ii2, int64_t ioffset1, int64_t ioffset2, int64_t m_step) mutable {
                            subtile_fetch_recv(A21,
                                               ii2, jj1, ioffset2, 0, m_step, nb1,
                                               tag, T21.data() + ioffset1, mb1);
                            tag += tag_increment;
                            subtile_fetch_recv(U2,
                                               ii2, 0, ioffset2, 0, m_step, 1,
                                               tag, TU2.data() + ioffset1, mb1);
                            tag += tag_increment;
                        },
                        // j_iter
                        [&] (int64_t jj2, int64_t joffset1, int64_t joffset2, int64_t n_step) mutable {
                            subtile_fetch_recv(A12,
                                               ii1, jj2, 0, joffset2, mb1, n_step,
                                               tag, T12.data() + joffset1*mb1, mb1);
                            tag += tag_increment;
                            subtile_fetch_recv(V2,
                                               jj2, 0, joffset2, 0, n_step, 1,
                                               tag, TV2.data() + joffset1, nb1);
                            tag += tag_increment;
                        },
                        // ij_iter
                        [&] (int64_t ii2, int64_t jj2,
                             int64_t ioffset1, int64_t ioffset2, int64_t joffset1, int64_t joffset2,
                             int64_t m_step, int64_t n_step) mutable {
                            subtile_fetch_recv(A22,
                                               ii2, jj2, ioffset2, joffset2, m_step, n_step,
                                               tag, T22.data() + ioffset1 + joffset1*mb1, mb1);
                            tag += tag_increment;
                        });

                    gerbt(mb1, nb1,
                          T11.data(), T11.stride(),
                          T12.data(), T12.stride(),
                          T21.data(), T21.stride(),
                          T22.data(), T22.stride(),
                          TU1.data(),
                          TU2.data(),
                          TV1.data(),
                          TV2.data());

                    tag = jj1 + ii1*nt1;
                    tile_iterator(
                        A11, A22,
                        ii1, ii2, jj1, jj2, ioffset2, joffset2,
                                  // i_iter
                        [&] (int64_t ii2, int64_t ioffset1, int64_t ioffset2, int64_t m_step) mutable {
                            subtile_return_send(A21,
                                                ii2, jj1, ioffset2, 0, m_step, nb1,
                                                tag, T21.data() + ioffset1, mb1);
                            tag += tag_increment;
                        },
                        [&] (int64_t jj2, int64_t joffset1, int64_t joffset2, int64_t n_step) mutable {
                            subtile_return_send(A12,
                                                ii1, jj2, 0, joffset2, mb1, n_step,
                                                tag, T12.data() + joffset1*mb1, mb1);
                            tag += tag_increment;
                        },
                        [&] (int64_t ii2, int64_t jj2,
                             int64_t ioffset1, int64_t ioffset2, int64_t joffset1, int64_t joffset2,
                             int64_t m_step, int64_t n_step) mutable {
                            subtile_return_send(A22,
                                                ii2, jj2, ioffset2, joffset2, m_step, n_step,
                                                tag, T22.data() + ioffset1 + joffset1*mb1, mb1);
                            tag += tag_increment;
                        });
                }
            } else {
                #pragma omp task firstprivate(A11, A12, A21, A22, U1, U2, V1, V2, \
                                              ii1, ii2, jj1, jj2, \
                                              ioffset2, joffset2, mb1, nb1)
                {
                    const int compute_rank = A11.tileRank(ii1, jj1);

                    const int tag_increment = mt1*nt1;
                    int tag = jj1 + ii1*nt1;
                    tile_iterator(
                        A11, A22,
                        ii1, ii2, jj1, jj2, ioffset2, joffset2,
                                  // i_iter
                        [&] (int64_t ii2, int64_t ioffset1, int64_t ioffset2, int64_t m_step) mutable {
                            subtile_return_recv(A21,
                                                ii2, jj1, ioffset2, 0, m_step, nb1,
                                                compute_rank, tag);
                            tag += tag_increment;
                        },
                        [&] (int64_t jj2, int64_t joffset1, int64_t joffset2, int64_t n_step) mutable {
                            subtile_return_recv(A12,
                                                ii1, jj2, 0, joffset2, mb1, n_step,
                                                compute_rank, tag);
                            tag += tag_increment;
                        },
                        [&] (int64_t ii2, int64_t jj2,
                             int64_t ioffset1, int64_t ioffset2, int64_t joffset1, int64_t joffset2,
                             int64_t m_step, int64_t n_step) mutable {
                            subtile_return_recv(A22,
                                                ii2, jj2, ioffset2, joffset2, m_step, n_step,
                                                compute_rank, tag);
                            tag += tag_increment;
                        });
                }
            }

            tile_step_j(A11, A22, jj1, jj2, joffset2);
        }

        tile_step_i(A11, A22, ii1, ii2, ioffset2);
    }
}

template
void gerbt(Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>);

template
void gerbt(Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>);

template
void gerbt(Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>);

template
void gerbt(Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>);

template<typename scalar_t>
void gerbt(bool transposed,
           Matrix<scalar_t> B1,
           Matrix<scalar_t> B2,
           Matrix<scalar_t> U1,
           Matrix<scalar_t> U2)
{
    const int64_t m = B1.m();
    slate_assert(m == B2.m());
    const int64_t n = B1.n();
    slate_assert(n == B2.n());

    const int64_t nt = B1.nt();
    slate_assert(B2.nt() == nt);
    const int64_t mt1 = B1.mt();

    // Send tiles that are needed
    #pragma omp task firstprivate(B1, B2, U1, U2) \
                     priority(1)
    {

        int64_t ii2 = 0, ioffset2 = 0;

        for (int64_t ii1 = 0; ii1 < mt1; ++ii1) {
            const int64_t mb1 = B1.tileMb(ii1);

            for (int64_t jj = 0; jj < nt; ++jj) {
                const int64_t nb = B1.tileNb(jj);
                const int compute_rank = B1.tileRank(ii1, jj);

                const int tag_increment = nt*mt1;
                int tag = jj + ii1*nt;

                subtile_fetch_send(U1,
                                   ii1, 0, 0, 0, mb1, 1,
                                   compute_rank, tag);
                tag += tag_increment;
                tile_iterator(B1, B2,
                              ii1, ii2, ioffset2,
                              [&] (int64_t ii2, int64_t, int64_t ioffset2, int64_t m_step) mutable {
                                  subtile_fetch_send(U2,
                                                     ii2, 0, ioffset2, 0, m_step, 1,
                                                     compute_rank, tag);
                                tag += tag_increment;

                                  subtile_fetch_send(B2,
                                                     ii2, jj, ioffset2, 0, m_step, nb,
                                                     compute_rank, tag);
                                tag += tag_increment;
                              });
            }
            tile_step_i(B1, B2, ii1, ii2, ioffset2);
        }
    }


    int64_t ii2 = 0, ioffset2 = 0;

    for (int64_t ii1 = 0; ii1 < mt1; ++ii1) {
        const int64_t mb1 = B1.tileMb(ii1);

        for (int64_t jj = 0; jj < nt; ++jj) {
            const int64_t nb = B1.tileNb(jj);

            #pragma omp task firstprivate(B1, B2, U1, U2, \
                                          ii1, ii2, jj, ioffset2, mb1, nb)
            if (B1.tileIsLocal(ii1, jj)) {

                const int tag_increment = nt*mt1;
                int tag = jj + ii1*nt;

                Tile<scalar_t> TB1 = B1(ii1, jj);
                std::vector<scalar_t> TB2_vector(mb1*nb);
                Tile<scalar_t> TB2 (mb1, nb, TB2_vector.data(), mb1,
                                    HostNum, TileKind::Workspace, Layout::ColMajor);
                std::vector<scalar_t> TU1_vector(mb1);
                Tile<scalar_t> TU1 (mb1, 1, TU1_vector.data(), mb1,
                                    HostNum, TileKind::Workspace, Layout::ColMajor);
                std::vector<scalar_t> TU2_vector(mb1);
                Tile<scalar_t> TU2 (mb1, 1, TU2_vector.data(), mb1,
                                    HostNum, TileKind::Workspace, Layout::ColMajor);


                subtile_fetch_recv(U1,
                                   ii1, 0, 0, 0, mb1, 1,
                                   tag, TU1.data(), mb1);
                tag += tag_increment;
                tile_iterator(B1, B2,
                              ii1, ii2, ioffset2,
                              [&] (int64_t ii2, int64_t ioffset1, int64_t ioffset2, int64_t m_step) mutable {
                    subtile_fetch_recv(U2,
                                       ii2, 0, ioffset2, 0, m_step, 1,
                                       tag, TU2.data() + ioffset1, mb1);
                    tag += tag_increment;
                    subtile_fetch_recv(B2,
                                       ii2, jj, ioffset2, 0, m_step, nb,
                                       tag, TB2.data() + ioffset1, mb1);
                    tag += tag_increment;
                });

                if (transposed) {
                    gerbt_trans(mb1, nb,
                                TB1.data(), TB1.stride(),
                                TB2.data(), TB2.stride(),
                                TU1.data(),
                                TU2.data());
                } else {
                    gerbt_notrans(mb1, nb,
                                  TB1.data(), TB1.stride(),
                                  TB2.data(), TB2.stride(),
                                  TU1.data(),
                                  TU2.data());
                }


                tag = jj + ii1*nt;
                tile_iterator(B1, B2,
                              ii1, ii2, ioffset2,
                              [&] (int64_t ii2, int64_t ioffset1, int64_t ioffset2, int64_t m_step) mutable {
                        subtile_return_send(B2,
                                            ii2, jj, ioffset2, 0, m_step, nb,
                                            tag, TB2.data() + ioffset1, mb1);
                        tag += tag_increment;
                    });
            } else {
                const int compute_rank = B1.tileRank(ii1, jj);

                const int tag_increment = nt*mt1;
                int tag = jj + ii1*nt;

                tile_iterator(B1, B2,
                              ii1, ii2, ioffset2,
                              [&] (int64_t ii2, int64_t ioffset1, int64_t ioffset2, int64_t m_step) mutable {
                        subtile_return_recv(B2,
                                            ii2, jj, ioffset2, 0, m_step, nb,
                                            compute_rank, tag);
                        tag += tag_increment;
                    });
            }
        }
        tile_step_i(B1, B2, ii1, ii2, ioffset2);
    }
}

template
void gerbt(bool,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>,
           Matrix<float>);

template
void gerbt(bool,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>,
           Matrix<double>);

template
void gerbt(bool,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>,
           Matrix<std::complex<float>>);

template
void gerbt(bool,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>,
           Matrix<std::complex<double>>);


} // namespace internal

} // namespace slate
