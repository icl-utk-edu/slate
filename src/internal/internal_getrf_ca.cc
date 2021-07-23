// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_getrf.hh"
#include "internal/Tile_getrf_tntpiv.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//TODO::RABAB, no need to pass A here
template <typename scalar_t>
void getrf_ca(
    Matrix<scalar_t>& A,
    std::vector< Tile<scalar_t> >& tiles,
    int64_t diag_len, int64_t ib, int stage,
    int nb, std::vector<int64_t>& tile_indices,
    std::vector< std::vector<AuxPivot<scalar_t>> >& aux_pivot,
    int mpi_rank, int max_panel_threads, int priority)
{

    // Launch the panel tasks.
    int thread_size = max_panel_threads;
    if (int(tiles.size()) < max_panel_threads)
        thread_size = tiles.size();

    ThreadBarrier thread_barrier;
    std::vector<scalar_t> max_value(thread_size);
    std::vector<int64_t> max_index(thread_size);
    std::vector<int64_t> max_offset(thread_size);
    std::vector<scalar_t> top_block(ib*nb);

   #if 1
       omp_set_nested(1);
       // Launching new threads for the panel guarantees progression.
       // This should never deadlock, but may be detrimental to performance.
       #pragma omp parallel for \
           num_threads(thread_size) \
           shared(thread_barrier, max_value, max_index, max_offset, \
                 top_block, aux_pivot)
    #else
      // Issuing panel operation as tasks may cause a deadlock.
       #pragma omp taskloop \
           num_tasks(thread_size) \
           shared(thread_barrier, max_value, max_index, max_offset, \
                top_block, aux_pivot)
    #endif

    for (int thread_rank = 0; thread_rank < thread_size; ++thread_rank) {
        // Factor the panel in parallel.
        getrf_tntpiv(diag_len, ib, stage,
        tiles, tile_indices,
        aux_pivot,
        mpi_rank, //mpi_root, MPI_COMM_SELF,
        thread_rank, thread_size,
        thread_barrier,
        max_value, max_index, max_offset, top_block);
    }
    #pragma omp taskwait

}
//------------------------------------------------------------------------------
/// LU factorization of a column of tiles.
/// Dispatches to target implementations.
/// @ingroup gesv_internal
///
template <Target target, typename scalar_t>
void getrf_ca(Matrix<scalar_t>&& A, Matrix<scalar_t>&& Awork,
           int64_t diag_len, int64_t ib,
           std::vector<Pivot>& pivot,
           int max_panel_threads, int priority)
{
    getrf_ca(internal::TargetType<target>(),
          A, Awork, diag_len, ib, pivot, max_panel_threads, priority);
}

//------------------------------------------------------------------------------
/// LU factorization of a column of tiles, host implementation.
/// @ingroup gesv_internal
///
template <typename scalar_t>
void getrf_ca(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A, Matrix<scalar_t>& Awork,
           int64_t diag_len, int64_t ib,
           std::vector<Pivot>& pivot,
           int max_panel_threads, int priority)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;

    // Assumes column major RABAB::Check the row major in case of getrf gpu as in old imp TODO
    const Layout layout = Layout::ColMajor;

    assert(A.nt() == 1);

    //slate::Matrix<scalar_t> Awork = A.emptyLike();
    //Awork.insertLocalTiles();
    internal::copy<Target::HostTask>( std::move(A), std::move(Awork) );

    // Move the panel to the host
    std::set<ij_tuple> A_tiles_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            A_tiles_set.insert({i, 0});
        }
    }
    A.tileGetForWriting(A_tiles_set, LayoutConvert::ColMajor);

    // lists of local tiles, indices, and offsets
    std::vector< Tile<scalar_t> > tiles, tiles_copy_poriginal;
    std::vector<int64_t> tile_indices;

    // Build the broadcast set.
    // Build lists of local tiles, indices, and offsets.
    int64_t tile_offset = 0;
    std::set<int> bcast_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        bcast_set.insert(A.tileRank(i, 0));
        if (A.tileIsLocal(i, 0)) {
            tiles.push_back(Awork(i, 0));
            tile_indices.push_back(i);
        }
        tile_offset += A.tileMb(i);
    }
    // Find each rank's first (top-most) row in this panel
    std::vector< std::pair<int, int64_t> > rank_rows;
    rank_rows.reserve(bcast_set.size());
    for (int r: bcast_set) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileRank(i, 0) == r) {
                rank_rows.push_back({r, i});
                break;
            }
        }
    }

    // Sort rank_rows by row.
    std::sort(rank_rows.begin(), rank_rows.end(), compareSecond<int, int64_t>);

    int index;
    for (index = 0; index < int(rank_rows.size()); ++index) {
        if (rank_rows[index].first == A.mpiRank())
            break;
    }

    // If participating in the panel factorization.
    if (index < int(rank_rows.size())) {

        int nranks = rank_rows.size();
        int nlevels = int( ceil( log2( nranks ) ) );

        std::vector< std::vector<AuxPivot<scalar_t>>> aux_pivot(2);
        aux_pivot[0].resize(diag_len);
        aux_pivot[1].resize(diag_len);

        #if 0
            for (int i=0; i<Awork.mt(); i++){
                if (A.tileIsLocal(i, 0)){
                    if( A.mpiRank() == 0){
                        std::cout<<"\n Tile: "<<i<<" of rank: "<<A.mpiRank()<<std::endl;
                        for(int m=0; m<Awork.tileMb(i);m++){
                            for(int n=0; n<Awork.tileMb(i);n++){
                               std::cout<<Awork(i,0)(m,n)<<",";
                            }
                            std::cout<<std::endl;
                        }
                   }
                }
           }
        #endif

        // Factor the panel locally in parallel.
        getrf_ca(A, tiles, diag_len, ib, 0,
            A.tileNb(0), tile_indices, aux_pivot,
            A.mpiRank(), max_panel_threads, priority);

        //TODO::RABAB pay attention when you are doing CALU for large matrix with will give wrong results
        //because we will get the orignial tile not the permuted one.
        internal::copy<Target::HostTask>( std::move(A), std::move(Awork) );


        std::vector< std::vector<std::pair<int, int64_t>> > global_tracking(tile_indices.size());

        for (int i=0; i < int(tile_indices.size()); i++) {
            global_tracking[i].reserve(A.tileMb(0));
            for (int64_t j = 0; j < A.tileMb(0); ++j) {
                global_tracking[i].push_back({tile_indices[i], j});
                       //  std::cout<<global_tracking[i][j].first<<", "<<global_tracking[i][j].second<<std::endl;
                }
            }


        std::pair<int, int64_t> global_pair;

        for(int j=0; j < diag_len ; ++j){
            if (aux_pivot[0][j].localTileIndex() > 0 ||
                aux_pivot[0][j].localOffset() > j){

                swapLocalRow(
                    0, A.tileNb(0),
                    tiles[0], j,
                    tiles[aux_pivot[0][j].localTileIndex()],
                    aux_pivot[0][j].localOffset());

                    int index = aux_pivot[0][j].localTileIndex();
                    int offset = aux_pivot[0][j].localOffset();

                    global_pair = global_tracking[0][j];
                    global_tracking[0][j] = global_tracking[index][offset];
                    global_tracking[index][offset]=global_pair;
            }
        }


        for(int j=0; j < diag_len ; ++j){
            aux_pivot[0][j].set_tileIndex(global_tracking[0][j].first);
            aux_pivot[0][j].set_elementOffset(global_tracking[0][j].second);
        }


        #if 0
            if(A.mpiRank()==1){
                for (int i=0; i<tile_indices.size(); i++) {
                    for (int64_t j = 0; j < A.tileMb(0); ++j) {
                        std::cout<<global_tracking[i][j].first<<", "<<global_tracking[i][j].second<<std::endl;
                    }
               }
            }
        #endif




       #if 0
           for (int i=0; i<A.mt(); i++){
               if (A.tileIsLocal(i, 0)){
                   if( A.mpiRank() == 3 ){
                       std::cout<<"\n After Tile: "<<i<<" of rank: "<<A.mpiRank()<<std::endl;
                       for(int m=0; m<A.tileMb(i);m++){
                           for(int n=0; n<A.tileMb(i);n++){
                               std::cout<<Awork(i,0)(m,n)<<",";
                           }
                      std::cout<<std::endl;
                      }
                   }
              }
          }
        #endif

       //Alocate workspace to copy tiles in the tree reduction.
       //These tiles will only be used during factorization.
       //The permoutations happen in the copy tiles of the work panel
       //TODO::RABAB all nodes will allocate those two tiles,
       //but only src nodes during tree reduction will use it
       //TODO::RABAB I am not sure what is the overhead of moving this inside the for loop. Need testing.

       std::vector< Tile<scalar_t> > local_tiles;
       std::vector<scalar_t> data1( A.tileMb(0) * A.tileNb(0) );
       std::vector<scalar_t> data2( A.tileMb(0) * A.tileNb(0) );

       Tile<scalar_t> tile1( A.tileMb(0), A.tileNb(0),
          &data1[0], A.tileMb(0), A.hostNum(), TileKind::Workspace );
       Tile<scalar_t> tile2( A.tileMb(0), A.tileNb(0),
          &data2[0], A.tileMb(0), A.hostNum(), TileKind::Workspace );

       local_tiles.push_back( tile1 );
       local_tiles.push_back( tile2 );

       int step =1;
       int src, dst;
       int64_t i_src, i_dst, i_current;

       for (int level = 0; level < nlevels; ++level){

           if(index % (2*step) == 0){
               if(index + step < nranks){
                   src = rank_rows[ index + step].first;
                   i_current = rank_rows[ index ].second;
                   i_dst = (rank_rows[ index ].second) + 1;

                 //std::cout<<"Recv ("<<src<<", "<<rank_rows[index].first<<")"<< " In: " <<i_dst<<std::endl;

                   Awork.tileRecv(i_dst, 0, src, layout);

                   MPI_Status status;
                   MPI_Recv(aux_pivot.at(1).data(),
                               sizeof(AuxPivot<scalar_t>)*aux_pivot.at(1).size(),
                               MPI_BYTE, src, 0, A.mpiComm(),  &status);


                   Awork(i_current, 0).copyData( &local_tiles[0]);
                   Awork(i_dst, 0).copyData( &local_tiles[1]);


            #if 0
                if( (A.mpiRank()==0) ){
                    for (int64_t i = 0; i < diag_len; ++i) {
                        std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[0][i].tileIndex()
                            <<","<<aux_pivot[0][i].elementOffset()<<", "
                            <<aux_pivot[0][i].localTileIndex()
                            <<","<<aux_pivot[0][i].localOffset()<<std::endl;
                    }

                    for (int64_t i = 0; i < diag_len; ++i) {
                        std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[1][i].tileIndex()
                            <<","<<aux_pivot[1][i].elementOffset()<<", "
                            <<aux_pivot[1][i].localTileIndex()
                            <<","<<aux_pivot[1][i].localOffset()<<std::endl;
                    }
               }

               if( A.mpiRank()==0 ){
                   for (int i=0; i< A.tileMb(0); i++){
                       for (int j=0; j< A.tileNb(0); j++){
                           std::cout<<local_tiles[0](i, j)<<", ";
                       }
                      std::cout<<std::endl;
                   }

                   std::cout<<std::endl;

                  for (int i=0; i< A.tileMb(0); i++){
                      for (int j=0; j< A.tileNb(0); j++){
                          std::cout<<local_tiles[1](i, j)<<", ";
                      }
                     std::cout<<std::endl;
                  }
                 std::cout<<std::endl;
              }
            #endif

            // Factor the panel locally in parallel.
            getrf_ca(A, local_tiles, diag_len, ib, 1,
                     A.tileNb(0), tile_indices, aux_pivot,
                     A.mpiRank(), max_panel_threads, priority);

            std::vector< Tile<scalar_t> > ptiles;
            ptiles.push_back(Awork(i_current, 0));
            ptiles.push_back(Awork(i_dst, 0));

            //int global, offset;
            for(int j=0; j < diag_len ; ++j){
                if (aux_pivot[0][j].localTileIndex() > 0 ||
                    aux_pivot[0][j].localOffset() > j){

                    swapLocalRow(
                        0, A.tileMb(0),
                        ptiles[0], j,
                        ptiles[aux_pivot[0][j].localTileIndex()],
                        aux_pivot[0][j].localOffset());
                }

                  /*      global = aux_pivot[0][j].tileIndex();
                        offset = aux_pivot[0][j].elementOffset();
                        aux_pivot[0][j].set_tileIndex(aux_pivot[aux_pivot[0][j].localTileIndex()][aux_pivot[0][j].localOffset()].tileIndex());
                        aux_pivot[0][j].set_elementOffset(aux_pivot[aux_pivot[0][j].localTileIndex()][aux_pivot[0][j].localOffset()].elementOffset());


                        aux_pivot[aux_pivot[0][j].localTileIndex()][aux_pivot[0][j].localOffset()].set_tileIndex(global);
                        aux_pivot[aux_pivot[0][j].localTileIndex()][aux_pivot[0][j].localOffset()].set_elementOffset(offset);
                    */
            }

            #if 0
               if( (A.mpiRank()==0) && level==nlevels-1 ){
                   for (int64_t i = 0; i < diag_len; ++i) {
                       std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[0][i].tileIndex()
                           <<","<<aux_pivot[0][i].elementOffset()<<", "
                           <<aux_pivot[0][i].localTileIndex()
                           <<","<<aux_pivot[0][i].localOffset()<<std::endl;
                   }

               /*
                    for (int64_t i = 0; i < diag_len; ++i) {
                        std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[1][i].tileIndex()
                            <<","<<aux_pivot[1][i].elementOffset()<<", "
                            <<aux_pivot[1][i].localTileIndex()
                            <<","<<aux_pivot[1][i].localOffset()<<std::endl;
                    }
                */
               }

               if( A.mpiRank()==0 && level==nlevels-1){
                   for (int i=0; i< A.tileMb(0); i++){
                       for (int j=0; j< A.tileNb(0); j++){
                           std::cout<<ptiles[0](i, j)<<", ";
                        }
                    std::cout<<std::endl;
                   }
               std::cout<<std::endl;

                   for (int i=0; i< A.tileMb(0); i++){
                       for (int j=0; j< A.tileNb(0); j++){
                           std::cout<<ptiles[1](i, j)<<", ";
                       }
                    std::cout<<std::endl;
                   }
               std::cout<<std::endl;
                }
            #endif
               if(level==nlevels-1){

                   //Copy the last factorization back to panel tile
                 local_tiles[0].copyData(&ptiles[0]);

               }
               Awork.tileTick(i_dst, 0);

              }
            }
            else{
                dst = rank_rows[ index - step ].first;
                i_src = rank_rows[ index ].second;
                //std::cout<<"Send ("<<rank_rows[index].first<<", "<<dst<<")"<< " From: " <<i_src<<std::endl;

                Awork.tileSend(i_src, 0, dst);

                MPI_Send(aux_pivot.at(0).data(),
                             sizeof(AuxPivot<scalar_t>)*aux_pivot.at(0).size(),
                             MPI_BYTE, dst, 0, A.mpiComm());
                break;
                }
          step *= 2;
        }// for loop over levels


      // Copy pivot information from aux_pivot to pivot.
      for (int64_t i = 0; i < diag_len; ++i) {
          pivot[i] = Pivot(aux_pivot[0][i].tileIndex(),
                     aux_pivot[0][i].elementOffset());
          if(A.mpiRank()==0 )
             std::cout<<aux_pivot[0][i].tileIndex()<<", "<< aux_pivot[0][i].elementOffset()<<std::endl;
      }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf_ca<Target::HostTask, float>(
    Matrix<float>&& A, Matrix<float>&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_ca<Target::HostTask, double>(
    Matrix<double>&& A, Matrix<double>&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_ca< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A, Matrix< std::complex<float> >&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_ca< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A, Matrix< std::complex<double> >&& Awork,
    int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

} // namespace internal
} // namespace slate
