// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_getrf.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh" 

namespace slate {
namespace internal {

template <typename scalar_t>
void getrf_ca(
    Matrix<scalar_t>& A,
    std::vector< Tile<scalar_t> >& tiles,    
    int64_t diag_len, int64_t ib, int nb,
    std::vector< Tile<scalar_t> >& tiles_copy,
    std::vector<int64_t>& tile_indices,
    std::vector< AuxPivot<scalar_t> >& aux_pivot,
    int mpi_rank, int mpi_root, MPI_Comm mpi_comm,
    int max_panel_threads, int priority)
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
        getrf(diag_len, ib,
        tiles_copy, tile_indices,
        aux_pivot,
        mpi_rank, mpi_root, MPI_COMM_SELF,
        thread_rank, thread_size,
        thread_barrier,
        max_value, max_index, max_offset, top_block);
    }
    #pragma omp taskwait

    // local swap

//    for (int i=0; i<A.mt(); i++){
//        if (A.tileIsLocal(i, 0)){
            /*for(int j=0; j < diag_len ;j++){
                swapLocalRow(
                    0, A.tileMb(j),
                    tiles[0], j,
                    //A(aux_pivot[j].tileIndex(), 0),
                    tiles[aux_pivot[j].tileIndex()],
                    aux_pivot[j].elementOffset());
            }*/
         //   break;
        //}
    //}

   #if 0
    // Print pivot information from aux_pivot.
        for (int64_t i = 0; i < diag_len; ++i) {
           std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[i].tileIndex()<<","<<aux_pivot[i].elementOffset()<<", "<<aux_pivot[i].localTileIndex()<<std::endl;
         }
   #endif
   //for (int i=0; i<A.mt(); i++){
    //   if (A.tileIsLocal(i, 0)){ 
        //if(A.mpiRank()==1){
          for(int j=0; j < diag_len ; ++j){
           std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[j].tileIndex()<<","<<aux_pivot[j].elementOffset()<<", "<<aux_pivot[j].localTileIndex()<<std::endl;
           swapLocalRow(
               0, nb,
               //A(i, 0), j,
               //A(aux_pivot[j].tileIndex(), 0),
               tiles[0], j, 
               tiles[aux_pivot[j].localTileIndex()],
               aux_pivot[j].elementOffset());
          }
        //}
      // break;
      // }
   //}
}
//------------------------------------------------------------------------------
/// LU factorization of a column of tiles.
/// Dispatches to target implementations.
/// @ingroup gesv_internal
///
template <Target target, typename scalar_t>
void getrf_ca(Matrix<scalar_t>&& A, int64_t diag_len, int64_t ib,
           std::vector<Pivot>& pivot,
           int max_panel_threads, int priority)
{
    getrf_ca(internal::TargetType<target>(),
          A, diag_len, ib, pivot, max_panel_threads, priority);
}

//------------------------------------------------------------------------------
/// LU factorization of a column of tiles, host implementation.
/// @ingroup gesv_internal
///
template <typename scalar_t>
void getrf_ca(internal::TargetType<Target::HostTask>,
           Matrix<scalar_t>& A, int64_t diag_len, int64_t ib,
           std::vector<Pivot>& pivot,
           int max_panel_threads, int priority)
{
    using ij_tuple = typename BaseMatrix<scalar_t>::ij_tuple;
 
    // Assumes column major RABAB::Check the row major in case of getrf gpu as in old imp TODO
    const Layout layout = Layout::ColMajor;

    assert(A.nt() == 1);

    slate::Matrix<scalar_t> A_work_panel = A.emptyLike();
    A_work_panel.insertLocalTiles();
    internal::copy<Target::HostTask>( std::move(A), std::move(A_work_panel) );

    // Move the panel to the host
    std::set<ij_tuple> A_tiles_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        if (A.tileIsLocal(i, 0)) {
            A_tiles_set.insert({i, 0});
        }
    }
    A.tileGetForWriting(A_tiles_set, LayoutConvert::ColMajor);

    // lists of local tiles, indices, and offsets
    std::vector< Tile<scalar_t> > tiles, tiles_copy;
    std::vector<int64_t> tile_indices;

    // Build the broadcast set.
    // Build lists of local tiles, indices, and offsets.
    int64_t tile_offset = 0;
    std::set<int> bcast_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        bcast_set.insert(A.tileRank(i, 0));
        if (A.tileIsLocal(i, 0)) {
            tiles.push_back(A(i, 0));
            tiles_copy.push_back(A_work_panel(i, 0));
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

        int self_rank; 
        MPI_Comm_rank(MPI_COMM_SELF, &self_rank);

        std::vector< AuxPivot<scalar_t> > aux_pivot(diag_len);

            #if 1
            for (int i=0; i<A_work_panel.mt(); i++){
                if (A.tileIsLocal(i, 0)){
                    if( A.mpiRank() == 1){
                    std::cout<<"\n Tile: "<<i<<" of rank: "<<A.mpiRank()<<std::endl;
                    for(int m=0; m<A_work_panel.tileMb(i);m++){
                        for(int n=0; n<A_work_panel.tileMb(i);n++){
                           std::cout<<A_work_panel(i,0)(m,n)<<",";
                        }
                        std::cout<<std::endl;
                    }
                   }
                }
           }
           #endif

                // Factor the panel locally in parallel.
                getrf_ca(A, tiles, diag_len, ib, A.tileNb(0),
                      tiles_copy, tile_indices,
                      aux_pivot,
                      //bcast_rank, bcast_root, bcast_comm,
                      self_rank, self_rank, MPI_COMM_SELF,
                      max_panel_threads, priority);
            #if 0
            // Print pivot information from aux_pivot.
            for (int64_t i = 0; i < diag_len; ++i) {
                
                std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[i].tileIndex()<<","<<aux_pivot[i].elementOffset()<<", "<<aux_pivot[i].localTileIndex()<<std::endl;
            }
            #endif
            /*internal::permuteRows<Target::HostTask>(
              Direction::Forward, std::move(A), pivot,
                        Layout::ColMajor, 1, 1);*/

            #if 1
            for (int i=0; i<A.mt(); i++){
                if (A.tileIsLocal(i, 0)){
                   if( A.mpiRank() == 1 ){
                     std::cout<<"\n After Tile: "<<i<<" of rank: "<<A.mpiRank()<<std::endl;
                     for(int m=0; m<A.tileMb(i);m++){
                         for(int n=0; n<A.tileMb(i);n++){                                          
                            std::cout<<A(i,0)(m,n)<<",";                                             
                         }
                      std::cout<<std::endl;                                                        
                      }   
                   }                                                                      
                 }                                                                                
           }
         #endif
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
                      A.tileRecv(i_dst, 0, src, layout);

                      std::vector< Tile<scalar_t> > local_tiles;
                      local_tiles.push_back( A( i_current, 0) );
                      local_tiles.push_back( A( i_dst, 0 ) );               

                      // Factor the panel locally in parallel.
                      getrf_ca(A, tiles, diag_len, ib, A.tileNb(0),
                            tiles_copy, tile_indices,
                            aux_pivot,
                            self_rank, self_rank, MPI_COMM_SELF,
                            max_panel_threads, priority);
 
                      A.tileTick(i_dst, 0);
                   }
                }else{
                     dst = rank_rows[ index - step ].first;
                     i_src = rank_rows[ index ].second;
                     //std::cout<<"Send ("<<rank_rows[index].first<<", "<<dst<<")"<< " From: " <<i_src<<std::endl;
                     A.tileSend(i_src, 0, dst);
                     break;   
                }
          step *= 2;
        }// for loop over levels

       //TODO RABAB: I need to broadcast pivot info
       
      // Copy pivot information from aux_pivot to pivot.
      for (int64_t i = 0; i < diag_len; ++i) {  
          pivot[i] = Pivot(aux_pivot[i].tileIndex(), 
                     aux_pivot[i].elementOffset());
      }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void getrf_ca<Target::HostTask, float>(
    Matrix<float>&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_ca<Target::HostTask, double>(
    Matrix<double>&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_ca< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

// ----------------------------------------
template
void getrf_ca< Target::HostTask, std::complex<double> >(
    Matrix< std::complex<double> >&& A, int64_t diag_len, int64_t ib,
    std::vector<Pivot>& pivot,
    int max_panel_threads, int priority);

} // namespace internal
} // namespace slate
