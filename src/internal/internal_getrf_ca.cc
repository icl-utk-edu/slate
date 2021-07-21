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

template <typename scalar_t>
void getrf_ca(
    Matrix<scalar_t>& A,
    std::vector< Tile<scalar_t> >& tiles,    
    int64_t diag_len, int64_t ib, int nb,
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
        getrf_tntpiv(diag_len, ib,
        tiles, tile_indices,
        aux_pivot,
        mpi_rank, //mpi_root, MPI_COMM_SELF,
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
    /*      for(int j=0; j < diag_len ; ++j){
           //std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[j].tileIndex()<<","<<aux_pivot[j].elementOffset()<<", "<<aux_pivot[j].localTileIndex()<<std::endl;
           //std::cout<<tiles.size()<<std::endl;
           swapLocalRow(
               0, nb,
               //A(i, 0), j,
               //A(aux_pivot[j].tileIndex(), 0),
               tiles[0], j, 
               tiles[aux_pivot[j].localTileIndex()],
               aux_pivot[j].elementOffset());
          }*/
         //std::cout<<"done"<<std::endl;
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
    std::vector< Tile<scalar_t> > tiles, tiles_copy_poriginal;
    std::vector<int64_t> tile_indices;

    // Build the broadcast set.
    // Build lists of local tiles, indices, and offsets.
    int64_t tile_offset = 0;
    std::set<int> bcast_set;
    for (int64_t i = 0; i < A.mt(); ++i) {
        bcast_set.insert(A.tileRank(i, 0));
        if (A.tileIsLocal(i, 0)) {
            tiles.push_back(A_work_panel(i, 0));
            //tiles_copy_poriginal.push_back(A_work_panel(i, 0));
            //tiles_copy.push_back(A_work_panel(i, 0));
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
                    if( A.mpiRank() == 0){
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
            //    std::vector< Tile<scalar_t> > tiles_copy(tiles_copy_poriginal.size());
                /*for(int i=0; i<tiles_copy_poriginal.size();i++){
                    tiles_copy_poriginal[i].copyData(&tiles_copy[i]);
                }*/
              //  std::copy(tiles_copy_poriginal.begin(), tiles_copy_poriginal.end(), back_inserter(tiles_copy));
                // Factor the panel locally in parallel.
                getrf_ca(A, tiles, diag_len, ib, A.tileNb(0),
                      tile_indices, aux_pivot,
                      //bcast_rank, bcast_root, bcast_comm,
                      self_rank, self_rank, MPI_COMM_SELF,
                      max_panel_threads, priority);


                internal::copy<Target::HostTask>( std::move(A), std::move(A_work_panel) );


                for(int j=0; j < diag_len ; ++j){
                    swapLocalRow(
                        0, A.tileNb(0),
                        tiles[0], j,
                        tiles[aux_pivot[j].localTileIndex()],
                        aux_pivot[j].elementOffset());
               }


            #if 0
            // Print pivot information from aux_pivot.
            for (int64_t i = 0; i < diag_len; ++i) {
                
                std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[i].tileIndex()<<","<<aux_pivot[i].elementOffset()<<", "<<aux_pivot[i].localTileIndex()<<std::endl;
            }
            #endif

            #if 1
            for (int i=0; i<A.mt(); i++){
                if (A.tileIsLocal(i, 0)){
                   if( A.mpiRank() == 0 ){
                     std::cout<<"\n After Tile: "<<i<<" of rank: "<<A.mpiRank()<<std::endl;
                     for(int m=0; m<A.tileMb(i);m++){
                         for(int n=0; n<A.tileMb(i);n++){                                          
                            std::cout<<A_work_panel(i,0)(m,n)<<",";                                             
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
         //Tile<scalar_t> tile( tileMb(i), tileNb(j), &data[0], tileMb(i), host_num_, TileKind::Workspace );
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
                      A_work_panel.tileRecv(i_dst, 0, src, layout);

                       //tiles[i_current].copyData( &local_tiles[0]);
                       //tiles[ i_dst ].copyData( &local_tiles[1]);
                       A_work_panel(i_current, 0).copyData( &local_tiles[0]);
                       A_work_panel(i_dst, 0).copyData( &local_tiles[1]);

                      //std::vector< Tile<scalar_t> > local_tiles;
                      //local_tiles.push_back( A_work_panel( i_current, 0) );
                      //local_tiles.push_back( A_work_panel( i_dst, 0 ) );               
                      //std::cout<<"local tiles size"<<local_tiles.size()<<std::endl;
                      // Factor the panel locally in parallel.
                      getrf_ca(A, local_tiles, diag_len, ib, A.tileNb(0),
                            //local_tiles, 
                            tile_indices, aux_pivot,
                            self_rank, self_rank, MPI_COMM_SELF,
                            max_panel_threads, priority);
                      /*for (int64_t i = 0; i < diag_len; ++i) {
                      std::cout<<"\n"<<A.mpiRank()<<","<<aux_pivot[i].tileIndex()<<","<<aux_pivot[i].elementOffset()<<", "<<aux_pivot[i].localTileIndex()<<std::endl;
                      }*/
/*                for(int j=0; j < diag_len ; ++j){
                    swapLocalRow(
                        0, A.tileNb(0),
                        tiles[0], j,
                        tiles[aux_pivot[j].localTileIndex()],
                        aux_pivot[j].elementOffset());
               }
*/

                       
                      A_work_panel.tileTick(i_dst, 0);
                   }
                }else{
                     dst = rank_rows[ index - step ].first;
                     i_src = rank_rows[ index ].second;
                     //std::cout<<"Send ("<<rank_rows[index].first<<", "<<dst<<")"<< " From: " <<i_src<<std::endl;
                     A_work_panel.tileSend(i_src, 0, dst);
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
