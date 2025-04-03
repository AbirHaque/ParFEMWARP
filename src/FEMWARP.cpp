/*
ParFEMWARP is a parallel mesh warping program

Copyright Notice:
    Copyright 2025 Abir Haque

License Notice:
    This file is part of ParFEWMARP

    ParFEMWARP is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License Version 3 as published by
    the Free Software Foundation.

    ParFEMWARP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License Version 3 for more details.

    You should have received a copy of the GNU Affero General Public License Version 3
    along with ParFEMWARP in the file labeled LICENSE.txt.  If not, see https://www.gnu.org/licenses/agpl-3.0.txt


Author:
    Abir Haque

Date Last Updated:
    April 3rd, 2025

Notes:
    This software was developed by Abir Haque in collaboration with Dr. Suzanne M. Shontz at the University of Kansas (KU).
    This work was supported by the following:
        HPC facilities operated by the Center for Research Computing at KU supported by NSF Grant OAC-2117449,
        REU Supplement to NSF Grant OAC-1808553,
        REU Supplement to NSF Grant CBET-2245153,
        KU School of Engineering Undergraduate Research Fellows Program
    If you wish to use this code in your own work, you must review the license at LICENSE.txt and cite the following paper:
        Abir Haque, Suzanne Shontz. Parallelization of the Finite Element-based Mesh Warping Algorithm Using Hybrid Parallel Programming. SIAM International Meshing Roundtable Workshop (SIAM IMR 25), March 2025
        Paper Link: https://internationalmeshingroundtable.com/assets/papers/2025/1018-compressed.pdf
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <unordered_map>
#include <set>
#include <vector>
#include <list>
#include <queue>
#include <numeric>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "./csr.hpp"
#include "./matrix_helper.hpp"

#define DEBUG (false)
#define MAX_TET_NODE_NEIGHBORS (120)
#define MAX_TET_NODE_NEIGHBORS_BUF (MAX_TET_NODE_NEIGHBORS+1)

using namespace std;


#define win_chunk_quantize(row_ind,chunk_size,num_chunks) row_ind/chunk_size>num_chunks-1?num_chunks-1:row_ind/chunk_size
#define inter_win_chunk_quantize(row_ind,win_ind,neighbors_chunks,neighbors_chunk_size,neighbors_chunk_size_last,num_chunks)win_ind!=neighbors_chunks-1? row_ind%neighbors_chunk_size:((row_ind-neighbors_chunk_size*(num_chunks-1))%neighbors_chunk_size_last)

void distributed_femwarp3d_RMA//works
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  MPI_Comm comm,
  int size,
  int rank,
  Eigen::MatrixXd& Z_femwarp_transformed
)
{double start,end;start = MPI_Wtime();
  /*register*/ int i,j,k,n_ele;
  int T_i_0;
  int T_i_1;
  int T_i_j;
  int T_i_k;
  int loc_num_vals_in_AI=0;
  int loc_num_vals_in_AB=0;
  int num_vals_in_AI=0;
  int num_vals_in_AB=0;
  int num_rows=0;
  int T_parts=num_eles/size;
  int low=rank*T_parts;
  int high=rank==size-1 ? num_eles : (rank+1)*T_parts;
  uint64_t str_lens[size];
  uint64_t recvcounts[size*2];
  uint64_t displs[size*2];
  uint64_t rec_str_len = 0;
  uint64_t total_received_elements = 0;
  int num_rows_arr[size];
  unsigned int T_ele[4];
  double Z_nums_0[3];
  double Z_nums_1[3];
  double Z_nums_2[3];
  double Z_nums_3[3];
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];
  unordered_map<int, set<int>> AI_dict;
  unordered_map<int, set<int>> AB_dict;
  unordered_map<int, set<int>> local_neighbors;
  unordered_map<int, set<int>> master_neighbor_dict[size];
  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  char* rec_str_buf=nullptr;
  char* all_gathered_chars=nullptr;


  int neighbors_chunks=1000;
  int neighbors_chunk_size;
  int neighbors_chunk_size_last;
  if(m<neighbors_chunks){
    neighbors_chunks=m;
    neighbors_chunk_size=1;
    neighbors_chunk_size_last=1;
  }
  else{
    neighbors_chunk_size=m/neighbors_chunks;
    neighbors_chunk_size_last=neighbors_chunk_size+m%neighbors_chunks;
  }
  int* neighbors[rank==1?neighbors_chunks:1];
  if(rank==1){
    for (i = 0; i < neighbors_chunks-1; i++){
      neighbors[i] = (int*)calloc(neighbors_chunk_size*MAX_TET_NODE_NEIGHBORS_BUF, sizeof(int));
    }
    neighbors[neighbors_chunks-1] = (int*)calloc(neighbors_chunk_size_last*MAX_TET_NODE_NEIGHBORS_BUF, sizeof(int));
  }

  int row_size=0;
  int N_val_1=0;
  int N_val_2=0;
  int N_ind=0;
  bool found=false;

  //cout<<"*"<<"\n";

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win T_win;
  MPI_Win Z_win;
  MPI_Win N_win;
  MPI_Win_create(rank!=0?NULL:T.data(), rank!=0?0:(sizeof(int) * num_eles * 4), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &T_win);
  MPI_Win_create(rank!=0?NULL:Z_original.data(), rank!=0?0:(sizeof(double) * n * 3), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &Z_win);
  MPI_Win_lock_all(0,T_win);
  MPI_Win_lock_all(0,Z_win);
  vector<MPI_Win> neighbors_wins(neighbors_chunks);
  //cout<<"//"<<"\n";
  for(i=0;i<neighbors_chunks-1;i++){
    MPI_Win_create(rank!=1?NULL:neighbors[i], rank!=1?0:(sizeof(int) * neighbors_chunk_size * MAX_TET_NODE_NEIGHBORS_BUF), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &neighbors_wins[i]);
  }
  MPI_Win_create(rank!=1?NULL:neighbors[neighbors_chunks-1], rank!=1?0:(sizeof(int) * neighbors_chunk_size_last * MAX_TET_NODE_NEIGHBORS_BUF), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &neighbors_wins[neighbors_chunks-1]);
  MPI_Barrier(MPI_COMM_WORLD);
  //cout<<rank<<"\n";
  int win_ind;
  int inter_win_offset;//Will need to quantize a row index within a window chunk
  int zero=0;
  int tmp_row_size_plus_one;
  int T_i_Xs[4];

  int T_chunk_Xs[(high-low)*4];


  int l_low;
  int l_high;
  int l_mid;
  int l;
  int elems[MAX_TET_NODE_NEIGHBORS_BUF];

  MPI_Get_accumulate(&T_chunk_Xs, (high-low)*4, MPI_INT, &T_chunk_Xs, (high-low)*4, MPI_INT, 0, low*4, (high-low)*4, MPI_INT, MPI_NO_OP, T_win);
  MPI_Win_flush_local(0,T_win);
  cout<<"MPI_Win_unlock_all(T_win);"<<rank<<"\n";
  MPI_Win_unlock_all(T_win);//MPI_Win_fence(0, T_win);

  int i_low_4x;
  for(i = low; i<high;i++){//O(n)
    //MPI_Get_accumulate(&T_i_Xs, 4, MPI_INT, &T_i_Xs, 4, MPI_INT, 0, i*4, 4, MPI_INT, MPI_NO_OP, T_win);
    //MPI_Win_flush_local(0,T_win);
    i_low_4x=(i-low)*4;
    T_i_Xs[0]=T_chunk_Xs[i_low_4x];
    T_i_Xs[1]=T_chunk_Xs[i_low_4x+1];
    T_i_Xs[2]=T_chunk_Xs[i_low_4x+2];
    T_i_Xs[3]=T_chunk_Xs[i_low_4x+3];
    T_i_0=T_i_Xs[0];
    T_i_1=T_i_Xs[1];
    if(T_i_0==T_i_1){
      break;
    }
    for(j=0;j<4;j++){
      T_i_j=T_i_Xs[j];
      for(k=0;k<4;k++){
        if(T_i_j<m){
          win_ind=win_chunk_quantize(T_i_j,neighbors_chunk_size,neighbors_chunks);
          inter_win_offset=inter_win_chunk_quantize(T_i_j,win_ind,neighbors_chunks,neighbors_chunk_size,neighbors_chunk_size_last,neighbors_chunks);

          //double lock_start = MPI_Wtime();
          MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, neighbors_wins[win_ind]);
          //double lock_end = MPI_Wtime(); cout<<rank<<" lock time: "<<lock_end-lock_start<<"\n";
          MPI_Get_accumulate(&elems, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, &elems, MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT, 1, inter_win_offset*MAX_TET_NODE_NEIGHBORS_BUF, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, MPI_NO_OP, neighbors_wins[win_ind]);
          MPI_Win_flush_local(1, neighbors_wins[win_ind]);
          T_i_k=T_i_Xs[k];
          row_size=elems[0];

          if(row_size==0){
            row_size=1;
            elems[0]=row_size;//First index of row contains size
            elems[1]=T_i_k;
            MPI_Accumulate(&elems, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, /*rank*/1, inter_win_offset*MAX_TET_NODE_NEIGHBORS_BUF, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, MPI_REPLACE, neighbors_wins[win_ind]);//MPI_Accumulate(&T_i_k, 1, MPI_INT, /*rank*/1, N_ind+inter_win_offset*MAX_TET_NODE_NEIGHBORS_BUF, 1, MPI_INT, MPI_REPLACE, neighbors_wins[win_ind]);
            //MPI_Win_flush_local(1,neighbors_wins[win_ind]);
            if(T_i_k<m){
              loc_num_vals_in_AI+=1;
            }
            else{
              loc_num_vals_in_AB+=1;
            }
          }
          else{
            found=false;
            l_low=1;
            l_high=row_size;
            while(l_low<=l_high){
              l_mid=l_low+(l_high-l_low)/2;
              N_val_1=elems[l_mid];
              if(N_val_1==T_i_k){
                found=true;
                break;
              }
              else if(N_val_1<T_i_k){
                l_low=l_mid+1;
              }
              else{
                l_high=l_mid-1;
              }
            }
            N_ind=l_low;
            if(!found){
              row_size+=1;
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
              else{
                //Insert element
                //local_neighbors[T_i_j].insert(T_i_k);
                elems[0]=row_size;
                for(l=row_size;l>=N_ind;l--){
                  elems[l]=elems[l-1];
                }
                elems[N_ind]=T_i_k;
                MPI_Accumulate(&elems, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, /*rank*/1, inter_win_offset*MAX_TET_NODE_NEIGHBORS_BUF, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, MPI_REPLACE, neighbors_wins[win_ind]);//MPI_Accumulate(&T_i_k, 1, MPI_INT, /*rank*/1, N_ind+inter_win_offset*MAX_TET_NODE_NEIGHBORS_BUF, 1, MPI_INT, MPI_REPLACE, neighbors_wins[win_ind]);
                //MPI_Win_flush_local(1,neighbors_wins[win_ind]);
                if(T_i_k<m){
                  loc_num_vals_in_AI+=1;
                }
                else{
                  loc_num_vals_in_AB+=1;
                }
              }
            }
          }
          MPI_Win_unlock(1, neighbors_wins[win_ind]);
        }
      }
    }
  }
  //cout<<"List "<<rank<<"\n";
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Lists time: "<<end-start<<"\n";}start = MPI_Wtime();
  MPI_Allreduce(&loc_num_vals_in_AI,&num_vals_in_AI,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&loc_num_vals_in_AB,&num_vals_in_AB,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);//exit(100);
  cout<<num_vals_in_AI<<" "<<num_vals_in_AB<<"\n";
  cout<<"MPI_Allreduce "<<rank<<"\n";
  CSR_Matrix AI_matrix(num_vals_in_AI,num_vals_in_AI,m+1);
  //exit(100);
  CSR_Matrix AB_matrix(num_vals_in_AB,num_vals_in_AB,/*b*/m+1);
  double tmpI[AI_matrix._num_vals];
  double tmpB[AB_matrix._num_vals];
  cout<<"CSR_Matrix "<<rank<<"\n";




  int AB_val_count=0;
  int AB_col_indices_pos=0;
  int AB_offset=m;


  int AI_val_count=0;
  int AI_col_indices_pos=0;
  int AI_offset=0;
  int tmp_neighbor_set[MAX_TET_NODE_NEIGHBORS_BUF];
  for(i = 0; i < neighbors_chunks; i++ ){ MPI_Win_lock_all(0, neighbors_wins[i]);/*MPI_Win_fence(0, neighbors_wins[i]);*/ }
  MPI_Barrier(MPI_COMM_WORLD);
  for (i = 0; i < m; i++){
    AI_matrix.setRowPtrsAt(i,AI_col_indices_pos);
    AB_matrix.setRowPtrsAt(i,AB_col_indices_pos);
    win_ind=win_chunk_quantize(i,neighbors_chunk_size,neighbors_chunks);
    inter_win_offset=inter_win_chunk_quantize(i,win_ind,neighbors_chunks,neighbors_chunk_size,neighbors_chunk_size_last,neighbors_chunks);
    MPI_Get_accumulate(&tmp_neighbor_set, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, &tmp_neighbor_set, MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT, 1, inter_win_offset*MAX_TET_NODE_NEIGHBORS_BUF, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, MPI_NO_OP, neighbors_wins[win_ind]);
    MPI_Win_flush_local(1,neighbors_wins[win_ind]);
    for(j=1;j<=tmp_neighbor_set[0];j++){
      if(tmp_neighbor_set[j]>=m){
        AB_matrix.setColIndicesAt(AB_col_indices_pos,tmp_neighbor_set[j]-AB_offset);
        AB_col_indices_pos+=1;
      }
      else{
        AI_matrix.setColIndicesAt(AI_col_indices_pos,tmp_neighbor_set[j]-AI_offset);
        AI_col_indices_pos+=1;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for(i = 0; i < neighbors_chunks; i++ ){ MPI_Win_unlock_all(neighbors_wins[i]);/*MPI_Win_fence(0, neighbors_wins[i]);*/ }
  cout<<"setRowPtrsAt "<<rank<<"\n";
  AI_matrix.setRowPtrsAt(m,AI_col_indices_pos);
  AB_matrix.setRowPtrsAt(m,AB_col_indices_pos);
  cout<<"FEM "<<rank<<"\n";



  //free(neighbors);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Matrix Allocation time: "<<end-start<<"\n";}start = MPI_Wtime();
  for (n_ele=low; n_ele<high; n_ele++){
    //cout<<rank<<":"<<low<<" "<<n_ele<<""<<high<<"\n";
    //MPI_Get(&T_ele, 4, MPI_UNSIGNED, 0, n_ele*4, 4, MPI_UNSIGNED, T_win);

    i_low_4x=(n_ele-low)*4;//MPI_Get_accumulate(&T_ele, 4, MPI_UNSIGNED, &T_ele, 4, MPI_UNSIGNED, 0, n_ele*4, 4, MPI_UNSIGNED, MPI_NO_OP, T_win);
    //MPI_Win_flush_local(0,T_win);
    T_ele[0]=(unsigned int)T_chunk_Xs[i_low_4x];
    T_ele[1]=(unsigned int)T_chunk_Xs[i_low_4x+1];
    T_ele[2]=(unsigned int)T_chunk_Xs[i_low_4x+2];
    T_ele[3]=(unsigned int)T_chunk_Xs[i_low_4x+3];

    MPI_Get_accumulate(&Z_nums_0, 3, MPI_DOUBLE, &Z_nums_0, 3, MPI_DOUBLE, 0, T_ele[0]*3, 3, MPI_DOUBLE, MPI_NO_OP, Z_win);
    MPI_Get_accumulate(&Z_nums_1, 3, MPI_DOUBLE, &Z_nums_1, 3, MPI_DOUBLE, 0, T_ele[1]*3, 3, MPI_DOUBLE, MPI_NO_OP, Z_win);
    MPI_Get_accumulate(&Z_nums_2, 3, MPI_DOUBLE, &Z_nums_2, 3, MPI_DOUBLE, 0, T_ele[2]*3, 3, MPI_DOUBLE, MPI_NO_OP, Z_win);
    MPI_Get_accumulate(&Z_nums_3, 3, MPI_DOUBLE, &Z_nums_3, 3, MPI_DOUBLE, 0, T_ele[3]*3, 3, MPI_DOUBLE, MPI_NO_OP, Z_win);

    MPI_Win_flush_local(0,Z_win);

    a_func[0]=Z_nums_0[0];
    a_func[1]=Z_nums_0[1];
    a_func[2]=Z_nums_0[2];
    b_func[0]=Z_nums_1[0];
    b_func[1]=Z_nums_1[1];
    b_func[2]=Z_nums_1[2];
    c_func[0]=Z_nums_2[0];
    c_func[1]=Z_nums_2[1];
    c_func[2]=Z_nums_2[2];
    d_func[0]=Z_nums_3[0];
    d_func[1]=Z_nums_3[1];
    d_func[2]=Z_nums_3[2];
    gen_basis_function_gradients_tetrahedron(a_func,b_func,c_func,d_func,basis_function_gradients);
    element_stiffness_matrix=gen_element_stiffness_matrix_tetrahedron(basis_function_gradients,a_func,b_func,c_func,d_func);
    for (i = 0; i<4; i++){
      for (j = 0; j<4; j++){
        if (T_ele[i]<m){
          if (T_ele[j]>=m){
            if (element_stiffness_matrix(i,j)!=0){
              AB_matrix.subractValAt(T_ele[i],T_ele[j]-m,element_stiffness_matrix(i,j));
            }
          }
          else{
            if (element_stiffness_matrix(i,j)!=0){
              AI_matrix.addValAt(T_ele[i],T_ele[j],element_stiffness_matrix(i,j));
            }
          }
        }
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"MPI_Win_unlock_all(Z_win);"<<rank<<"\n";
  MPI_Win_unlock_all(Z_win);//MPI_Win_fence(0, Z_win);
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"MPI_Allreduce "<<rank<<"\n";

  for(i = 0; i < AI_matrix._num_vals;i++){
    tmpI[i]=AI_matrix._vals[i];
  }
  for(i = 0; i < AB_matrix._num_vals;i++){
    tmpB[i]=AB_matrix._vals[i];
  }
  MPI_Allreduce(tmpI,AI_matrix._vals,AI_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(tmpB,AB_matrix._vals,AB_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" FEM time: "<<end-start<<"\n";}start = MPI_Wtime();




  MPI_Barrier(MPI_COMM_WORLD);
  CSR_Matrix local_A_I(0,0,0);
  CSR_Matrix local_A_B(0,0,0);
  cout<<"csr_to_dist "<<rank<<"\n";
  csr_to_dist_csr(AI_matrix,local_A_I,m,comm,size,rank);
  csr_to_dist_csr(AB_matrix,local_A_B,m,comm,size,rank);



  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Distribute Matrix time: "<<end-start<<"\n";}

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  cout<<"MPI_Allgather "<<rank<<"\n";
  int local_num_rows=local_A_B._num_row_ptrs-1;
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(MPI_COMM_WORLD);
  num_rows = accumulate(num_rows_arr, num_rows_arr+size, num_rows);
  A_B_hat.resize(3,num_rows);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();
  MPI_Barrier(MPI_COMM_WORLD);start = MPI_Wtime();
  cout<<"parallel_csr_x_matrix "<<rank<<"\n";
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Times Matrix time: "<<end-start<<"\n";}start = MPI_Wtime();

  cout<<"parallel_sparse_block_conjugate_gradient_v2 "<<rank<<"\n";

  parallel_sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows);

  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CG time: "<<end-start<<"\n";}
}


#define REDUNDANT 0
#define RMA 1

void distributed_femwarp3d_SHM_RMA //works
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  MPI_Comm comm,
  int size,
  int rank,
  Eigen::MatrixXd& Z_femwarp_transformed
)
{
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,0, MPI_INFO_NULL,&shmcomm);
  MPI_Group world_group;
  MPI_Group shared_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Comm_group(shmcomm, &shared_group);
  int shmrank;
  int shmsize;
  MPI_Comm_rank(shmcomm, &shmrank);
  MPI_Comm_size(shmcomm, &shmsize);
  int num_nodes=size/shmsize;


  MPI_Comm inter_comm;
  int inter_rank=-1;
  int inter_size=-1;
  MPI_Comm_split(MPI_COMM_WORLD, shmrank==0, rank, &inter_comm);
  MPI_Comm_rank(inter_comm, &inter_rank);
  MPI_Comm_size(inter_comm, &inter_size);


  double start,end;start = MPI_Wtime();

  /*register*/ int i,j,k,n_ele;
  int T_i_0;
  int T_i_1;
  int T_i_j;
  int T_i_k;
  int loc_num_vals_in_AI=0;
  int loc_num_vals_in_AB=0;
  int num_vals_in_AI=0;
  int num_vals_in_AB=0;
  int num_rows=0;
  int T_parts=num_eles/size;
  int low=rank*T_parts;
  int high=rank==size-1 ? num_eles : (rank+1)*T_parts;
  uint64_t str_lens[size];
  uint64_t recvcounts[size*2];
  uint64_t displs[size*2];
  uint64_t rec_str_len = 0;
  uint64_t total_received_elements = 0;
  int num_rows_arr[size];
  unsigned int T_ele[4];
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];
  unordered_map<int, set<int>> AI_dict;
  unordered_map<int, set<int>> AB_dict;
  unordered_map<int, set<int>> local_neighbors;
  unordered_map<int, set<int>> master_neighbor_dict[size];


  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  char* rec_str_buf=nullptr;
  char* all_gathered_chars=nullptr;


  int neighbors_chunks=size;
  int neighbors_chunk_size;
  int neighbors_chunk_size_last;
  neighbors_chunk_size=m/neighbors_chunks;
  neighbors_chunk_size_last=neighbors_chunk_size+m%neighbors_chunks;
  //int* neighbors[neighbors_chunk_size];


  MPI_Aint size_of_shmem;
  int disp_unit;

  if(rank==0){
    cout<<"Job Information:"<<"\n";
    cout<<"\tRanks: "<<size<<"\n";
    cout<<"\tNodes: "<<num_nodes<<"\n";
    cout<<"\tRanks per node: "<<shmsize<<"\n";
    cout<<"\tn: "<<n<<"\n";
    cout<<"\tm: "<<m<<"\n";
    cout<<"\tb: "<<b<<"\n";
    cout<<"\tnum_eles: "<<num_eles<<"\n";
  }




  int neighbors[shmrank==0?m:1][shmrank==0?MAX_TET_NODE_NEIGHBORS_BUF:1]={0};//neighbors[rank==size-1?neighbors_chunk_size_last:neighbors_chunk_size][MAX_TET_NODE_NEIGHBORS_BUF];
  vector<int*> shared_neigbors_arr(shmsize);


  vector<MPI_Win> shared_win(shmsize);



  if(shmrank==0){
    for(i=0;i<shmsize;i++){
      if(MPI_Win_allocate_shared(m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &shared_neigbors_arr[i], &shared_win[i])!=0){
        cout<<"Shared window failed to allocate on rank "<<rank<<"\n";
        exit(100);
      }
    }
  }
  else{
    for(i=0;i<shmsize;i++){
      MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &shared_neigbors_arr[i], &shared_win[i]);
      MPI_Win_shared_query(shared_win[i], 0, &size_of_shmem, &disp_unit, &shared_neigbors_arr[i]);
    }
  }






  vector<int*> tmp_neighbors(num_nodes);
  vector<MPI_Win> tmp_neighbors_shared_win(num_nodes);
  if(num_nodes>1){
    if(shmrank==0){
      for(i = 0; i < num_nodes; i++){
        if(MPI_Win_allocate_shared(m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &tmp_neighbors[i], &tmp_neighbors_shared_win[i])!=0){
          cout<<"Shared window failed to allocate on rank "<<rank<<"\n";
          exit(100);
        }
    }
    }
    else{
      for(i = 0; i < num_nodes; i++){
        MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &tmp_neighbors[i], &tmp_neighbors_shared_win[i]);
        MPI_Win_shared_query(tmp_neighbors_shared_win[i], 0, &size_of_shmem, &disp_unit, &tmp_neighbors[i]);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);












  int row_size=0;
  int N_val_1=0;
  int N_val_2=0;
  int N_ind=0;
  bool found=false;

  //cout<<"*"<<"\n";

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win T_win;
  MPI_Win Z_win;
  MPI_Win N_win;
  MPI_Win_create(rank!=0?NULL:T.data(), rank!=0?0:(sizeof(int) * num_eles * 4), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &T_win);
  //MPI_Win_create(rank!=0?NULL:Z_original.data(), rank!=0?0:(sizeof(double) * n * 3), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &Z_win);
  MPI_Win_lock_all(0,T_win);
  //MPI_Win_lock_all(0,Z_win);

  MPI_Barrier(MPI_COMM_WORLD);
  //cout<<rank<<"\n";
  int win_ind;
  int inter_win_offset;//Will need to quantize a row index within a window chunk
  int zero=0;
  int tmp_row_size_plus_one;
  int T_i_Xs[4];

  int T_chunk_Xs[(high-low)*4];


  int l_low;
  int l_high;
  int l_mid;
  int l;
  int elems[MAX_TET_NODE_NEIGHBORS_BUF];

  MPI_Get_accumulate(&T_chunk_Xs, (high-low)*4, MPI_INT, &T_chunk_Xs, (high-low)*4, MPI_INT, 0, low*4, (high-low)*4, MPI_INT, MPI_NO_OP, T_win);
  MPI_Win_flush_local(0,T_win);

  MPI_Win_unlock_all(T_win);

  int i_low_4x;
  int rank_to_send_val;
  int local_index;
  int num_out=0;

  for(i = low; i<high;i++){//O(n)
    i_low_4x=(i-low)*4;
    T_i_Xs[0]=T_chunk_Xs[i_low_4x];
    T_i_Xs[1]=T_chunk_Xs[i_low_4x+1];
    T_i_Xs[2]=T_chunk_Xs[i_low_4x+2];
    T_i_Xs[3]=T_chunk_Xs[i_low_4x+3];
    T_i_0=T_i_Xs[0];
    T_i_1=T_i_Xs[1];
    if(T_i_0==T_i_1){
      break;
    }
    for(j=0;j<4;j++){
      T_i_j=T_i_Xs[j];
      for(k=0;k<4;k++){
        if(T_i_j<m){
          local_index=T_i_j%neighbors_chunk_size;
          if(rank==size-1){
            local_index=T_i_j-rank*neighbors_chunk_size;
          }

          T_i_k=T_i_Xs[k];
          row_size=shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+0];//neighbors[T_i_j/*local_index*/][0];

          if(row_size==0){
            row_size=1;
            shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;//neighbors[T_i_j/*local_index*/][0]=row_size;//First index of row contains size
            shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+1]=T_i_k;//neighbors[T_i_j/*local_index*/][1]=T_i_k;
            if(T_i_k<m && (REDUNDANT?rank==0:(num_nodes==1?rank==0:false))){
              loc_num_vals_in_AI+=1;
            }
            else if(T_i_k>=m && (REDUNDANT?rank==0:(num_nodes==1?rank==0:false))){
              loc_num_vals_in_AB+=1;
            }
          }
          else{
            found=false;
            l_low=1;
            l_high=row_size;
            while(l_low<=l_high){
              l_mid=l_low+(l_high-l_low)/2;
              N_val_1=shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];//neighbors[T_i_j/*local_index*/][l_mid];
              if(N_val_1==T_i_k){
                found=true;
                break;
              }
              else if(N_val_1<T_i_k){
                l_low=l_mid+1;
              }
              else{
                l_high=l_mid-1;
              }
            }
            N_ind=l_low;
            if(!found){
              row_size+=1;
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
              else{
                //Insert element
                //local_neighbors[T_i_j].insert(T_i_k);
                shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;//neighbors[T_i_j/*local_index*/][0]=row_size;
                for(l=row_size;l>=N_ind;l--){
                  shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+l-1];//neighbors[T_i_j/*local_index*/][l]=neighbors[T_i_j/*local_index*/][l-1];
                }
                //memcpy(shared_neigbors_arr[shmrank]+T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+N_ind+1,shared_neigbors_arr[shmrank]+T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+N_ind,(row_size-N_ind+1)*sizeof(int));
                shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=T_i_k;//neighbors[T_i_j/*local_index*/][N_ind]=T_i_k;
                if(T_i_k<m && (REDUNDANT?rank==0:(num_nodes==1?rank==0:false))){//was rank==0
                  loc_num_vals_in_AI+=1;
                }
                else if(T_i_k>=m && (REDUNDANT?rank==0:(num_nodes==1?rank==0:false))){//was rank==0
                  loc_num_vals_in_AB+=1;
                }
              }
            }
          }
        }
      }
    }
  }


  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Lists time: "<<end-start<<"\n";}start = MPI_Wtime();


  int shm_parts=m/shmsize;
  int shm_start=shmrank*shm_parts;
  int shm_end=shmrank==shmsize-1?m:(shmrank+1)*shm_parts;


  for(i=shm_start;i<shm_end;i++){
    for(j=0;j<shmsize;j++){
      for(k=1;k<=shared_neigbors_arr[j][i*MAX_TET_NODE_NEIGHBORS_BUF+0];k++){
        row_size=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0];
        int val_to_find=shared_neigbors_arr[j][i*MAX_TET_NODE_NEIGHBORS_BUF+k];
        found=false;
        l_low=1;
        l_high=row_size;
        while(l_low<=l_high){
          l_mid=l_low+(l_high-l_low)/2;
          N_val_1=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
          if(N_val_1==val_to_find){
            found=true;
            break;
          }
          else if(N_val_1<val_to_find){
            l_low=l_mid+1;
          }
          else{
            l_high=l_mid-1;
          }
        }
        N_ind=l_low;
        if(!found){
          row_size+=1;
          if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
          else{
            //Insert element
            //local_neighbors[T_i_j].insert(T_i_k);
            shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;

            //shift old elements forward by 1
            for(l=row_size;l>=N_ind;l--){
              shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
            }
            //emcpy(shared_neigbors_arr[0]+i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind+1,shared_neigbors_arr[0]+i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind,(row_size-N_ind+1)*sizeof(int));
            shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
            if(val_to_find<m && (REDUNDANT?rank<shmsize:(num_nodes==1?rank<shmsize:false))){
              loc_num_vals_in_AI+=1;
            }
            else if(val_to_find>=m && (REDUNDANT?rank<shmsize:(num_nodes==1?rank<shmsize:false))){
              loc_num_vals_in_AB+=1;
            }
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Combine lists 1 time: "<<end-start<<"\n";}start = MPI_Wtime();


  //390
  //1194
  ///////////////////Combine across multiple nodes///////////////
  if(num_nodes>1){
    if(!REDUNDANT){
      if(!RMA){

        int m_parts=m/size;
        int m_start=rank*m_parts;
        int m_end=rank==size-1?m:(rank+1)*m_parts;

        int start_end_nodes[num_nodes+1];
        int disps[num_nodes];
        int counts_send[num_nodes];
        for(i=0;i<num_nodes;i++){
          start_end_nodes[i]=(i*shmsize)*m_parts;
          disps[i]=start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF;
          if(i>=1){
            counts_send[i-1]=(start_end_nodes[i]-start_end_nodes[i-1])*MAX_TET_NODE_NEIGHBORS_BUF;
          }
        }
        counts_send[num_nodes-1]=(start_end_nodes[num_nodes]-start_end_nodes[num_nodes-1])*MAX_TET_NODE_NEIGHBORS_BUF;
        start_end_nodes[num_nodes]=m;

        if(shmrank==0){
          int tmp_tmp_neighbors[num_nodes][m][MAX_TET_NODE_NEIGHBORS_BUF]={0};
          MPI_Allgather(shared_neigbors_arr[0],m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,tmp_tmp_neighbors,m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,inter_comm);
          MPI_Barrier(inter_comm);if(rank==0){cout<<rank<<" MPI_Allgather 1 time: "<<MPI_Wtime()-start<<"\n";}
          for(i=0;i<num_nodes;i++){
            memcpy(tmp_neighbors[i]+m_start*MAX_TET_NODE_NEIGHBORS_BUF /*technically start for node*/,tmp_tmp_neighbors[i][m_start/*technically start for node*/],(start_end_nodes[rank/shmsize+1]-m_start)*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
          }
        }




        MPI_Barrier(MPI_COMM_WORLD);if(rank==0){cout<<rank<<" memcpy 1 time: "<<MPI_Wtime()-start<<"\n";}




        for(i=m_start;i<m_end;i++){
          row_size=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0];
          for(j=0;j<num_nodes;j++){
            for(k=1;k<=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+0]/*[j][i][0]*/;k++){
              int val_to_find=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+k];//[j][i][k];
              found=false;
              l_low=1;
              l_high=row_size;
              while(l_low<=l_high){
                l_mid=l_low+(l_high-l_low)/2;
                N_val_1=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
                if(N_val_1==val_to_find){
                  found=true;
                  break;
                }
                else if(N_val_1<val_to_find){
                  l_low=l_mid+1;
                }
                else{
                  l_high=l_mid-1;
                }
              }
              N_ind=l_low;
              if(!found){
                row_size+=1;
                if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
                else{
                  //Insert element
                  //local_neighbors[T_i_j].insert(T_i_k);
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;

                  for(l=row_size;l>=N_ind;l--){
                    shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
                  }
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
                }
              }
            }
          }
          int mid_div = std::lower_bound(&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+1],&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+row_size+1],m)-&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+1];
          loc_num_vals_in_AI+=mid_div;
          loc_num_vals_in_AB+=row_size-mid_div;

        }

        MPI_Barrier(MPI_COMM_WORLD);if(rank==0){cout<<rank<<" Merge time: "<<MPI_Wtime()-start<<"\n";}

        if(shmrank==0){
          int tmp_tmp_neighbors[num_nodes][m][MAX_TET_NODE_NEIGHBORS_BUF]={0};
          MPI_Allgather(shared_neigbors_arr[0],m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,tmp_tmp_neighbors,m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,inter_comm);
          MPI_Barrier(inter_comm);if(rank==0){cout<<rank<<" MPI_Allgather 2 time: "<<MPI_Wtime()-start<<"\n";}
          for(i=0;i<num_nodes;i++){
            memcpy(shared_neigbors_arr[0]+start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF,tmp_tmp_neighbors[i][start_end_nodes[i]],(start_end_nodes[i+1]-start_end_nodes[i])*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" NRCombine lists 2 time: "<<end-start<<"\n";}start = MPI_Wtime();
      }
      else{
        int m_parts=m/size;
        int m_start=rank*m_parts;
        int m_end=rank==size-1?m:(rank+1)*m_parts;

        int start_end_nodes[num_nodes+1];
        int disps[num_nodes];
        int counts_send[num_nodes];
        for(i=0;i<num_nodes;i++){
          start_end_nodes[i]=(i*shmsize)*m_parts;
          disps[i]=start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF;
          if(i>=1){
            counts_send[i-1]=(start_end_nodes[i]-start_end_nodes[i-1])*MAX_TET_NODE_NEIGHBORS_BUF;
          }
        }
        start_end_nodes[num_nodes]=m;
        counts_send[num_nodes-1]=(start_end_nodes[num_nodes]-start_end_nodes[num_nodes-1])*MAX_TET_NODE_NEIGHBORS_BUF;


        int neighors_T_size=shmrank==0?counts_send[inter_rank]*num_nodes:1;


        int neighors_T[neighors_T_size]={0};
        MPI_Win neighors_T_win;
        MPI_Win local_master_neighors_win;

        if(shmrank==0){
          MPI_Win_create(neighors_T, neighors_T_size*sizeof(int), sizeof(int), MPI_INFO_NULL, inter_comm, &neighors_T_win);
          MPI_Win_create(shared_neigbors_arr[0], m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int), sizeof(int), MPI_INFO_NULL, inter_comm, &local_master_neighors_win);
        }MPI_Barrier(MPI_COMM_WORLD);

        if(shmrank==0){
          MPI_Win_lock_all(0, neighors_T_win);//MPI_Win_fence(0, neighors_T_win);
          //MPI_Allgather(shared_neigbors_arr[0],m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,tmp_tmp_neighbors,m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,inter_comm);}
          //for(i=0;i<num_nodes;i++){
          //  memcpy(tmp_neighbors[i]+m_start*MAX_TET_NODE_NEIGHBORS_BUF /*technically start for node*/,tmp_tmp_neighbors[i][m_start/*technically start for node*/],(start_end_nodes[rank/shmsize+1]-m_start)*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
          //}
          for(i=num_nodes-1;i>=0;i--){
            MPI_Put(shared_neigbors_arr[0]+start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF, counts_send[i], MPI_INT, i, inter_rank*counts_send[i],counts_send[i], MPI_INT, neighors_T_win);
          }
          MPI_Win_flush_local_all(neighors_T_win);
          MPI_Win_flush_all(neighors_T_win);

          MPI_Win_unlock_all(neighors_T_win);//MPI_Win_fence(0, neighors_T_win);

          MPI_Barrier(inter_comm);

          for(i=0;i<num_nodes;i++){
            //debug cout<<"!! "<<(start_end_nodes[rank/shmsize+1]-m_start)*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int)<<" "<<counts_send[i]*sizeof(int)<<"\n";
            memcpy(tmp_neighbors[i]+m_start*MAX_TET_NODE_NEIGHBORS_BUF,neighors_T+i*counts_send[inter_rank],(start_end_nodes[rank/shmsize+1]-m_start)*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
            //debug cout<<rank<<" "<<m_start<<" "<<counts_send[inter_rank]<<" "<<counts_send[i]<<"\n";
          }
        }




        MPI_Barrier(MPI_COMM_WORLD);if(rank==0){cout<<rank<<" DEBUG RMA memcpy 1 time: "<<MPI_Wtime()-start<<"\n";}



        for(i=m_start;i<m_end;i++){
          row_size=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0];
          for(j=0;j<num_nodes;j++){
            for(k=1;k<=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+0]/*[j][i][0]*/;k++){
              int val_to_find=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+k];//[j][i][k];
              found=false;
              l_low=1;
              l_high=row_size;
              while(l_low<=l_high){
                l_mid=l_low+(l_high-l_low)/2;
                N_val_1=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
                if(N_val_1==val_to_find){
                  found=true;
                  break;
                }
                else if(N_val_1<val_to_find){
                  l_low=l_mid+1;
                }
                else{
                  l_high=l_mid-1;
                }
              }
              N_ind=l_low;
              if(!found){
                row_size+=1;
                if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
                else{
                  //Insert element
                  //local_neighbors[T_i_j].insert(T_i_k);
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;

                  for(l=row_size;l>=N_ind;l--){
                    shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
                  }
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
                }
              }
            }
          }
          int mid_div = std::lower_bound(&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+1],&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+row_size+1],m)-&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+1];
          loc_num_vals_in_AI+=mid_div;
          loc_num_vals_in_AB+=row_size-mid_div;

        }

        MPI_Barrier(MPI_COMM_WORLD);if(rank==0){cout<<rank<<" DEBUG Merge time: "<<MPI_Wtime()-start<<"\n";}

        if(shmrank==0){
          MPI_Win_lock_all(0, local_master_neighors_win);
          //int tmp_tmp_neighbors[num_nodes][m][MAX_TET_NODE_NEIGHBORS_BUF]={0};
          //MPI_Allgather(shared_neigbors_arr[0],m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,tmp_tmp_neighbors,m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,inter_comm);
          for(i=num_nodes-1;i>=0;i--){
            //Need to just put node's portion of shared_neigbors_arr[0] into first part of neighors_T_win or maybe another rma window
            MPI_Put(shared_neigbors_arr[0]+start_end_nodes[inter_rank]*MAX_TET_NODE_NEIGHBORS_BUF, counts_send[inter_rank], MPI_INT, i, start_end_nodes[inter_rank]*MAX_TET_NODE_NEIGHBORS_BUF,counts_send[inter_rank], MPI_INT, local_master_neighors_win);
          }
          MPI_Win_flush_local_all(local_master_neighors_win);
          MPI_Win_flush_all(local_master_neighors_win);

          MPI_Win_unlock_all(local_master_neighors_win);

          MPI_Barrier(inter_comm);if(rank==0){cout<<rank<<" DEBUG MPI_Allgather 2 time: "<<MPI_Wtime()-start<<"\n";}
          for(i=0;i<num_nodes;i++){
            //debug cout<<start_end_nodes[i]<<" "<<start_end_nodes[i+1]<<"\n";
            //memcpy(shared_neigbors_arr[0]+start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF,tmp_tmp_neighbors[i][start_end_nodes[i]],(start_end_nodes[i+1]-start_end_nodes[i])*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
            //memcpy(tmp_neighbors[i]+m_start*MAX_TET_NODE_NEIGHBORS_BUF,neighors_T+i*counts_send[inter_rank],(start_end_nodes[rank/shmsize+1]-m_start)*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
            //Just copy from first part of neighors_T_win or maybe another rma window into appropriate part of shared_neigbors_arr[0]
            //memcpy(shared_neigbors_arr[0]+start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF,/*tmp_tmp_neighbors[i][start_end_nodes[i]]*/neighors_T+i*counts_send[inter_rank]+start_end_nodes[i],(start_end_nodes[i+1]-start_end_nodes[i])*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));

          }
        }
        MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" NRCombine lists 2 time: "<<end-start<<"\n";}start = MPI_Wtime();
      }
    }

    else{
      //bad, redundant across nodes

      //want to do something with shared memory
        /*So have neighbors(rank0)and tmp_neighbors(rank0) be shared across all cores in a node.

        each core is responsible for WRITING to subset of rows in neighbors, but responsible for READING ALL rows in tmp_neighbors.


      */



      if(shmrank==0){
        int tmp_tmp_neighbors[num_nodes][m][MAX_TET_NODE_NEIGHBORS_BUF]={0};
        MPI_Allgather(shared_neigbors_arr[0],m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,tmp_tmp_neighbors,m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,inter_comm);
        MPI_Barrier(inter_comm);
        for(i=0;i<num_nodes;i++){
          memcpy(tmp_neighbors[i],tmp_tmp_neighbors[i],m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      //only want first node assembling
      if(1){//rank<shmsize){//rank==0){

        int shm_parts=m/shmsize;
        int shm_start=shmrank*shm_parts;
        int shm_end=shmrank==shmsize-1?m:(shmrank+1)*shm_parts;
        for(i=shm_start;i<shm_end;i++){
        //for(i=0;i<m;i++){
          for(j=0;j<num_nodes;j++){
            for(k=1;k<=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+0]/*[j][i][0]*/;k++){
              row_size=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0];
              int val_to_find=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+k];//[j][i][k];
              found=false;
              l_low=1;
              l_high=row_size;
              while(l_low<=l_high){
                l_mid=l_low+(l_high-l_low)/2;
                N_val_1=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
                if(N_val_1==val_to_find){
                  found=true;
                  break;
                }
                else if(N_val_1<val_to_find){
                  l_low=l_mid+1;
                }
                else{
                  l_high=l_mid-1;
                }
              }
              N_ind=l_low;
              if(!found){
                row_size+=1;
                if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
                else{
                  //Insert element
                  //local_neighbors[T_i_j].insert(T_i_k);
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;

                  for(l=row_size;l>=N_ind;l--){
                    shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
                  }
                  //memcpy(neighbors[i]+N_ind+1,neighbors[i]+N_ind,(row_size-N_ind+1)*sizeof(int));
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
                  if(val_to_find<m && rank<shmsize){
                    loc_num_vals_in_AI+=1;
                  }
                  else if(val_to_find>=m && rank<shmsize){
                    loc_num_vals_in_AB+=1;
                  }
                }
              }
            }
          }
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" RCombine lists 2 time: "<<end-start<<"\n";}start = MPI_Wtime();
    }
  }

  ///////////////////Combine across multiple nodes///////////////



  if(shmrank==0){
    memcpy(neighbors,shared_neigbors_arr[0], m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
  }





  MPI_Allreduce(&loc_num_vals_in_AI,&num_vals_in_AI,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&loc_num_vals_in_AB,&num_vals_in_AB,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0) cout<<"num_vals_in_AI: "<<num_vals_in_AI<<"\n";
  if(rank==0) cout<<"num_vals_in_AB: "<<num_vals_in_AB<<"\n";

  //cout<<"MPI_Allreduce "<<rank<<"\n";
  CSR_Matrix AI_matrix(num_vals_in_AI,num_vals_in_AI,m+1);
  CSR_Matrix AB_matrix(num_vals_in_AB,num_vals_in_AB,m+1);

  //cout<<"CSR_Matrix "<<rank<<"\n";




  int AB_val_count=0;
  int AB_col_indices_pos=0;
  int AB_offset=m;


  int AI_val_count=0;
  int AI_col_indices_pos=0;
  int AI_offset=0;
  if(shmrank==0){
    for(i=0;i<m;i++){
      AI_matrix.setRowPtrsAt(i,AI_col_indices_pos);
      AB_matrix.setRowPtrsAt(i,AB_col_indices_pos);
      for(j=1;j<=neighbors[i][0];j++){
        if(neighbors[i][j]>=m){
          AB_matrix.setColIndicesAt(AB_col_indices_pos,neighbors[i][j]-AB_offset);
          AB_col_indices_pos+=1;
        }
        else{
          AI_matrix.setColIndicesAt(AI_col_indices_pos,neighbors[i][j]-AI_offset);
          AI_col_indices_pos+=1;
        }
      }
    }
    //cout<<"setRowPtrsAt "<<rank<<"\n";
    AI_matrix.setRowPtrsAt(m,AI_col_indices_pos);
    AB_matrix.setRowPtrsAt(m,AB_col_indices_pos);
  }
  //MPI_Barrier(MPI_COMM_WORLD);
  //if(shmrank==0){cout<<rank<<" [Part 0] CSR Matrix Allocation time: "<<MPI_Wtime()-start<<"\n";}





  double tmpI[AI_matrix._num_vals];
  double tmpB[AB_matrix._num_vals];
  int* shared_tmpI_c;
  MPI_Win shared_tmpI_c_win;
  int* shared_tmpB_c;
  MPI_Win shared_tmpB_c_win;



  if(shmrank==0){
    if(MPI_Win_allocate_shared(AI_matrix._num_col_indices*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &shared_tmpI_c, &shared_tmpI_c_win)!=0){
      cout<<"Shared window failed to allocate on rank "<<rank<<"\n";
      exit(100);
    }
    if(MPI_Win_allocate_shared(AB_matrix._num_col_indices*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &shared_tmpB_c, &shared_tmpB_c_win)!=0){
      cout<<"Shared window failed to allocate on rank "<<rank<<"\n";
      exit(100);
    }
  }
  else{
    MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &shared_tmpI_c, &shared_tmpI_c_win);
    MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &shared_tmpB_c, &shared_tmpB_c_win);
    MPI_Win_shared_query(shared_tmpI_c_win, 0, &size_of_shmem, &disp_unit, &shared_tmpI_c);
    MPI_Win_shared_query(shared_tmpB_c_win, 0, &size_of_shmem, &disp_unit, &shared_tmpB_c);
  }














  int tmpI_r[AI_matrix._num_row_ptrs];
  int tmpB_r[AB_matrix._num_row_ptrs];

  if(shmrank==0){
    memcpy(tmpI_r,AI_matrix._row_ptrs,AI_matrix._num_row_ptrs*sizeof(int));
    memcpy(tmpB_r,AB_matrix._row_ptrs,AB_matrix._num_row_ptrs*sizeof(int));
  }
  MPI_Barrier(shmcomm);
  //if(shmrank==0){cout<<rank<<" [Part 1] CSR Matrix Allocation time: "<<MPI_Wtime()-start<<"\n";}
  MPI_Bcast(tmpI_r, AI_matrix._num_row_ptrs, MPI_INT, 0, shmcomm);
  MPI_Bcast(tmpB_r, AB_matrix._num_row_ptrs, MPI_INT, 0, shmcomm);
  MPI_Barrier(shmcomm);
  //if(shmrank==0){cout<<rank<<" [Part 2] CSR Matrix Allocation time: "<<MPI_Wtime()-start<<"\n";}
  if(shmrank!=0){
    memcpy(AI_matrix._row_ptrs,tmpI_r,AI_matrix._num_row_ptrs*sizeof(int));
    memcpy(AB_matrix._row_ptrs,tmpB_r,AB_matrix._num_row_ptrs*sizeof(int));
  }


  if(shmrank==0){
    memcpy(shared_tmpI_c,AI_matrix._col_indices,AI_matrix._num_col_indices*sizeof(int));
    memcpy(shared_tmpB_c,AB_matrix._col_indices,AB_matrix._num_col_indices*sizeof(int));
  }
  MPI_Barrier(shmcomm);
  //if(shmrank==0){cout<<rank<<" [Part 3+4] CSR Matrix Allocation time: "<<MPI_Wtime()-start<<"\n";}
  if(shmrank!=0){
    memcpy(AI_matrix._col_indices,shared_tmpI_c,AI_matrix._num_col_indices*sizeof(int));
    memcpy(AB_matrix._col_indices,shared_tmpB_c,AB_matrix._num_col_indices*sizeof(int));
  }








  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Matrix Allocation time: "<<end-start<<"\n";}start = MPI_Wtime();
  if(num_nodes>1){for(i=0;i<num_nodes;i++){MPI_Win_free(&tmp_neighbors_shared_win[i]);}}
  for(i=0;i<shmsize;i++){MPI_Win_free(&shared_win[i]);}
  MPI_Win_free(&shared_tmpI_c_win);
  MPI_Win_free(&shared_tmpB_c_win);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" MPI_Win_free time: "<<end-start<<"\n";}start = MPI_Wtime();


  for (n_ele=low; n_ele<high; n_ele++){

    i_low_4x=(n_ele-low)*4;
    T_ele[0]=(unsigned int)T_chunk_Xs[i_low_4x];
    T_ele[1]=(unsigned int)T_chunk_Xs[i_low_4x+1];
    T_ele[2]=(unsigned int)T_chunk_Xs[i_low_4x+2];
    T_ele[3]=(unsigned int)T_chunk_Xs[i_low_4x+3];



    a_func[0]=Z_original(T_ele[0],0);
    a_func[1]=Z_original(T_ele[0],1);
    a_func[2]=Z_original(T_ele[0],2);
    b_func[0]=Z_original(T_ele[1],0);
    b_func[1]=Z_original(T_ele[1],1);
    b_func[2]=Z_original(T_ele[1],2);
    c_func[0]=Z_original(T_ele[2],0);
    c_func[1]=Z_original(T_ele[2],1);
    c_func[2]=Z_original(T_ele[2],2);
    d_func[0]=Z_original(T_ele[3],0);
    d_func[1]=Z_original(T_ele[3],1);
    d_func[2]=Z_original(T_ele[3],2);


    gen_basis_function_gradients_tetrahedron(a_func,b_func,c_func,d_func,basis_function_gradients);
    element_stiffness_matrix=gen_element_stiffness_matrix_tetrahedron(basis_function_gradients,a_func,b_func,c_func,d_func);
    for (i = 0; i<4; i++){
      for (j = 0; j<4; j++){
        if (T_ele[i]<m){
          if (T_ele[j]>=m){
            if (element_stiffness_matrix(i,j)!=0){
              AB_matrix.subractValAt(T_ele[i],T_ele[j]-m,element_stiffness_matrix(i,j));
            }
          }
          else{
            if (element_stiffness_matrix(i,j)!=0){
              AI_matrix.addValAt(T_ele[i],T_ele[j],element_stiffness_matrix(i,j));
            }
          }
        }
      }
    }
  }


  MPI_Barrier(MPI_COMM_WORLD);


  memcpy(tmpI,AI_matrix._vals,AI_matrix._num_vals*sizeof(double));
  memcpy(tmpB,AB_matrix._vals,AB_matrix._num_vals*sizeof(double));


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(tmpI,AI_matrix._vals,AI_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(tmpB,AB_matrix._vals,AB_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" FEM time: "<<end-start<<"\n";}start = MPI_Wtime();



  MPI_Barrier(MPI_COMM_WORLD);
  CSR_Matrix local_A_I(0,0,0);
  CSR_Matrix local_A_B(0,0,0);
  //cout<<"csr_to_dist "<<rank<<"\n";
  csr_to_dist_csr(AI_matrix,local_A_I,m,comm,size,rank);
  csr_to_dist_csr(AB_matrix,local_A_B,m,comm,size,rank);


  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Distribute Matrix time: "<<end-start<<"\n";}

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  //cout<<"MPI_Allgather "<<rank<<"\n";
  int local_num_rows=local_A_B._num_row_ptrs-1;
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(MPI_COMM_WORLD);
  num_rows = accumulate(num_rows_arr, num_rows_arr+size, num_rows);
  A_B_hat.resize(3,num_rows);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();
  MPI_Barrier(MPI_COMM_WORLD);start = MPI_Wtime();
  //cout<<"parallel_csr_x_matrix "<<rank<<"\n";
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  //parallel_csr_x_matrix_eigen(local_A_B_matrix_eigen,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Times Matrix time: "<<end-start<<"\n";}start = MPI_Wtime();

  //cout<<"parallel_sparse_block_conjugate_gradient_v3 "<<rank<<"\n";




  int block_size=1;
  block_size=local_num_rows<block_size?local_num_rows:block_size;
  //parallel_sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows); //40
  parallel_sparse_block_conjugate_gradient_v4(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows,block_size); //19


  //parallel_preconditioned_sparse_block_conjugate_gradient_v2_icf(local_A_I,A_B_hat,Z_femwarp_transformed,L_inv,L_inv_transpose,comm,rank,size,num_rows);

  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CG time: "<<end-start<<"\n";}
}



#define REDUNDANT 0
#define RMA 1

void distributed_multistep_femwarp3d_SHM_RMA //works
(
  Eigen::MatrixXd& Z_original,
  void (*deformation_functions[])(Eigen::MatrixXd&),int num_deformations,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  MPI_Comm comm,
  int size,
  int rank,
  Eigen::MatrixXd& Z_femwarp_transformed
)
{
  MPI_Comm shmcomm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,0, MPI_INFO_NULL,&shmcomm);
  MPI_Group world_group;
  MPI_Group shared_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Comm_group(shmcomm, &shared_group);
  int shmrank;
  int shmsize;
  MPI_Comm_rank(shmcomm, &shmrank);
  MPI_Comm_size(shmcomm, &shmsize);
  int num_nodes=size/shmsize;


  MPI_Comm inter_comm;
  int inter_rank=-1;
  int inter_size=-1;
  MPI_Comm_split(MPI_COMM_WORLD, shmrank==0, rank, &inter_comm);
  MPI_Comm_rank(inter_comm, &inter_rank);
  MPI_Comm_size(inter_comm, &inter_size);


  double start,end;start = MPI_Wtime();

  int i,j,k,n_ele;
  int T_i_0;
  int T_i_1;
  int T_i_j;
  int T_i_k;
  int loc_num_vals_in_AI=0;
  int loc_num_vals_in_AB=0;
  int num_vals_in_AI=0;
  int num_vals_in_AB=0;
  int num_rows=0;
  int T_parts=num_eles/size;
  int low=rank*T_parts;
  int high=rank==size-1 ? num_eles : (rank+1)*T_parts;
  uint64_t str_lens[size];
  uint64_t recvcounts[size*2];
  uint64_t displs[size*2];
  uint64_t rec_str_len = 0;
  uint64_t total_received_elements = 0;
  int num_rows_arr[size];
  unsigned int T_ele[4];
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];
  unordered_map<int, set<int>> AI_dict;
  unordered_map<int, set<int>> AB_dict;
  unordered_map<int, set<int>> local_neighbors;
  unordered_map<int, set<int>> master_neighbor_dict[size];


  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  char* rec_str_buf=nullptr;
  char* all_gathered_chars=nullptr;


  int neighbors_chunks=size;
  int neighbors_chunk_size;
  int neighbors_chunk_size_last;
  neighbors_chunk_size=m/neighbors_chunks;
  neighbors_chunk_size_last=neighbors_chunk_size+m%neighbors_chunks;


  MPI_Aint size_of_shmem;
  int disp_unit;


  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_cur_boundary;
  Z_cur_boundary=Z_original.block(n-b,0,b,3);//Get only boundary nodes

  if(rank==0){
    cout<<"Job Information:"<<"\n";
    cout<<"\tRanks: "<<size<<"\n";
    cout<<"\tNodes: "<<num_nodes<<"\n";
    cout<<"\tRanks per node: "<<shmsize<<"\n";
    cout<<"\tn: "<<n<<"\n";
    cout<<"\tm: "<<m<<"\n";
    cout<<"\tb: "<<b<<"\n";
    cout<<"\tnum_eles: "<<num_eles<<"\n";
  }




  int neighbors[shmrank==0?m:1][shmrank==0?MAX_TET_NODE_NEIGHBORS_BUF:1]={0};
  vector<int*> shared_neigbors_arr(shmsize);


  vector<MPI_Win> shared_win(shmsize);



  if(shmrank==0){
    for(i=0;i<shmsize;i++){
      if(MPI_Win_allocate_shared(m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &shared_neigbors_arr[i], &shared_win[i])!=0){
        cout<<"Shared window failed to allocate on rank "<<rank<<"\n";
        exit(100);
      }
    }
  }
  else{
    for(i=0;i<shmsize;i++){
      MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &shared_neigbors_arr[i], &shared_win[i]);
      MPI_Win_shared_query(shared_win[i], 0, &size_of_shmem, &disp_unit, &shared_neigbors_arr[i]);
    }
  }

  vector<int*> tmp_neighbors(num_nodes);
  vector<MPI_Win> tmp_neighbors_shared_win(num_nodes);
  if(num_nodes>1){
    if(shmrank==0){
      for(i = 0; i < num_nodes; i++){
        if(MPI_Win_allocate_shared(m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &tmp_neighbors[i], &tmp_neighbors_shared_win[i])!=0){
          cout<<"Shared window failed to allocate on rank "<<rank<<"\n";
          exit(100);
        }
    }
    }
    else{
      for(i = 0; i < num_nodes; i++){
        MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &tmp_neighbors[i], &tmp_neighbors_shared_win[i]);
        MPI_Win_shared_query(tmp_neighbors_shared_win[i], 0, &size_of_shmem, &disp_unit, &tmp_neighbors[i]);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);



  int row_size=0;
  int N_val_1=0;
  int N_val_2=0;
  int N_ind=0;
  bool found=false;


  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win T_win;
  MPI_Win Z_win;
  MPI_Win N_win;
  MPI_Win_create(rank!=0?NULL:T.data(), rank!=0?0:(sizeof(int) * num_eles * 4), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &T_win);
  MPI_Win_lock_all(0,T_win);

  MPI_Barrier(MPI_COMM_WORLD);
  int win_ind;
  int inter_win_offset;
  int zero=0;
  int tmp_row_size_plus_one;
  int T_i_Xs[4];

  int T_chunk_Xs[(high-low)*4];


  int l_low;
  int l_high;
  int l_mid;
  int l;
  int elems[MAX_TET_NODE_NEIGHBORS_BUF];

  MPI_Get_accumulate(&T_chunk_Xs, (high-low)*4, MPI_INT, &T_chunk_Xs, (high-low)*4, MPI_INT, 0, low*4, (high-low)*4, MPI_INT, MPI_NO_OP, T_win);
  MPI_Win_flush_local(0,T_win);

  MPI_Win_unlock_all(T_win);

  int i_low_4x;
  int rank_to_send_val;
  int local_index;
  int num_out=0;

  for(i = low; i<high;i++){
    i_low_4x=(i-low)*4;
    T_i_Xs[0]=T_chunk_Xs[i_low_4x];
    T_i_Xs[1]=T_chunk_Xs[i_low_4x+1];
    T_i_Xs[2]=T_chunk_Xs[i_low_4x+2];
    T_i_Xs[3]=T_chunk_Xs[i_low_4x+3];
    T_i_0=T_i_Xs[0];
    T_i_1=T_i_Xs[1];
    if(T_i_0==T_i_1){
      break;
    }
    for(j=0;j<4;j++){
      T_i_j=T_i_Xs[j];
      for(k=0;k<4;k++){
        if(T_i_j<m){
          local_index=T_i_j%neighbors_chunk_size;
          if(rank==size-1){
            local_index=T_i_j-rank*neighbors_chunk_size;
          }

          T_i_k=T_i_Xs[k];
          row_size=shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+0];

          if(row_size==0){
            row_size=1;
            shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;
            shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+1]=T_i_k;
            if(T_i_k<m && (REDUNDANT?rank==0:(num_nodes==1?rank==0:false))){
              loc_num_vals_in_AI+=1;
            }
            else if(T_i_k>=m && (REDUNDANT?rank==0:(num_nodes==1?rank==0:false))){
              loc_num_vals_in_AB+=1;
            }
          }
          else{
            found=false;
            l_low=1;
            l_high=row_size;
            while(l_low<=l_high){
              l_mid=l_low+(l_high-l_low)/2;
              N_val_1=shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
              if(N_val_1==T_i_k){
                found=true;
                break;
              }
              else if(N_val_1<T_i_k){
                l_low=l_mid+1;
              }
              else{
                l_high=l_mid-1;
              }
            }
            N_ind=l_low;
            if(!found){
              row_size+=1;
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
              else{
                shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;
                for(l=row_size;l>=N_ind;l--){
                  shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
                }

                shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=T_i_k;
                if(T_i_k<m && (REDUNDANT?rank==0:(num_nodes==1?rank==0:false))){
                  loc_num_vals_in_AI+=1;
                }
                else if(T_i_k>=m && (REDUNDANT?rank==0:(num_nodes==1?rank==0:false))){
                  loc_num_vals_in_AB+=1;
                }
              }
            }
          }
        }
      }
    }
  }


  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Lists time: "<<end-start<<"\n";}start = MPI_Wtime();


  int shm_parts=m/shmsize;
  int shm_start=shmrank*shm_parts;
  int shm_end=shmrank==shmsize-1?m:(shmrank+1)*shm_parts;


  for(i=shm_start;i<shm_end;i++){
    for(j=0;j<shmsize;j++){
      for(k=1;k<=shared_neigbors_arr[j][i*MAX_TET_NODE_NEIGHBORS_BUF+0];k++){
        row_size=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0];
        int val_to_find=shared_neigbors_arr[j][i*MAX_TET_NODE_NEIGHBORS_BUF+k];
        found=false;
        l_low=1;
        l_high=row_size;
        while(l_low<=l_high){
          l_mid=l_low+(l_high-l_low)/2;
          N_val_1=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
          if(N_val_1==val_to_find){
            found=true;
            break;
          }
          else if(N_val_1<val_to_find){
            l_low=l_mid+1;
          }
          else{
            l_high=l_mid-1;
          }
        }
        N_ind=l_low;
        if(!found){
          row_size+=1;
          if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
          else{
            //Insert element
            shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;
            //shift old elements forward by 1
            for(l=row_size;l>=N_ind;l--){
              shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
            }
            shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
            if(val_to_find<m && (REDUNDANT?rank<shmsize:(num_nodes==1?rank<shmsize:false))){
              loc_num_vals_in_AI+=1;
            }
            else if(val_to_find>=m && (REDUNDANT?rank<shmsize:(num_nodes==1?rank<shmsize:false))){
              loc_num_vals_in_AB+=1;
            }
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Combine lists 1 time: "<<end-start<<"\n";}start = MPI_Wtime();




  ///////////////////Combine across multiple nodes///////////////
  if(num_nodes>1){
    if(!REDUNDANT){
      if(!RMA){

        int m_parts=m/size;
        int m_start=rank*m_parts;
        int m_end=rank==size-1?m:(rank+1)*m_parts;

        int start_end_nodes[num_nodes+1];
        int disps[num_nodes];
        int counts_send[num_nodes];
        for(i=0;i<num_nodes;i++){
          start_end_nodes[i]=(i*shmsize)*m_parts;
          disps[i]=start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF;
          if(i>=1){
            counts_send[i-1]=(start_end_nodes[i]-start_end_nodes[i-1])*MAX_TET_NODE_NEIGHBORS_BUF;
          }
        }
        counts_send[num_nodes-1]=(start_end_nodes[num_nodes]-start_end_nodes[num_nodes-1])*MAX_TET_NODE_NEIGHBORS_BUF;
        start_end_nodes[num_nodes]=m;

        if(shmrank==0){
          int tmp_tmp_neighbors[num_nodes][m][MAX_TET_NODE_NEIGHBORS_BUF]={0};
          MPI_Allgather(shared_neigbors_arr[0],m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,tmp_tmp_neighbors,m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,inter_comm);
          MPI_Barrier(inter_comm);if(rank==0){cout<<rank<<" MPI_Allgather 1 time: "<<MPI_Wtime()-start<<"\n";}
          for(i=0;i<num_nodes;i++){
            memcpy(tmp_neighbors[i]+m_start*MAX_TET_NODE_NEIGHBORS_BUF /*technically start for node*/,tmp_tmp_neighbors[i][m_start/*technically start for node*/],(start_end_nodes[rank/shmsize+1]-m_start)*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
          }
        }


        MPI_Barrier(MPI_COMM_WORLD);if(rank==0){cout<<rank<<" memcpy 1 time: "<<MPI_Wtime()-start<<"\n";}


        for(i=m_start;i<m_end;i++){
          row_size=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0];
          for(j=0;j<num_nodes;j++){
            for(k=1;k<=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+0];k++){
              int val_to_find=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+k];
              found=false;
              l_low=1;
              l_high=row_size;
              while(l_low<=l_high){
                l_mid=l_low+(l_high-l_low)/2;
                N_val_1=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
                if(N_val_1==val_to_find){
                  found=true;
                  break;
                }
                else if(N_val_1<val_to_find){
                  l_low=l_mid+1;
                }
                else{
                  l_high=l_mid-1;
                }
              }
              N_ind=l_low;
              if(!found){
                row_size+=1;
                if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
                else{
                  //Insert element
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;

                  for(l=row_size;l>=N_ind;l--){
                    shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
                  }
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
                }
              }
            }
          }
          int mid_div = std::lower_bound(&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+1],&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+row_size+1],m)-&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+1];
          loc_num_vals_in_AI+=mid_div;
          loc_num_vals_in_AB+=row_size-mid_div;

        }

        MPI_Barrier(MPI_COMM_WORLD);if(rank==0){cout<<rank<<" Merge time: "<<MPI_Wtime()-start<<"\n";}

        if(shmrank==0){
          int tmp_tmp_neighbors[num_nodes][m][MAX_TET_NODE_NEIGHBORS_BUF]={0};
          MPI_Allgather(shared_neigbors_arr[0],m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,tmp_tmp_neighbors,m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,inter_comm);
          MPI_Barrier(inter_comm);if(rank==0){cout<<rank<<" MPI_Allgather 2 time: "<<MPI_Wtime()-start<<"\n";}
          for(i=0;i<num_nodes;i++){
            memcpy(shared_neigbors_arr[0]+start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF,tmp_tmp_neighbors[i][start_end_nodes[i]],(start_end_nodes[i+1]-start_end_nodes[i])*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" NRCombine lists 2 time: "<<end-start<<"\n";}start = MPI_Wtime();
      }
      else{
        int m_parts=m/size;
        int m_start=rank*m_parts;
        int m_end=rank==size-1?m:(rank+1)*m_parts;

        int start_end_nodes[num_nodes+1];
        int disps[num_nodes];
        int counts_send[num_nodes];
        for(i=0;i<num_nodes;i++){
          start_end_nodes[i]=(i*shmsize)*m_parts;
          disps[i]=start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF;
          if(i>=1){
            counts_send[i-1]=(start_end_nodes[i]-start_end_nodes[i-1])*MAX_TET_NODE_NEIGHBORS_BUF;
          }
        }
        start_end_nodes[num_nodes]=m;
        counts_send[num_nodes-1]=(start_end_nodes[num_nodes]-start_end_nodes[num_nodes-1])*MAX_TET_NODE_NEIGHBORS_BUF;


        int neighors_T_size=shmrank==0?counts_send[inter_rank]*num_nodes:1;


        int neighors_T[neighors_T_size]={0};
        MPI_Win neighors_T_win;
        MPI_Win local_master_neighors_win;

        if(shmrank==0){
          MPI_Win_create(neighors_T, neighors_T_size*sizeof(int), sizeof(int), MPI_INFO_NULL, inter_comm, &neighors_T_win);
          MPI_Win_create(shared_neigbors_arr[0], m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int), sizeof(int), MPI_INFO_NULL, inter_comm, &local_master_neighors_win);
        }MPI_Barrier(MPI_COMM_WORLD);

        if(shmrank==0){
          MPI_Win_lock_all(0, neighors_T_win);
          for(i=num_nodes-1;i>=0;i--){
            MPI_Put(shared_neigbors_arr[0]+start_end_nodes[i]*MAX_TET_NODE_NEIGHBORS_BUF, counts_send[i], MPI_INT, i, inter_rank*counts_send[i],counts_send[i], MPI_INT, neighors_T_win);
          }
          MPI_Win_flush_local_all(neighors_T_win);
          MPI_Win_flush_all(neighors_T_win);

          MPI_Win_unlock_all(neighors_T_win);
          MPI_Barrier(inter_comm);

          for(i=0;i<num_nodes;i++){
            memcpy(tmp_neighbors[i]+m_start*MAX_TET_NODE_NEIGHBORS_BUF,neighors_T+i*counts_send[inter_rank],(start_end_nodes[rank/shmsize+1]-m_start)*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
          }
        }




        MPI_Barrier(MPI_COMM_WORLD);if(rank==0){cout<<rank<<" DEBUG RMA memcpy 1 time: "<<MPI_Wtime()-start<<"\n";}


        for(i=m_start;i<m_end;i++){
          row_size=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0];
          for(j=0;j<num_nodes;j++){
            for(k=1;k<=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+0];k++){
              int val_to_find=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+k];
              found=false;
              l_low=1;
              l_high=row_size;
              while(l_low<=l_high){
                l_mid=l_low+(l_high-l_low)/2;
                N_val_1=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
                if(N_val_1==val_to_find){
                  found=true;
                  break;
                }
                else if(N_val_1<val_to_find){
                  l_low=l_mid+1;
                }
                else{
                  l_high=l_mid-1;
                }
              }
              N_ind=l_low;
              if(!found){
                row_size+=1;
                if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
                else{
                  //Insert element
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;

                  for(l=row_size;l>=N_ind;l--){
                    shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
                  }
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
                }
              }
            }
          }
          int mid_div = std::lower_bound(&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+1],&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+row_size+1],m)-&shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+1];
          loc_num_vals_in_AI+=mid_div;
          loc_num_vals_in_AB+=row_size-mid_div;

        }

        MPI_Barrier(MPI_COMM_WORLD);if(rank==0){cout<<rank<<" DEBUG Merge time: "<<MPI_Wtime()-start<<"\n";}

        if(shmrank==0){
          MPI_Win_lock_all(0, local_master_neighors_win);
          for(i=num_nodes-1;i>=0;i--){
            //Need to just put node's portion of shared_neigbors_arr[0] into first part of neighors_T_win or maybe another rma window
            MPI_Put(shared_neigbors_arr[0]+start_end_nodes[inter_rank]*MAX_TET_NODE_NEIGHBORS_BUF, counts_send[inter_rank], MPI_INT, i, start_end_nodes[inter_rank]*MAX_TET_NODE_NEIGHBORS_BUF,counts_send[inter_rank], MPI_INT, local_master_neighors_win);
          }
          MPI_Win_flush_local_all(local_master_neighors_win);
          MPI_Win_flush_all(local_master_neighors_win);

          MPI_Win_unlock_all(local_master_neighors_win);

          MPI_Barrier(inter_comm);if(rank==0){cout<<rank<<" DEBUG MPI_Allgather 2 time: "<<MPI_Wtime()-start<<"\n";}
        }
        MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" NRCombine lists 2 time: "<<end-start<<"\n";}start = MPI_Wtime();
      }
    }

    else{
      if(shmrank==0){
        int tmp_tmp_neighbors[num_nodes][m][MAX_TET_NODE_NEIGHBORS_BUF]={0};
        MPI_Allgather(shared_neigbors_arr[0],m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,tmp_tmp_neighbors,m*MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,inter_comm);
        MPI_Barrier(inter_comm);
        for(i=0;i<num_nodes;i++){
          memcpy(tmp_neighbors[i],tmp_tmp_neighbors[i],m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      int shm_parts=m/shmsize;
      int shm_start=shmrank*shm_parts;
      int shm_end=shmrank==shmsize-1?m:(shmrank+1)*shm_parts;
      for(i=shm_start;i<shm_end;i++){
        for(j=0;j<num_nodes;j++){
          for(k=1;k<=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+0];k++){
            row_size=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0];
            int val_to_find=tmp_neighbors[j][i*MAX_TET_NODE_NEIGHBORS_BUF+k];
            found=false;
            l_low=1;
            l_high=row_size;
            while(l_low<=l_high){
              l_mid=l_low+(l_high-l_low)/2;
              N_val_1=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l_mid];
              if(N_val_1==val_to_find){
                found=true;
                break;
              }
              else if(N_val_1<val_to_find){
                l_low=l_mid+1;
              }
              else{
                l_high=l_mid-1;
              }
            }
            N_ind=l_low;
            if(!found){
              row_size+=1;
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
              else{
                //Insert element
                shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;

                for(l=row_size;l>=N_ind;l--){
                  shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
                }
                shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
                if(val_to_find<m && rank<shmsize){
                  loc_num_vals_in_AI+=1;
                }
                else if(val_to_find>=m && rank<shmsize){
                  loc_num_vals_in_AB+=1;
                }
              }
            }
          }
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" RCombine lists 2 time: "<<end-start<<"\n";}start = MPI_Wtime();
    }
  }

  ///////////////////Combine across multiple nodes///////////////



  if(shmrank==0){
    memcpy(neighbors,shared_neigbors_arr[0], m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
  }





  MPI_Allreduce(&loc_num_vals_in_AI,&num_vals_in_AI,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&loc_num_vals_in_AB,&num_vals_in_AB,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0) cout<<"num_vals_in_AI: "<<num_vals_in_AI<<"\n";
  if(rank==0) cout<<"num_vals_in_AB: "<<num_vals_in_AB<<"\n";

  //cout<<"MPI_Allreduce "<<rank<<"\n";
  CSR_Matrix AI_matrix(num_vals_in_AI,num_vals_in_AI,m+1);
  CSR_Matrix AB_matrix(num_vals_in_AB,num_vals_in_AB,m+1);

  //cout<<"CSR_Matrix "<<rank<<"\n";




  int AB_val_count=0;
  int AB_col_indices_pos=0;
  int AB_offset=m;


  int AI_val_count=0;
  int AI_col_indices_pos=0;
  int AI_offset=0;
  if(shmrank==0){
    for(i=0;i<m;i++){
      AI_matrix.setRowPtrsAt(i,AI_col_indices_pos);
      AB_matrix.setRowPtrsAt(i,AB_col_indices_pos);
      for(j=1;j<=neighbors[i][0];j++){
        if(neighbors[i][j]>=m){
          AB_matrix.setColIndicesAt(AB_col_indices_pos,neighbors[i][j]-AB_offset);
          AB_col_indices_pos+=1;
        }
        else{
          AI_matrix.setColIndicesAt(AI_col_indices_pos,neighbors[i][j]-AI_offset);
          AI_col_indices_pos+=1;
        }
      }
    }
    AI_matrix.setRowPtrsAt(m,AI_col_indices_pos);
    AB_matrix.setRowPtrsAt(m,AB_col_indices_pos);
  }



  double tmpI[AI_matrix._num_vals];
  double tmpB[AB_matrix._num_vals];
  int* shared_tmpI_c;
  MPI_Win shared_tmpI_c_win;
  int* shared_tmpB_c;
  MPI_Win shared_tmpB_c_win;



  if(shmrank==0){
    if(MPI_Win_allocate_shared(AI_matrix._num_col_indices*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &shared_tmpI_c, &shared_tmpI_c_win)!=0){
      cout<<"Shared window failed to allocate on rank "<<rank<<"\n";
      exit(100);
    }
    if(MPI_Win_allocate_shared(AB_matrix._num_col_indices*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &shared_tmpB_c, &shared_tmpB_c_win)!=0){
      cout<<"Shared window failed to allocate on rank "<<rank<<"\n";
      exit(100);
    }
  }
  else{
    MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &shared_tmpI_c, &shared_tmpI_c_win);
    MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &shared_tmpB_c, &shared_tmpB_c_win);
    MPI_Win_shared_query(shared_tmpI_c_win, 0, &size_of_shmem, &disp_unit, &shared_tmpI_c);
    MPI_Win_shared_query(shared_tmpB_c_win, 0, &size_of_shmem, &disp_unit, &shared_tmpB_c);
  }



  int tmpI_r[AI_matrix._num_row_ptrs];
  int tmpB_r[AB_matrix._num_row_ptrs];

  if(shmrank==0){
    memcpy(tmpI_r,AI_matrix._row_ptrs,AI_matrix._num_row_ptrs*sizeof(int));
    memcpy(tmpB_r,AB_matrix._row_ptrs,AB_matrix._num_row_ptrs*sizeof(int));
  }
  MPI_Barrier(shmcomm);
  MPI_Bcast(tmpI_r, AI_matrix._num_row_ptrs, MPI_INT, 0, shmcomm);
  MPI_Bcast(tmpB_r, AB_matrix._num_row_ptrs, MPI_INT, 0, shmcomm);
  MPI_Barrier(shmcomm);
  if(shmrank!=0){
    memcpy(AI_matrix._row_ptrs,tmpI_r,AI_matrix._num_row_ptrs*sizeof(int));
    memcpy(AB_matrix._row_ptrs,tmpB_r,AB_matrix._num_row_ptrs*sizeof(int));
  }


  if(shmrank==0){
    memcpy(shared_tmpI_c,AI_matrix._col_indices,AI_matrix._num_col_indices*sizeof(int));
    memcpy(shared_tmpB_c,AB_matrix._col_indices,AB_matrix._num_col_indices*sizeof(int));
  }
  MPI_Barrier(shmcomm);
  if(shmrank!=0){
    memcpy(AI_matrix._col_indices,shared_tmpI_c,AI_matrix._num_col_indices*sizeof(int));
    memcpy(AB_matrix._col_indices,shared_tmpB_c,AB_matrix._num_col_indices*sizeof(int));
  }








  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Matrix Allocation time: "<<end-start<<"\n";}start = MPI_Wtime();
  if(num_nodes>1){for(i=0;i<num_nodes;i++){MPI_Win_free(&tmp_neighbors_shared_win[i]);}}
  for(i=0;i<shmsize;i++){MPI_Win_free(&shared_win[i]);}
  MPI_Win_free(&shared_tmpI_c_win);
  MPI_Win_free(&shared_tmpB_c_win);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" MPI_Win_free time: "<<end-start<<"\n";}start = MPI_Wtime();


  for (n_ele=low; n_ele<high; n_ele++){

    i_low_4x=(n_ele-low)*4;
    T_ele[0]=(unsigned int)T_chunk_Xs[i_low_4x];
    T_ele[1]=(unsigned int)T_chunk_Xs[i_low_4x+1];
    T_ele[2]=(unsigned int)T_chunk_Xs[i_low_4x+2];
    T_ele[3]=(unsigned int)T_chunk_Xs[i_low_4x+3];



    a_func[0]=Z_original(T_ele[0],0);
    a_func[1]=Z_original(T_ele[0],1);
    a_func[2]=Z_original(T_ele[0],2);
    b_func[0]=Z_original(T_ele[1],0);
    b_func[1]=Z_original(T_ele[1],1);
    b_func[2]=Z_original(T_ele[1],2);
    c_func[0]=Z_original(T_ele[2],0);
    c_func[1]=Z_original(T_ele[2],1);
    c_func[2]=Z_original(T_ele[2],2);
    d_func[0]=Z_original(T_ele[3],0);
    d_func[1]=Z_original(T_ele[3],1);
    d_func[2]=Z_original(T_ele[3],2);


    gen_basis_function_gradients_tetrahedron(a_func,b_func,c_func,d_func,basis_function_gradients);
    element_stiffness_matrix=gen_element_stiffness_matrix_tetrahedron(basis_function_gradients,a_func,b_func,c_func,d_func);
    for (i = 0; i<4; i++){
      for (j = 0; j<4; j++){
        if (T_ele[i]<m){
          if (T_ele[j]>=m){
            if (element_stiffness_matrix(i,j)!=0){
              AB_matrix.subractValAt(T_ele[i],T_ele[j]-m,element_stiffness_matrix(i,j));
            }
          }
          else{
            if (element_stiffness_matrix(i,j)!=0){
              AI_matrix.addValAt(T_ele[i],T_ele[j],element_stiffness_matrix(i,j));
            }
          }
        }
      }
    }
  }


  MPI_Barrier(MPI_COMM_WORLD);


  memcpy(tmpI,AI_matrix._vals,AI_matrix._num_vals*sizeof(double));
  memcpy(tmpB,AB_matrix._vals,AB_matrix._num_vals*sizeof(double));


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(tmpI,AI_matrix._vals,AI_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(tmpB,AB_matrix._vals,AB_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" FEM time: "<<end-start<<"\n";}start = MPI_Wtime();



  MPI_Barrier(MPI_COMM_WORLD);
  CSR_Matrix local_A_I(0,0,0);
  CSR_Matrix local_A_B(0,0,0);
  csr_to_dist_csr(AI_matrix,local_A_I,m,comm,size,rank);
  csr_to_dist_csr(AB_matrix,local_A_B,m,comm,size,rank);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Distribute Matrix time: "<<end-start<<"\n";}
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;
  int local_num_rows=local_A_B._num_row_ptrs-1;
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(MPI_COMM_WORLD);
  num_rows = accumulate(num_rows_arr, num_rows_arr+size, num_rows);

  for(i=0;i<num_deformations;i++){
    A_B_hat.resize(3,AI_matrix._num_row_ptrs-1);//Should be (num_rows,3), but mpi gets mad
    A_B_hat.setZero();
    Z_femwarp_transformed.setZero();
    MPI_Barrier(MPI_COMM_WORLD);start = MPI_Wtime();
    (*deformation_functions[i])(Z_cur_boundary);
    parallel_csr_x_matrix(local_A_B,Z_cur_boundary,A_B_hat,comm,rank,size,num_rows_arr);
    MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Times Matrix time: "<<end-start<<"\n";}start = MPI_Wtime();
    int block_size=1;
    block_size=local_num_rows<block_size?local_num_rows:block_size;
    parallel_sparse_block_conjugate_gradient_v3(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows,block_size);
  }




  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CG time: "<<end-start<<"\n";}

  if(rank==0){
    Z_original.block(0,0,m,3)=Z_femwarp_transformed.eval();
    Z_original.block(m,0,b,3)=Z_cur_boundary.eval();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void serial_femwarp3d
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  Eigen::MatrixXd& Z_femwarp_transformed
)
{
  int rank = 0;
  double start,end;start = MPI_Wtime();
  /*register*/ int i,j,k,n_ele;
  int T_i_0;
  int T_i_1;
  int T_i_j;
  int T_i_k;
  int loc_num_vals_in_AI=0;
  int loc_num_vals_in_AB=0;
  int num_vals_in_AI=0;
  int num_vals_in_AB=0;
  int num_rows=0;
  unsigned int T_ele[4];
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];

  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  char* rec_str_buf=nullptr;
  char* all_gathered_chars=nullptr;



  cout<<"Job Information:"<<"\n";
  cout<<"\tRanks: "<<1<<"\n";
  cout<<"\tNodes: "<<1<<"\n";
  cout<<"\tRanks per node: "<<1<<"\n";
  cout<<"\tn: "<<n<<"\n";
  cout<<"\tm: "<<m<<"\n";
  cout<<"\tb: "<<b<<"\n";
  cout<<"\tnum_eles: "<<num_eles<<"\n";





  int neighbors[m][MAX_TET_NODE_NEIGHBORS_BUF]={0};




  int row_size=0;
  int N_val_1=0;
  int N_val_2=0;
  int N_ind=0;
  bool found=false;

  //cout<<"*"<<"\n";

  int win_ind;
  int inter_win_offset;//Will need to quantize a row index within a window chunk
  int zero=0;
  int tmp_row_size_plus_one;


  int l_low;
  int l_high;
  int l_mid;
  int l;
  int elems[MAX_TET_NODE_NEIGHBORS_BUF];

  int i_low_4x;
  int rank_to_send_val;
  int local_index;
  int num_out=0;

  for(i = 0; i<num_eles;i++){//O(n)

    if(T(i,0)==T(i,1)){
      break;
    }
    for(j=0;j<4;j++){
      T_i_j=T(i,j);
      for(k=0;k<4;k++){
        if(T_i_j<m){

          T_i_k=T(i,k);
          row_size=neighbors[T_i_j][0];//neighbors[T_i_j/*local_index*/][0];

          if(row_size==0){
            row_size=1;
            neighbors[T_i_j][0]=row_size;//neighbors[T_i_j/*local_index*/][0]=row_size;//First index of row contains size
            neighbors[T_i_j][1]=T_i_k;//neighbors[T_i_j/*local_index*/][1]=T_i_k;
            if(T_i_k<m){
              num_vals_in_AI+=1;
            }
            else if(T_i_k>=m) {
              num_vals_in_AB+=1;
            }
          }
          else{
            found=false;
            l_low=1;
            l_high=row_size;
            while(l_low<=l_high){
              l_mid=l_low+(l_high-l_low)/2;
              N_val_1=neighbors[T_i_j][l_mid];//neighbors[T_i_j/*local_index*/][l_mid];
              if(N_val_1==T_i_k){
                found=true;
                break;
              }
              else if(N_val_1<T_i_k){
                l_low=l_mid+1;
              }
              else{
                l_high=l_mid-1;
              }
            }
            N_ind=l_low;
            if(!found){
              row_size+=1;
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
              else{
                //Insert element
                //local_neighbors[T_i_j].insert(T_i_k);
                neighbors[T_i_j][0]=row_size;//neighbors[T_i_j/*local_index*/][0]=row_size;
                for(l=row_size;l>=N_ind;l--){
                  neighbors[T_i_j][l]=neighbors[T_i_j][l-1];//neighbors[T_i_j/*local_index*/][l]=neighbors[T_i_j/*local_index*/][l-1];
                }
                //memcpy(shared_neigbors_arr[shmrank]+T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+N_ind+1,shared_neigbors_arr[shmrank]+T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+N_ind,(row_size-N_ind+1)*sizeof(int));
                neighbors[T_i_j][N_ind]=T_i_k;//neighbors[T_i_j/*local_index*/][N_ind]=T_i_k;
                if(T_i_k<m){
                  num_vals_in_AI+=1;
                }
                else if(T_i_k>=m){
                  num_vals_in_AB+=1;
                }
              }
            }
          }
        }
      }
    }
  }



  end = MPI_Wtime();if(rank==0){cout<<rank<<" Lists time: "<<end-start<<"\n";}start = MPI_Wtime();

  CSR_Matrix AI_matrix(num_vals_in_AI,num_vals_in_AI,m+1);
  CSR_Matrix AB_matrix(num_vals_in_AB,num_vals_in_AB,m+1);




  int AB_val_count=0;
  int AB_col_indices_pos=0;
  int AB_offset=m;


  int AI_val_count=0;
  int AI_col_indices_pos=0;
  int AI_offset=0;
  for(i=0;i<m;i++){
    AI_matrix.setRowPtrsAt(i,AI_col_indices_pos);
    AB_matrix.setRowPtrsAt(i,AB_col_indices_pos);
    for(j=1;j<=neighbors[i][0];j++){
      if(neighbors[i][j]>=m){
        AB_matrix.setColIndicesAt(AB_col_indices_pos,neighbors[i][j]-AB_offset);
        AB_col_indices_pos+=1;
      }
      else{
        AI_matrix.setColIndicesAt(AI_col_indices_pos,neighbors[i][j]-AI_offset);
        AI_col_indices_pos+=1;
      }
    }
  }
  //cout<<"setRowPtrsAt "<<rank<<"\n";
  AI_matrix.setRowPtrsAt(m,AI_col_indices_pos);
  AB_matrix.setRowPtrsAt(m,AB_col_indices_pos);


  end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Matrix Allocation time: "<<end-start<<"\n";}start = MPI_Wtime();




  for (n_ele=0; n_ele<num_eles; n_ele++){

    T_ele[0]=T(n_ele,0);
    T_ele[1]=T(n_ele,1);
    T_ele[2]=T(n_ele,2);
    T_ele[3]=T(n_ele,3);



    a_func[0]=Z_original(T_ele[0],0);
    a_func[1]=Z_original(T_ele[0],1);
    a_func[2]=Z_original(T_ele[0],2);
    b_func[0]=Z_original(T_ele[1],0);
    b_func[1]=Z_original(T_ele[1],1);
    b_func[2]=Z_original(T_ele[1],2);
    c_func[0]=Z_original(T_ele[2],0);
    c_func[1]=Z_original(T_ele[2],1);
    c_func[2]=Z_original(T_ele[2],2);
    d_func[0]=Z_original(T_ele[3],0);
    d_func[1]=Z_original(T_ele[3],1);
    d_func[2]=Z_original(T_ele[3],2);


    gen_basis_function_gradients_tetrahedron(a_func,b_func,c_func,d_func,basis_function_gradients);
    element_stiffness_matrix=gen_element_stiffness_matrix_tetrahedron(basis_function_gradients,a_func,b_func,c_func,d_func);
    for (i = 0; i<4; i++){
      for (j = 0; j<4; j++){
        if (T_ele[i]<m){
          if (T_ele[j]>=m){
            if (element_stiffness_matrix(i,j)!=0){
              AB_matrix.subractValAt(T_ele[i],T_ele[j]-m,element_stiffness_matrix(i,j));
            }
          }
          else{
            if (element_stiffness_matrix(i,j)!=0){
              AI_matrix.addValAt(T_ele[i],T_ele[j],element_stiffness_matrix(i,j));
            }
          }
        }
      }
    }
  }
  end = MPI_Wtime();if(rank==0){cout<<rank<<" FEM time: "<<end-start<<"\n";}

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  A_B_hat.resize(3,AI_matrix._num_row_ptrs-1);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();


  start = MPI_Wtime();
  csr_x_matrix(AB_matrix,xy_prime,A_B_hat);
  end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Times Matrix time: "<<end-start<<"\n";}start = MPI_Wtime();

  //sparse_block_conjugate_gradient_v2(AI_matrix,A_B_hat,Z_femwarp_transformed);
  /* 19 *///sparse_block_conjugate_gradient_v3(AI_matrix,A_B_hat,Z_femwarp_transformed);
  /* ?? */sparse_block_conjugate_gradient_v4(AI_matrix,A_B_hat,Z_femwarp_transformed);


  end = MPI_Wtime();if(rank==0){cout<<rank<<" CG time: "<<end-start<<"\n";}
}




void serial_multistep_femwarp3d
(
  Eigen::MatrixXd& Z_original,
  void (*deformation_functions[])(Eigen::MatrixXd&),int num_deformations,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  Eigen::MatrixXd& Z_femwarp_transformed
)
{
  int rank = 0;
  double start,end;start = MPI_Wtime();
  int i,j,k,n_ele;
  int T_i_0, T_i_1, T_i_j, T_i_k;
  int loc_num_vals_in_AI=0;
  int loc_num_vals_in_AB=0;
  int num_vals_in_AI=0;
  int num_vals_in_AB=0;
  int num_rows=0;
  unsigned int T_ele[4];
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];

  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  char* rec_str_buf=nullptr;
  char* all_gathered_chars=nullptr;


  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_cur_boundary;
  Z_cur_boundary=Z_original.block(n-b,0,b,3);//Get only boundary nodes


  cout<<"Job Information:"<<"\n";
  cout<<"\tRanks: "<<1<<"\n";
  cout<<"\tNodes: "<<1<<"\n";
  cout<<"\tRanks per node: "<<1<<"\n";
  cout<<"\tn: "<<n<<"\n";
  cout<<"\tm: "<<m<<"\n";
  cout<<"\tb: "<<b<<"\n";
  cout<<"\tnum_eles: "<<num_eles<<"\n";





  int neighbors[m][MAX_TET_NODE_NEIGHBORS_BUF]={0};




  int row_size=0;
  int N_val_1=0;
  int N_val_2=0;
  int N_ind=0;
  bool found=false;

  int win_ind;
  int inter_win_offset;
  int zero=0;
  int tmp_row_size_plus_one;


  int l_low;
  int l_high;
  int l_mid;
  int l;
  int elems[MAX_TET_NODE_NEIGHBORS_BUF];

  int i_low_4x;
  int rank_to_send_val;
  int local_index;
  int num_out=0;

  for(i = 0; i<num_eles;i++){

    if(T(i,0)==T(i,1)){
      break;
    }
    for(j=0;j<4;j++){
      T_i_j=T(i,j);
      for(k=0;k<4;k++){
        if(T_i_j<m){

          T_i_k=T(i,k);
          row_size=neighbors[T_i_j][0];

          if(row_size==0){
            row_size=1;
            neighbors[T_i_j][0]=row_size;
            neighbors[T_i_j][1]=T_i_k;
            if(T_i_k<m){
              num_vals_in_AI+=1;
            }
            else if(T_i_k>=m) {
              num_vals_in_AB+=1;
            }
          }
          else{
            found=false;
            l_low=1;
            l_high=row_size;
            while(l_low<=l_high){
              l_mid=l_low+(l_high-l_low)/2;
              N_val_1=neighbors[T_i_j][l_mid];
              if(N_val_1==T_i_k){
                found=true;
                break;
              }
              else if(N_val_1<T_i_k){
                l_low=l_mid+1;
              }
              else{
                l_high=l_mid-1;
              }
            }
            N_ind=l_low;
            if(!found){
              row_size+=1;
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<"\n";exit(MAX_TET_NODE_NEIGHBORS_BUF);}
              else{
                neighbors[T_i_j][0]=row_size;
                for(l=row_size;l>=N_ind;l--){
                  neighbors[T_i_j][l]=neighbors[T_i_j][l-1];
                }
                neighbors[T_i_j][N_ind]=T_i_k;
                if(T_i_k<m){
                  num_vals_in_AI+=1;
                }
                else if(T_i_k>=m){
                  num_vals_in_AB+=1;
                }
              }
            }
          }
        }
      }
    }
  }



  end = MPI_Wtime();if(rank==0){cout<<rank<<" Lists time: "<<end-start<<"\n";}start = MPI_Wtime();

  CSR_Matrix AI_matrix(num_vals_in_AI,num_vals_in_AI,m+1);
  CSR_Matrix AB_matrix(num_vals_in_AB,num_vals_in_AB,m+1);




  int AB_val_count=0;
  int AB_col_indices_pos=0;
  int AB_offset=m;


  int AI_val_count=0;
  int AI_col_indices_pos=0;
  int AI_offset=0;
  for(i=0;i<m;i++){
    AI_matrix.setRowPtrsAt(i,AI_col_indices_pos);
    AB_matrix.setRowPtrsAt(i,AB_col_indices_pos);
    for(j=1;j<=neighbors[i][0];j++){
      if(neighbors[i][j]>=m){
        AB_matrix.setColIndicesAt(AB_col_indices_pos,neighbors[i][j]-AB_offset);
        AB_col_indices_pos+=1;
      }
      else{
        AI_matrix.setColIndicesAt(AI_col_indices_pos,neighbors[i][j]-AI_offset);
        AI_col_indices_pos+=1;
      }
    }
  }
  AI_matrix.setRowPtrsAt(m,AI_col_indices_pos);
  AB_matrix.setRowPtrsAt(m,AB_col_indices_pos);


  end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Matrix Allocation time: "<<end-start<<"\n";}start = MPI_Wtime();

  for (n_ele=0; n_ele<num_eles; n_ele++){
    T_ele[0]=T(n_ele,0);
    T_ele[1]=T(n_ele,1);
    T_ele[2]=T(n_ele,2);
    T_ele[3]=T(n_ele,3);



    a_func[0]=Z_original(T_ele[0],0);
    a_func[1]=Z_original(T_ele[0],1);
    a_func[2]=Z_original(T_ele[0],2);
    b_func[0]=Z_original(T_ele[1],0);
    b_func[1]=Z_original(T_ele[1],1);
    b_func[2]=Z_original(T_ele[1],2);
    c_func[0]=Z_original(T_ele[2],0);
    c_func[1]=Z_original(T_ele[2],1);
    c_func[2]=Z_original(T_ele[2],2);
    d_func[0]=Z_original(T_ele[3],0);
    d_func[1]=Z_original(T_ele[3],1);
    d_func[2]=Z_original(T_ele[3],2);

    gen_basis_function_gradients_tetrahedron(a_func,b_func,c_func,d_func,basis_function_gradients);
    element_stiffness_matrix=gen_element_stiffness_matrix_tetrahedron(basis_function_gradients,a_func,b_func,c_func,d_func);
    for (i = 0; i<4; i++){
      for (j = 0; j<4; j++){
        if (T_ele[i]<m){
          if (T_ele[j]>=m){
            if (element_stiffness_matrix(i,j)!=0){
              AB_matrix.subractValAt(T_ele[i],T_ele[j]-m,element_stiffness_matrix(i,j));
            }
          }
          else{
            if (element_stiffness_matrix(i,j)!=0){
              AI_matrix.addValAt(T_ele[i],T_ele[j],element_stiffness_matrix(i,j));
            }
          }
        }
      }
    }
  }
  end = MPI_Wtime();if(rank==0){cout<<rank<<" FEM time: "<<end-start<<"\n";}
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  for(i=0;i<num_deformations;i++){
    A_B_hat.resize(3,AI_matrix._num_row_ptrs-1);//Should be (num_rows,3), but mpi gets mad
    A_B_hat.setZero();
    Z_femwarp_transformed.setZero();
    start = MPI_Wtime();
    (*deformation_functions[i])(Z_cur_boundary);
    end = MPI_Wtime();if(rank==0){cout<<rank<<" Deformation time: "<<end-start<<"\n";}start = MPI_Wtime();
    csr_x_matrix(AB_matrix,Z_cur_boundary,A_B_hat);
    end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Times Matrix time: "<<end-start<<"\n";}start = MPI_Wtime();
    sparse_block_conjugate_gradient_v3(AI_matrix,A_B_hat,Z_femwarp_transformed);
    end = MPI_Wtime();if(rank==0){cout<<rank<<" CG time: "<<end-start<<"\n";}
  }
  Z_original.block(0,0,m,3)=Z_femwarp_transformed.eval();
  Z_original.block(m,0,b,3)=Z_cur_boundary.eval();
}