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
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <vector>
#include <list>
#include <queue>
#include <numeric>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <csr.hpp>


using namespace std;


double volume_tetrahedron
(
  double a[],
  double b[],
  double c[],
  double d[]
)
{
Eigen::Matrix4d vol_matrix;
vol_matrix << a[0],b[0],c[0],d[0],a[1],b[1],c[1],d[1],a[2],b[2],c[2],d[2],1,1,1,1;
return abs(vol_matrix.determinant());
}

Eigen::Matrix4d gen_element_stiffness_matrix_tetrahedron
(
  Eigen::Matrix4d& gradients,
  double a[],
  double b[],
  double c[],
  double d[]
)
  {
    Eigen::Matrix4d k;
    double V=volume_tetrahedron(a,b,c,d);
    for(int i = 0; i < 4; i++){
      for(int j = i; j < 4; j++){
        k(i,j)=(gradients(i,0)*gradients(j,0)+gradients(i,1)*gradients(j,1)+gradients(i,2)*gradients(j,2))*V;
        k(j,i)=k(i,j);
      }
    }
    return k;
  }

void gen_basis_function_gradients_tetrahedron
(
  double* a,
  double* b,
  double* c,
  double* d,
  Eigen::Matrix4d& ret
)
{
  Eigen::MatrixXd A(16,16);
  Eigen::VectorXd vector_b(16);
  A<<
    a[0],a[1],a[2],1, 0,0,0,0,        0,0,0,0,        0,0,0,0,
    b[0],b[1],b[2],1, 0,0,0,0,        0,0,0,0,        0,0,0,0,
    c[0],c[1],c[2],1, 0,0,0,0,        0,0,0,0,        0,0,0,0,
    d[0],d[1],d[2],1, 0,0,0,0,        0,0,0,0,        0,0,0,0,
    0,0,0,0,       a[0],a[1],a[2],1,  0,0,0,0,        0,0,0,0,
    0,0,0,0,       b[0],b[1],b[2],1,  0,0,0,0,        0,0,0,0,
    0,0,0,0,       c[0],c[1],c[2],1,  0,0,0,0,        0,0,0,0,
    0,0,0,0,       d[0],d[1],d[2],1,  0,0,0,0,        0,0,0,0,
    0,0,0,0,       0,0,0,0,       a[0],a[1],a[2],1,   0,0,0,0,
    0,0,0,0,       0,0,0,0,       b[0],b[1],b[2],1,   0,0,0,0,
    0,0,0,0,       0,0,0,0,       c[0],c[1],c[2],1,   0,0,0,0,
    0,0,0,0,       0,0,0,0,       d[0],d[1],d[2],1,   0,0,0,0,
    0,0,0,0,       0,0,0,0,       0,0,0,0,         a[0],a[1],a[2],1,
    0,0,0,0,       0,0,0,0,       0,0,0,0,         b[0],b[1],b[2],1,
    0,0,0,0,       0,0,0,0,       0,0,0,0,         c[0],c[1],c[2],1,
    0,0,0,0,       0,0,0,0,       0,0,0,0,         d[0],d[1],d[2],1;
  vector_b<<1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1;
  vector_b << A.partialPivLu().solve(vector_b);//TODO need to convert into matrix
  for(int i = 0; i < 16; i++){
    ret(i/4,i%4)=vector_b(i);
  }
}

int gen_neighbors_csr
(
  unordered_map<int, set<int>>& neighbors_dict,
  CSR_Matrix& csr_matrix,
  int offset=0
)
{
  int n=neighbors_dict.size();
  int val_count=0;
  int num_vals_in_set;
  int col_indices_pos=0;
  for (int i = 0; i < n; i++){
    csr_matrix.setRowPtrsAt(i,col_indices_pos);
    num_vals_in_set=neighbors_dict[i].size();
    val_count+=num_vals_in_set;
    for(set<int>::iterator val_node=neighbors_dict[i].begin(); val_node!= neighbors_dict[i].end(); val_node++){
      csr_matrix.setColIndicesAt(col_indices_pos,*val_node-offset);
      col_indices_pos+=1;
    }
  }
  csr_matrix.setRowPtrsAt(n,col_indices_pos);
  return val_count;
}


void csr_to_dist_csr
(
  CSR_Matrix& csr_matrix,
  CSR_Matrix& dist_csr_matrix_part,
  int rows,
  MPI_Comm comm,
  int size,
  int rank
)
{
  int num_rows_per_part=rows/size;
  int vtxdist[size+1];
  for(int i = 0; i < size; i++){
    vtxdist[i]=i*num_rows_per_part;
  }
  vtxdist[size]=rows;
  MPI_Barrier(comm);
  vector<vector<int>> local_row_ptrs;
  vector<vector<int>> local_col_inds;
  vector<vector<double>> local_datas;
  int tmp1_size=0;
  int tmp2_size=0;
  int tmp3_size=0;

  int* local_row_ptr_buf;
  int* local_col_ind_buf;
  double* local_data_buf;

  int local_row_ptr_buf_sendcounts[size];
  int local_row_ptr_buf_displs[size]={0};
  int local_row_ptr_buf_recvcount;

  int local_col_ind_buf_sendcounts[size];
  int local_col_ind_buf_displs[size]={0};
  int local_col_ind_buf_recvcount;

  int local_data_buf_sendcounts[size];
  int local_data_buf_displs[size]={0};
  int local_data_buf_recvcount;

  if (rank==0){
    int col_offset=0;
    for(int i = 0; i < size;i++){
      int start_offset=vtxdist[i];
      int end_offset=vtxdist[i+1]+1;
      tmp1_size+=end_offset-start_offset;
      local_row_ptr_buf_sendcounts[i]=end_offset-start_offset;
      vector<int> tmp1(local_row_ptr_buf_sendcounts[i]);
      for(int j = start_offset; j < end_offset; j++){
        tmp1[j-start_offset]=csr_matrix._row_ptrs[j]-csr_matrix._row_ptrs[start_offset];
      }
      local_row_ptrs.push_back(tmp1);
      int tmp23_size=local_row_ptrs[i].back()-local_row_ptrs[i][0];
      tmp2_size+=tmp23_size;
      tmp3_size+=tmp23_size;
      local_col_ind_buf_sendcounts[i]=tmp23_size;
      local_data_buf_sendcounts[i]=tmp23_size;
      vector<int> tmp2(tmp23_size);
      vector<double> tmp3(tmp23_size);
      for(int j = local_row_ptrs[i][0]; j < local_row_ptrs[i].back(); j++){
        tmp2[j]=csr_matrix._col_indices[j+col_offset];
        tmp3[j]=csr_matrix._vals[j+col_offset];
      }
      local_col_inds.push_back(tmp2);
      local_datas.push_back(tmp3);
      col_offset+=local_row_ptrs.back().back();
    }
    local_row_ptr_buf=(int*)malloc(tmp1_size*sizeof(int));
    local_col_ind_buf=(int*)malloc(tmp2_size*sizeof(int));
    local_data_buf=(double*) malloc(tmp3_size*sizeof(double));
    memcpy(local_row_ptr_buf, local_row_ptrs[0].data(), local_row_ptrs[0].size()*sizeof(int));
    memcpy(local_col_ind_buf, local_col_inds[0].data(), local_col_inds[0].size()*sizeof(int));
    memcpy(local_data_buf, local_datas[0].data(), local_datas[0].size()*sizeof(double));
    for(int i = 1; i < size; i++){
      local_row_ptr_buf_displs[i]=local_row_ptr_buf_displs[i-1]+local_row_ptr_buf_sendcounts[i-1];
      local_col_ind_buf_displs[i]=local_col_ind_buf_displs[i-1]+local_col_ind_buf_sendcounts[i-1];
      local_data_buf_displs[i]=local_data_buf_displs[i-1]+local_data_buf_sendcounts[i-1];
      memcpy(&local_row_ptr_buf[local_row_ptr_buf_displs[i]], local_row_ptrs[i].data(), local_row_ptrs[i].size()*sizeof(int));
      memcpy(&local_col_ind_buf[local_col_ind_buf_displs[i]], local_col_inds[i].data(), local_col_inds[i].size()*sizeof(int));
      memcpy(&local_data_buf[local_data_buf_displs[i]], local_datas[i].data(), local_datas[i].size()*sizeof(double));
    }
  }
  MPI_Barrier(comm);
  MPI_Bcast(local_row_ptr_buf_sendcounts,size,MPI_INT,0,comm);
  MPI_Bcast(local_row_ptr_buf_displs,size,MPI_INT,0,comm);
  MPI_Bcast(local_col_ind_buf_sendcounts,size,MPI_INT,0,comm);
  MPI_Bcast(local_col_ind_buf_displs,size,MPI_INT,0,comm);
  MPI_Bcast(local_data_buf_sendcounts,size,MPI_INT,0,comm);
  MPI_Bcast(local_data_buf_displs,size,MPI_INT,0,comm);
  vector<int> local_row_ptr(local_row_ptr_buf_sendcounts[rank]);
  vector<int> local_col_ind(local_col_ind_buf_sendcounts[rank]);
  vector<double> local_data(local_data_buf_sendcounts[rank]);
  MPI_Scatterv(local_row_ptr_buf,local_row_ptr_buf_sendcounts,local_row_ptr_buf_displs,MPI_INT,local_row_ptr.data(),local_row_ptr_buf_sendcounts[rank],MPI_INT,0,comm);
  MPI_Scatterv(local_col_ind_buf,local_col_ind_buf_sendcounts,local_col_ind_buf_displs,MPI_INT,local_col_ind.data(),local_col_ind_buf_sendcounts[rank],MPI_INT,0,comm);
  MPI_Scatterv(local_data_buf,local_data_buf_sendcounts,local_data_buf_displs,MPI_DOUBLE,local_data.data(),local_data_buf_sendcounts[rank],MPI_DOUBLE,0,comm);
  MPI_Barrier(comm);

  if(rank==0){
    free(local_row_ptr_buf);
    free(local_col_ind_buf);
    free(local_data_buf);
  }



  dist_csr_matrix_part._num_vals=local_data.size();
  dist_csr_matrix_part._num_col_indices=local_col_ind.size();
  dist_csr_matrix_part._num_row_ptrs=local_row_ptr.size();
  dist_csr_matrix_part._vals=new double[dist_csr_matrix_part._num_vals];
  dist_csr_matrix_part._col_indices=new int[dist_csr_matrix_part._num_col_indices];
  dist_csr_matrix_part._row_ptrs=new int[dist_csr_matrix_part._num_row_ptrs];
  for(int i = 0; i < dist_csr_matrix_part._num_vals; i++){
    dist_csr_matrix_part._vals[i]=local_data[i];
  }
  for(int i = 0; i < dist_csr_matrix_part._num_col_indices; i++){
    dist_csr_matrix_part._col_indices[i]=local_col_ind[i];
  }
  for(int i = 0; i < dist_csr_matrix_part._num_row_ptrs; i++){
    dist_csr_matrix_part._row_ptrs[i]=local_row_ptr[i];
  }
}


void parallel_csr_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result,
  MPI_Comm comm,
  int rank,
  int size,
  int* num_rows_arr
)
{
  int dists[size];
  int N=csr._num_row_ptrs-1;
  int num_cols_in_matrix=static_cast<int>(matrix.cols());
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> tmp_result;

  tmp_result.resize(3,N);
  tmp_result.setZero();
  for(int i = 0; i < N; i++){
      for(int k = csr._row_ptrs[i]; k < csr._row_ptrs[i+1]; k++){
      for(int j = 0; j<num_cols_in_matrix;j++){
          tmp_result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
      }
      }
  }
  MPI_Barrier(comm);
  vector<int> sizes(size);
  for(int i = 0; i < size; i++){
      sizes[i]=num_rows_arr[i]*3;
  }


  dists[0]=0;
  for(int i = 1; i < size; i++){
    dists[i]=dists[i-1]+sizes[i-1];
  }


  MPI_Allgatherv(tmp_result.data(), 3*N, MPI_DOUBLE, result.data(), sizes.data(),dists,MPI_DOUBLE, MPI_COMM_WORLD);
  result.transposeInPlace();
}



void parallel_csr_x_matrix_optimized
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result,
  Eigen::MatrixXd& tmp_result,
  MPI_Comm comm,
  int rank,
  int size,
  int* num_rows_arr
)
{
  int dists[size];
  int N=csr._num_row_ptrs-1;
  int num_cols_in_matrix=static_cast<int>(matrix.cols());
  for(int i = 0; i < N; i++){
      tmp_result(0,i)=0;
      tmp_result(1,i)=0;
      tmp_result(2,i)=0;
      for(int k = csr._row_ptrs[i]; k < csr._row_ptrs[i+1]; k++){
      for(int j = 0; j<num_cols_in_matrix;j++){
          tmp_result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
      }
      }
  }
  vector<int> sizes(size);
  for(int i = 0; i < size; i++){
      sizes[i]=num_rows_arr[i]*3;
  }


  dists[0]=0;
  for(int i = 1; i < size; i++){
    dists[i]=dists[i-1]+sizes[i-1];
  }

  MPI_Allgatherv(tmp_result.data(), 3*N, MPI_DOUBLE, result.data(), sizes.data(),dists,MPI_DOUBLE, MPI_COMM_WORLD);
  result.transposeInPlace();
}



void diag_matrix_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& result
)
{
  int N=csr._num_row_ptrs-1;
  int num_cols_in_matrix=static_cast<int>(B.cols());
  double diag_val;

  result.resize(3,N);
  for(int i = 0; i < N; i++){
      diag_val=csr.getValAt(i,i);
      for(int j = 0; j<num_cols_in_matrix;j++){
        result(j,i)=B(i,j)/diag_val;
      }
  }
  result.transposeInPlace();
}


void parallel_diag_matrix_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& result,
  MPI_Comm comm,
  int rank,
  int size,
  int* num_rows_arr,
  int upto_num_rows_dist
)
{
  int N=csr._num_row_ptrs-1;
  int num_cols_in_matrix=static_cast<int>(B.cols());
  double diag_val;
  int dists[size];
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> tmp_result;
  vector<int> sizes(size);
  for(int i = 0; i < size; i++){
      sizes[i]=num_rows_arr[i]*3;
  }

  dists[0]=0;
  for(int i = 1; i < size; i++){
    dists[i]=dists[i-1]+sizes[i-1];
  }

  tmp_result.resize(3,N);
  tmp_result.setZero();
  for(int i = 0; i < N; i++){
      diag_val=csr.getValAt(i,i+upto_num_rows_dist);
      for(int j = 0; j<num_cols_in_matrix;j++){
        tmp_result(j,i)=B(i+upto_num_rows_dist,j)/diag_val;
      }
  }
  MPI_Allgatherv(tmp_result.data(), 3*N, MPI_DOUBLE, result.data(), sizes.data(),dists,MPI_DOUBLE, MPI_COMM_WORLD);
  result.transposeInPlace();
}


void csr_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result
)
{
    int N=csr._num_row_ptrs-1;
    int num_cols_in_matrix=static_cast<int>(matrix.cols());
    for(int i = 0; i < N; i++){
        for(int k = csr._row_ptrs[i]; k < csr._row_ptrs[i+1]; k++){
        for(int j = 0; j<num_cols_in_matrix;j++){
            result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
        }
        }
    }
    result.transposeInPlace();
}


void parallel_ATxA
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& ATxA,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  ATxA.resize(m,m);
  ATxA.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }
  ATxA=A.block(start,0,end-start,m).transpose()*A.block(start,0,end-start,m);
  MPI_Allreduce(MPI_IN_PLACE,ATxA.data(),m*m,MPI_DOUBLE,MPI_SUM,comm);
}


void parallel_ATxB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& ATxB,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  int n=static_cast<int>(B.cols());
  ATxB.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }

  ATxB=A.block(start,0,end-start,m).transpose()*B.block(start,0,end-start,m);;
  MPI_Allreduce(MPI_IN_PLACE,ATxB.data(),m*n,MPI_DOUBLE,MPI_SUM,comm);
}

void parallel_ATxB_CTxD
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& D,
  Eigen::MatrixXd& ATxB_CTxD,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  int n=static_cast<int>(B.cols());
  ATxB_CTxD.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }

  ATxB_CTxD.block(0,0,3,3)=A.block(start,0,end-start,m).transpose()*B.block(start,0,end-start,m);
  ATxB_CTxD.block(3,0,3,3)=C.block(start,0,end-start,m).transpose()*D.block(start,0,end-start,m);
  MPI_Allreduce(MPI_IN_PLACE,ATxB_CTxD.data(),m*n*2,MPI_DOUBLE,MPI_SUM,comm);
}

void parallel_matrix_multiplication
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& AxB,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  int n=static_cast<int>(B.cols());
  AxB.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }
  AxB.block(start,0,end-start,m)=A.block(start,0,end-start,m)*B;
  MPI_Allreduce(MPI_IN_PLACE,AxB.data(),l*n,MPI_DOUBLE,MPI_SUM,comm);
}
void parallel_C_add_AB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& C_add_AB,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  int n=static_cast<int>(B.cols());
  C_add_AB.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }
  C_add_AB.block(start,0,end-start,m)=C.block(start,0,end-start,m)+A.block(start,0,end-start,m)*B;
  MPI_Allreduce(MPI_IN_PLACE,C_add_AB.data(),l*n,MPI_DOUBLE,MPI_SUM,comm);
}
void parallel_C_sub_AB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& C_sub_AB,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  int n=static_cast<int>(B.cols());
  C_sub_AB.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }
  C_sub_AB.block(start,0,end-start,m)=C.block(start,0,end-start,m)-A.block(start,0,end-start,m)*B;
  MPI_Allreduce(MPI_IN_PLACE,C_sub_AB.data(),l*n,MPI_DOUBLE,MPI_SUM,comm);
}

void parallel_C_add_AB_F_sub_DE
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& D,
  Eigen::MatrixXd& E,
  Eigen::MatrixXd& F,
  Eigen::MatrixXd& C_add_AB_F_sub_DE,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  int n=static_cast<int>(B.cols());
  C_add_AB_F_sub_DE.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }
  C_add_AB_F_sub_DE.block(start,0,end-start,m)=C.block(start,0,end-start,m)+A.block(start,0,end-start,m)*B;
  C_add_AB_F_sub_DE.block(l+start,0,end-start,m)=F.block(start,0,end-start,m)-D.block(start,0,end-start,m)*E;
  MPI_Allreduce(MPI_IN_PLACE,C_add_AB_F_sub_DE.data(),l*n*2,MPI_DOUBLE,MPI_SUM,comm);
}

void parallel_matrix_addition
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }
  A_B.setZero();
  A_B.block(start,0,end-start,m)=A.block(start,0,end-start,m)+B.block(start,0,end-start,m);
  MPI_Allreduce(MPI_IN_PLACE,A_B.data(),l*m,MPI_DOUBLE,MPI_SUM,comm);
}

void parallel_matrix_subtraction
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  MPI_Comm comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  A_B.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size);
  if(rank==size-1){
    end=l;
  }
  A_B.block(start,0,end-start,m)=A.block(start,0,end-start,m)-B.block(start,0,end-start,m);
  MPI_Allreduce(MPI_IN_PLACE,A_B.data(),l*m,MPI_DOUBLE,MPI_SUM,comm);
}


void parallel_sparse_block_conjugate_gradient_v2
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  MPI_Comm comm,
  int rank,
  int size,
  int num_rows
)
{
  double threshold=1E-10;
  Eigen::MatrixXd R;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Rold;
  Eigen::MatrixXd Rnew;
  Eigen::MatrixXd PAP_inv;
  Eigen::MatrixXd alpha;
  Eigen::MatrixXd AP;
  Eigen::MatrixXd PAP;
  Eigen::MatrixXd Palpha;
  Eigen::MatrixXd APalpha;
  Eigen::MatrixXd beta;
  Eigen::MatrixXd P_beta;

  int num_rows_dist=0;
  int local_num_rows=local_A._num_row_ptrs-1;
  int num_rows_arr[size];
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(comm);
  num_rows_dist = accumulate(num_rows_arr, num_rows_arr+size, num_rows_dist);



  R=global_b;
  P=R;
  parallel_ATxA(R,Rold,comm,rank,size);
  Rnew=Rold;
  MPI_Barrier(comm);
double start;
double end;

  Eigen::MatrixXd wkspc;
  wkspc.resize(X.rows(),X.cols());
  Palpha.resize(R.rows(),3);
  APalpha.resize(R.rows(),3);
  wkspc.setZero();
  int i = 0;
  double cur_err = Rnew.trace();
  while (cur_err>=threshold){
    start = MPI_Wtime();
    AP.resize(3,num_rows_dist);//Should be (num_rows_dist,3), but mpi gets mad
    AP.setZero();
    parallel_csr_x_matrix(local_A,P,AP,comm,rank,size,num_rows_arr);
    parallel_ATxB(P,AP,PAP,comm,rank,size);
    PAP_inv=PAP.inverse();
    alpha=PAP_inv*Rold;
    parallel_matrix_multiplication(P,alpha,Palpha,comm,rank,size);
    parallel_matrix_addition(X,Palpha,wkspc,comm,rank,size);X=wkspc;
    parallel_matrix_multiplication(AP,alpha,APalpha,comm,rank,size);
    parallel_matrix_subtraction(R,APalpha,wkspc,comm,rank,size);R=wkspc;
    parallel_ATxA(R,Rnew,comm,rank,size);
    beta=Rold.inverse()*Rnew;
    Rold=Rnew;
    P_beta=P*beta;
    parallel_matrix_addition(R,P_beta,P,comm,rank,size);
    cur_err = Rnew.trace();
    i++;
    end = MPI_Wtime();
    if(rank==0)cout<<"\t\tIteration "<<i<<" time: "<<end-start<<endl;
    if(rank==0)cout<<"\t\tIteration error: "<<cur_err<<"|"<<threshold<<endl;
  }
  if(rank==0)cout<<"\tIterations: "<<i<<endl;
}



void parallel_sparse_block_conjugate_gradient_v3//preconditioning
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  MPI_Comm comm,
  int rank,
  int size,
  int num_rows,
  int block_size
)
{
  double threshold=1E-10;
  Eigen::MatrixXd R;
  Eigen::MatrixXd Z;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Rold;
  Eigen::MatrixXd Rnew;
  Eigen::MatrixXd PAP_inv;
  Eigen::MatrixXd alpha;
  Eigen::MatrixXd AP;
  Eigen::MatrixXd PAP;
  Eigen::MatrixXd Palpha;
  Eigen::MatrixXd APalpha;
  Eigen::MatrixXd beta;
  Eigen::MatrixXd P_beta;
  Eigen::MatrixXd tmp_result;
  Eigen::MatrixXd wkspc;



  int num_rows_dist=0;
  int upto_num_rows_dist=0;
  int local_num_rows=local_A._num_row_ptrs-1;
  int num_rows_arr[size];
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(comm);
  num_rows_dist = accumulate(num_rows_arr, num_rows_arr+size, num_rows_dist);
  upto_num_rows_dist=accumulate(num_rows_arr, num_rows_arr+rank, upto_num_rows_dist);

double start = MPI_Wtime();
double end;
  int num_blocks=local_num_rows/block_size;




  R=global_b;
  Z.resize(3,num_rows_dist);

  parallel_diag_matrix_x_matrix(local_A,R,Z,comm,rank,size,num_rows_arr,upto_num_rows_dist);

  P=Z;
  parallel_ATxA(R,Rold,comm,rank,size);
  Rnew=Rold;
  MPI_Barrier(comm);


  tmp_result.resize(3,local_num_rows);
  wkspc.resize(X.rows(),X.cols());
  Palpha.resize(R.rows(),3);
  APalpha.resize(R.rows(),3);
  P_beta.resize(R.rows(),3);
  tmp_result.setZero();
  wkspc.setZero();
  PAP.resize(3,3);
  int i = 0;
  double cur_err = Rnew.norm();
  while (cur_err>=threshold){
    start = MPI_Wtime();
    AP.resize(3,num_rows_dist);//Should be (num_rows_dist,3), but mpi gets mad
    AP.setZero();
    parallel_ATxB(R,Z,Rold,comm,rank,size);
    parallel_csr_x_matrix_optimized(local_A,P,AP,tmp_result,comm,rank,size,num_rows_arr);
    parallel_ATxB(P,AP,PAP,comm,rank,size);
    PAP_inv=PAP.inverse();
    alpha=PAP_inv*Rold;
    parallel_matrix_multiplication(P,alpha,Palpha,comm,rank,size);
    parallel_matrix_addition(X,Palpha,wkspc,comm,rank,size);X=wkspc;
    parallel_matrix_multiplication(AP,alpha,APalpha,comm,rank,size);
    parallel_matrix_subtraction(R,APalpha,wkspc,comm,rank,size);R=wkspc;
    Z.resize(3,num_rows_dist);
    parallel_diag_matrix_x_matrix(local_A,R,Z,comm,rank,size,num_rows_arr,upto_num_rows_dist);
    parallel_ATxB(R,Z,Rnew,comm,rank,size);
    beta=Rold.inverse()*Rnew;
    Rold=Rnew;
    parallel_matrix_multiplication(P,beta,P_beta,comm,rank,size);//P_beta=P*beta;
    parallel_matrix_addition(Z,P_beta,P,comm,rank,size);
    cur_err = Rnew.norm();
    i++;
    end = MPI_Wtime();if(rank==0)cout<<"\t\tIteration "<<i<<" time: "<<end-start<<"\n";
    if(rank==0)cout<<"\t\tIteration error: "<<cur_err<<"|"<<threshold<<"\n";
  }
  if(rank==0)cout<<"\tIterations: "<<i<<endl;
}

void parallel_sparse_block_conjugate_gradient_v4//preconditioning
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  MPI_Comm comm,
  int rank,
  int size,
  int num_rows,
  int block_size
)
{
  double threshold=1E-10;
  Eigen::MatrixXd R;
  Eigen::MatrixXd Z;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Rold;
  Eigen::MatrixXd Rnew;
  Eigen::MatrixXd PAP_inv;
  Eigen::MatrixXd alpha;
  Eigen::MatrixXd AP;
  Eigen::MatrixXd PAP;
  Eigen::MatrixXd Palpha;
  Eigen::MatrixXd APalpha;
  Eigen::MatrixXd beta;
  Eigen::MatrixXd P_beta;
  Eigen::MatrixXd tmp_result;
  Eigen::MatrixXd PAP_Rold_wkspc;
  Eigen::MatrixXd X_R_wkspc;
  Eigen::MatrixXd wkspc;




  int num_rows_dist=0;
  int upto_num_rows_dist=0;
  int local_num_rows=local_A._num_row_ptrs-1;
  int num_rows_arr[size];
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(comm);
  num_rows_dist = accumulate(num_rows_arr, num_rows_arr+size, num_rows_dist);
  upto_num_rows_dist=accumulate(num_rows_arr, num_rows_arr+rank, upto_num_rows_dist);

double start = MPI_Wtime();
double end;
  int num_blocks=local_num_rows/block_size;




  R=global_b;
  Z.resize(3,num_rows_dist);

  parallel_diag_matrix_x_matrix(local_A,R,Z,comm,rank,size,num_rows_arr,upto_num_rows_dist);

  P=Z;
  parallel_ATxA(R,Rold,comm,rank,size);
  Rnew=Rold;
  MPI_Barrier(comm);


  tmp_result.resize(3,local_num_rows);
  wkspc.resize(X.rows(),X.cols());
  Palpha.resize(R.rows(),3);
  APalpha.resize(R.rows(),3);
  P_beta.resize(R.rows(),3);
  tmp_result.setZero();
  wkspc.setZero();
  PAP.resize(3,3);
  PAP_Rold_wkspc.resize(6,3);
  X_R_wkspc.resize(X.rows()+R.rows(),X.cols());
  int i = 0;
  double cur_err = Rnew.norm();
  while (cur_err>=threshold){
    start = MPI_Wtime();
    AP.resize(3,num_rows_dist);//Should be (num_rows_dist,3), but mpi gets mad
    AP.setZero();
    parallel_csr_x_matrix_optimized(local_A,P,AP,tmp_result,comm,rank,size,num_rows_arr);
    parallel_ATxB_CTxD(P,AP,R,Z,PAP_Rold_wkspc,comm,rank,size);PAP=PAP_Rold_wkspc.block(0,0,3,3);Rold=PAP_Rold_wkspc.block(3,0,3,3);
    PAP_inv=PAP.inverse();
    alpha=PAP_inv*Rold;
    parallel_C_add_AB_F_sub_DE(P,alpha,X,AP,alpha,R,X_R_wkspc,comm,rank,size);X=X_R_wkspc.block(0,0,X.rows(),X.cols());R=X_R_wkspc.block(X.rows(),0,R.rows(),R.cols());
    Z.resize(3,num_rows_dist);
    parallel_diag_matrix_x_matrix(local_A,R,Z,comm,rank,size,num_rows_arr,upto_num_rows_dist);
    parallel_ATxB(R,Z,Rnew,comm,rank,size);
    beta=Rold.inverse()*Rnew;
    Rold=Rnew;
    parallel_C_add_AB(P,beta,Z,wkspc,comm,rank,size);P=wkspc;
    cur_err = Rnew.norm();
    i++;
    end = MPI_Wtime();if(rank==0)cout<<"\t\tIteration "<<i<<" time: "<<end-start<<endl;
    if(rank==0)cout<<"\t\tIteration error: "<<cur_err<<"|"<<threshold<<endl;
  }
  if(rank==0)cout<<"\tIterations: "<<i<<endl;
}




void sparse_block_conjugate_gradient_v2
(
  CSR_Matrix& A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X
)
{
  double threshold=1E-10;
  Eigen::MatrixXd R;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Rold;
  Eigen::MatrixXd Rnew;
  Eigen::MatrixXd PAP_inv;
  Eigen::MatrixXd alpha;
  Eigen::MatrixXd AP;
  Eigen::MatrixXd PAP;
  Eigen::MatrixXd Palpha;
  Eigen::MatrixXd APalpha;
  Eigen::MatrixXd beta;
  Eigen::MatrixXd P_beta;

  R=global_b;
  P=R;
  Rold=R.transpose()*R;
  Rnew=Rold;
  int i=0;
  double cur_err = Rnew.trace();
  double start;
  double end;

  while (cur_err>=threshold){
    start = MPI_Wtime();
    AP.resize(3,A._num_row_ptrs-1);
    AP.setZero();
    csr_x_matrix(A,P,AP);
    PAP=P.transpose()*AP;
    PAP_inv=PAP.inverse();
    alpha=PAP_inv*Rold;
    Palpha=P*alpha;
    X+=Palpha;
    APalpha=AP*alpha;
    R-=APalpha;
    Rnew=R.transpose()*R;
    beta=Rold.inverse()*Rnew;
    Rold=Rnew;
    P_beta=P*beta;
    P=R+P_beta;
    cur_err = Rnew.trace();
    i++;
    end = MPI_Wtime();cout<<"\t\tIteration "<<i<<" time: "<<end-start<<endl;
    cout<<"\t\tIteration error: "<<cur_err<<"|"<<threshold<<endl;
  }
  cout<<"\tIterations: "<<i<<endl;
}

void sparse_block_conjugate_gradient_v3//preconditioning jacobi
(
  CSR_Matrix& A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X
)
{
  double threshold=1E-10;
  Eigen::MatrixXd R;
  Eigen::MatrixXd Z;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Rold;
  Eigen::MatrixXd Rnew;
  Eigen::MatrixXd PAP_inv;
  Eigen::MatrixXd alpha;
  Eigen::MatrixXd AP;
  Eigen::MatrixXd PAP;
  Eigen::MatrixXd Palpha;
  Eigen::MatrixXd APalpha;
  Eigen::MatrixXd beta;
  Eigen::MatrixXd P_beta;

  R=global_b;
  diag_matrix_x_matrix(A,R,Z);
  P=Z;
  Rold=R.transpose()*Z;
  Rnew=Rold;
  int i=0;
  double cur_err = Rnew.trace();
  double start;
  double end;


  while (cur_err>=threshold){
    start = MPI_Wtime();
    AP.resize(3,A._num_row_ptrs-1);
    AP.setZero();
    csr_x_matrix(A,P,AP);
    PAP=P.transpose()*AP;
    PAP_inv=PAP.inverse();
    alpha=PAP_inv*Rold;
    Palpha=P*alpha;
    X+=Palpha;
    APalpha=AP*alpha;
    R-=APalpha;
    diag_matrix_x_matrix(A,R,Z);
    Rnew=R.transpose()*Z;
    beta=Rold.inverse()*Rnew;
    Rold=Rnew;
    P_beta=P*beta;
    P=Z+P_beta;
    cur_err = Rnew.norm();
    i++;
    end = MPI_Wtime();cout<<"\t\tIteration "<<i<<" time: "<<end-start<<endl;
    cout<<"\t\tIteration error: "<<cur_err<<"|"<<threshold<<endl;
  }
  cout<<"\tIterations: "<<i<<endl;
}

void icf_forward_solve
(
  CSR_Matrix& L,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& Y
)
{

/* TODO */
}

void icf_backward_solve
(
  CSR_Matrix& U,
  Eigen::MatrixXd& Y,
  Eigen::MatrixXd& X
)
{

/* TODO */
}

void icf_solve
(
  CSR_Matrix& M,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& Y,
  Eigen::MatrixXd& X
)
{
  icf_forward_solve(M,B,Y);
  icf_backward_solve(M,Y,X);
}

void icf
(
  CSR_Matrix& A,
  CSR_Matrix& M
)
{

/* TODO */
}


void sparse_block_conjugate_gradient_v4//preconditioning ????
(
  CSR_Matrix& A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X
)
{
/* TODO */
}