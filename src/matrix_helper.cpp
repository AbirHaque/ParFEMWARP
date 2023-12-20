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
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/mpi/collectives/all_gather.hpp>
#include <boost/mpi/collectives/all_gatherv.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include "./csr.hpp"


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
  CSR_Matrix<double>& csr_matrix,
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
  CSR_Matrix<double>& csr_matrix,
  CSR_Matrix<double>& dist_csr_matrix_part,
  int rows,
  boost::mpi::communicator comm,
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
  comm.barrier();
  vector<vector<int>> local_row_ptrs;
  vector<vector<int>> local_col_inds;
  vector<vector<double>> local_datas;
  if (rank==0){
    int col_offset=0;
    for(int i = 0; i < size;i++){
      int start_offset=vtxdist[i];
      int end_offset=vtxdist[i+1]+1;
      vector<int> tmp1(end_offset-start_offset);
      for(int j = start_offset; j < end_offset; j++){
        tmp1[j-start_offset]=csr_matrix._row_ptrs[j]-csr_matrix._row_ptrs[start_offset];
      }
      local_row_ptrs.push_back(tmp1);
      int tmp23_size=local_row_ptrs[i].back()-local_row_ptrs[i][0];
      vector<int> tmp2(tmp23_size);
      vector<double> tmp3(tmp23_size);
      //cout<<local_row_ptrs[i][0]+col_offset<<" "<<local_row_ptrs[i].back()+col_offset<<endl;
      for(int j = local_row_ptrs[i][0]; j < local_row_ptrs[i].back(); j++){
        tmp2[j]=csr_matrix._col_indices[j+col_offset];
        tmp3[j]=csr_matrix._vals[j+col_offset];
      }
      local_col_inds.push_back(tmp2);
      local_datas.push_back(tmp3);
      col_offset+=local_row_ptrs.back().back();
    }
  }
  comm.barrier();
  vector<int> local_row_ptr;
  vector<int> local_col_ind;
  vector<double> local_data;
  boost::mpi::scatter(comm,local_row_ptrs,local_row_ptr,0);
  boost::mpi::scatter(comm,local_col_inds,local_col_ind,0);
  boost::mpi::scatter(comm,local_datas,local_data,0);
  //cout<<local_data.size()<<endl;
  dist_csr_matrix_part._num_vals=local_data.size();
  dist_csr_matrix_part._num_col_indices=local_col_ind.size();
  dist_csr_matrix_part._num_row_ptrs=local_row_ptr.size();
  dist_csr_matrix_part._vals=new double[dist_csr_matrix_part._num_vals];
  dist_csr_matrix_part._col_indices=new int[dist_csr_matrix_part._num_col_indices];
  dist_csr_matrix_part._row_ptrs=new int[dist_csr_matrix_part._num_row_ptrs];
  for(int i = 0; i < dist_csr_matrix_part._num_vals; i++){
    dist_csr_matrix_part._vals[i]=local_data[i];
  }
  //cout<<dist_csr_matrix_part._num_vals<<endl;
  for(int i = 0; i < dist_csr_matrix_part._num_col_indices; i++){
    dist_csr_matrix_part._col_indices[i]=local_col_ind[i];
  }
  for(int i = 0; i < dist_csr_matrix_part._num_row_ptrs; i++){
    dist_csr_matrix_part._row_ptrs[i]=local_row_ptr[i];
  }
}


void parallel_csr_x_matrix
(
  CSR_Matrix<double>& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result,
  boost::mpi::communicator comm,
  int rank,
  int size,
  int num_rows_arr[]
)
{
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
    comm.barrier();
    vector<int> sizes(size);
    for(int i = 0; i < size; i++){
        sizes[i]=num_rows_arr[i]*3;
    }
    boost::mpi::all_gatherv(comm, tmp_result.data(), result.data(),sizes);
    comm.barrier();
    result.transposeInPlace();
}






void parallel_csr_x_matrix_shared_mem
(
  CSR_Matrix<double>& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result,
  boost::mpi::communicator comm,
  int rank,
  int size,
  int num_rows_arr[],
  int num_rows
)
{
    /*TODO int N=csr._num_row_ptrs-1;
    int num_cols_in_matrix=static_cast<int>(matrix.cols());
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> tmp_result;
    if(rank==0){
      tmp_result.resize(3,num_rows);
      tmp_result.setZero();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win tmp_result_win;
    MPI_Win_create(tmp_result.data(), sizeof(double) * num_rows * 3, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &tmp_result);
    MPI_Win_fence(0, tmp_result);
    /*need to offset.
    for(int i = 0; i < N; i++){
        for(int k = csr._row_ptrs[i]; k < csr._row_ptrs[i+1]; k++){
        for(int j = 0; j<num_cols_in_matrix;j++){
            tmp_result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
        }
        }
    }
    comm.barrier();
    vector<int> sizes(size);
    for(int i = 0; i < size; i++){
        sizes[i]=num_rows_arr[i]*3;
    }
    boost::mpi::all_gatherv(comm, tmp_result.data(), result.data(),sizes);
    comm.barrier();
    result.transposeInPlace();*/
}


void csr_x_matrix
(
  CSR_Matrix<double>& csr,
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
            result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
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
  boost::mpi::communicator comm,
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
  for(int i = start; i < end; i++){
    for(int j = 0; j < m; j++){
      for(int k = 0; k < m; k++){
        ATxA(j,k)+=A(i,j)*A(i,k);
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,ATxA.data(),m*m,MPI_DOUBLE,MPI_SUM,comm);
}


void parallel_ATxB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& ATxB,
  boost::mpi::communicator comm,
  int rank,
  int size
)
{
  int l=static_cast<int>(A.rows());
  int m=static_cast<int>(A.cols());
  int n=static_cast<int>(B.cols());
  ATxB.resize(m,n);
  ATxB.setZero();
  int start = rank*(l/size);
  int end = (rank+1)*(l/size); 
  if(rank==size-1){
    end=l;
  }
  for(int k = start;k<end;k++){
    for(int i = 0;i<m;i++){
      for(int j = 0;j<n;j++){
        ATxB(i,j)+=A(k,i)*B(k,j);
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,ATxB.data(),m*n,MPI_DOUBLE,MPI_SUM,comm);
}


void parallel_matrix_multiplication
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& AxB,
  boost::mpi::communicator comm,
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
  for(int i = start; i < end; i++){
    for(int j = 0; j < n; j++){
      for(int k = 0; k < m; k++){
        AxB(i,j)+=A(i,k)*B(k,j);
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,AxB.data(),l*n,MPI_DOUBLE,MPI_SUM,comm);
}


void parallel_matrix_addition
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  boost::mpi::communicator comm,
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
  for(int i = start; i < end; i++){
    for(int j = 0; j < m; j++){
      A_B(i,j)=A(i,j)+B(i,j);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,A_B.data(),l*m,MPI_DOUBLE,MPI_SUM,comm);
}

void parallel_matrix_subtraction
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  boost::mpi::communicator comm,
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
  for(int i = start; i < end; i++){
    for(int j = 0; j < m; j++){
      A_B(i,j)=A(i,j)-B(i,j);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,A_B.data(),l*m,MPI_DOUBLE,MPI_SUM,comm);
}


void parallel_sparse_block_conjugate_gradient_v2
(
  CSR_Matrix<double>& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  boost::mpi::communicator comm,
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
  comm.barrier();
  num_rows_dist = accumulate(num_rows_arr, num_rows_arr+size, num_rows_dist);



  R=global_b;
  P=R;
  parallel_ATxA(R,Rold,comm,rank,size);
  Rnew=Rold;
  comm.barrier();


  Eigen::MatrixXd wkspc;
  wkspc.resize(X.rows(),X.cols());
  Palpha.resize(R.rows(),3);
  APalpha.resize(R.rows(),3);
  wkspc.setZero();
  int i = 0;
  while (Rnew.trace()>=threshold){
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
    i++;
  }
  //cout<<i<<endl;
}



void parallel_preconditioned_sparse_block_conjugate_gradient_v2_icf
(
  CSR_Matrix<double>& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  Eigen::MatrixXd& L_inv,
  Eigen::MatrixXd& L_inv_transpose,
  boost::mpi::communicator comm,
  int rank,
  int size,
  int num_rows
)
{
  double threshold=1E-10;
  Eigen::MatrixXd R;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Z;
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
  comm.barrier();
  num_rows_dist = accumulate(num_rows_arr, num_rows_arr+size, num_rows_dist);


  R=global_b;//No initial guess.
  Z=L_inv_transpose*(L_inv*R);
  P=/*R;*/Z;
  parallel_ATxA(R,Rold,comm,rank,size);
  Rnew=Rold;
  comm.barrier();


  Eigen::MatrixXd wkspc;
  wkspc.resize(X.rows(),X.cols());
  Palpha.resize(R.rows(),3);
  APalpha.resize(R.rows(),3);
  wkspc.setZero();
  int i = 0;
  while(Rnew.trace()>=threshold){
    AP.resize(3,num_rows_dist);//Should be (num_rows_dist,3), but mpi gets mad
    AP.setZero();
    parallel_ATxB(R,Z,Rold,comm,rank,size);
    parallel_csr_x_matrix(local_A,P,AP,comm,rank,size,num_rows_arr);
    parallel_ATxB(P,AP,PAP,comm,rank,size);
    PAP_inv=PAP.inverse();
    alpha=PAP_inv*Rold;
    parallel_matrix_multiplication(P,alpha,Palpha,comm,rank,size);
    parallel_matrix_addition(X,Palpha,wkspc,comm,rank,size);X=wkspc;
    parallel_matrix_multiplication(AP,alpha,APalpha,comm,rank,size);
    parallel_matrix_subtraction(R,APalpha,wkspc,comm,rank,size);R=wkspc;
    Z=L_inv_transpose*(L_inv*R);
    /*parallel_ATxA(R,Rnew,comm,rank,size);*/parallel_ATxB(R,Z,Rnew,comm,rank,size);
    beta=Rold.inverse()*Rnew;
    Rold=Rnew;
    P_beta=P*beta;
    parallel_matrix_addition(Z/*R*/,P_beta,P,comm,rank,size);
    i++;
  }
  cout<<i<<endl;

}




void sparse_block_conjugate_gradient_v2
(
  CSR_Matrix<double>& A,
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


  while (Rnew.trace()>=threshold){
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
  }
}

