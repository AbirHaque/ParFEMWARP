
#define _USE_MATH_DEFINES


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

#define M_PI_3 M_PI/3


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
  int num_rows_arr[],
  bool transpose=true
)
{
  int N=csr._num_row_ptrs-1;
  int num_cols_in_matrix=static_cast<int>(matrix.cols());
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> tmp_result;
  
  if(transpose){
    tmp_result.resize(3,N);
    tmp_result.setZero();
    for(int i = 0; i < N; i++){
      for(int k = csr._row_ptrs[i]; k < csr._row_ptrs[i+1]; k++){
        for(int j = 0; j<num_cols_in_matrix;j++){
          tmp_result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
          tmp_result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
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
  else{
    tmp_result.resize(3,N);
    tmp_result.setZero();
    for(int i = 0; i < N; i++){
      for(int k = csr._row_ptrs[i]; k < csr._row_ptrs[i+1]; k++){
        for(int j = 0; j<num_cols_in_matrix;j++){
          tmp_result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
          tmp_result(j,i)+=matrix(csr._col_indices[k],j)*csr._vals[k];
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
  }
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
  AxB.resize(l,n);
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
  A_B.resize(l,m);
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
  A_B.resize(l,m);
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
void sparse_block_conjugate_gradient_v2
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
  wkspc.setZero();
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
  }
}

void femwarp3d
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  boost::mpi::communicator comm,
  int size,
  int rank,
  Eigen::MatrixXd& Z_femwarp_transformed
)
{
  //Generate global stiffness matrix:
  int T_parts=num_eles/size;
  int low=rank*T_parts;
  int high=(rank+1)*T_parts;
  if (rank==size-1){
    high=num_eles;
  }
  unordered_map<int, set<int>> local_neighbors;
  bool done = false;
  //#pragma omp parallel private(done)
  {
    //#pragma omp for
    for(int i = low; i<high;i++){//O(n)
      if(done){
        continue;
      }
      if(T(i,0)==T(i,1)){
        done=true;
        break;
      }
      for(int j=0;j<4;j++){
        if(local_neighbors.find(T(i,j))==local_neighbors.end() && T(i,j)<m){ //Hashtable look-up is O(1) average
          local_neighbors[T(i,j)]; //O(1) average
        }
      }
      for(int j=0;j<4;j++){
        for(int k=0;k<4;k++){
          if(T(i,j)<m){
            local_neighbors[T(i,j)].insert(T(i,k));
          }
        }
      }
    }
  }
  
  unordered_map<int, set<int>> master_neighbor_dict[size];
  ostringstream oss;
  boost::archive::text_oarchive archive_out(oss);
  archive_out << local_neighbors;
  string send_str = oss.str();
  vector<string> rec_str;
  boost::mpi::all_gather(comm, send_str,rec_str);
  comm.barrier();
  for(int i = 0; i < size;i++){
    istringstream iss(rec_str[i]);
    boost::archive::text_iarchive archive_in(iss);
    archive_in>>master_neighbor_dict[i];      
  };
  unordered_map<int, set<int>> AI_dict;
  unordered_map<int, set<int>> AB_dict;
  int num_vals_in_AI=0;
  int num_vals_in_AB=0;
  for(int i = 0; i < size; i++){
    for (unordered_map<int, set<int>>::iterator key_node = master_neighbor_dict[i].begin(); key_node!= master_neighbor_dict[i].end(); key_node++){
      if(AI_dict.find(key_node->first)==AI_dict.end()){//(key_node not in AI_dict:
        AI_dict[key_node->first];
      }
      if(AB_dict.find(key_node->first)==AB_dict.end()){
        AB_dict[key_node->first];
      }
      for(set<int>::iterator val_node = master_neighbor_dict[i][key_node->first].begin(); val_node!= master_neighbor_dict[i][key_node->first].end(); val_node++){
        if (*val_node <m){
          if (AI_dict[key_node->first].insert(*val_node).second){ //Union of the sets
            num_vals_in_AI+=1;
          }
        }
        else{
          if(AB_dict[key_node->first].insert(*val_node).second){
            num_vals_in_AB+=1;
          }
        }
      }
    }
  }
  comm.barrier();
  CSR_Matrix<double> AI_matrix(num_vals_in_AI,num_vals_in_AI,AI_dict.size()+1);
  CSR_Matrix<double> AB_matrix(num_vals_in_AB,num_vals_in_AB,AB_dict.size()+1);
  gen_neighbors_csr(AI_dict,AI_matrix);
  gen_neighbors_csr(AB_dict,AB_matrix,m);
  comm.barrier();

  low=rank*T_parts;
  high= rank!=(size-1) ? (rank+1)*T_parts : num_eles;
  
  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];
  
  for (int n_ele=low; n_ele<high; n_ele++){
    a_func[0]=  Z_original(T(n_ele,0),0);
    a_func[1]=  Z_original(T(n_ele,0),1);
    a_func[2]=  Z_original(T(n_ele,0),2);
    b_func[0]=  Z_original(T(n_ele,1),0);
    b_func[1]=  Z_original(T(n_ele,1),1);
    b_func[2]=  Z_original(T(n_ele,1),2);
    c_func[0]=  Z_original(T(n_ele,2),0);
    c_func[1]=  Z_original(T(n_ele,2),1);
    c_func[2]=  Z_original(T(n_ele,2),2);
    d_func[0]=  Z_original(T(n_ele,3),0);
    d_func[1]=  Z_original(T(n_ele,3),1);
    d_func[2]=  Z_original(T(n_ele,3),2);
    gen_basis_function_gradients_tetrahedron(a_func,b_func,c_func,d_func,basis_function_gradients);
    element_stiffness_matrix=gen_element_stiffness_matrix_tetrahedron(basis_function_gradients,a_func,b_func,c_func,d_func);
    for (int i = 0; i<4; i++){
      for (int j = 0; j<4; j++){
        if (T(n_ele,i)<m){
          if (T(n_ele,j)>=m){
            if (element_stiffness_matrix(i,j)!=0){
              AB_matrix.subractValAt(T(n_ele,i),T(n_ele,j)-m,element_stiffness_matrix(i,j));
            }
          }
          else{
            if (element_stiffness_matrix(i,j)!=0){
              AI_matrix.addValAt(T(n_ele,i),T(n_ele,j),element_stiffness_matrix(i,j));
            }
          }
        }
      }
    }
  }
  comm.barrier();
  double tmpI[AI_matrix._num_vals];
  for(int i = 0; i < AI_matrix._num_vals;i++){
    tmpI[i]=AI_matrix._vals[i];
  }
  double tmpB[AB_matrix._num_vals];
  for(int i = 0; i < AB_matrix._num_vals;i++){
    tmpB[i]=AB_matrix._vals[i];
  }
  MPI_Allreduce(tmpI,AI_matrix._vals,AI_matrix._num_vals,MPI_DOUBLE,MPI_SUM,comm);
  MPI_Allreduce(tmpB,AB_matrix._vals,AB_matrix._num_vals,MPI_DOUBLE,MPI_SUM,comm);
  comm.barrier();

  CSR_Matrix<double> local_A_I(0,0,0);
  CSR_Matrix<double> local_A_B(0,0,0);
  csr_to_dist_csr(AI_matrix,local_A_I,m,comm,size,rank);
  csr_to_dist_csr(AB_matrix,local_A_B,m,comm,size,rank);
  comm.barrier();

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  int num_rows=0;
  int local_num_rows=local_A_B._num_row_ptrs-1;
  int num_rows_arr[size];
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  comm.barrier();
  num_rows = accumulate(num_rows_arr, num_rows_arr+size, num_rows);
  A_B_hat.resize(3,num_rows);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();
  comm.barrier();
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  comm.barrier();


  sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows);
  comm.barrier();
}


int main(int argc, char *argv[]){

  boost::mpi::environment env{argc, argv};
  boost::mpi::communicator comm;
  
  int n;//Number nodes
  int m;//Number of interior nodes
  int b;//Number of boundary nodes
  int offset;//n-1
  int num_eles;
  int i;//index variable
  int j;//index variable
  int k;//index variable
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_full_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::AngleAxisd rotate_transform (M_PI_3,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> rotate_transform_matrix_transpose;

  int size = comm.size();
  int rank = comm.rank();


  if(rank==0){
    fstream in_file;
    stringstream ss;
    string word;
    string orig_mesh_name=/*"formatted_heart_5M";*/"body_has_heart";
    string new_mesh_name=/*"new_formatted_heart_5M";*/"new_body_has_heart";
    string in_line="#";

    in_file.open("./tetgen_meshes/"+orig_mesh_name+".smesh", ios::in);
    while(in_line[0]=='#'||in_line[0]==' '||in_line[0]=='\0'||in_line[0]=='0'||in_line[0]=='\n'||in_line[0]=='\r'){
      getline(in_file,in_line);
    }
    ss.str(in_line);
    ss>>word;
    in_file.close();  
    b=stoi(word);
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".node", ios::in);
    in_line="#";
    while(in_line[0]=='#'||in_line[0]==' '||in_line[0]=='\0'){
      getline(in_file,in_line);
    }
    ss.clear();
    ss.str(in_line);
    ss>>word;
    n=stoi(word);
    m=n-b;
    offset = n-1;
    Z_original.resize(n,3);
    Z_boundary_transformation.resize(b,3);
    Z_full_transformation.resize(n,3);
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
    for(i = 0; i < n;i++){
      getline(in_file,in_line);
      ss.clear();
      ss.str(in_line);
      ss>>word;//skip index
      ss>>word;Z_original(offset-i,0)=stod(word);
      ss>>word;Z_original(offset-i,1)=stod(word);
      ss>>word;Z_original(offset-i,2)=stod(word);
      }

    in_file.close();  

    in_file.open("./tetgen_meshes/"+orig_mesh_name+".ele", ios::in);
    in_line="#";
    while(in_line[0]=='#'||in_line[0]==' '||in_line[0]=='\0'){
      getline(in_file,in_line);
    }
    ss.clear();
    ss.str(in_line);
    ss>>word;
    num_eles=stoi(word);
    T.resize(num_eles,4);
    for(i = 0; i < num_eles;i++){
      getline(in_file,in_line);
      ss.clear();
      ss.str(in_line);
      ss>>word;//skip index
      for(j = 0; j < 4; j++){
        ss>>word;T(i,j)=offset-stoi(word);
      }
      
    }

    in_file.close();
    rotate_transform_matrix_transpose = rotate_transform.matrix().transpose();
    Z_full_transformation=Z_original*rotate_transform_matrix_transpose;
    Z_boundary_transformation=Z_original.block(n-b,0,b,3);//Get only boundary nodes
    Z_boundary_transformation*=rotate_transform_matrix_transpose;
  }

  comm.barrier();
  broadcast(comm,n,0);
  broadcast(comm,m,0);
  broadcast(comm,b,0);
  broadcast(comm,num_eles,0);
  comm.barrier();
  if(rank!=0){
    Z_original.resize(n,3);
    Z_boundary_transformation.resize(b,3);
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
    T.resize(num_eles,4);
  }
  comm.barrier();
  broadcast(comm,Z_original.data(),n*3,0);
  broadcast(comm,Z_boundary_transformation.data(),b*3,0);
  broadcast(comm,T.data(),num_eles*4,0); 
  double start,end;
  comm.barrier();
  start = MPI_Wtime();
  
  femwarp3d(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed);
  comm.barrier();
  end  = MPI_Wtime();
  
  if(rank==0){
    //cout<<Z_femwarp_transformed<<endl; */
    cout<<end-start<<endl;
  }



  MPI_Finalize();
}