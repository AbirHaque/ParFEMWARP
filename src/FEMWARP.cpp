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
#include <Eigen/Sparse> 
#include "./csr.hpp"
#include "./matrix_helper.hpp"

#define MAX_TET_NODE_NEIGHBORS (40)
#define MAX_TET_NODE_NEIGHBORS_BUF (MAX_TET_NODE_NEIGHBORS+1)

using namespace std;

void distributed_femwarp3d
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
  for(int i = low; i<high;i++){//O(n)
    if(T(i,0)==T(i,1)){
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
  /*comm.barrier();
  cout<<"master_neighbor_dict"<<endl;*/
  unordered_map<int, set<int>> master_neighbor_dict[size];
  ostringstream oss;
  boost::archive::text_oarchive archive_out(oss);
  archive_out << local_neighbors;
  
  /*cout<<"clearing local_neighbors"<<endl;
  for (auto& local_set : local_neighbors) {
    local_set.second.clear();
    set<int>().swap(local_set.second);
  }
  local_neighbors.clear();
  comm.barrier();
  
  cout<<"all_gather"<<endl;*/
  string send_str = oss.str();
  vector<string> rec_str;
  boost::mpi::all_gather(comm, send_str,rec_str);
  /*send_str.clear();*/
  comm.barrier();
  /*cout<<"archive_in"<<endl;*/
  for(int i = 0; i < size;i++){
    istringstream iss(rec_str[i]);
    /*rec_str[i].clear();*/
    boost::archive::text_iarchive archive_in(iss);
    archive_in>>master_neighbor_dict[i];      
  };
  /*rec_str.clear();*/
  comm.barrier();
  //cout<<"master_neighbor_dict[i]"<<endl;
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
  //cout<<"FEM"<<endl;
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

  //cout<<"csr_to_dist_csr"<<endl;
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
  //cout<<"parallel_csr_x_matrix"<<endl;
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  comm.barrier();


  //cout<<"parallel_sparse_block_conjugate_gradient_v2"<<endl;
  parallel_sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows);
  comm.barrier();
}


void distributed_femwarp3d_shared_mem
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
  MPI_Win T_win;
  MPI_Win_create(T.data(), sizeof(int) * num_eles * 4, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &T_win);
  //cout<<"local_neighbors"<<endl;
  MPI_Win_fence(0, T_win);
  int T_i_0;
  int T_i_1;
  int T_i_j;
  int T_i_k;
  for(int i = low; i<high;i++){//O(n)
    MPI_Get(&T_i_0, 1, MPI_INT, 0, i*4, 1, MPI_INT, T_win);
    MPI_Get(&T_i_1, 1, MPI_INT, 0, i*4+1, 1, MPI_INT, T_win);
    if(T_i_0==T_i_1){
      break;
    }
    for(int j=0;j<4;j++){
      MPI_Get(&T_i_j, 1, MPI_INT, 0, i*4+j, 1, MPI_INT, T_win);
      if(local_neighbors.find(T_i_j)==local_neighbors.end() && T_i_j<m){ //Hashtable look-up is O(1) average
        local_neighbors[T_i_j]; //O(1) average
      }
    }
    for(int j=0;j<4;j++){
      MPI_Get(&T_i_j, 1, MPI_INT, 0, i*4+j, 1, MPI_INT, T_win);
      for(int k=0;k<4;k++){
        if(T_i_j<m){
          MPI_Get(&T_i_k, 1, MPI_INT, 0, i*4+k, 1, MPI_INT, T_win);
          local_neighbors[T_i_j].insert(T_i_k);
        }
      }
    }
  }
  /*comm.barrier();
  cout<<"master_neighbor_dict"<<endl;*/
  unordered_map<int, set<int>> master_neighbor_dict[size];
  ostringstream oss;
  boost::archive::text_oarchive archive_out(oss);
  archive_out << local_neighbors;
  
  /*cout<<"clearing local_neighbors"<<endl;
  for (auto& local_set : local_neighbors) {
    local_set.second.clear();
    set<int>().swap(local_set.second);
  }
  local_neighbors.clear();
  comm.barrier();
  
  cout<<"all_gather"<<endl;*/
  string send_str = oss.str();
  vector<string> rec_str;
  boost::mpi::all_gather(comm, send_str,rec_str);
  /*send_str.clear();*/
  comm.barrier();
  /*cout<<"archive_in"<<endl;*/
  for(int i = 0; i < size;i++){
    istringstream iss(rec_str[i]);
    /*rec_str[i].clear();*/
    boost::archive::text_iarchive archive_in(iss);
    archive_in>>master_neighbor_dict[i];      
  };
  /*rec_str.clear();*/
  comm.barrier();
  //cout<<"master_neighbor_dict[i]"<<endl;
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
  int T_ele[4];
  //cout<<"FEM"<<endl;
  for (int n_ele=low; n_ele<high; n_ele++){
    
    MPI_Get(&T_ele, 4, MPI_INT, 0, n_ele*4, 4, MPI_INT, T_win);
    a_func[0]=  Z_original(T_ele[0],0);
    a_func[1]=  Z_original(T_ele[0],1);
    a_func[2]=  Z_original(T_ele[0],2);
    b_func[0]=  Z_original(T_ele[1],0);
    b_func[1]=  Z_original(T_ele[1],1);
    b_func[2]=  Z_original(T_ele[1],2);
    c_func[0]=  Z_original(T_ele[2],0);
    c_func[1]=  Z_original(T_ele[2],1);
    c_func[2]=  Z_original(T_ele[2],2);
    d_func[0]=  Z_original(T_ele[3],0);
    d_func[1]=  Z_original(T_ele[3],1);
    d_func[2]=  Z_original(T_ele[3],2);
    gen_basis_function_gradients_tetrahedron(a_func,b_func,c_func,d_func,basis_function_gradients);
    element_stiffness_matrix=gen_element_stiffness_matrix_tetrahedron(basis_function_gradients,a_func,b_func,c_func,d_func);
    for (int i = 0; i<4; i++){
      for (int j = 0; j<4; j++){
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
  MPI_Win_fence(0, T_win);
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

  //cout<<"csr_to_dist_csr"<<endl;
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
  //cout<<"parallel_csr_x_matrix"<<endl;
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  comm.barrier();


  //cout<<"parallel_sparse_block_conjugate_gradient_v2"<<endl;
  parallel_sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows);
  comm.barrier();
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
  //Generate global stiffness matrix:
  unordered_map<int, set<int>> master_neighbor_dict;
  bool done = false;
  //#pragma omp parallel private(done)
  {
    //#pragma omp for
    for(int i = 0; i<num_eles;i++){//O(n)
      if(done){
        continue;
      }
      if(T(i,0)==T(i,1)){
        done=true;
        break;
      }
      for(int j=0;j<4;j++){
        if(master_neighbor_dict.find(T(i,j))==master_neighbor_dict.end() && T(i,j)<m){ //Hashtable look-up is O(1) average
          master_neighbor_dict[T(i,j)]; //O(1) average
        }
      }
      for(int j=0;j<4;j++){
        for(int k=0;k<4;k++){
          if(T(i,j)<m){
            master_neighbor_dict[T(i,j)].insert(T(i,k));
          }
        }
      }
    }
  }
  unordered_map<int, set<int>> AI_dict;
  unordered_map<int, set<int>> AB_dict;
  int num_vals_in_AI=0;
  int num_vals_in_AB=0;
  for (unordered_map<int, set<int>>::iterator key_node = master_neighbor_dict.begin(); key_node!= master_neighbor_dict.end(); key_node++){
    if(AI_dict.find(key_node->first)==AI_dict.end()){//(key_node not in AI_dict:
      AI_dict[key_node->first];
    }
    if(AB_dict.find(key_node->first)==AB_dict.end()){
      AB_dict[key_node->first];
    }
    for(set<int>::iterator val_node = master_neighbor_dict[key_node->first].begin(); val_node!= master_neighbor_dict[key_node->first].end(); val_node++){
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
  CSR_Matrix<double> AI_matrix(num_vals_in_AI,num_vals_in_AI,AI_dict.size()+1);
  CSR_Matrix<double> AB_matrix(num_vals_in_AB,num_vals_in_AB,AB_dict.size()+1);
  gen_neighbors_csr(AI_dict,AI_matrix);
  gen_neighbors_csr(AB_dict,AB_matrix,m);

  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];
  
  for (int n_ele=0; n_ele<num_eles; n_ele++){
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
  
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  A_B_hat.resize(3,AI_matrix._num_row_ptrs-1);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();

  csr_x_matrix(AB_matrix,xy_prime,A_B_hat);

  sparse_block_conjugate_gradient_v2(AI_matrix,A_B_hat,Z_femwarp_transformed);
}

void serial_femwarp3d_no_precompute_eigen
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  Eigen::MatrixXd& Z_femwarp_transformed
)
{
  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  cout<<"!"<<endl;
  Eigen::SparseMatrix<double,Eigen::RowMajor> AI_matrix(m,m);
  Eigen::SparseMatrix<double,Eigen::RowMajor> AB_matrix(m,b);
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];
  cout<<"!"<<endl;
  for (int n_ele=0; n_ele<num_eles; n_ele++){
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
              AB_matrix.coeffRef(T(n_ele,i),T(n_ele,j)-m)-=element_stiffness_matrix(i,j);
            }
          }
          else{
            if (element_stiffness_matrix(i,j)!=0){
              AI_matrix.coeffRef(T(n_ele,i),T(n_ele,j))+=element_stiffness_matrix(i,j);
            }
          }
        }
      }
    }
  }
  cout<<"!"<<endl;
  
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  ///A_B_hat.resize(3,AI_matrix._num_row_ptrs-1);//Should be (num_rows,3), but mpi gets mad
  //A_B_hat.setZero();
  cout<<"!"<<endl;
  A_B_hat=AB_matrix*xy_prime;

  cout<<"!"<<endl;
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
  cout<<"!"<<endl;
  cg.compute(AI_matrix);
  cout<<"!"<<endl;
  Z_femwarp_transformed = cg.solve(A_B_hat);
  cout<<"!"<<endl;
}


void distributed_femwarp3d_shared_mem_2
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
  unordered_map<int, set<int>> master_neighbor_dict[size];
  uint64_t str_lens[size];
  uint64_t recvcounts[size*2];
  uint64_t displs[size*2];
  cout<<"Start"<<endl;
  //Generate global stiffness matrix:
  int T_parts=num_eles/size;
  int low=rank*T_parts;
  int high=(rank+1)*T_parts;
  if (rank==size-1){
    high=num_eles;
  }
  unordered_map<int, set<int>> local_neighbors;
  MPI_Win T_win;
  MPI_Win_create(T.data(), sizeof(int) * num_eles * 4, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &T_win);
  cout<<"local_neighbors"<<endl;
  MPI_Win_fence(0, T_win);
  int T_i_0;
  int T_i_1;
  int T_i_j;
  int T_i_k;
  for(int i = low; i<high;i++){//O(n)
    MPI_Get(&T_i_0, 1, MPI_INT, 0, i*4, 1, MPI_INT, T_win);
    MPI_Get(&T_i_1, 1, MPI_INT, 0, i*4+1, 1, MPI_INT, T_win);
    if(T_i_0==T_i_1){
      break;
    }
    for(int j=0;j<4;j++){
      MPI_Get(&T_i_j, 1, MPI_INT, 0, i*4+j, 1, MPI_INT, T_win);
      if(local_neighbors.find(T_i_j)==local_neighbors.end() && T_i_j<m){ //Hashtable look-up is O(1) average
        local_neighbors[T_i_j]; //O(1) average
      }
    }
    for(int j=0;j<4;j++){
      MPI_Get(&T_i_j, 1, MPI_INT, 0, i*4+j, 1, MPI_INT, T_win);
      for(int k=0;k<4;k++){
        if(T_i_j<m){
          MPI_Get(&T_i_k, 1, MPI_INT, 0, i*4+k, 1, MPI_INT, T_win);
          local_neighbors[T_i_j].insert(T_i_k);
        }
      }
    }
  }
  cout<<"master_neighbor_dict"<<endl;
  ostringstream oss;
  cout<<rank<<" archive_out"<<endl;
  boost::archive::text_oarchive archive_out(oss);
  cout<<rank<<" local_neighbors"<<endl;
  archive_out << local_neighbors;
  

  
  cout<<rank<<" .str"<<endl;
  string send_str = oss.str();
  //vector<string> rec_str;
  

  cout<<rank<<" local_str_len"<<endl;
  uint64_t local_str_len=send_str.length()+1;
  cout<<rank<<" str_lens"<<endl;
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"all_gather"<<endl;
  MPI_Allgather(&local_str_len,1,MPI_UINT64_T,str_lens,1,MPI_UINT64_T,comm);
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"accumulate"<<endl;
  uint64_t rec_str_len = 0;
  //rec_str_len = accumulate(str_lens, str_lens+size, rec_str_len);
  for(int i = 0; i < size; i++){
    rec_str_len+=str_lens[i];
  } 



  uint64_t total_received_elements = 0;
  cout<<"recvcounts displs"<<endl;
  for (int i = 0; i < size; i++) {
      recvcounts[i] = str_lens[i];
      displs[i] = total_received_elements;
      total_received_elements += str_lens[i];
  }
  //char all_gathered_chars[rec_str_len];
  char* all_gathered_chars = (char*) malloc(sizeof(char)*rec_str_len);
  if(all_gathered_chars==NULL){
    cout<<rank<<" malloc failed for size of "<<rec_str_len<<endl;
  }
  //boost::mpi::all_gather(comm, send_str,rec_str);
  //instead:
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win send_str_win;
  MPI_Win_create((void *)send_str.data(), sizeof(char)*str_lens[rank], sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &send_str_win);
  MPI_Win_fence(0, send_str_win);
  char* rec_str_buf=nullptr;
  for(int i = 0; i < size;i++){
    rec_str_buf=(char*) malloc(sizeof(char)*str_lens[i]);
    if(rec_str_buf==NULL){
      cout<<rank<<" malloc failed for size of "<<str_lens[i]<<endl;
    }
    MPI_Get(rec_str_buf, str_lens[i], MPI_CHAR, i, 0, str_lens[i], MPI_CHAR, send_str_win);
    istringstream iss(rec_str_buf);
    boost::archive::text_iarchive archive_in(iss);
    archive_in>>master_neighbor_dict[i];
  }
  MPI_Win_fence(0, send_str_win);
  free(all_gathered_chars);
  MPI_Barrier(MPI_COMM_WORLD);

  cout<<"master_neighbor_dict[i]"<<endl;
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
  MPI_Barrier(MPI_COMM_WORLD);
  CSR_Matrix<double> AI_matrix(num_vals_in_AI,num_vals_in_AI,AI_dict.size()+1);
  CSR_Matrix<double> AB_matrix(num_vals_in_AB,num_vals_in_AB,AB_dict.size()+1);
  gen_neighbors_csr(AI_dict,AI_matrix);
  gen_neighbors_csr(AB_dict,AB_matrix,m);
  MPI_Barrier(MPI_COMM_WORLD);

  low=rank*T_parts;
  high= rank!=(size-1) ? (rank+1)*T_parts : num_eles;
  
  Eigen::Matrix4d basis_function_gradients;
  Eigen::Matrix4d element_stiffness_matrix;
  double a_func[3];
  double b_func[3];
  double c_func[3];
  double d_func[3];
  unsigned int T_ele[4];
  cout<<"FEM"<<endl;
  for (int n_ele=low; n_ele<high; n_ele++){
    
    //if((n_ele-low)%10000==0)cout<<rank<<" "<<(double)((double)(n_ele-low))/(double)((high-low))<<endl;
    MPI_Get(&T_ele, 4, MPI_UNSIGNED, 0, n_ele*4, 4, MPI_UNSIGNED, T_win);
    //https://htor.inf.ethz.ch/publications/img/mpi_mpi_hybrid_programming.pdf
    //https://www.cmor-faculty.rice.edu/~mk51/presentations/SIAMPP2016_4.pdf
    //https://www.intel.com/content/dam/develop/external/us/en/documents/an-introduction-to-mpi-3-597891.pdf
    //https://stackoverflow.com/questions/39912588/can-i-use-mpi-with-shared-memory
    //https://wgropp.cs.illinois.edu/index.html
    a_func[0]=  Z_original(T_ele[0],0);
    a_func[1]=  Z_original(T_ele[0],1);
    a_func[2]=  Z_original(T_ele[0],2);
    b_func[0]=  Z_original(T_ele[1],0);
    b_func[1]=  Z_original(T_ele[1],1);
    b_func[2]=  Z_original(T_ele[1],2);
    c_func[0]=  Z_original(T_ele[2],0);
    c_func[1]=  Z_original(T_ele[2],1);
    c_func[2]=  Z_original(T_ele[2],2);
    d_func[0]=  Z_original(T_ele[3],0);
    d_func[1]=  Z_original(T_ele[3],1);
    d_func[2]=  Z_original(T_ele[3],2);
    gen_basis_function_gradients_tetrahedron(a_func,b_func,c_func,d_func,basis_function_gradients);
    element_stiffness_matrix=gen_element_stiffness_matrix_tetrahedron(basis_function_gradients,a_func,b_func,c_func,d_func);
    for (int i = 0; i<4; i++){
      for (int j = 0; j<4; j++){
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
  MPI_Win_fence(0, T_win);
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"tmpI[i]"<<endl;
  double tmpI[AI_matrix._num_vals];
  for(int i = 0; i < AI_matrix._num_vals;i++){
    tmpI[i]=AI_matrix._vals[i];
  }
  cout<<"tmpB[i]"<<endl;
  double tmpB[AB_matrix._num_vals];
  for(int i = 0; i < AB_matrix._num_vals;i++){
    tmpB[i]=AB_matrix._vals[i];
  }
  cout<<"Allreduce1"<<endl;
  MPI_Allreduce(tmpI,AI_matrix._vals,AI_matrix._num_vals,MPI_DOUBLE,MPI_SUM,comm);
  cout<<"Allreduce2"<<endl;
  MPI_Allreduce(tmpB,AB_matrix._vals,AB_matrix._num_vals,MPI_DOUBLE,MPI_SUM,comm);
  MPI_Barrier(MPI_COMM_WORLD);

  cout<<"csr_to_dist_csr"<<endl;
  CSR_Matrix<double> local_A_I(0,0,0);
  CSR_Matrix<double> local_A_B(0,0,0);
  csr_to_dist_csr(AI_matrix,local_A_I,m,comm,size,rank);
  csr_to_dist_csr(AB_matrix,local_A_B,m,comm,size,rank);
  MPI_Barrier(MPI_COMM_WORLD);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  int num_rows=0;
  int local_num_rows=local_A_B._num_row_ptrs-1;
  int num_rows_arr[size];
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(MPI_COMM_WORLD);
  num_rows = accumulate(num_rows_arr, num_rows_arr+size, num_rows);
  A_B_hat.resize(3,num_rows);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"parallel_csr_x_matrix"<<endl;
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  MPI_Barrier(MPI_COMM_WORLD);


  cout<<"parallel_sparse_block_conjugate_gradient_v2"<<endl;
  parallel_sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows);
  MPI_Barrier(MPI_COMM_WORLD);
}




void distributed_femwarp3d_shared_mem_3
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
  
  
  int* neighbors=nullptr;
  int row_size=0;
  int N_val_1=0;
  int N_val_2=0;
  int N_ind=0;
  bool found=false;
  cout<<"!"<<endl;
  if(rank==1){
    neighbors=(int*)calloc(m*MAX_TET_NODE_NEIGHBORS_BUF,sizeof(int));
    cout<<neighbors<<endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win T_win;
  MPI_Win Z_win;
  MPI_Win N_win;
  MPI_Win_create(T.data(), sizeof(int) * num_eles * 4, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &T_win);
  MPI_Win_create(Z_original.data(), sizeof(double) * n * 3, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &Z_win);
  MPI_Win_create(neighbors, sizeof(int) * m * MAX_TET_NODE_NEIGHBORS_BUF, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &N_win);
  MPI_Win_fence(0, T_win);
  vector<MPI_Win> neighbors_wins(m);
  for(i=0;i<m;i++){
    MPI_Win_create(neighbors+i*MAX_TET_NODE_NEIGHBORS_BUF, rank==1?sizeof(int) * MAX_TET_NODE_NEIGHBORS_BUF:0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &neighbors_wins[i]);
  }
  cout<<"e"<<endl;
  
//6.4
  MPI_Barrier(MPI_COMM_WORLD);
  for(i = low; i<high;i++){//O(n)
    MPI_Get(&T_i_0, 1, MPI_INT, 0, i*4, 1, MPI_INT, T_win);
    MPI_Get(&T_i_1, 1, MPI_INT, 0, i*4+1, 1, MPI_INT, T_win);
    if(T_i_0==T_i_1){
      break;
    }
    for(j=0;j<4;j++){
      MPI_Get(&T_i_j, 1, MPI_INT, 0, i*4+j, 1, MPI_INT, T_win);
      for(k=0;k<4;k++){
        if(T_i_j<m){
          MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, neighbors_wins[T_i_j]); 
          MPI_Get(&T_i_k, 1, MPI_INT, 0, i*4+k, 1, MPI_INT, T_win);
          MPI_Get(&row_size, 1, MPI_INT, 1, 0, 1, MPI_INT, neighbors_wins[T_i_j]);
          if(row_size==0){
            row_size=1;
            MPI_Put(&row_size, 1, MPI_INT, 1, 0, 1, MPI_INT, neighbors_wins[T_i_j]);//First index of row contains size
            MPI_Put(&T_i_k, 1, MPI_INT, 1, row_size, 1, MPI_INT, neighbors_wins[T_i_j]);//First index of row contains size
            if(T_i_k<m){
              loc_num_vals_in_AI+=1;
            }
            else{
              loc_num_vals_in_AB+=1;
            }
          }/*
          
          Collecting information, read paper, implementation
          
          
          */
          else{
            found=false;
            int low=1;
            int high=row_size;
            int mid;
            while(low<=high){
              mid=low+(high-low)/2;
              MPI_Get(&N_val_1, 1, MPI_INT, 1, mid, 1, MPI_INT, neighbors_wins[T_i_j]);
              if(N_val_1==T_i_k){
                found=true;
                break;
              }
              else if(N_val_1<T_i_k){
                low=mid+1;
              }
              else{
                high=mid-1;
              }
            }
            N_ind=low;
            if(!found){
              row_size+=1;
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){
                cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<endl;
                exit(MAX_TET_NODE_NEIGHBORS_BUF);
              }
              else{
                //Insert element
                //local_neighbors[T_i_j].insert(T_i_k);
                MPI_Put(&row_size, 1, MPI_INT, 1, 0, 1, MPI_INT, neighbors_wins[T_i_j]);//First index of row contains size
                for(int l=row_size;l>=N_ind;l--){
                  MPI_Get(&N_val_2, 1, MPI_INT, 1, l-1, 1, MPI_INT, neighbors_wins[T_i_j]);
                  MPI_Put(&N_val_2, 1, MPI_INT, 1, l, 1, MPI_INT, neighbors_wins[T_i_j]);
                }
                MPI_Put(&T_i_k, 1, MPI_INT, 1, N_ind, 1, MPI_INT, neighbors_wins[T_i_j]);
                if(T_i_k<m){
                  loc_num_vals_in_AI+=1;
                }
                else{
                  loc_num_vals_in_AB+=1;
                }
              }
            }
          }
          MPI_Win_unlock(1, neighbors_wins[T_i_j]);
        }
      }
    }
  }
  cout<<"List "<<rank<<endl;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(&loc_num_vals_in_AI,&num_vals_in_AI,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&loc_num_vals_in_AB,&num_vals_in_AB,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Win_fence(0, T_win);

  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"MPI_Allreduce "<<rank<<endl;
  CSR_Matrix<double> AI_matrix(num_vals_in_AI,num_vals_in_AI,m+1);
  //exit(100);
  CSR_Matrix<double> AB_matrix(num_vals_in_AB,num_vals_in_AB,b+1);
  double tmpI[AI_matrix._num_vals];
  double tmpB[AB_matrix._num_vals];
  cout<<"CSR_Matrix "<<rank<<endl;
/*390,1194
390,1194
390,1194
390,1194
390,1194
390,1194
390,1194
390,1194*/

  //gen_neighbors_csr_win(AI_dict,AI_matrix);
  //gen_neighbors_csr_win(AB_dict,AB_matrix,m);





  int AB_val_count=0;
  int AB_col_indices_pos=0;
  int AB_offset=m;


  int AI_val_count=0;
  int AI_col_indices_pos=0;
  int AI_offset=0;
  int tmp_neighbor_set[MAX_TET_NODE_NEIGHBORS_BUF];
  MPI_Win_fence(0, N_win);
  for (i = 0; i < m; i++){
    AI_matrix.setRowPtrsAt(i,AI_col_indices_pos);
    AB_matrix.setRowPtrsAt(i,AB_col_indices_pos);
    
    
    //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, N_win); 
    MPI_Get(&tmp_neighbor_set, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, 1, i*MAX_TET_NODE_NEIGHBORS_BUF, MAX_TET_NODE_NEIGHBORS_BUF, MPI_INT, N_win);
    //MPI_Win_unlock(1, N_win);
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
  MPI_Win_fence(0, N_win);
  AI_matrix.setRowPtrsAt(m,AI_col_indices_pos);
  AB_matrix.setRowPtrsAt(m,AB_col_indices_pos);
  cout<<"FEM "<<rank<<endl;





  MPI_Barrier(MPI_COMM_WORLD);

  free(neighbors);
  
  MPI_Win_fence(0, Z_win);
  MPI_Barrier(MPI_COMM_WORLD);
  for (n_ele=low; n_ele<high; n_ele++){
    MPI_Get(&T_ele, 4, MPI_UNSIGNED, 0, n_ele*4, 4, MPI_UNSIGNED, T_win);
    MPI_Get(&Z_nums_0, 3, MPI_DOUBLE, 0, T_ele[0]*3, 3, MPI_DOUBLE, Z_win);
    MPI_Get(&Z_nums_1, 3, MPI_DOUBLE, 0, T_ele[1]*3, 3, MPI_DOUBLE, Z_win);
    MPI_Get(&Z_nums_2, 3, MPI_DOUBLE, 0, T_ele[2]*3, 3, MPI_DOUBLE, Z_win);
    MPI_Get(&Z_nums_3, 3, MPI_DOUBLE, 0, T_ele[3]*3, 3, MPI_DOUBLE, Z_win);
    
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
  MPI_Win_fence(0, T_win);
  MPI_Win_fence(0, Z_win);
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"csr_to_dist "<<rank<<endl;
  
  for(i = 0; i < AI_matrix._num_vals;i++){
    tmpI[i]=AI_matrix._vals[i];
  }
  for(i = 0; i < AB_matrix._num_vals;i++){
    tmpB[i]=AB_matrix._vals[i];
  }
  MPI_Allreduce(tmpI,AI_matrix._vals,AI_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(tmpB,AB_matrix._vals,AB_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  CSR_Matrix<double> local_A_I(0,0,0);
  CSR_Matrix<double> local_A_B(0,0,0);
  csr_to_dist_csr(AI_matrix,local_A_I,m,comm,size,rank);
  csr_to_dist_csr(AB_matrix,local_A_B,m,comm,size,rank);
  
 
  MPI_Barrier(MPI_COMM_WORLD);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  int local_num_rows=local_A_B._num_row_ptrs-1;
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(MPI_COMM_WORLD);
  num_rows = accumulate(num_rows_arr, num_rows_arr+size, num_rows);
  A_B_hat.resize(3,num_rows);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();
  MPI_Barrier(MPI_COMM_WORLD);
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  MPI_Barrier(MPI_COMM_WORLD);

  parallel_sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows);
  MPI_Barrier(MPI_COMM_WORLD);
}

