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

#define DEBUG (false)

#define MAX_TET_NODE_NEIGHBORS (40)
#define MAX_TET_NODE_NEIGHBORS_BUF (MAX_TET_NODE_NEIGHBORS+1)

using namespace std;

void distributed_femwarp3d_costly
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



void serial_femwarp3d_costly
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

void serial_femwarp3d_no_precompute_eigen_costly
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




#define win_chunk_quantize(row_ind,chunk_size,num_chunks) row_ind/chunk_size>num_chunks-1?num_chunks-1:row_ind/chunk_size
#define inter_win_chunk_quantize(row_ind,win_ind,neighbors_chunks,neighbors_chunk_size,neighbors_chunk_size_last,num_chunks)win_ind!=neighbors_chunks-1? row_ind%neighbors_chunk_size:((row_ind-neighbors_chunk_size*(num_chunks-1))%neighbors_chunk_size_last)

void distributed_femwarp3d_RMA//works
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

  //cout<<"*"<<endl;

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win T_win;
  MPI_Win Z_win;
  MPI_Win N_win;
  MPI_Win_create(rank!=0?NULL:T.data(), rank!=0?0:(sizeof(int) * num_eles * 4), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &T_win);
  MPI_Win_create(rank!=0?NULL:Z_original.data(), rank!=0?0:(sizeof(double) * n * 3), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &Z_win);
  MPI_Win_lock_all(0,T_win);
  MPI_Win_lock_all(0,Z_win);
  vector<MPI_Win> neighbors_wins(neighbors_chunks);
  //cout<<"//"<<endl;
  for(i=0;i<neighbors_chunks-1;i++){
    MPI_Win_create(rank!=1?NULL:neighbors[i], rank!=1?0:(sizeof(int) * neighbors_chunk_size * MAX_TET_NODE_NEIGHBORS_BUF), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &neighbors_wins[i]);
  }
  MPI_Win_create(rank!=1?NULL:neighbors[neighbors_chunks-1], rank!=1?0:(sizeof(int) * neighbors_chunk_size_last * MAX_TET_NODE_NEIGHBORS_BUF), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &neighbors_wins[neighbors_chunks-1]);
  MPI_Barrier(MPI_COMM_WORLD);
  //cout<<rank<<endl;
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
  cout<<"MPI_Win_unlock_all(T_win);"<<rank<<endl;
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
          //double lock_end = MPI_Wtime(); cout<<rank<<" lock time: "<<lock_end-lock_start<<endl;
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
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<endl;exit(MAX_TET_NODE_NEIGHBORS_BUF);}
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
  //cout<<"List "<<rank<<endl;
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Lists time: "<<end-start<<endl;}start = MPI_Wtime();
  MPI_Allreduce(&loc_num_vals_in_AI,&num_vals_in_AI,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&loc_num_vals_in_AB,&num_vals_in_AB,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);//exit(100);
  cout<<num_vals_in_AI<<" "<<num_vals_in_AB<<endl;
  cout<<"MPI_Allreduce "<<rank<<endl;
  CSR_Matrix<double> AI_matrix(num_vals_in_AI,num_vals_in_AI,m+1);
  //exit(100);
  CSR_Matrix<double> AB_matrix(num_vals_in_AB,num_vals_in_AB,/*b*/m+1);
  double tmpI[AI_matrix._num_vals];
  double tmpB[AB_matrix._num_vals];
  cout<<"CSR_Matrix "<<rank<<endl;




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
  cout<<"setRowPtrsAt "<<rank<<endl;
  AI_matrix.setRowPtrsAt(m,AI_col_indices_pos);
  AB_matrix.setRowPtrsAt(m,AB_col_indices_pos);
  cout<<"FEM "<<rank<<endl;



  //free(neighbors);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Matrix Allocation time: "<<end-start<<endl;}start = MPI_Wtime();
  for (n_ele=low; n_ele<high; n_ele++){
    //cout<<rank<<":"<<low<<" "<<n_ele<<""<<high<<endl;
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
  cout<<"MPI_Win_unlock_all(Z_win);"<<rank<<endl;
  MPI_Win_unlock_all(Z_win);//MPI_Win_fence(0, Z_win);
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"MPI_Allreduce "<<rank<<endl;
  
  for(i = 0; i < AI_matrix._num_vals;i++){
    tmpI[i]=AI_matrix._vals[i];
  }
  for(i = 0; i < AB_matrix._num_vals;i++){
    tmpB[i]=AB_matrix._vals[i];
  }
  MPI_Allreduce(tmpI,AI_matrix._vals,AI_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(tmpB,AB_matrix._vals,AB_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" FEM time: "<<end-start<<endl;}start = MPI_Wtime();
  



  MPI_Barrier(MPI_COMM_WORLD);
  CSR_Matrix<double> local_A_I(0,0,0);
  CSR_Matrix<double> local_A_B(0,0,0);
  cout<<"csr_to_dist "<<rank<<endl;
  csr_to_dist_csr(AI_matrix,local_A_I,m,comm,size,rank);
  csr_to_dist_csr(AB_matrix,local_A_B,m,comm,size,rank);



  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Distribute Matrix time: "<<end-start<<endl;}

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  cout<<"MPI_Allgather "<<rank<<endl;
  int local_num_rows=local_A_B._num_row_ptrs-1;
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(MPI_COMM_WORLD);
  num_rows = accumulate(num_rows_arr, num_rows_arr+size, num_rows);
  A_B_hat.resize(3,num_rows);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();
  MPI_Barrier(MPI_COMM_WORLD);start = MPI_Wtime();
  cout<<"parallel_csr_x_matrix "<<rank<<endl;
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Times Matrix time: "<<end-start<<endl;}start = MPI_Wtime();

  cout<<"parallel_sparse_block_conjugate_gradient_v2 "<<rank<<endl;
  
  parallel_sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows);

  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CG time: "<<end-start<<endl;}
}



void distributed_femwarp3d_SMH_RMA //Forget this exists
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
{double start,end;start = MPI_Wtime();
  MPI_Comm shmcomm;
  MPI_Comm_split_type (MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,0, MPI_INFO_NULL,&shmcomm);
  MPI_Group world_group;
  MPI_Group shared_group;
  MPI_Comm_group (MPI_COMM_WORLD, &world_group);
  MPI_Comm_group (shmcomm, &shared_group);
  int shmrank;
  int shmsize;
  MPI_Comm_rank (shmcomm, &shmrank);
  MPI_Comm_size (shmcomm, &shmsize);
  int num_nodes=size/shmsize;


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
  

  int neighbors_chunks=size;
  int neighbors_chunk_size;
  int neighbors_chunk_size_last;
  neighbors_chunk_size=m/neighbors_chunks;
  neighbors_chunk_size_last=neighbors_chunk_size+m%neighbors_chunks;
  //int* neighbors[neighbors_chunk_size];
  int neighbors[m][MAX_TET_NODE_NEIGHBORS_BUF]={0};//neighbors[rank==size-1?neighbors_chunk_size_last:neighbors_chunk_size][MAX_TET_NODE_NEIGHBORS_BUF];
  vector<int*> shared_neigbors_arr(shmsize);

  
  vector<MPI_Win> shared_win(shmsize);
  
  MPI_Aint size_of_shmem;
  int disp_unit;
  

  if(shmrank==0){
    for(i=0;i<shmsize;i++){
      cout<<MPI_Win_allocate_shared(m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int), sizeof(int), MPI_INFO_NULL, shmcomm, &shared_neigbors_arr[i], &shared_win[i])<<endl;
    }
  }
  else{
    for(i=0;i<shmsize;i++){
      MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shmcomm, &shared_neigbors_arr[i], &shared_win[i]);
      MPI_Win_shared_query(shared_win[i], 0, &size_of_shmem, &disp_unit, &shared_neigbors_arr[i]);
    }
  }





  int row_size=0;
  int N_val_1=0;
  int N_val_2=0;
  int N_ind=0;
  bool found=false;

  //cout<<"*"<<endl;

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Win T_win;
  MPI_Win Z_win;
  MPI_Win N_win;
  MPI_Win_create(rank!=0?NULL:T.data(), rank!=0?0:(sizeof(int) * num_eles * 4), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &T_win);
  //MPI_Win_create(rank!=0?NULL:Z_original.data(), rank!=0?0:(sizeof(double) * n * 3), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &Z_win);
  MPI_Win_lock_all(0,T_win);
  //MPI_Win_lock_all(0,Z_win);

  MPI_Barrier(MPI_COMM_WORLD);
  //cout<<rank<<endl;
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
  MPI_Win_free(T_win);

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
          rank_to_send_val=T_i_j/neighbors_chunk_size;
          if(rank_to_send_val>=neighbors_chunks){
            rank_to_send_val=neighbors_chunks-1;
          }
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
            if(T_i_k<m && rank==0){
              loc_num_vals_in_AI+=1;
            }
            else if(T_i_k>=m && rank==0) {
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
              if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<endl;exit(MAX_TET_NODE_NEIGHBORS_BUF);}
              else{
                //Insert element
                //local_neighbors[T_i_j].insert(T_i_k);
                shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;//neighbors[T_i_j/*local_index*/][0]=row_size;
                for(l=row_size;l>=N_ind;l--){
                  shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+l-1];//neighbors[T_i_j/*local_index*/][l]=neighbors[T_i_j/*local_index*/][l-1];
                }
                shared_neigbors_arr[shmrank][T_i_j*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=T_i_k;//neighbors[T_i_j/*local_index*/][N_ind]=T_i_k;
                if(T_i_k<m && rank==0){
                  loc_num_vals_in_AI+=1;
                }
                else if(T_i_k>=m && rank==0){
                  loc_num_vals_in_AB+=1;
                }
              }
            }
          }

        }
      }
    }
  }

  
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Lists time: "<<end-start<<endl;}start = MPI_Wtime();

  
  if(shmrank==0){
    for(i=0;i<m;i++){
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
            if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<endl;exit(MAX_TET_NODE_NEIGHBORS_BUF);}
            else{
              //Insert element
              //local_neighbors[T_i_j].insert(T_i_k);
              shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+0]=row_size;
              for(l=row_size;l>=N_ind;l--){
                shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l]=shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+l-1];
              }
              shared_neigbors_arr[0][i*MAX_TET_NODE_NEIGHBORS_BUF+N_ind]=val_to_find;
              if(val_to_find<m && rank==0){
                loc_num_vals_in_AI+=1;
              }
              else if(val_to_find>=m && rank==0){
                loc_num_vals_in_AB+=1;
              }
            }
          }
        }
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Combine lists 1 time: "<<end-start<<endl;}start = MPI_Wtime();

  if(shmrank==0){
    memcpy(neighbors,shared_neigbors_arr[0], m*MAX_TET_NODE_NEIGHBORS_BUF*sizeof(int));
  }

  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" memcpy: "<<end-start<<endl;}start = MPI_Wtime();
  MPI_Request req;



  for(i=0;i<shmsize;i++){MPI_Win_free(&shared_win[i]);}


  ///////////////////Combine across multiple nodes///////////////
  if(num_nodes>1){
    int tmp_neighbors[num_nodes][MAX_TET_NODE_NEIGHBORS_BUF]={0};
    MPI_Barrier(MPI_COMM_WORLD);


    for(i=0;i<m;i++){
      if(shmrank==0 && rank!=0){
        MPI_Isend(neighbors[i],MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,0,0,MPI_COMM_WORLD,&req);
      }
      else if(rank==0){
        for(j=0;j<neighbors[i][0];j++){
          tmp_neighbors[0][j]=neighbors[i][j];
        }
        for(j=1;j<num_nodes;j++){
          MPI_Recv(tmp_neighbors[j],MAX_TET_NODE_NEIGHBORS_BUF,MPI_INT,j*shmsize,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      
      for(j=0;j<num_nodes;j++){
        for(k=1;k<=tmp_neighbors[j][0];k++){
          row_size=neighbors[i][0];
          int val_to_find=tmp_neighbors[j][k];
          found=false;
          l_low=1;
          l_high=row_size;
          while(l_low<=l_high){
            l_mid=l_low+(l_high-l_low)/2;
            N_val_1=neighbors[i][l_mid];
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
            if(row_size==MAX_TET_NODE_NEIGHBORS_BUF){cout<<"Node "<<i<<" has more than "<<MAX_TET_NODE_NEIGHBORS<<" neighbors! "<<endl;exit(MAX_TET_NODE_NEIGHBORS_BUF);}
            else{
              //Insert element
              //local_neighbors[T_i_j].insert(T_i_k);
              neighbors[i][0]=row_size;
              for(l=row_size;l>=N_ind;l--){
                neighbors[i][l]=neighbors[i][l-1];
              }
              neighbors[i][N_ind]=val_to_find;
              if(val_to_find<m && rank==0){
                loc_num_vals_in_AI+=1;
              }
              else if(val_to_find>=m && rank==0){
                loc_num_vals_in_AB+=1;
              }
            }
          }
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  ///////////////////Combine across multiple nodes///////////////







  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Combine lists 2 time: "<<end-start<<endl;}start = MPI_Wtime();




  
  MPI_Allreduce(&loc_num_vals_in_AI,&num_vals_in_AI,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&loc_num_vals_in_AB,&num_vals_in_AB,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"MPI_Allreduce "<<rank<<endl;
  CSR_Matrix<double> AI_matrix(num_vals_in_AI,num_vals_in_AI,m+1);
  CSR_Matrix<double> AB_matrix(num_vals_in_AB,num_vals_in_AB,m+1);

  cout<<"CSR_Matrix "<<rank<<endl;




  int AB_val_count=0;
  int AB_col_indices_pos=0;
  int AB_offset=m;


  int AI_val_count=0;
  int AI_col_indices_pos=0;
  int AI_offset=0;
  if(rank==0){
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
    cout<<"setRowPtrsAt "<<rank<<endl;
    AI_matrix.setRowPtrsAt(m,AI_col_indices_pos);
    AB_matrix.setRowPtrsAt(m,AB_col_indices_pos); 
  }
  MPI_Barrier(MPI_COMM_WORLD);



  double tmpI[AI_matrix._num_vals];
  double tmpB[AB_matrix._num_vals];
  int tmpI_c[AI_matrix._num_col_indices];
  int tmpB_c[AB_matrix._num_col_indices];
  int tmpI_r[AI_matrix._num_row_ptrs];
  int tmpB_r[AB_matrix._num_row_ptrs];

  if(rank==0){
    memcpy(tmpI_r,AI_matrix._row_ptrs,AI_matrix._row_ptrs*sizeof(int));
    memcpy(tmpB_r,AB_matrix._row_ptrs,AB_matrix._row_ptrs*sizeof(int));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(tmpI_r, AI_matrix._num_row_ptrs, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(tmpB_r, AB_matrix._num_row_ptrs, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank!=0){
    memcpy(AI_matrix._num_row_ptrs,tmpI_r,AI_matrix._num_row_ptrs*sizeof(int));
    memcpy(AB_matrix._num_row_ptrs,tmpB_r,AB_matrix._num_row_ptrs*sizeof(int));
  }


  if(rank==0){
    memcpy(tmpI_c,AI_matrix._col_indices,AI_matrix._num_col_indices*sizeof(int));
    memcpy(tmpB_c,AB_matrix._col_indices,AB_matrix._num_col_indices*sizeof(int));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(tmpI_c, AI_matrix._num_col_indices, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(tmpB_c, AB_matrix._num_col_indices, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank!=0){
    memcpy(AI_matrix._col_indices,tmpI_c,AI_matrix._num_col_indices*sizeof(int));
    memcpy(AB_matrix._col_indices,tmpB_c,AB_matrix._num_col_indices*sizeof(int));
  }
  
  MPI_Barrier(MPI_COMM_WORLD);



  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Matrix Allocation time: "<<end-start<<endl;}start = MPI_Wtime();
  
  
  

  cout<<"FEM "<<rank<<endl;

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
  cout<<"MPI_Allreduce "<<rank<<endl;

  
  memcpy(tmpI,AI_matrix._vals,AI_matrix._num_vals*sizeof(double));
  memcpy(tmpB,AB_matrix._vals,AB_matrix._num_vals*sizeof(double));

  MPI_Allreduce(tmpI,AI_matrix._vals,AI_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(tmpB,AB_matrix._vals,AB_matrix._num_vals,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" FEM time: "<<end-start<<endl;}start = MPI_Wtime();
  


  MPI_Barrier(MPI_COMM_WORLD);
  CSR_Matrix<double> local_A_I(0,0,0);
  CSR_Matrix<double> local_A_B(0,0,0);
  cout<<"csr_to_dist "<<rank<<endl;
  csr_to_dist_csr(AI_matrix,local_A_I,m,comm,size,rank);
  csr_to_dist_csr(AB_matrix,local_A_B,m,comm,size,rank);


  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" Distribute Matrix time: "<<end-start<<endl;}

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> A_B_hat;

  cout<<"MPI_Allgather "<<rank<<endl;
  int local_num_rows=local_A_B._num_row_ptrs-1;
  MPI_Allgather(&local_num_rows,1,MPI_INT,num_rows_arr,1,MPI_INT,comm);
  MPI_Barrier(MPI_COMM_WORLD);
  num_rows = accumulate(num_rows_arr, num_rows_arr+size, num_rows);
  A_B_hat.resize(3,num_rows);//Should be (num_rows,3), but mpi gets mad
  A_B_hat.setZero();
  MPI_Barrier(MPI_COMM_WORLD);start = MPI_Wtime();
  cout<<"parallel_csr_x_matrix "<<rank<<endl;
  parallel_csr_x_matrix(local_A_B,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  //parallel_csr_x_matrix_eigen(local_A_B_matrix_eigen,xy_prime,A_B_hat,comm,rank,size,num_rows_arr);
  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CSR Times Matrix time: "<<end-start<<endl;}start = MPI_Wtime();

  cout<<"parallel_sparse_block_conjugate_gradient_v2 "<<rank<<endl;
  
  parallel_sparse_block_conjugate_gradient_v2(local_A_I,A_B_hat,Z_femwarp_transformed,comm,rank,size,num_rows);
  //parallel_preconditioned_sparse_block_conjugate_gradient_v2_icf(local_A_I,A_B_hat,Z_femwarp_transformed,L_inv,L_inv_transpose,comm,rank,size,num_rows);

  MPI_Barrier(MPI_COMM_WORLD);end = MPI_Wtime();if(rank==0){cout<<rank<<" CG time: "<<end-start<<endl;}
}
