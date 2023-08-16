
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
#include "./matrix_helper.hpp"

#define M_PI_3 M_PI/3


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

void test1(int argc, char *argv[]){

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
  //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_full_transformation;
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
    string orig_mesh_name=/*"formatted_heart_5M";//*/"body_has_heart";
    string new_mesh_name=/*"new_formatted_heart_5M";//*/"new_body_has_heart";
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
   //Z_full_transformation.resize(n,3);
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
    cout<<"Start ele"<<endl;
    
    for(i = 0; i < num_eles;i++){
      getline(in_file,in_line);
      ss.clear();
      ss.str(in_line);
      ss>>word;//skip index
      for(j = 0; j < 4; j++){
        ss>>word;T(i,j)=offset-stoi(word);
      }
    }
    T.transposeInPlace();
    cout<<"End ele"<<endl;

    in_file.close();


    rotate_transform_matrix_transpose = rotate_transform.matrix().transpose();
   //Z_full_transformation=Z_original*rotate_transform_matrix_transpose;
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
    //T.resize(num_eles,4);
  }
  comm.barrier();


  broadcast(comm,Z_original.data(),n*3,0);
  broadcast(comm,Z_boundary_transformation.data(),b*3,0);
  //broadcast(comm,T.data(),num_eles*4,0); 
  double start,end;
  comm.barrier();
  start = MPI_Wtime();
  
  distributed_femwarp3d(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed);
  comm.barrier();
  end  = MPI_Wtime();
  
  if(rank==0){
    //cout<<Z_femwarp_transformed<<endl; 
    cout<<end-start<<endl;
  }


  MPI_Finalize();
}


void serial_test(int argc, char *argv[]){


  
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
  //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_full_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::AngleAxisd rotate_transform (M_PI_3,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> rotate_transform_matrix_transpose;


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
 //Z_full_transformation.resize(n,3);
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
 //Z_full_transformation=Z_original*rotate_transform_matrix_transpose;
  Z_boundary_transformation=Z_original.block(n-b,0,b,3);//Get only boundary nodes
  Z_boundary_transformation*=rotate_transform_matrix_transpose;

  auto start = std::chrono::high_resolution_clock::now();
  serial_femwarp3d(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,Z_femwarp_transformed);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  //cout<<Z_femwarp_transformed<<endl;
  cout<<((double)(ms))/1000000<<endl;

}


int main(int argc, char *argv[]){
  test1(argc,argv);
  //serial_test(argc,argv);
}