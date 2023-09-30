
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
#include "./FEMWARP.hpp"

#define M_PI_3 M_PI/3
#define IN_MESH_NAME "formatted_heart_5M"//*/"body_has_heart"
#define OUT_MESH_NAME "new_formatted_heart_5M"//*/"new_body_has_heart"


using namespace std;

void distributed_test_packed_3(int argc, char *argv[]){

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
    string orig_mesh_name=IN_MESH_NAME;
    string new_mesh_name=OUT_MESH_NAME;
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
    int min = T.minCoeff();
    if(min!=0){
      T.array()-=min;
    }
    T.transposeInPlace();
    cout<<"End ele"<<endl;

    in_file.close();


    rotate_transform_matrix_transpose = rotate_transform.matrix().transpose();
   //Z_full_transformation=Z_original*rotate_transform_matrix_transpose;
    Z_boundary_transformation=Z_original.block(n-b,0,b,3);//Get only boundary nodes
    Z_boundary_transformation*=rotate_transform_matrix_transpose;

    Z_original.transposeInPlace();
  }

  comm.barrier();
  broadcast(comm,n,0);
  broadcast(comm,m,0);
  broadcast(comm,b,0);
  broadcast(comm,num_eles,0);
  comm.barrier();
  if(rank!=0){
    //Z_original.resize(n,3);
    Z_boundary_transformation.resize(b,3);
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
    //T.resize(num_eles,4);
  }
  comm.barrier();


  //broadcast(comm,Z_original.data(),n*3,0);
  broadcast(comm,Z_boundary_transformation.data(),b*3,0);
  //broadcast(comm,T.data(),num_eles*4,0); 
  double start,end;
  comm.barrier();
  start = MPI_Wtime();
  distributed_femwarp3d_shared_mem_3(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed);
  comm.barrier();
  end  = MPI_Wtime();
  
  if(rank==0){
    cout<<Z_femwarp_transformed<<endl; 
    cout<<end-start<<endl;
  }


  MPI_Finalize();
}

void distributed_test_packed_2(int argc, char *argv[]){

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
    string orig_mesh_name=IN_MESH_NAME;
    string new_mesh_name=OUT_MESH_NAME;
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
    int min = T.minCoeff();
    if(min!=0){
      T.array()-=min;
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
  distributed_femwarp3d_shared_mem_2(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed);
  comm.barrier();
  end  = MPI_Wtime();
  
  if(rank==0){
    //cout<<Z_femwarp_transformed<<endl; 
    cout<<end-start<<endl;
  }


  MPI_Finalize();
}
/*
void distributed_test_packed(int argc, char *argv[]){

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
    string orig_mesh_name=IN_MESH_NAME;
    string new_mesh_name=OUT_MESH_NAME;
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
    //cout<<"Start ele"<<endl;
    
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
    //cout<<"End ele"<<endl;

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
  
  distributed_femwarp3d_shared_mem(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed);
  comm.barrier();
  end  = MPI_Wtime();
  
  if(rank==0){
    //cout<<Z_femwarp_transformed<<endl; 
    cout<<end-start<<endl;
  }


  MPI_Finalize();
}

void distributed_test(int argc, char *argv[]){

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
    string orig_mesh_name=IN_MESH_NAME;
    string new_mesh_name=OUT_MESH_NAME;
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
    //cout<<"Start ele"<<endl;
    
    for(i = 0; i < num_eles;i++){
      getline(in_file,in_line);
      ss.clear();
      ss.str(in_line);
      ss>>word;//skip index
      for(j = 0; j < 4; j++){
        ss>>word;T(i,j)=offset-stoi(word);
      }
    }
    //T.transposeInPlace();
    //cout<<"End ele"<<endl;

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
    T.resize(num_eles,4);
  }
  comm.barrier();


  broadcast(comm,Z_original.data(),n*3,0);
  broadcast(comm,Z_boundary_transformation.data(),b*3,0);
  broadcast(comm,T.data(),num_eles*4,0); 
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
  string orig_mesh_name=IN_MESH_NAME;
  string new_mesh_name=OUT_MESH_NAME;
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
*/


void serial_test_no_precompute_eigen(int argc, char *argv[]){
//TODO subtract by one in element list

  
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
  string orig_mesh_name=IN_MESH_NAME;
  string new_mesh_name=OUT_MESH_NAME;
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


  int min = T.minCoeff();
    if(min!=0){
      T.array()-=min;
    }


    
  rotate_transform_matrix_transpose = rotate_transform.matrix().transpose();
 //Z_full_transformation=Z_original*rotate_transform_matrix_transpose;
  Z_boundary_transformation=Z_original.block(n-b,0,b,3);//Get only boundary nodes
  Z_boundary_transformation*=rotate_transform_matrix_transpose;

  auto start = std::chrono::high_resolution_clock::now();
  serial_femwarp3d_no_precompute_eigen(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,Z_femwarp_transformed);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  //cout<<Z_femwarp_transformed<<endl;
  cout<<((double)(ms))/1000000<<endl;

}

int main(int argc, char *argv[]){
  //distributed_test_packed(argc,argv);
  //distributed_test_packed_2(argc,argv);
  distributed_test_packed_3(argc,argv);
  //distributed_test_packed_3(argc,argv);
  //distributed_test(argc,argv);
  //serial_test(argc,argv);
  
  //serial_test_no_precompute_eigen(argc,argv);
}