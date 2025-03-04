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
    March 5th, 2025

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

#define _USE_MATH_DEFINES
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

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
#include <algorithm>
#include <ctime>
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
#include "./FEMWARP.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>


#define DEF_ANGLE 0.01745
//(M_PI/81)
#define IN_MESH_NAME "formatted_heart_5M"//*/"body_has_heart"
#define OUT_MESH_NAME "owo"//"new_formatted_heart_5M"//*/"new_body_has_heart"


using namespace std;



void distributed_test_packed_6(int argc, char *argv[]){

  boost::mpi::environment env{argc, argv};
  boost::mpi::communicator comm;
  
  int n;//Number nodes
  int m;//Number of interior nodes
  int b;//Number of boundary nodes
  int offset;//n-1
  int num_eles;
  int num_faces;
  int i;//index variable
  int j;//index variable
  int k;//index variable
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original_natural_ordering;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_full_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;

  int rank;
  int size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);


  if(rank==0){
    cout<<"Start"<<endl;
    ifstream in_file;
    stringstream ss;
    string word;
    string orig_mesh_name=argv[1];//IN_MESH_NAME;
    string new_mesh_name=OUT_MESH_NAME;
    string in_line="#";
    bool no_smesh=false;

   


    
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_meta", ios::binary);
    int meta[4];
    in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
    in_file.close();

    m=meta[0];
    b=meta[1];
    n=meta[2];
    num_eles=meta[3];


    Z_original.resize(3,n);
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_nodes", ios::binary);
    in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
    in_file.close();
    Z_original.transposeInPlace();
    
    

    T.resize(4,num_eles);
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_eles", ios::binary);
    in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
    in_file.close();
    //T.transposeInPlace();
    
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  broadcast(comm,n,0);
  broadcast(comm,m,0);
  broadcast(comm,b,0);
  broadcast(comm,num_eles,0);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank!=0){
    Z_original.resize(n,3);
    Z_boundary_transformation.resize(b,3);
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
    //T.resize(num_eles,4);//Comment out if we want to make T a giant window
  }
  MPI_Barrier(MPI_COMM_WORLD);


  broadcast(comm,Z_original.data(),n*3,0);
  broadcast(comm,Z_boundary_transformation.data(),b*3,0);
  //broadcast(comm,T.data(),num_eles*4,0); //Comment out if we want to make T a giant window
  double start,end;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  distributed_femwarp3d_SHM_RMA(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed); //distributed_femwarp3d_RMA works
  MPI_Barrier(MPI_COMM_WORLD);
  end  = MPI_Wtime();
  
  if(rank==0){
    cout<<Z_femwarp_transformed<<endl;//(0,0)<<" "<<-124.835<<endl; 
    cout<<size<<" time: "<<end-start<<endl;
  }


  MPI_Finalize();
}


void distributed_test_packed_3(int argc, char *argv[]){

  boost::mpi::environment env{argc, argv};
  boost::mpi::communicator comm;
  
  int n;//Number nodes
  int m;//Number of interior nodes
  int b;//Number of boundary nodes
  int offset;//n-1
  int num_eles;
  int num_faces;
  int i;//index variable
  int j;//index variable
  int k;//index variable
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_full_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;

  int rank;
  int size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);


  if(rank==0){
    cout<<"Start"<<endl;
    ifstream in_file;
    stringstream ss;
    string word;
    string orig_mesh_name=argv[1];//IN_MESH_NAME;
    string new_mesh_name=OUT_MESH_NAME;
    string in_line="#";



    
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_meta", ios::binary);
    int meta[4];
    in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
    in_file.close();

    m=meta[0];
    b=meta[1];
    n=meta[2];
    num_eles=meta[3];


    Z_original.resize(3,n);
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_nodes", ios::binary);
    in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
    in_file.close();
    Z_original.transposeInPlace();
    
    

    T.resize(4,num_eles);
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_eles", ios::binary);
    in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
    in_file.close();
    //T.transposeInPlace();
    
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();

  }

  MPI_Barrier(MPI_COMM_WORLD);
  broadcast(comm,n,0);
  broadcast(comm,m,0);
  broadcast(comm,b,0);
  broadcast(comm,num_eles,0);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank!=0){
    //Z_original.resize(n,3);
    Z_boundary_transformation.resize(b,3);
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
    //T.resize(num_eles,4);//Comment out if we want to make T a giant window
  }
  MPI_Barrier(MPI_COMM_WORLD);


  //broadcast(comm,Z_original.data(),n*3,0);
  broadcast(comm,Z_boundary_transformation.data(),b*3,0);
  //broadcast(comm,T.data(),num_eles*4,0); //Comment out if we want to make T a giant window
  double start,end;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  distributed_femwarp3d_RMA(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed); //distributed_femwarp3d_RMA works
  MPI_Barrier(MPI_COMM_WORLD);
  end  = MPI_Wtime();
  
  if(rank==0){
    cout<<Z_femwarp_transformed<<endl;//(0,0)<<" "<<-124.835<<endl; 
    cout<<size<<" time: "<<end-start<<endl;
  }


  MPI_Finalize();
}




void serial_test(int argc, char *argv[]){
  int n;//Number nodes
  int m;//Number of interior nodes
  int b;//Number of boundary nodes
  int offset;//n-1
  int num_eles;
  int num_faces;
  int i;//index variable
  int j;//index variable
  int k;//index variable
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_full_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;


  ifstream in_file;
  stringstream ss;
  string word;
  string orig_mesh_name=argv[1];//IN_MESH_NAME;
  string new_mesh_name=OUT_MESH_NAME;
  string in_line="#";
  bool no_smesh=false;
  b=0;



  in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_meta", ios::binary);
  int meta[4];
  in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
  in_file.close();

  m=meta[0];
  b=meta[1];
  n=meta[2];
  num_eles=meta[3];


  Z_original.resize(3,n);
  in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_nodes", ios::binary);
  in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
  in_file.close();
  Z_original.transposeInPlace();
  
  

  T.resize(4,num_eles);
  in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_eles", ios::binary);
  in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
  in_file.close();
  T.transposeInPlace();
  
  Z_femwarp_transformed.resize(m,3);
  Z_femwarp_transformed.setZero();
    
  transformation_matrix_transpose = transformation.transpose();
 //Z_full_transformation=Z_original*transformation_matrix_transpose;
  Z_boundary_transformation=Z_original.block(n-b,0,b,3);//Get only boundary nodes
  Z_boundary_transformation*=transformation_matrix_transpose;

  double start,end;
  start = MPI_Wtime();
  serial_femwarp3d(Z_original,Z_boundary_transformation,T,n,m,b,num_eles,Z_femwarp_transformed);
  end  = MPI_Wtime();
  cout<<Z_femwarp_transformed<<endl;//(0,0)<<" "<<-124.835<<endl; 
  cout<<1<<" time: "<<end-start<<endl;



}

void affine_transformation(Eigen::MatrixXd& Z_cur_boundary){
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;
  transformation_matrix_transpose = transformation.transpose();
  Z_cur_boundary*=transformation_matrix_transpose;
}

void serial_multistep_test(int argc, char *argv[]){
  int n;//Number nodes
  int m;//Number of interior nodes
  int b;//Number of boundary nodes
  int offset;//n-1
  int num_eles;
  int num_faces;
  int i;//index variable
  int j;//index variable
  int k;//index variable
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_full_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;


  ifstream in_file;
  string orig_mesh_name=argv[1];//IN_MESH_NAME;
  string new_mesh_name=OUT_MESH_NAME;

  
  in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_meta", ios::binary);
  int meta[4];
  in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
  in_file.close();

  m=meta[0];
  b=meta[1];
  n=meta[2];
  num_eles=meta[3];


  Z_original.resize(3,n);
  in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_nodes", ios::binary);
  in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
  in_file.close();
  Z_original.transposeInPlace();
  
  

  T.resize(4,num_eles);
  in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_eles", ios::binary);
  in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
  in_file.close();
  T.transposeInPlace();
  
  Z_femwarp_transformed.resize(m,3);
  Z_femwarp_transformed.setZero();


  int num_deformations = 1;
  void (*deformation_functions[])(Eigen::MatrixXd&) = {
    affine_transformation/*,
    affine_transformation,
    affine_transformation,
    affine_transformation,
    affine_transformation,
    affine_transformation*/
  };


  double start,end;
  start = MPI_Wtime();
  serial_multistep_femwarp3d(Z_original,deformation_functions,num_deformations,T,n,m,b,num_eles,Z_femwarp_transformed);
  end  = MPI_Wtime();
  //cout<<Z_original.block(0,0,m,3)<<endl; 
  //cout<<"\n\n\n\n\n\n";

  //cout<<Z_femwarp_transformed<<endl; 

  fstream out_file;
  time_t ctime_cur = time(NULL);
  struct tm ctime_data = *localtime(&ctime_cur);
  string str_time(asctime(&ctime_data));
  str_time.erase(std::remove(str_time.begin(), str_time.end(), '\n'), str_time.end());

  out_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+str_time+" "+orig_mesh_name+".bin_nodes", ios::out | ios::binary);
  out_file.write(reinterpret_cast<char *>(Z_original.data()), n*3*sizeof(double));
  out_file.close();



  //cout<<"\n\n\n\n\n\n";
  //cout<<Z_original.block(0,0,m,3)-Z_femwarp_transformed<<endl; 
  cout<<1<<" time: "<<end-start<<endl;



}

void parallel_affine_transformation(Eigen::MatrixXd& Z_cur_boundary){
  int rank;
  int size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  int N=static_cast<int>(Z_cur_boundary.rows());
  int n_parts=N/size;
  int n_start=rank*n_parts;
  int n_end=rank==size-1?N:(rank+1)*n_parts;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;
  transformation_matrix_transpose = transformation.transpose();
  if(rank==0){
    Z_cur_boundary.block(n_end,0,N-(n_end-n_start),3).setZero();
  }
  else if(rank==size-1){
    Z_cur_boundary.block(0,0,N-(n_end-n_start),3).setZero();
  }
  else{
    Z_cur_boundary.block(0,0,n_start,3).setZero();
    Z_cur_boundary.block(n_end,0,N-n_end,3).setZero();
  }
  Z_cur_boundary.block(n_start,0,n_end-n_start,3)*=transformation_matrix_transpose;
  MPI_Allreduce(MPI_IN_PLACE,Z_cur_boundary.data(),N*3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

void distributed_multistep_test_packed_6(int argc, char *argv[]){

  boost::mpi::environment env{argc, argv};
  boost::mpi::communicator comm;
  
  int n;//Number nodes
  int m;//Number of interior nodes
  int b;//Number of boundary nodes
  int offset;//n-1
  int num_eles;
  int num_faces;
  int i;//index variable
  int j;//index variable
  int k;//index variable
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original_natural_ordering;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;

  int rank;
  int size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);


  if(rank==0){
    cout<<"Start"<<endl;
    ifstream in_file;
    stringstream ss;
    string word;
    string orig_mesh_name=argv[1];//IN_MESH_NAME;
    string new_mesh_name=OUT_MESH_NAME;
    string in_line="#";

    
    
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_meta", ios::binary);
    int meta[4];
    in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
    in_file.close();

    m=meta[0];
    b=meta[1];
    n=meta[2];
    num_eles=meta[3];
    
    Z_original.resize(3,n);
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_nodes", ios::binary);
    in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
    in_file.close();
    Z_original.transposeInPlace();
    
    

    T.resize(4,num_eles);
    in_file.open("./tetgen_meshes/"+orig_mesh_name+".bin_eles", ios::binary);
    in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
    in_file.close();
    //T.transposeInPlace();
    
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  broadcast(comm,n,0);
  broadcast(comm,m,0);
  broadcast(comm,b,0);
  broadcast(comm,num_eles,0);
  MPI_Barrier(MPI_COMM_WORLD);
  int num_deformations=1;
  void (*deformation_functions[])(Eigen::MatrixXd&) = {
    parallel_affine_transformation,
    /*parallel_affine_transformation,
    parallel_affine_transformation,
    parallel_affine_transformation,
    parallel_affine_transformation,
    parallel_affine_transformation*/
  };
  if(rank!=0){
    Z_original.resize(n,3);
    //Z_boundary_transformation.resize(b,3);
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
    //T.resize(num_eles,4);//Comment out if we want to make T a giant window
  }
  MPI_Barrier(MPI_COMM_WORLD);


  broadcast(comm,Z_original.data(),n*3,0);
  //broadcast(comm,Z_boundary_transformation.data(),b*3,0);
  
  //broadcast(comm,T.data(),num_eles*4,0); //Comment out if we want to make T a giant window
  double start,end;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  distributed_multistep_femwarp3d_SHM_RMA(Z_original,deformation_functions,num_deformations,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed); //distributed_femwarp3d_RMA works
  MPI_Barrier(MPI_COMM_WORLD);
  end  = MPI_Wtime();
  
  if(rank==0){
    //cout<<Z_original<<endl;
    //cout<<"=========="<<endl;
    //cout<<Z_femwarp_transformed<<endl;//(0,0)<<" "<<-124.835<<endl; 
    cout<<size<<" time: "<<end-start<<endl;
    string orig_mesh_name=argv[1];//IN_MESH_NAME;
    fstream out_file;
    time_t ctime_cur = time(NULL);
    struct tm ctime_data = *localtime(&ctime_cur);
    string str_time(asctime(&ctime_data));
    str_time.erase(std::remove(str_time.begin(), str_time.end(), '\n'), str_time.end());

    out_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+str_time+" "+orig_mesh_name+".bin_nodes", ios::out | ios::binary);
    out_file.write(reinterpret_cast<char *>(Z_original.data()), n*3*sizeof(double));
    out_file.close();
  }


  MPI_Finalize();
}




int* global_bound_ids;
int global_m;


double beam_deflection(double x,double p){
  return p*(x*x)*(3.0-x)/6.0;
}



void wing_deformation_crm(Eigen::MatrixXd& Z_cur_boundary){
  int i;
  int N=static_cast<int>(Z_cur_boundary.rows());
  double min_max_vals[2];
  double l;
  double p=0.02;
  double theta=p*0.5;//0.0000001;
  min_max_vals[0]=Z_cur_boundary.col(1).minCoeff();
  min_max_vals[1]=Z_cur_boundary.col(1).maxCoeff();
  l=abs(min_max_vals[1]-min_max_vals[0]);
  for(i=0;i<N;i++){
    if(   global_bound_ids[i+global_m] == 3
      ||  global_bound_ids[i+global_m] == 5
      ||  global_bound_ids[i+global_m] == 7
      ||  global_bound_ids[i+global_m] == 9
    )
    {
      Z_cur_boundary(i,2)+=beam_deflection(((Z_cur_boundary(i,1)-min_max_vals[0])/l),p)*l;
      Z_cur_boundary(i,1)=(Z_cur_boundary(i,1)-min_max_vals[0])*cos(theta)+min_max_vals[0];
    }
  }
}



void parallel_wing_deformation_crm(Eigen::MatrixXd& Z_cur_boundary){
  int rank;
  int size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  int N=static_cast<int>(Z_cur_boundary.rows());
  int n_parts=N/size;
  int n_start=rank*n_parts;
  int n_end=rank==size-1?N:(rank+1)*n_parts;
  double min_max_vals[2];
  double l;
  double p=0.02;
  double theta=p*0.5;//0.0000001;
  int i;
  if(rank==0){
    min_max_vals[0]=Z_cur_boundary.col(1).minCoeff();
    min_max_vals[1]=Z_cur_boundary.col(1).maxCoeff();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(min_max_vals, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  l=abs(min_max_vals[1]-min_max_vals[0]);
  if(rank==0){
    Z_cur_boundary.block(n_end,0,N-(n_end-n_start),3).setZero();
  }
  else if(rank==size-1){
    Z_cur_boundary.block(0,0,N-(n_end-n_start),3).setZero();
  }
  else{
    Z_cur_boundary.block(0,0,n_start,3).setZero();
    Z_cur_boundary.block(n_end,0,N-n_end,3).setZero();
  }
  for(i=n_start;i<n_end;i++){
    if(   global_bound_ids[i+global_m] == 3
      ||  global_bound_ids[i+global_m] == 5
      ||  global_bound_ids[i+global_m] == 7
      ||  global_bound_ids[i+global_m] == 9
    )
    {
      Z_cur_boundary(i,2)+=beam_deflection(((Z_cur_boundary(i,1)-min_max_vals[0])/l),p)*l;
      Z_cur_boundary(i,1)=(Z_cur_boundary(i,1)-min_max_vals[0])*cos(theta)+min_max_vals[0];
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,Z_cur_boundary.data(),N*3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

void serial_multistep_test_crm(int argc, char *argv[]){
  int n;//Number nodes
  int m;//Number of interior nodes
  int b;//Number of boundary nodes
  int offset;//n-1
  int num_eles;
  int num_faces;
  int i;//index variable
  int j;//index variable
  int k;//index variable
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_full_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;


  ifstream in_file;
  string orig_mesh_name=argv[1];//IN_MESH_NAME;
  string new_mesh_name=OUT_MESH_NAME;

  
  in_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+orig_mesh_name+".bin_meta", ios::binary);
  int meta[4];
  in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
  in_file.close();

  m=meta[0];
  b=meta[1];
  n=meta[2];
  num_eles=meta[3];




  global_m=m;
  global_bound_ids=(int*)malloc(n*sizeof(int));
  in_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+orig_mesh_name+".bin_bounds", ios::binary);
  in_file.read(reinterpret_cast<char*>(global_bound_ids), n*sizeof(int));
  in_file.close();


  Z_original.resize(3,n);
  in_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+orig_mesh_name+".bin_nodes", ios::binary);
  in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
  in_file.close();
  Z_original.transposeInPlace();
  
  

  T.resize(4,num_eles);
  in_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+orig_mesh_name+".bin_eles", ios::binary);
  in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
  in_file.close();
  T.transposeInPlace();
  
  Z_femwarp_transformed.resize(m,3);
  Z_femwarp_transformed.setZero();

  int num_deformations = 1;
  void (*deformation_functions[])(Eigen::MatrixXd&) = {
    wing_deformation_crm/*,
    affine_transformation,
    affine_transformation,
    affine_transformation,
    affine_transformation,
    affine_transformation*/
  };


  double start,end;
  start = MPI_Wtime();
  serial_multistep_femwarp3d(Z_original,deformation_functions,num_deformations,T,n,m,b,num_eles,Z_femwarp_transformed);
  end  = MPI_Wtime();
  //cout<<Z_original.block(0,0,m,3)<<endl; 
  //cout<<"\n\n\n\n\n\n";
  //cout<<Z_femwarp_transformed<<endl; 
  //cout<<"\n\n\n\n\n\n";
  //cout<<Z_original.block(0,0,m,3)-Z_femwarp_transformed<<endl; 


  fstream out_file;
  time_t ctime_cur = time(NULL);
  struct tm ctime_data = *localtime(&ctime_cur);
  string str_time(asctime(&ctime_data));
  str_time.erase(std::remove(str_time.begin(), str_time.end(), '\n'), str_time.end());

  out_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+str_time+" "+orig_mesh_name+".bin_nodes", ios::out | ios::binary);
  out_file.write(reinterpret_cast<char *>(Z_original.data()), n*3*sizeof(double));
  out_file.close();
  cout<<str_time<<endl;

  cout<<1<<" time: "<<end-start<<endl;
  free(global_bound_ids);


}

void distributed_multistep_test_packed_6_crm(int argc, char *argv[]){

  boost::mpi::environment env{argc, argv};
  boost::mpi::communicator comm;
  
  int n;//Number nodes
  int m;//Number of interior nodes
  int b;//Number of boundary nodes
  int offset;//n-1
  int num_eles;
  int num_faces;
  int i;//index variable
  int j;//index variable
  int k;//index variable
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original_natural_ordering;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);  //Eigen::AngleAxisd transformation (DEF_ANGLE,Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;

  int rank;
  int size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  string orig_mesh_name=argv[1];//IN_MESH_NAME;

  if(rank==0){
    cout<<"Start"<<endl;
    ifstream in_file;
    stringstream ss;
    string word;
    string new_mesh_name=OUT_MESH_NAME;
    string in_line="#";

    



    in_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+orig_mesh_name+".bin_meta", ios::binary);
    int meta[4];
    in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
    in_file.close();

    m=meta[0];
    b=meta[1];
    n=meta[2];
    num_eles=meta[3];


    global_m=m;
    global_bound_ids=(int*)malloc(n*sizeof(int));
    while(global_bound_ids==NULL){
      cout<<"malloc failed on rank "<<rank<<endl;
      global_bound_ids=(int*)malloc(n*sizeof(int));
    }
    in_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+orig_mesh_name+".bin_bounds", ios::binary);
    in_file.read(reinterpret_cast<char*>(global_bound_ids), n*sizeof(int));
    in_file.close();

    
    Z_original.resize(3,n);
    in_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+orig_mesh_name+".bin_nodes", ios::binary);
    in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
    in_file.close();
    Z_original.transposeInPlace();
    
    

    T.resize(4,num_eles);
    in_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+orig_mesh_name+".bin_eles", ios::binary);
    in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
    in_file.close();
    //T.transposeInPlace();
    
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  broadcast(comm,n,0);
  broadcast(comm,m,0);
  broadcast(comm,b,0);
  broadcast(comm,num_eles,0);
  MPI_Barrier(MPI_COMM_WORLD);
  int num_deformations=1;
  void (*deformation_functions[])(Eigen::MatrixXd&) = {
    parallel_wing_deformation_crm,
    /*parallel_wing_deformation_crm,
    /*parallel_affine_transformation,
    parallel_affine_transformation,
    parallel_affine_transformation,
    parallel_affine_transformation*/
  };
  if(rank!=0){
    Z_original.resize(n,3);
    //Z_boundary_transformation.resize(b,3);
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
    //T.resize(num_eles,4);//Comment out if we want to make T a giant window


    
    global_bound_ids=(int*)malloc(n*sizeof(int));
    while(global_bound_ids==NULL){
      cout<<"malloc failed on rank "<<rank<<endl;
      global_bound_ids=(int*)malloc(n*sizeof(int));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);



  global_m=m;
  broadcast(comm,global_bound_ids,n,0);
  broadcast(comm,Z_original.data(),n*3,0);
  //broadcast(comm,Z_boundary_transformation.data(),b*3,0);
  
  //broadcast(comm,T.data(),num_eles*4,0); //Comment out if we want to make T a giant window
  double start,end;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  distributed_multistep_femwarp3d_SHM_RMA(Z_original,deformation_functions,num_deformations,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed); //distributed_femwarp3d_RMA works
  MPI_Barrier(MPI_COMM_WORLD);
  end  = MPI_Wtime();
  
  if(rank==0){
    /*cout<<Z_original<<endl;
    cout<<"=========="<<endl;
    cout<<Z_femwarp_transformed<<endl;//(0,0)<<" "<<-124.835<<endl; */
    fstream out_file;
    time_t ctime_cur = time(NULL);
    struct tm ctime_data = *localtime(&ctime_cur);
    string str_time(asctime(&ctime_data));
    str_time.erase(std::remove(str_time.begin(), str_time.end(), '\n'), str_time.end());

    out_file.open("/kuhpc/work/bigjay/a976h261/tetgen_meshes/"+str_time+" "+orig_mesh_name+".bin_nodes", ios::out | ios::binary);
    out_file.write(reinterpret_cast<char *>(Z_original.data()), n*3*sizeof(double));
    out_file.close();
    //cout<<Z_femwarp_transformed<<endl;

    cout<<str_time<<endl;
    cout<<size<<" time: "<<end-start<<endl;
  }
  free(global_bound_ids);


  MPI_Finalize();
}














int main(int argc, char *argv[]){
  MPI_Init(NULL,NULL);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size>1){
    //distributed_test_packed_6(argc,argv);
    //distributed_multistep_test_packed_6(argc,argv);
    distributed_multistep_test_packed_6_crm(argc,argv);
  }
  else{
    //serial_test(argc,argv);
    MPI_Finalize();
    //serial_multistep_test(argc,argv);
    serial_multistep_test_crm(argc,argv);
  }
}
