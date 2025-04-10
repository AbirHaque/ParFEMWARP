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
#include <algorithm>
#include <ctime>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <csr.hpp>
#include <matrix_helper.hpp>
#include <FEMWARP.hpp>

using namespace std;


void parallel_expand(Eigen::MatrixXd& Z_cur_boundary){
  int rank;
  int size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  int N=static_cast<int>(Z_cur_boundary.rows());
  int n_parts=N/size;
  int n_start=rank*n_parts;
  int n_end=rank==size-1?N:(rank+1)*n_parts;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(1.01,1.01,1.0); 
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


void parallel_contract(Eigen::MatrixXd& Z_cur_boundary){
  int rank;
  int size;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  int N=static_cast<int>(Z_cur_boundary.rows());
  int n_parts=N/size;
  int n_start=rank*n_parts;
  int n_end=rank==size-1?N:(rank+1)*n_parts;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(1.0/1.01,1.0/1.01,1.0);
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


void serial_expand(Eigen::MatrixXd& Z_cur_boundary){
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(1.01,1.01,1.01); 
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;
  transformation_matrix_transpose = transformation.transpose();
  Z_cur_boundary*=transformation_matrix_transpose;
}

void serial_contract(Eigen::MatrixXd& Z_cur_boundary){
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(1/1.01,1/1.01,1/1.01); 
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;
  transformation_matrix_transpose = transformation.transpose();
  Z_cur_boundary*=transformation_matrix_transpose;
}


void distributed_breathing(int argc, char *argv[]){

  //MPI_Init(&argc,&argv);
  MPI_Comm comm = MPI_COMM_WORLD;
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
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original_const;
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
    string orig_mesh_name=argv[1];
    string in_line="#";



 in_file.open(orig_mesh_name+".bin_meta", ios::binary);
    int meta[4];
    in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
    in_file.close();

    m=meta[0];
    b=meta[1];
    n=meta[2];
    num_eles=meta[3];

    Z_original.resize(3,n);
    in_file.open(orig_mesh_name+".bin_nodes", ios::binary);
    in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
    in_file.close();
    Z_original.transposeInPlace();

    Z_original_const.resize(n,3);
    Z_original_const=Z_original.eval();



    T.resize(4,num_eles);
    in_file.open(orig_mesh_name+".bin_eles", ios::binary);
    in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
    in_file.close();

    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
  }

 MPI_Barrier(MPI_COMM_WORLD);

  MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&b,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&num_eles,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);


int num_cycles=2;
int num_steps_per_cycle=8;
int num_deformations=num_cycles*num_steps_per_cycle;

  void (*deformation_functions[num_deformations])(Eigen::MatrixXd&);
  for(i=0;i<num_cycles;i++){
    for(j=0;j<num_steps_per_cycle;j++){
      if(i%2==0){
        deformation_functions[i*num_steps_per_cycle+j]=parallel_expand;
      }
      else{
        deformation_functions[i*num_steps_per_cycle+j]=parallel_contract;
      }
    }
  }
  if(rank!=0){
    Z_original.resize(n,3);
    Z_femwarp_transformed.resize(m,3);
    Z_femwarp_transformed.setZero();
  }
  MPI_Barrier(MPI_COMM_WORLD);
MPI_Bcast(Z_original.data(),n*3,MPI_DOUBLE,0,MPI_COMM_WORLD);


double start,end;
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  distributed_multistep_femwarp3d_SHM_RMA(Z_original,deformation_functions,num_deformations,T,n,m,b,num_eles,comm,size,rank,Z_femwarp_transformed); //distributed_femwarp3d_RMA works
  MPI_Barrier(MPI_COMM_WORLD);
  end  = MPI_Wtime();

  if(rank==0){
    cout<<size<<" Ranks overall time: "<<end-start<<endl;
    double residual=(Z_original_const-Z_original).norm();
    cout<<"||A-A'||_2: "<<residual<<endl;
    if(residual<10E-5){
      cout<<"||A-A'||_2 < 10^-5: Test passed"<<endl;
    }
    else{
      cout<<"||A-A'||_2 >= 10^-5: Test failed"<<endl;
    }
  }


  MPI_Finalize();
}




void serial_breathing(int argc, char *argv[]){
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
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_original_const;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_boundary_transformation;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Z_femwarp_transformed;
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> T;
  Eigen::Matrix3d transformation; transformation=Eigen::Scaling(2.0,2.0,2.0);
  Eigen::Matrix<double,3,3> transformation_matrix_transpose;


  ifstream in_file;
  string orig_mesh_name=argv[1];


  in_file.open(orig_mesh_name+".bin_meta", ios::binary);
  int meta[4];
  in_file.read(reinterpret_cast<char*>(meta), 4*sizeof(int));
  in_file.close();

  m=meta[0];
  b=meta[1];
  n=meta[2];
  num_eles=meta[3];


  Z_original.resize(3,n);
  in_file.open(orig_mesh_name+".bin_nodes", ios::binary);
  in_file.read(reinterpret_cast<char*>(Z_original.data()), n*3*sizeof(double));
  in_file.close();
  Z_original.transposeInPlace();
  Z_original_const.resize(n,3);
  Z_original_const=Z_original.eval();



  T.resize(4,num_eles);
  in_file.open(orig_mesh_name+".bin_eles", ios::binary);
  in_file.read(reinterpret_cast<char*>(T.data()), num_eles*4*sizeof(int));
  in_file.close();
  T.transposeInPlace();

  Z_femwarp_transformed.resize(m,3);
  Z_femwarp_transformed.setZero();


  int num_cycles=2;
  int num_steps_per_cycle=8;
  int num_deformations=num_cycles*num_steps_per_cycle;

  void (*deformation_functions[num_deformations])(Eigen::MatrixXd&);
  for(i=0;i<num_cycles;i++){
    for(j=0;j<num_steps_per_cycle;j++){
      if(i%2==0){
        deformation_functions[i*num_steps_per_cycle+j]=parallel_expand;
      }
      else{
        deformation_functions[i*num_steps_per_cycle+j]=parallel_contract;
      }
    }
  }

  double start,end;
  start = MPI_Wtime();
  serial_multistep_femwarp3d(Z_original,deformation_functions,num_deformations,T,n,m,b,num_eles,Z_femwarp_transformed);
  end  = MPI_Wtime();
  cout<<1<<" Ranks overall time: "<<end-start<<endl;
  double residual=(Z_original_const-Z_original).norm();
  cout<<"||A-A'||_2: "<<residual<<endl;
  if(residual<10E-5){
    cout<<"||A-A'||_2 < 10^-5: Test passed"<<endl;
  }
  else{
    cout<<"||A-A'||_2 >= 10^-5: Test failed"<<endl;
  }
}

    

int main(int argc, char *argv[]){
  MPI_Init(NULL,NULL);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size>1){
    distributed_breathing(argc,argv);
  }
  else{
    serial_breathing(argc,argv);
    MPI_Finalize();
  }
}