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


using namespace std;

void distributed_femwarp3d_RMA
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  boost::mpi::communicator comm,
  int size,
  int rank,
  Eigen::MatrixXd& Z_femwarp_transformed
);

void distributed_femwarp3d_SHM_RMA
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  boost::mpi::communicator comm,
  int size,
  int rank,
  Eigen::MatrixXd& Z_femwarp_transformed
);

void serial_femwarp3d
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  Eigen::MatrixXd& Z_femwarp_transformed
);

void serial_multistep_femwarp3d
(
  Eigen::MatrixXd& Z_original,
  void (*deformation_functions[])(Eigen::MatrixXd&),int num_deformations,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles, 
  Eigen::MatrixXd& Z_femwarp_transformed
);

void distributed_multistep_femwarp3d_SHM_RMA //works
(
  Eigen::MatrixXd& Z_original,
  void (*deformation_functions[])(Eigen::MatrixXd&),int num_deformations,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  boost::mpi::communicator comm,
  int size,
  int rank,
  Eigen::MatrixXd& Z_femwarp_transformed
);
