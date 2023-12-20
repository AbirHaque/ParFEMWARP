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
);

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

void serial_femwarp3d_costly
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  Eigen::MatrixXd& Z_femwarp_transformed
);

void serial_femwarp3d_no_precompute_eigen_costly
(
  Eigen::MatrixXd& Z_original,
  Eigen::MatrixXd& xy_prime,
  Eigen::MatrixXi& T,
  int n,int m,int b,int num_eles,
  Eigen::MatrixXd& Z_femwarp_transformed
);
