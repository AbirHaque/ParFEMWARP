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


double volume_tetrahedron
(
  double a[], 
  double b[], 
  double c[], 
  double d[]
);

Eigen::Matrix4d gen_element_stiffness_matrix_tetrahedron
(
  Eigen::Matrix4d& gradients,
  double a[],
  double b[],
  double c[],
  double d[]
);

void gen_basis_function_gradients_tetrahedron
(
  double* a, 
  double* b, 
  double* c, 
  double* d,
  Eigen::Matrix4d& ret
);

int gen_neighbors_csr
(
  unordered_map<int, set<int>>& neighbors_dict,
  CSR_Matrix<double>& csr_matrix,
  int offset=0
);

void csr_to_dist_csr
(
  CSR_Matrix<double>& csr_matrix,
  CSR_Matrix<double>& dist_csr_matrix_part,
  int rows,
  boost::mpi::communicator comm,
  int size,
  int rank
);

void parallel_csr_x_matrix
(
  CSR_Matrix<double>& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result,
  boost::mpi::communicator comm,
  int rank,
  int size,
  int num_rows_arr[]
);

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
);

void csr_x_matrix
(
  CSR_Matrix<double>& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result
);

void parallel_ATxA
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& ATxA,
  boost::mpi::communicator comm,
  int rank,
  int size
);

void parallel_ATxB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& ATxB,
  boost::mpi::communicator comm,
  int rank,
  int size
);

void parallel_matrix_multiplication
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& AxB,
  boost::mpi::communicator comm,
  int rank,
  int size
);

void parallel_matrix_addition
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  boost::mpi::communicator comm,
  int rank,
  int size
);

void parallel_matrix_subtraction
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  boost::mpi::communicator comm,
  int rank,
  int size
);

void parallel_sparse_block_conjugate_gradient_v2
(
  CSR_Matrix<double>& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  boost::mpi::communicator comm,
  int rank,
  int size,
  int num_rows
);

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
);

void sparse_block_conjugate_gradient_v2
(
  CSR_Matrix<double>& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X
);
