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
  CSR_Matrix& csr_matrix,
  int offset=0
);

void csr_to_dist_csr
(
  CSR_Matrix& csr_matrix,
  CSR_Matrix& dist_csr_matrix_part,
  int rows,
  boost::mpi::communicator& comm,
  int size,
  int rank
);

void parallel_csr_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result,
  boost::mpi::communicator& comm,
  int rank,
  int size,
  int* num_rows_arr
);

void parallel_csr_x_matrix_optimized
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result,
  Eigen::MatrixXd& tmp_result,
  boost::mpi::communicator& comm,
  int rank,
  int size,
  int* num_rows_arr
);

void parallel_block_diag_matrix_x_matrix
(
  Eigen::MatrixXd& local_A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& result,
  boost::mpi::communicator& comm,
  int rank,
  int size,
  int* num_rows_arr,
  int upto_num_rows_dist
);

void parallel_block_diag_matrices_x_matrix
(
  vector<Eigen::MatrixXd>& local_A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& result,
  boost::mpi::communicator& comm,
  int rank,
  int size,
  int* num_rows_arr,
  int* starting_indices
);

void parallel_diag_matrix_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& result,
  boost::mpi::communicator& comm,
  int rank,
  int size,
  int* num_rows_arr,
  int upto_num_rows_dist
);

void csr_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result
);

void parallel_ATxA
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& ATxA,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_ATxB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& ATxB,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_ATxB_CTxD
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& D,
  Eigen::MatrixXd& ATxB_CTxD,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_matrix_multiplication
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& AxB,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_matrix_addition
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_matrix_subtraction
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_C_add_AB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& C_add_AB,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_C_sub_AB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& C_sub_AB,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_C_add_AB_F_sub_DE
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& D,
  Eigen::MatrixXd& E,
  Eigen::MatrixXd& F,
  Eigen::MatrixXd& C_add_AB_F_sub_DE,
  boost::mpi::communicator& comm,
  int rank,
  int size
);

void parallel_sparse_block_conjugate_gradient_v2
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  boost::mpi::communicator& comm,
  int rank,
  int size,
  int num_rows
);

void parallel_sparse_block_conjugate_gradient_v3
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  boost::mpi::communicator& comm,
  int rank,
  int size,
  int num_rows,
  int block_size
);

void parallel_sparse_block_conjugate_gradient_v4
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  boost::mpi::communicator& comm,
  int rank,
  int size,
  int num_rows,
  int block_size
);

void sparse_block_conjugate_gradient_v2
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X
);


void sparse_block_conjugate_gradient_v3
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X
);

void sparse_block_conjugate_gradient_v4
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X
);


void icf_forward_solve
(
  CSR_Matrix& L,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& Y
);


void icf_backward_solve
(
  CSR_Matrix& U,
  Eigen::MatrixXd& Y,
  Eigen::MatrixXd& X
);


void icf_solve
(
  CSR_Matrix& M,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& Y,
  Eigen::MatrixXd& X
);


void icf
(
  CSR_Matrix& A,
  CSR_Matrix& M
);

