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
  MPI_Comm comm,
  int size,
  int rank
);

void parallel_csr_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& matrix,
  Eigen::MatrixXd& result,
  MPI_Comm comm,
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
  MPI_Comm comm,
  int rank,
  int size,
  int* num_rows_arr
);


void parallel_diag_matrix_x_matrix
(
  CSR_Matrix& csr,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& result,
  MPI_Comm comm,
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
  MPI_Comm comm,
  int rank,
  int size
);

void parallel_ATxB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& ATxB,
  MPI_Comm comm,
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
  MPI_Comm comm,
  int rank,
  int size
);

void parallel_matrix_multiplication
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& AxB,
  MPI_Comm comm,
  int rank,
  int size
);

void parallel_matrix_addition
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  MPI_Comm comm,
  int rank,
  int size
);

void parallel_matrix_subtraction
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& A_B,
  MPI_Comm comm,
  int rank,
  int size
);

void parallel_C_add_AB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& C_add_AB,
  MPI_Comm comm,
  int rank,
  int size
);

void parallel_C_sub_AB
(
  Eigen::MatrixXd& A,
  Eigen::MatrixXd& B,
  Eigen::MatrixXd& C,
  Eigen::MatrixXd& C_sub_AB,
  MPI_Comm comm,
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
  MPI_Comm comm,
  int rank,
  int size
);

void parallel_sparse_block_conjugate_gradient_v2
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  MPI_Comm comm,
  int rank,
  int size,
  int num_rows
);

void parallel_sparse_block_conjugate_gradient_v3
(
  CSR_Matrix& local_A,
  Eigen::MatrixXd& global_b,
  Eigen::MatrixXd& X,
  MPI_Comm comm,
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
  MPI_Comm comm,
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