# ParFEMWARP
ParFEMWARP is a parallel mesh warping program

## Copyright Notice:
Copyright 2025 Abir Haque

## License Notice:
This file is part of ParFEMWARP

ParFEMWARP is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License Version 3 as published by
the Free Software Foundation.

ParFEMWARP is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License Version 3 for more details.

You should have received a copy of the GNU Affero General Public License Version 3
along with ParFEMWARP in the file labeled LICENSE.txt.  If not, see https://www.gnu.org/licenses/agpl-3.0.txt

## Author: 
Abir Haque

## Date Last Updated: 
April 3rd, 2025

## Acknowledgements: 
- This software was developed by Abir Haque in collaboration with Dr. Suzanne M. Shontz at the University of Kansas (KU). 
- This work was supported by the following:
    - HPC facilities operated by the Center for Research Computing at KU supported by NSF Grant OAC-2117449,
    - REU Supplement to NSF Grant OAC-1808553,
    - REU Supplement to NSF Grant CBET-2245153,
    - KU School of Engineering Undergraduate Research Fellows Program

## Citing ParFEMWARP: 
If you wish to use this code in your own work, you must review the license at LICENSE.txt and cite the following paper:
- Citation: _Abir Haque, Suzanne Shontz. Parallelization of the Finite Element-based Mesh Warping Algorithm Using Hybrid Parallel Programming. SIAM International Meshing Roundtable Workshop (SIAM IMR 25), March 2025_
- Paper Link: https://internationalmeshingroundtable.com/assets/papers/2025/1018-compressed.pdf

## Setting Up ParFEMWARP
To utilize ParFEMWARP in your own program, you must build the ParFEMWARP library.

### 1. Install MPI
Note: ParFEMWARP has only been tested with OpenMPI 4.1
At the bare minimum, your MPI implementation **must** support the MPI 3.0 standard, as ParFEMWARP utilizes several RMA functions (including shared memory).

### 2. Install Eigen in your desired location
```
cd <INSERT_LOCATION>
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xvzf eigen-3.4.0.tar.gz
```
### 3. Build the ParFEMWARP static library
```
cd <PARFEMWARP_LOCATION>/ParFEMWARP/src
make EIGEN_DIR=<INSERT_LOCATION>/eigen-3.4.0/
```
Note: ParFEMWARP sets EIGEN_DIR as `/usr/include/eigen-3.4.0 by default`

## Compiling programs using ParFEMWARP
You simply need to include Eigen headers, ParFEMWARP headers (FEMWARP.hpp, matrix_helper.hpp, csr.hpp), and link ParFEMWARP
```
mpic++ -std=c++11 \
    -I <INSERT_LOCATION>/eigen-3.4.0/ \
    -I <PARFEMWARP_LOCATION>/ParFEMWARP/src/include \
    -o YOUR_BINARY.o YOUR_PROGRAM.cpp \
    -L<PARFEMWARP_LOCATION>/ParFEMWARP/src/lib -l_parfemwarp
```

## Executing applications using ParFEMWARP
- If your application only uses serial ParFEMWARP functions (i.e. does not utilize MPI), then you can execute your application without `mpirun`.
- If your application utilizes parallel ParFEMWARP functions (i.e. does utilize MPI), then you must use `mpirun`.
- Note: Depending on your hardware and system configuration, you may need to explicitly supply `mpirun` with various MCA parameters to support shared memory and/or inter-node communication.
