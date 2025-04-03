CXX = mpic++
EIGEN_DIR = /usr/include/eigen3/
CXXFLAGS = -g -std=c++11 -I $(EIGEN_DIR)
SRC = main.cpp matrix_helper.cpp FEMWARP.cpp
OBJ = ParFEMWARP.o

all: $(OBJ)
$(OBJ): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(OBJ) $(SRC)
clean:
	rm -f $(OBJ)