# Change EIGEN_DIR to location of eigen
EIGEN_DIR = /usr/include/eigen3/




PARFEMWARP_INC_DIR = ../include
PARFEMWARP_BIN_DIR = ../bin
OUT_DIR = ../lib
CXX = mpic++

CXXFLAGS = -std=c++11 -I $(EIGEN_DIR) -I $(PARFEMWARP_INC_DIR) -fPIC
SRC = matrix_helper.cpp FEMWARP.cpp
OBJ = matrix_helper.o FEMWARP.o
LIB = $(OUT_DIR)/lib_parfemwarp.a

%.o: %.cpp
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB): $(OBJ)
	ar rcs $@ $^


clean:
	rm -f $(LIB) $(OBJ)