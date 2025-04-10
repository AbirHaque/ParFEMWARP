CXX = mpic++
EIGEN_DIR ?= /usr/include/eigen-3.4.0
PARFEMWARP_INC_DIR = ../../include
PARFEMWARP_LIB_DIR = ../../lib
CXXFLAGS = -std=c++11 -I $(EIGEN_DIR) -I $(PARFEMWARP_INC_DIR)
LIBS = -L$(PARFEMWARP_LIB_DIR) -l_parfemwarp
TARGET = breathing1.o
SRC = breathing1.cpp

all: $(TARGET)
 
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LIBS)
 
clean:
	rm -f $(TARGET)
