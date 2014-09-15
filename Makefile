
ifndef CXX
CXX=clang++
endif

CPP_FLAG = -O3 -std=c++11 -stdlib=libc++
CNN_INCLUDE = -I. -I/Users/amir/Desktop/Spring2014/deepdive/lib/dw_mac/lib/tclap/include -I/lfs/local/0/amir/deepdive/lib/dw_mac/lib/tclap/include
CNN_FLAG = -O3 -std=c++0x
CPP_INCLUDE = -I./src
NUMA_FLAG= -I./lib/libunwind-1.1/include -L./lib/numactl-2.0.9 -I./lib/numactl-2.0.9 
CPP_LAST = -lnuma -lrt

cnn: clean
	$(CXX) $(CNN_FLAG) $(CPP_INCLUDE) $(CNN_INCLUDE) $(NUMA_FLAG) src/main.cpp -o cnn $(CPP_LAST)

local: clean
	$(CXX) $(CNN_FLAG) $(CPP_INCLUDE) $(CNN_INCLUDE) $(NUMA_FLAG) src/main.cpp -o cnn

exp:
	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) examples/logistic_regression_dense_sgd.cpp -o example


test_dep:

	$(CXX) -O3 -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -c ./lib/gtest-1.7.0/src/gtest_main.cc

	$(CXX) -O3 -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -c ./lib/gtest-1.7.0/src/gtest-all.cc
	
runtest:

	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) -I./test -I./lib/gtest-1.7.0/include/ -I./lib/gtest-1.7.0/ -c test/glm.cc

	$(CXX) $(CPP_FLAG) gtest_main.o  glm.o gtest-all.o -o run_test

	./run_test

julia:

	$(CXX) $(CPP_FLAG) $(CPP_INCLUDE) -I./src -I./lib/julia/src/ -I./lib/libsupport/ -I./lib/libuv/include/ \
			-dynamiclib src/helper/julia_helper.cpp -o libdw_julia.dylib

clean: 
	-rm cnn 
	-rm example
