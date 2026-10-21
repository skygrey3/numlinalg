cmake_minimum_required(VERSION 3.16)
project(numlinalg VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(numlinalg src/lu.cpp src/sparse.cpp src/cg.cpp src/cholesky.cpp
            src/eigen.cpp)
target_include_directories(numlinalg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()
add_executable(test_lu tests/test_lu.cpp)
target_link_libraries(test_lu PRIVATE numlinalg)
add_test(NAME lu COMMAND test_lu)

add_executable(test_sparse tests/test_sparse.cpp)
target_link_libraries(test_sparse PRIVATE numlinalg)
add_test(NAME sparse COMMAND test_sparse)

add_executable(test_cg tests/test_cg.cpp)
target_link_libraries(test_cg PRIVATE numlinalg)
add_test(NAME cg COMMAND test_cg)

add_executable(test_cholesky tests/test_cholesky.cpp)
target_link_libraries(test_cholesky PRIVATE numlinalg)
add_test(NAME cholesky COMMAND test_cholesky)

add_executable(test_eigen tests/test_eigen.cpp)
target_link_libraries(test_eigen PRIVATE numlinalg)
add_test(NAME eigen COMMAND test_eigen)
