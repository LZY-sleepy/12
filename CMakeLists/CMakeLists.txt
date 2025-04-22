#下面引入你刚才写的CMake文件
include(./FindGUROBI.cmake)
include_directories(${GUROBI_INCLUDE_DIRS})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#下面将你的代码链接到动态链接库
#把mip替换为你的代码的名字
ADD_EXECUTABLE(tsp3 ../tsp3.cpp)
TARGET_LINK_LIBRARIES(tsp3 ${catkin_LIBRARIES} ${GUROBI_LIBRARIES} OpenMP::OpenMP_CXX)

ADD_EXECUTABLE(tsp2 ../tsp2.cpp)
TARGET_LINK_LIBRARIES(tsp2 ${catkin_LIBRARIES} ${GUROBI_LIBRARIES} OpenMP::OpenMP_CXX)

ADD_EXECUTABLE(tsp1 ../tsp1.cpp)
TARGET_LINK_LIBRARIES(tsp1 ${catkin_LIBRARIES} ${GUROBI_LIBRARIES} OpenMP::OpenMP_CXX)

