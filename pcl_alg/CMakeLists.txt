cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(planar_segmentation)

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp

)

add_definitions(${PCL_DEFINITIONS})
include_directories(  ${catkin_INCLUDE_DIRS}  ${PCL_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} ${VTK_USE_FILE})
#sinclude(  )

link_directories(${PCL_LIBRARY_DIRS} ${VTK_INCLUDE_DIRS})


add_executable (viewer viewer.cpp)
target_link_libraries (viewer ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES})
add_executable (segment segment.cpp)
target_link_libraries (segment ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES})
add_executable (euc euc_cluster_extr.cpp)
target_link_libraries (euc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES})
add_executable (conv gen_convex.cpp)
target_link_libraries (conv ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES})
add_executable (conc ConcaveHull.cpp)
target_link_libraries (conc ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES})
add_executable (smooth smooth.cpp)
target_link_libraries (smooth ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES})
add_executable (wrapped wrapped.cpp)
target_link_libraries (wrapped ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES})
