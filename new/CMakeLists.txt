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


add_executable (planar_segmentation planar_segmentation.cpp)
target_link_libraries (planar_segmentation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES})

