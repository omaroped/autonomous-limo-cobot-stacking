cmake_minimum_required(VERSION 3.10)
project(gazebo_grasp_plugin)

find_package(gazebo REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(${PROJECT_NAME} SHARED gazebo_grasp_fix.cpp)
target_link_libraries(${PROJECT_NAME} ${GAZEBO_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ament_index_marker" "")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ament_index_marker"
        DESTINATION share/ament_index/resource_index/packages
        RENAME ${PROJECT_NAME})

install(FILES package.xml DESTINATION share/${PROJECT_NAME})
