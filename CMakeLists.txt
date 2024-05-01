cmake_minimum_required(VERSION 2.8.3)
project(relocalization_disco)

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  rospy 
  sensor_msgs
  geometry_msgs
  std_msgs
  nav_msgs
  message_generation  
 )

 ## Generate services in the 'srv' folder
add_service_files(
  DIRECTORY srv
  FILES
  relocalize_pointcloud.srv
)

#catkin_python_setup()
add_message_files(
  DIRECTORY msg
  FILES
  DiSCO.msg
)

generate_messages(
  DEPENDENCIES
  nav_msgs
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS    
  geometry_msgs
  sensor_msgs
  std_msgs  
  nav_msgs
  message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
