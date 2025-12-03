# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(edopt)

if(NOT CMAKE_BUILD_TYPE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
endif()

#include(GNUInstallDirs)

# mandatory use of these packages
find_package(YCM REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver QUIET)
find_package(SuperimposeMesh REQUIRED)
find_package(OpenCV REQUIRED)
find_package(YARP COMPONENTS OS sig math dev idl_tools REQUIRED) #rosmsg
find_package(event-driven REQUIRED)

#include(AddInstallRPATHSupport)
#add_install_rpath_support(BIN_DIRS "${CMAKE_INSTALL_FULL_BINDIR}"
#                          LIB_DIRS "${CMAKE_INSTALL_FULL_LIBDIR}"
#                          INSTALL_NAME_DIR "${CMAKE_INSTALL_FULL_LIBDIR}"
#                          USE_LINK_PATH)

# yarp_add_idl(ROS_MSG SharedData.msg)

add_executable(${PROJECT_NAME} main.cpp image_processing.h projection.h erosdirect.h comparison.h ROScommunication.h ${ROS_MSG})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
#target_compile_options(sixdofobject PRIVATE -Werror -Wall -Wextra)
target_link_libraries(${PROJECT_NAME} PRIVATE  YARP_init
                                            YARP_os
                                            event-driven
                                            ${OpenCV_LIBS}
                                            MetavisionSDK::core
                                            MetavisionSDK::driver
                                            SI::SuperimposeMesh)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})

