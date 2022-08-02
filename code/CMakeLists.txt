# requires minimum cmake version
cmake_minimum_required(VERSION 3.5)

# produce the cmake var PROJECT_NAME
project(sixdofobject)

if(NOT CMAKE_BUILD_TYPE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
endif()

#include(GNUInstallDirs)

# mandatory use of these packages
find_package(YCM REQUIRED)
find_package(MetavisionSDK COMPONENTS core driver QUIET)
find_package(SuperimposeMesh REQUIRED)
find_package(OpenCV REQUIRED)
find_package(YARP COMPONENTS os REQUIRED)
find_package(event-driven REQUIRED)

#include(AddInstallRPATHSupport)
#add_install_rpath_support(BIN_DIRS "${CMAKE_INSTALL_FULL_BINDIR}"
#                          LIB_DIRS "${CMAKE_INSTALL_FULL_LIBDIR}"
#                          INSTALL_NAME_DIR "${CMAKE_INSTALL_FULL_LIBDIR}"
#                          USE_LINK_PATH)

add_executable(sixdofobject main.cpp)
target_compile_options(sixdofobject PRIVATE -Werror -Wall -Wextra)
target_link_libraries(sixdofobject PRIVATE  YARP_init
                                            YARP_os
                                            event-driven
                                            ${OpenCV_LIBS}
                                            MetavisionSDK::core
                                            MetavisionSDK::driver
                                            SI::SuperimposeMesh)

install(TARGETS sixdofobject DESTINATION ${CMAKE_INSTALL_BINDIR})

