cmake_minimum_required(VERSION 3.22.1)
project(nanobot_probe C)

set(CMAKE_SKIP_RPATH TRUE)

add_library(nanobot_probe SHARED probe.c)
target_link_libraries(nanobot_probe log)

add_library(nanobot_network SHARED IMPORTED)
set_target_properties(nanobot_network PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libnanobot_network.so"
    IMPORTED_NO_SONAME TRUE)

add_library(nanobot_network_jni SHARED native_network_jni.c)
target_include_directories(nanobot_network_jni PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../native-network/include")
find_library(log-lib log)
target_link_libraries(nanobot_network_jni nanobot_network ${log-lib})
