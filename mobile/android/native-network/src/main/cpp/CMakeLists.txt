cmake_minimum_required(VERSION 3.22.1)
project(nanobot_network_jni C)

set(CMAKE_SKIP_RPATH TRUE)

set(go_library_dir "${GO_NETWORK_LIB_DIR}/${ANDROID_ABI}")

add_library(nanobot_network_jni SHARED native_network_jni.c)
target_include_directories(nanobot_network_jni PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../native-network/include")
target_compile_definitions(nanobot_network_jni PRIVATE JNI_CLASS_NAME="${JNI_CLASS_NAME}")
set_target_properties(nanobot_network_jni PROPERTIES C_VISIBILITY_PRESET hidden)
set_property(TARGET nanobot_network_jni APPEND PROPERTY
    LINK_DEPENDS "${go_library_dir}/libnanobot_network.so")
# Gradle packages the Go library from its generated JNI source directory.
target_link_directories(nanobot_network_jni PRIVATE "${go_library_dir}")
target_link_libraries(nanobot_network_jni PRIVATE nanobot_network)
