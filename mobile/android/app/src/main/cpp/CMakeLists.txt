cmake_minimum_required(VERSION 3.22.1)
project(nanobot_probe C)

add_library(nanobot_probe SHARED probe.c)
target_compile_definitions(nanobot_probe PRIVATE JNI_CLASS_NAME="${JNI_CLASS_NAME}")
set_target_properties(nanobot_probe PROPERTIES C_VISIBILITY_PRESET hidden)
