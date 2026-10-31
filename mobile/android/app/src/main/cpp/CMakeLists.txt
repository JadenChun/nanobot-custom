cmake_minimum_required(VERSION 3.22.1)
project(nanobot_probe C)

add_library(nanobot_probe SHARED probe.c)
target_link_libraries(nanobot_probe log)
