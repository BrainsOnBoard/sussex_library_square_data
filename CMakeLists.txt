cmake_minimum_required(VERSION 3.1)
include(get_submodule.cmake)

# Default to building vector_field
if(NOT TARGET)
    set(TARGET vector_field)
endif()

# Use solid angle units
add_definitions(-DENABLE_PREDEFINED_SOLID_ANGLE_UNITS)

if(${TARGET} STREQUAL vector_field)
    BoB_project(EXECUTABLE vector_field
                SOURCES vector_field.cc memory.cc
                BOB_MODULES common imgproc navigation)
else()
    BoB_project(EXECUTABLE ridf
                SOURCES ridf.cc memory.cc
                BOB_MODULES common imgproc navigation)
endif()
