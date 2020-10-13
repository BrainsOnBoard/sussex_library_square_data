# If USE_LOCAL_BOB_ROBOTICS is set, then look for it at $BOB_ROBOTICS_PATH
if(NOT DEFINED ENV{USE_LOCAL_BOB_ROBOTICS} OR "$ENV{USE_LOCAL_BOB_ROBOTICS}" STREQUAL "" OR "$ENV{USE_LOCAL_BOB_ROBOTICS}" STREQUAL "0")
    # Automatically check out git submodule and include main CMake file
    macro(local_exec_or_fail)
        execute_process(COMMAND ${ARGV} RESULT_VARIABLE rv OUTPUT_VARIABLE SHELL_OUTPUT)
        if(NOT ${rv} EQUAL 0)
            message(FATAL_ERROR "Error while executing: ${ARGV}")
        endif()
    endmacro()

    # Checkout git submodules under this path
    find_package(Git REQUIRED)
    local_exec_or_fail(${GIT_EXECUTABLE} submodule update --init bob_robotics
                    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    set(ENV{BOB_ROBOTICS_PATH} ${CMAKE_CURRENT_LIST_DIR}/bob_robotics)
endif()

message("BoB robotics path: $ENV{BOB_ROBOTICS_PATH}")

# Load BoB robotics CMake file
include($ENV{BOB_ROBOTICS_PATH}/cmake/bob_robotics.cmake)
