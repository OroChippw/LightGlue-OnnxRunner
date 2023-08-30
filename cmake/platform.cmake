message(STATUS "Checking Build Platform for: ${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    set(PLATFORM_NAME windows)
else()
    message(FATAL_ERROR "Not support for ${CMAKE_SYSTEM_NAME} now!")
endif()

message(STATUS "Checking Build Platform Done!")