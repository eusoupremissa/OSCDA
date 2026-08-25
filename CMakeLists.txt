cmake_minimum_required(VERSION 3.20)

project(UniWarp VERSION 0.1.0 LANGUAGES C)

include(GNUInstallDirs)
include(CTest)
find_package(Threads REQUIRED)

add_library(uniwarp SHARED src/runtime.c)
target_compile_features(uniwarp PUBLIC c_std_11)
target_compile_definitions(uniwarp PRIVATE UNI_WARP_BUILD)
target_compile_options(uniwarp PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(uniwarp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(uniwarp PRIVATE Threads::Threads)
set_target_properties(uniwarp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(BUILD_TESTING)
    add_executable(runtime_test tests/runtime_test.c)
    target_compile_options(runtime_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(runtime_test PRIVATE uniwarp)
    add_test(NAME runtime COMMAND runtime_test)
endif()

install(TARGETS uniwarp
    EXPORT UniWarpTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT UniWarpTargets
    FILE UniWarpTargets.cmake
    NAMESPACE UniWarp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/UniWarp
)
