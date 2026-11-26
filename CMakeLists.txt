cmake_minimum_required(VERSION 3.20)

project(OSCDA VERSION 0.2.3 LANGUAGES C)

include(GNUInstallDirs)
include(CTest)
if(NOT MINGW)
    find_package(Threads REQUIRED)
endif()

add_library(osc SHARED src/runtime.c src/cuda_registration.c)
target_compile_features(osc PUBLIC c_std_11)
target_compile_definitions(osc PRIVATE OSCDA_BUILD)
target_compile_options(osc PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(osc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(MINGW)
    target_link_libraries(osc PRIVATE winpthread)
else()
    target_link_libraries(osc PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()
set_target_properties(osc PROPERTIES
    C_VISIBILITY_PRESET hidden
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_executable(amd_vector_add examples/amd_vector_add.c)
target_compile_options(amd_vector_add PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(amd_vector_add PRIVATE osc)

add_executable(amd_matmul examples/amd_matmul.c)
target_compile_options(amd_matmul PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(amd_matmul PRIVATE osc)

if(BUILD_TESTING)
    add_executable(runtime_test tests/runtime_test.c)
    target_compile_options(runtime_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(runtime_test PRIVATE osc)
    add_test(NAME runtime COMMAND runtime_test)
    set_tests_properties(runtime PROPERTIES ENVIRONMENT "OSC_BACKEND=cpu")

    add_executable(cuda_registration_test tests/cuda_registration_test.c)
    target_compile_options(cuda_registration_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(cuda_registration_test PRIVATE osc)
    add_test(NAME cuda_registration COMMAND cuda_registration_test)

    add_executable(amd_runtime_test tests/amd_runtime_test.c)
    target_compile_options(amd_runtime_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(amd_runtime_test PRIVATE osc)
    add_test(NAME amd_runtime COMMAND amd_runtime_test)
    set_tests_properties(amd_runtime PROPERTIES SKIP_RETURN_CODE 77)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME ptx_frontend COMMAND ${Python3_EXECUTABLE} -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/ptx_test.py")
    endif()
endif()

install(TARGETS osc
    EXPORT OSCDATargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(PROGRAMS tools/osc_ptx.py DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME osc-ptx)
install(EXPORT OSCDATargets
    FILE OSCDATargets.cmake
    NAMESPACE OSC::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OSCDA
)
