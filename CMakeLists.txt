cmake_minimum_required(VERSION 3.20)

project(OSCDA VERSION 0.3.0 LANGUAGES C)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "OSCDA's Intel backend currently supports Linux only")
endif()

include(GNUInstallDirs)
include(CTest)
if(NOT MINGW)
    find_package(Threads REQUIRED)
endif()

add_library(osc SHARED src/runtime.c src/intel_backend.c src/cuda_registration.c)
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

    if(UNIX)
        add_library(mock_level_zero SHARED tests/mock_level_zero.c)
        target_compile_options(mock_level_zero PRIVATE -Wall -Wextra -Wpedantic -Werror)

        add_executable(intel_runtime_test tests/intel_runtime_test.c)
        target_compile_options(intel_runtime_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_link_libraries(intel_runtime_test PRIVATE osc)
        add_test(NAME intel_runtime COMMAND intel_runtime_test)
        set_tests_properties(intel_runtime PROPERTIES ENVIRONMENT
            "OSC_LEVEL_ZERO_LIBRARY=$<TARGET_FILE:mock_level_zero>;OSC_INTEL_MEMORY_BUDGET_MB=64;OSC_INTEL_TIMEOUT_MS=25")

        add_test(NAME intel_runtime_discrete COMMAND intel_runtime_test discrete)
        set_tests_properties(intel_runtime_discrete PROPERTIES ENVIRONMENT
            "OSC_LEVEL_ZERO_LIBRARY=$<TARGET_FILE:mock_level_zero>;OSC_INTEL_MEMORY_BUDGET_MB=64;OSC_INTEL_TIMEOUT_MS=25;MOCK_ZE_DISCRETE=1")

        add_executable(intel_timeout_test tests/intel_timeout_test.c)
        target_compile_options(intel_timeout_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_link_libraries(intel_timeout_test PRIVATE osc)
        add_test(NAME intel_timeout COMMAND intel_timeout_test)
        set_tests_properties(intel_timeout PROPERTIES ENVIRONMENT
            "OSC_LEVEL_ZERO_LIBRARY=$<TARGET_FILE:mock_level_zero>;OSC_INTEL_MEMORY_BUDGET_MB=64;OSC_INTEL_TIMEOUT_MS=1;MOCK_ZE_TIMEOUT=1")
    endif()

endif()

install(TARGETS osc
    EXPORT OSCDATargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT OSCDATargets
    FILE OSCDATargets.cmake
    NAMESPACE OSC::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OSCDA
)
