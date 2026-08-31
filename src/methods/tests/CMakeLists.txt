if (BUILD_UNIT_TESTS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

    set(TEST_ID methods_mbpt)

    add_executable(test_${TEST_ID} test_mbpt.cpp)
    target_link_libraries(test_${TEST_ID} catch_main utils numerics meanfield methods_lib)

    add_mpi_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}" ${CTEST_NPROC})
    set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
endif()
