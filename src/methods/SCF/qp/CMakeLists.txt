#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_library(qp_lib linearized_qp.cpp linearized_qp.hpp qp_solvers.hpp qp_params_t.h)
target_link_libraries(qp_lib PUBLIC utils numerics)
install(TARGETS qp_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
