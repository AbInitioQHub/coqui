#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_library(scf_lib simple_dyson.cpp dca_dyson.cpp dca_dyson.h scf_driver.hpp scf_driver.cpp rpa.cpp scf_common.hpp scf_common.cpp qp_scf_common.cpp mb_solver_t.h qp_params_t.h)
target_link_libraries(scf_lib PUBLIC utils numerics meanfield hamilt ac_lib hf_lib gw_lib gf2_lib methods_tools_lib)
install(TARGETS scf_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
