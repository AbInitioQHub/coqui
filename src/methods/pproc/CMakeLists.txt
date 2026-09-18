#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_library(pproc_lib pproc_t.cpp hamiltonians.cpp wavefunction_utils.cpp)
if(ENABLE_NEVANLINNA)
    target_link_libraries(pproc_lib PUBLIC utils numerics meanfield hamilt embed_lib ac_lib qp_lib)
else()
    target_link_libraries(pproc_lib PUBLIC utils numerics meanfield hamilt embed_lib qp_lib)
endif()
install(TARGETS pproc_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
