#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

add_subdirectory(fft)
add_subdirectory(device_kernels)
add_subdirectory(ac)
add_subdirectory(imag_axes_ft)

add_library(numerics INTERFACE)
if(ENABLE_SLATE)
target_link_libraries(numerics INTERFACE arch_lib fft_lib nda_c h5_c slate_lapack_api slate device_kernels iaft_utils_lib)
ELSE(ENABLE_SLATE)
target_link_libraries(numerics INTERFACE arch_lib fft_lib nda_c h5_c device_kernels iaft_utils_lib)
ENDIF(ENABLE_SLATE)

if(ENABLE_DLR)
  target_link_libraries(numerics INTERFACE cppdlr::cppdlr_c)
endif()

install(TARGETS numerics
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(ENABLE_TBLIS)
add_subdirectory(sandbox)
endif(ENABLE_TBLIS)

if (BUILD_UNIT_TESTS)
  add_subdirectory(fft/tests)
  add_subdirectory(distributed_array/tests)
  add_subdirectory(shared_array/tests)
  add_subdirectory(sparse/tests)

  if(ENABLE_SLATE)
    add_subdirectory(slate/tests)
  endif(ENABLE_SLATE)

  add_subdirectory(tests)
endif()

