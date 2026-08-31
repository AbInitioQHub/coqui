#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(SRCS
  iaft_utils.cpp
)

if(USE_DEVICE_COMPILER)
if(ENABLE_CUDA)
  foreach( X ${SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()
endif()

add_library(iaft_utils_lib ${SRCS})
target_link_libraries(iaft_utils_lib PUBLIC h5_c nda_c utils)
if(ENABLE_DLR)
  target_link_libraries(iaft_utils_lib PUBLIC cppdlr::cppdlr_c)
endif()

# Install ir data directory to the install directory under 'share/ir/data'
set(DATA_DIR "${PROJECT_SOURCE_DIR}/src/numerics/imag_axes_ft/ir/data")
file(GLOB H5_FILES "${DATA_DIR}/*.h5")
install(FILES ${H5_FILES}
        DESTINATION data/ir)

if (BUILD_UNIT_TESTS)
  if(ENABLE_DLR)
    add_subdirectory(dlr/tests)
  endif(ENABLE_DLR)
  add_subdirectory(ir/tests)
  add_subdirectory(tests)
endif()
