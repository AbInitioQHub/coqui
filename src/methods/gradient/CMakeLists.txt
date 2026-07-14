#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRCS
  gradient_common.cpp
  gradient_driver.cpp
  )

add_library(gradient_lib ${SRCS})
target_link_libraries(gradient_lib PUBLIC io_lib mb_state_lib nda_c)
install(TARGETS gradient_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
