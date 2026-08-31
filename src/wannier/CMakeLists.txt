#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRC
  wan90.cpp
)
ADD_LIBRARY(wannier_lib ${SRC})
TARGET_LINK_LIBRARIES(wannier_lib PUBLIC utils_f meanfield)
install(TARGETS wannier_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()

