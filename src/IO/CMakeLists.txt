#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

SET(SRCS_IO
  ptree/InputParser.hpp
  app_loggers.cpp
)

IF(COQUI_PYTHON_SUPPORT)
  ADD_LIBRARY(io_lib SHARED ${SRCS_IO})
ELSE()
  ADD_LIBRARY(io_lib ${SRCS_IO})
ENDIF()

TARGET_LINK_LIBRARIES(io_lib PUBLIC tomlplusplus::tomlplusplus nda_c h5_c nda_warnings)
IF(ENABLE_SPDLOG)
  TARGET_LINK_LIBRARIES(io_lib PUBLIC spdlog::spdlog)  
ENDIF()
IF(ENABLE_CPPTRACE)
  TARGET_LINK_LIBRARIES(io_lib PUBLIC cpptrace::cpptrace)  
ENDIF()
install(TARGETS io_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

