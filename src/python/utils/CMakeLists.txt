
# If GENERATE_PYTHON_BINDINGS,
#   1. we generate python bindings coqui.wrap.cxx from CoQuí.cpp using clair
#   2. then we compile coqui.wrap.cxx
# else
#   1. we directly compile coqui.wrap.cxx provided by the source codes

Python_add_library(mpi_handler MODULE mpi_handler.cpp)

# clair is needed only if we want to generate python bindings from scratch
target_link_libraries(mpi_handler PRIVATE c2py::c2py nda::nda_py utils)
set_target_properties(mpi_handler PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_PATH}/_lib")

install(TARGETS mpi_handler
        LIBRARY DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}/_lib)


Python_add_library(utils_module MODULE utils_module.cpp)

# clair is needed only if we want to generate python bindings from scratch
target_link_libraries(utils_module PRIVATE c2py::c2py nda::nda_py utils)
set_target_properties(utils_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_PATH}/_lib")

install(TARGETS utils_module
        LIBRARY DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}/_lib)

Python_add_library(iaft_module MODULE imag_axes_ft/iaft_module.cpp)
target_link_libraries(iaft_module PRIVATE c2py::c2py nda::nda_py numerics)
set_target_properties(iaft_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PYTHON_PACKAGE_PATH}/_lib")
install(TARGETS iaft_module
        LIBRARY DESTINATION ${PYTHON_PACKAGE_INSTALL_PATH}/_lib)

if (COQUI_UPDATE_PYTHON_BINDINGS)
    add_custom_command(
        COMMAND ${clair-c2py} -p ${PROJECT_BINARY_DIR} mpi_handler.cpp  # -p path/to/compile/commands
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}                   # Execute in source directory
        OUTPUT mpi_handler.wrap.cxx                                     # Generates the .wrap.cxx file
        DEPENDS mpi_handler.cpp                    # which depends on the source (hence will be regenerated if it changes)
    )
    add_custom_target(mpi_bindings_generation DEPENDS mpi_handler.wrap.cxx) # We make the module depend on the bindings
    add_dependencies(mpi_handler mpi_bindings_generation)                   # so that the generation happens before the compilation

    add_custom_command(
        COMMAND ${clair-c2py} -p ${PROJECT_BINARY_DIR} utils_module.cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT utils_module.wrap.cxx
        DEPENDS utils_module.cpp
    )
    add_custom_target(utils_bindings_generation DEPENDS utils_module.wrap.cxx)
    add_dependencies(utils_module utils_bindings_generation mpi_handler)

    add_custom_command(
        COMMAND ${clair-c2py} -p ${PROJECT_BINARY_DIR} imag_axes_ft/iaft_module.cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT imag_axes_ft/iaft_module.wrap.cxx
        DEPENDS imag_axes_ft/iaft_module.cpp
    )
    add_custom_target(iaft_bindings_generation DEPENDS imag_axes_ft/iaft_module.wrap.cxx)
    add_dependencies(iaft_module iaft_bindings_generation)
endif()
