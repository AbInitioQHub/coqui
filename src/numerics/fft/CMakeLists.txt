#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(fft_src
     fftw.cpp
   )

if(ENABLE_CUDA)
  set(fft_src 
       ${fft_src}
       cufft.cpp 
     )
endif()


# Shared under Python so fftw.cpp lives in one libfft_lib.so: as a static
# archive it is duplicated per c2py module and (under RTLD_LOCAL) can bind
# fftw_* to MKL's shim in one module and real FFTW3 in another -> plan/exec
# mismatch -> crash. Gated on COQUI_PYTHON_SUPPORT like io_lib (see 56645d9).
if(COQUI_PYTHON_SUPPORT)
  add_library(fft_lib SHARED ${fft_src})
else()
  add_library(fft_lib ${fft_src})
endif()
if(COQUI_FFT_BACKEND STREQUAL "FFTW")
  # List FFTW::Double first so libfftw3 precedes MKL in the dynamical linking search order. 
  # FIXME Technically, it does not guarantee mkl's fftw api will not be linked first... 
  target_link_libraries(fft_lib PUBLIC FFTW::Double utils nda_c)
else() # MKL
  target_link_libraries(fft_lib PUBLIC utils nda_c)  # fftw_* resolves to MKL (only provider)
  target_include_directories(fft_lib PUBLIC ${MKL_FFTW_INCLUDE_DIR})
endif()
if(ENABLE_CUDA)
  target_link_libraries(fft_lib PUBLIC CUDA::cufft) 
endif()
install(TARGETS fft_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
