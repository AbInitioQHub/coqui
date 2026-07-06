#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(fft_src
   )

if(ENABLE_FFTW)
  set(fft_src 
       ${fft_src}
       fftw.cpp 
     )
endif()

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
target_link_libraries(fft_lib PUBLIC utils nda_c)
if(ENABLE_FFTW)
  target_link_libraries(fft_lib PUBLIC FFTW::Double) 
endif()
if(ENABLE_CUDA)
  target_link_libraries(fft_lib PUBLIC CUDA::cufft) 
endif()
install(TARGETS fft_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
