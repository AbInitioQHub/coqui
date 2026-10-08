Quantum ESPRESSO Converter for CoQuí
------------------------------------
**Last Updated:** Sep. 1, 2025

This directory provides the Fortran source code `pw2coqui.f90` required to 
interface **Quantum ESPRESSO (QE)** with **CoQuí**. The converter extracts 
the necessary data from QE calculations and makes it available for use in 
CoQuí workflows.

## 📦 Installation

The converter is distributed with the CoQuí source code, but it is **not 
compiled as part of CoQuí itself**. Instead, you need to integrate it into 
your local QE source tree and compile it together with QE.

### Step 1: Locate the converter source files
Copy `pw2coqui.f90` into the **`PP/src/` folder** (the *PostProc* package in the
QE suite) of your QE source tree:  
```bash
cp -r pw2coqui.f90 /path/to/qe-source/PP/src/
```

### Step 2: Modify QE’s CMakeLists.txt
Update the QE build system so that the converter is compiled. Inside the
`PostProc` package folder within your QE source tree (e.g. `/path/to/qe-source/PP/`):
1. Open the `CMakeLists.txt` file.
2. Insert the following block: 
   ```cmake
   ###########################################################
   # pw2coqui.x
   ###########################################################
   set(src_pw2coqui_x src/pw2coqui.f90)
   qe_add_executable(qe_pp_pw2coqui_exe ${src_pw2coqui_x})
   set_target_properties(qe_pp_pw2coqui_exe PROPERTIES OUTPUT_NAME pw2coqui.x)
   target_link_libraries(qe_pp_pw2coqui_exe
       PRIVATE
           qe_pw
           qe_modules
           qe_pp
           qe_upflib
           qe_fftx
           qe_mpi_fortran
           qe_xclib)
   ```
   above the section where `PP_EXE_TARGETS` is defined: 
   ```cmake 
   set(PP_EXE_TARGETS
       ...
   ) 
   ```
3. Add the new executable target to the list of `PP_EXE_TARGETS`:
   ```cmake
   set(PP_EXE_TARGETS
       ...
       qe_pp_pw2coqui_exe
       ...
   )
   ```

### Step 3: Recompile QE
Rebuild QE with the modified source and CMake configuration.
After successful compilation, the executable `pw2coqui.x` will be available
in your QE build directory (typically `bin/` inside the build tree).


## Usage

> **`pw2coqui.x` is serial only — run it on a single MPI rank.**
> ```shell
> pw2coqui.x < svo.pw2coqui.in          # or: mpirun -np 1 pw2coqui.x -in svo.pw2coqui.in
> ```
> The preceding `pw.x` scf/nscf steps are of course run in parallel as usual;
> only the conversion is serial. The converter opens its HDF5 file, and every
> group inside it, on `ionode` alone, so additional ranks have nothing valid to
> write through. The program now refuses to start on more than one rank rather
> than produce a damaged file. See issue
> [#11](https://github.com/AbInitioQHub/coqui/issues/11).

Once QE has been recompiled with the converter, you can proceed with the 
CoQuí [Quickstart tutorials](https://github.com/AbInitioQHub/coqui-tutorial/blob/main/quickstart/01s_dft_to_coqui_converter.ipynb) 
to generate CoQuí inputs from QE.