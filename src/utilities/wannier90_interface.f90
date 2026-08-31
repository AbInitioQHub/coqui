!=============================================================================
! CoQuí: Correlated Quantum ínterface
!
! Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!=============================================================================


! CoQui already checked that atoms and lattice vectors are consistent 
subroutine wann90_setup(prefix, clen, nb, nw, nat, at_cart, max_sym_str_len, at_syms, sym_str_len, &
                        eval, uccart, nk, nkabc, kpt,  &
                        nn, nnkp_size, nnkp_out, auto_proj, nproj, max_proj_str_len, proj_str, &
                        proj_str_len, proj_ints, proj_reals,  &
                        write_nnkp, exclude_string, exclude_str_len, ierr)
  use mpi
  use w90_constants, only: dp, maxlen
  use w90_library
  use w90_library_extra, only: write_kmesh 
  USE ISO_C_BINDING, ONLY: C_CHAR, C_INT, C_DOUBLE, C_DOUBLE_COMPLEX, C_BOOL

  implicit none

  character(kind=C_CHAR,len=256), intent(in) :: prefix, exclude_string
  integer(C_INT), intent(IN)  :: clen, nb, nw, nnkp_size, nat, max_sym_str_len, auto_proj
  integer(C_INT), intent(in)  :: nk, nkabc(3), nproj, proj_str_len(nproj), max_proj_str_len
  real(C_DOUBLE), intent(in)  :: kpt(3,nk), uccart(3,3), eval(nb,nk)
  real(C_DOUBLE), intent(in) :: at_cart(3,nat)
  character(kind=C_CHAR,len=max_sym_str_len), intent(in) :: at_syms(nat)
  integer(C_INT), intent(out) :: nn, nnkp_out(4,nnkp_size,nk)
  integer(C_INT), intent(in)  :: sym_str_len(nat), exclude_str_len
  character(kind=C_CHAR,len=max_proj_str_len), intent(in) :: proj_str(nproj)
  ! for each projector: l, mr, rad, s: 4 
  integer(C_INT), intent(out) :: proj_ints(4,nw) 
  ! for each projector: site(3),zaxis(3),xaxis(3),sqa(3),zona(1) : 13
  real(C_DOUBLE), intent(out) :: proj_reals(13,nw) 
  logical(C_BOOL), intent(in) :: write_nnkp 
  integer(C_INT), intent(out) :: ierr


  ! local variables
  integer, allocatable :: nnkp(:, :), gkpb(:,:,:)
  integer, allocatable :: l(:), m(:), s(:), rad(:)
  real(kind=dp), allocatable :: site(:, :)
  real(8), allocatable :: sqa(:, :), z(:, :), x(:, :), zona(:)
  integer :: stdout, stderr
  CHARACTER(len=maxlen) :: sym_f_string(nat)
  
!  integer, allocatable :: distk(:)
  integer :: i, j, ik, nkl, np
  integer :: mpisize, mpirank
  type(lib_common_type), target :: w90main

  ! wannier interface starts
  ! stdout/err
  call w90_get_fortran_stdout(stdout)
  call w90_get_fortran_stderr(stderr)

  ! crude k distribution
!  allocate (distk(nk))
!  nkl = nk ! number of kpoints per rank
!  if (mod(nk, mpisize) > 0) nkl = nkl + 1
!  do i = 1, nk
!    distk(i) = (i - 1)/nkl ! contiguous blocks with potentially fewer processes on last rank
!  enddo

  if(max_sym_str_len .gt. maxlen) then
    ierr = 3001
    write(stderr,*) 'Error in wann90_setup: Atomic label string are too long:',max_sym_str_len,maxlen
    return
  endif
  if(max_proj_str_len .gt. maxlen) then
    ierr = 3002
    write(stderr,*) 'Error in wann90_setup: Projection strings are too long:',max_sym_str_len,maxlen
    return
  endif

  ! required settings
  call w90_set_option(w90main, 'kpoints', kpt)
  call w90_set_option(w90main, 'mp_grid', nkabc)
  call w90_set_option(w90main, 'num_bands', nb)
  call w90_set_option(w90main, 'num_kpts', nk)
  call w90_set_option(w90main, 'num_wann', nw)
  call w90_set_option(w90main, 'unit_cell_cart', uccart)
  if(write_nnkp .and. exclude_str_len.gt.0) then
    call w90_set_option(w90main, 'exclude_bands', TRIM(exclude_string(1:exclude_str_len)))
  endif

  do i=1,nat
    sym_f_string(i)(:) = ' '
    sym_f_string(i)(1:sym_str_len(i)) = TRIM(at_syms(i)(1:sym_str_len(i)))
  enddo
  call w90_set_option(w90main, 'symbols', sym_f_string)
  call w90_set_option(w90main, 'atoms_cart', at_cart)

  if(nproj .gt. 0) then
! library has a bug in src/readwrite.F90, line: 3912, should also cycle if txtdata is 'bohr' 
!    call w90_set_option(w90main, 'projections', 'bohr')
    do i=1,nproj
      call w90_set_option(w90main, 'projections', TRIM(proj_str(i)(1:proj_str_len(i))))
    enddo
  elseif(auto_proj .gt. 0) then
    call w90_set_option(w90main, 'auto_projections', .true.)
  endif

  ! enable mpi later on
  call w90_set_comm(w90main, mpi_comm_self)
  call w90_input_setopt(w90main, TRIM(prefix(1:clen)), stdout, stderr, ierr) ! apply settings
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_input_setopt: ',ierr
    return
  endif

  ! optional settings read from *.win file
  call w90_input_reader(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_input_reader: ',ierr
    return
  endif

  call w90_get_nn(w90main, nn, stdout, stderr, ierr); 
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_get_nn: ',ierr
    return
  endif
  if(nn .gt. nnkp_size) then
    ierr = 1001
    return
  endif

  if(write_nnkp) then
    call write_kmesh(w90main, stdout, stderr, ierr)
    if(ierr .ne. 0) then
      write(stderr,*) ' Error in write_kmesh: ',ierr
      return
    endif
  endif 

  allocate (nnkp(nk, nn), stat=ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in allocate: ',ierr
    return
  endif
  call w90_get_nnkp(w90main, nnkp, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_get_nnkp: ',ierr
    return
  endif

  !! gkpb must be dimensioned (3,nk,nnb)
  allocate (gkpb(3, nk, nn), stat=ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in allocate: ',ierr
    return
  endif
  call w90_get_gkpb(w90main, gkpb, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_get_nnkp: ',ierr
    return
  endif

  do i = 1, nn
    do ik = 1, nk
      nnkp_out(1,i,ik) = nnkp(ik,i)
      nnkp_out(2:4,i,ik) = gkpb(1:3,ik,i)
    enddo
  enddo

  if(allocated(nnkp)) deallocate( nnkp )
  if(allocated(gkpb)) deallocate( gkpb )

  if( nproj .gt. 0 ) then
    allocate( site(3,nw), l(nw), m(nw), s(nw), rad(nw), x(3,nw), &
              z(3, nw), sqa(3,nw), zona(nw), stat=ierr )
    if(ierr .ne. 0) then
      write(stderr,*) ' Error in allocate: ',ierr
      return
    endif
    call w90_get_proj(w90main, np, site, l, m, s, rad, x, z, sqa, zona, &
                            stdout, stderr, ierr)
    if(ierr .ne. 0) then
      write(stderr,*) ' Error in w90_get_proj: ',ierr
      return
    endif
    if(np .ne. nw) then
      ierr = 2002
      write(stderr,*) ' Error in w90_get_proj: nw incompatible with w90_get_proj. ',ierr
      return
    endif
   
    do i=1,nw
      proj_ints(1,i)=l(i)
      proj_ints(2,i)=m(i)
      proj_ints(3,i)=rad(i)
      proj_ints(4,i)=s(i)
      proj_reals(1:3,i) = site(:,i) 
      proj_reals(4:6,i) = z(:,i) 
      proj_reals(7:9,i) = x(:,i) 
      proj_reals(10:12,i) = sqa(:,i) 
      proj_reals(13,i) = zona(i) 
    enddo
  
    deallocate(site, l, m, s, rad, x, z, sqa, zona)
  endif

end subroutine wann90_setup

! CoQui already checked that atoms and lattice vectors are consistent 
subroutine wann90_run(prefix, clen, nb, nw, nat, at_cart, max_sym_str_len, at_syms, sym_str_len, &
                      eval, uccart, nk, nkabc, kpt, nn,  &
                      auto_proj, nproj, max_proj_str_len, proj_str, proj_str_len, &
                      m_matrix, u_matrix_opt, centers, spreads, ierr) 
  use mpi
  use w90_constants, only: dp, maxlen
  use w90_library
  use w90_utility, only: utility_zgemm_new
  USE ISO_C_BINDING, ONLY: C_CHAR, C_INT, C_DOUBLE, C_DOUBLE_COMPLEX

  implicit none

  character(kind=C_CHAR,len=256), intent(in) :: prefix
  integer(C_INT), intent(IN)  :: clen, nb, nw, nn, nat, max_sym_str_len
  integer(C_INT), intent(IN)  :: auto_proj, nproj, proj_str_len(nproj), max_proj_str_len
  integer(C_INT), intent(in)  :: nk, nkabc(3) 
  real(C_DOUBLE), intent(in)  :: kpt(3,nk), uccart(3,3), eval(nb,nk)
  real(C_DOUBLE), intent(in) :: at_cart(3,nat)
  character(kind=C_CHAR,len=max_sym_str_len), intent(in) :: at_syms(nat)
  integer(C_INT), intent(in)  :: sym_str_len(nat)
  character(kind=C_CHAR,len=max_proj_str_len), intent(in) :: proj_str(nproj)
  integer(C_INT), intent(out) :: ierr
  ! m_matrix(nb, nb, nn, nk)
  complex(C_DOUBLE_COMPLEX), intent(inout) :: m_matrix(nb, nb, nn, nk)
  ! u_matrix_opt(nb, nw, nk)
  complex(C_DOUBLE_COMPLEX), intent(inout) :: u_matrix_opt(nb, nw, nk)
  real(C_DOUBLE), intent(out) :: centers(3,nw), spreads(nw)

  ! local variables
  integer :: stdout, stderr
  CHARACTER(len=maxlen) :: sym_f_string(nat)
!  CHARACTER(len=maxlen) :: proj_f_string

  integer, allocatable :: distk(:)
  integer :: i, j, ib, ik, nkl, np
  integer :: mpisize, mpirank
  complex(8), allocatable :: u_matrix(:,:,:)
  complex(8), allocatable :: t_matrix(:,:)
  type(lib_common_type), target :: w90main

  ! wannier interface starts
  ! stdout/err
  call w90_get_fortran_stdout(stdout)
  call w90_get_fortran_stderr(stderr)

  ! crude k distribution
  allocate (distk(nk), stat=ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in allocate: ',ierr
    return
  endif
  !nkl = nk ! number of kpoints per rank
  !if (mod(nk, mpisize) > 0) nkl = nkl + 1
  do i = 1, nk
    distk(i) = 0 !(i - 1)/nkl ! contiguous blocks with potentially fewer processes on last rank
  enddo

  if(max_sym_str_len .gt. maxlen) then
    ierr = 3001
    write(stderr,*) 'Error in wann90_setup: Atomic label string are too long:',max_sym_str_len,maxlen
    return
  endif
!  if(max_proj_str_len .gt. maxlen) then
!    ierr = 3002
!    write(stderr,*) 'Error in wann90_setup: Projection strings are too long:',max_sym_str_len,maxlen
!    return
!  endif
  if(nb < nw) then
    ierr = 3003
    write(stderr,*) 'Error in wann90_setup: num_band < num_wann: ',nb,nw
    return
  endif

  ! required settings
  call w90_set_option(w90main, 'kpoints', kpt)
  call w90_set_option(w90main, 'mp_grid', nkabc)
  call w90_set_option(w90main, 'num_bands', nb)
  call w90_set_option(w90main, 'num_kpts', nk)
  call w90_set_option(w90main, 'num_wann', nw)
  call w90_set_option(w90main, 'distk', distk)
  call w90_set_option(w90main, 'unit_cell_cart', uccart)

  do i=1,nat
    sym_f_string(i)(:) = ' '
    sym_f_string(i)(1:sym_str_len(i)) = TRIM(at_syms(i)(1:sym_str_len(i)))
  enddo
  call w90_set_option(w90main, 'symbols', sym_f_string)
  call w90_set_option(w90main, 'atoms_cart', at_cart)

  if(nproj .gt. 0) then
    do i=1,nproj
      call w90_set_option(w90main, 'projections', TRIM(proj_str(i)(1:proj_str_len(i))))
    enddo
  endif

  if(auto_proj .gt. 0) then
    call w90_set_option(w90main, 'auto_projections', .true.)
  endif

  ! enable mpi later on
  call w90_set_comm(w90main, mpi_comm_self)
  call w90_input_setopt(w90main, TRIM(prefix(1:clen)), stdout, stderr, ierr) ! apply settings
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_input_setopt: ',ierr
    return
  endif

  ! optional settings read from *.win file
  call w90_input_reader(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_input_reader: ',ierr
    return
  endif

  call w90_create_kmesh(w90main,stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_create_kmesh: ',ierr
    return
  endif

  call w90_print_info(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_print_info: ',ierr
    return
  endif

  call w90_get_nn(w90main, i, stdout, stderr, ierr); 
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_get_nn: ',ierr
    return
  endif
  if(i .ne. nn) then
    write(stderr,*) ' Error in w90_get_nn: Incompatible n_neighbors with CoQui ',i,nn,ierr
    ierr = 4001
    return
  endif

  call w90_set_m_local(w90main, m_matrix) 
  call w90_set_u_opt(w90main, u_matrix_opt)

  ! pass pointer to eval array
  call w90_set_eigval(w90main, eval)

  ! final u matrix
  allocate (u_matrix(nw, nw, nk), stat=ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in allocate: ',ierr
    return
  endif
  call w90_set_u_matrix(w90main, u_matrix)

  call w90_disentangle(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_disentangle: ',ierr
    return
  endif
  call w90_project_overlap(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_project_overlap: ',ierr
    return
  endif
  call w90_wannierise(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_wannierise: ',ierr
    return
  endif

  call w90_get_centres(w90main, centers)
  call w90_get_spreads(w90main, spreads)

  call w90_plot(w90main, stdout, stderr, ierr)

  ! use m_matrix for temporary storage 
  if(nb > nw) then
    allocate( t_matrix(nb,nw), stat=ierr )
    if(ierr .ne. 0) then
      write(stderr,*) ' Error in allocate: ',ierr
      return
    endif
    do ik=1,nk
      t_matrix(:,:) = u_matrix_opt(:,:,ik)
      call utility_zgemm_new( t_matrix(:,:), u_matrix(:,:,ik), u_matrix_opt(:,:,ik) )
    enddo
  else
    u_matrix_opt(:,:,:) = u_matrix(:,:,:) 
  endif

  if(allocated(t_matrix)) deallocate(t_matrix)
  if(allocated(u_matrix)) deallocate(u_matrix)
  if(allocated(distk)) deallocate(distk)

end subroutine wann90_run


!
! This version assumumes that *win, *mmn, and *amn exist.
! Sets up w90 and calls disentangle + wannierize
!
subroutine wann90_run_from_files(prefix, clen, nb, nw, eval, uccart, nk, nkabc, kpt,  &
                       nn, u_matrix_opt, centers, spreads, ierr)
  use mpi
  use w90_library
  use w90_library_extra, only: overlaps  
  use w90_utility, only: utility_zgemm_new
  USE ISO_C_BINDING, ONLY: C_CHAR, C_INT, C_DOUBLE, C_DOUBLE_COMPLEX

  implicit none

  character(kind=C_CHAR,len=256), intent(in) :: prefix
  integer(C_INT), intent(IN) :: clen, nb, nw 
  integer(C_INT), intent(in) :: nk, nkabc(3)
  real(C_DOUBLE), intent(in) :: kpt(3,nk), uccart(3,3), eval(nb,nk)
  integer(C_INT), intent(out) :: nn 
  integer(C_INT), intent(out) :: ierr
  ! u_matrix_opt(nb, nw, nk)
  complex(C_DOUBLE_COMPLEX), intent(inout) :: u_matrix_opt(nb, nw, nk)
  real(C_DOUBLE), intent(out) :: centers(3,nw), spreads(nw)

  integer :: stdout, stderr
  integer, allocatable :: distk(:)
  integer :: i, j, ib, ik, nkl
  integer :: mpisize, mpirank
  ! m_matrix(nb, nb, nn, nk)
  complex(8), allocatable :: m_matrix(:,:,:,:)
  complex(8), allocatable :: u_matrix(:,:,:)
  type(lib_common_type), target :: w90main

  ! crude k distribution
  allocate (distk(nk), stat=ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in allocate: ',ierr
    return
  endif
!  nkl = nk ! number of kpoints per rank
!  if (mod(nk, mpisize) > 0) nkl = nkl + 1
  do i = 1, nk
    distk(i) = 0 ! contiguous blocks with potentially fewer processes on last rank
  enddo

  ! wannier interface starts
  ! stdout/err
  call w90_get_fortran_stdout(stdout)
  call w90_get_fortran_stderr(stderr)

  ! required settings
  call w90_set_option(w90main, 'kpoints', kpt)
  call w90_set_option(w90main, 'mp_grid', nkabc)
  call w90_set_option(w90main, 'num_bands', nb)
  call w90_set_option(w90main, 'num_kpts', nk)
  call w90_set_option(w90main, 'num_wann', nw)
!  call w90_set_option(w90main, 'num_proj', nw)
  call w90_set_option(w90main, 'distk', distk)
  call w90_set_option(w90main, 'unit_cell_cart', uccart)

  ! enable mpi later on
  call w90_set_comm(w90main, mpi_comm_self)
  call w90_input_setopt(w90main, TRIM(prefix(1:clen)), stdout, stderr, ierr) ! apply settings
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_input_setopt: ',ierr
    return
  endif

  ! optional settings read from *.win file
  call w90_input_reader(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_input_reader: ',ierr
    return
  endif

  call w90_get_nn(w90main, i, stdout, stderr, ierr); 
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_get_nn: ',ierr
    return
  endif
  if(i .ne. nn) then
    write(stderr,*) ' Error in w90_get_nn: Incompatible number of nearest neighbors ',i,nn
    ierr = 2001
    return
  endif

  call w90_print_info(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_print_info: ',ierr
    return
  endif

  allocate (m_matrix(nb, nb, nn, nk), stat=ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in allocate: ',ierr
    return
  endif
  m_matrix(:,:,:,:) = (0.d0,0.d0)
  call w90_set_m_local(w90main, m_matrix) ! m_matrix_local_orig
  call w90_set_u_opt(w90main, u_matrix_opt)

  ! read from ".mmn" and ".amn"
  ! and assign to m and a (now called u)
  ! a dft code would calculate the overlaps here instead
  call overlaps(w90main, stdout, stderr, ierr) ! from library-extra
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in overlaps: ',ierr
    return
  endif

  ! pass pointer to eval array
  call w90_set_eigval(w90main, eval)

  ! final u matrix
  allocate (u_matrix(nw, nw, nk), stat=ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in allocate: ',ierr
    return
  endif
  call w90_set_u_matrix(w90main, u_matrix)

  call w90_disentangle(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_disentangle: ',ierr
    return
  endif
  call w90_project_overlap(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_project_overlap: ',ierr
    return
  endif
  call w90_wannierise(w90main, stdout, stderr, ierr)
  if(ierr .ne. 0) then
    write(stderr,*) ' Error in w90_wannierise: ',ierr
    return
  endif

  call w90_get_centres(w90main, centers)
  call w90_get_spreads(w90main, spreads)

  call w90_plot(w90main, stdout, stderr, ierr)

  ! use m_matrix for temporary storage 
  if( nb > nw ) then
    do ik=1,nk
      m_matrix(1:nb,1:nw,1,1) = u_matrix_opt(:,:,ik)
      call utility_zgemm_new( m_matrix(1:nb,1:nw,1,1), u_matrix(:,:,ik), u_matrix_opt(:,:,ik) ) 
    enddo
  else
    u_matrix_opt(:,:,:) = u_matrix(:,:,:) 
  endif

  if(allocated(u_matrix)) deallocate(u_matrix)
  if(allocated(m_matrix)) deallocate(m_matrix)
  if(allocated(distk)) deallocate(distk)

end subroutine wann90_run_from_files

