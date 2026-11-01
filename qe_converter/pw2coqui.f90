!
!
!
!-------------------------------------------------------------------------------

PROGRAM pw2coqui

  USE control_flags, ONLY : gamma_only
  USE environment, ONLY : environment_start, environment_end
  USE io_files, ONLY : prefix, tmp_dir
  USE io_global, ONLY : ionode, ionode_id
  USE kinds, ONLY : DP
  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_openfile, qeh5_close
  USE mp, ONLY : mp_sum, mp_bcast, mp_min, mp_max, mp_get, mp_barrier
  USE mp_global, ONLY : mp_startup
  USE mp_pools, ONLY : me_pool, root_pool,npool, nproc_pool, &
    intra_pool_comm, inter_pool_comm
  USE mp_world, ONLY : mpime, nproc, world_comm
  USE mp_bands, ONLY : intra_bgrp_comm, nbgrp
  USE paw_variables, ONLY : okpaw
  USE uspp, ONLY : okvan
  USE klist , ONLY: nks, nkstot, ngk, igk_k
  USE gvect, ONLY: ig_l2g 

  IMPLICIT NONE

  character(len=9) :: codename = 'PW2COQUI'

  character ( len = 256 ) :: outdir
  character ( len = 256 ) :: fname 
  type(qeh5_file) :: h5f
  character ( len = 256 ) :: h5name
  logical :: add_system, add_pp, add_orbs 
  integer :: ios, iks, ike, i, k, npwx_g, maxg
  integer, allocatable :: npw_g(:)
  character (len=256), external :: trimcheck
  character (len=1), external :: lowercase
  real(dp), external :: ewald
  integer, external :: global_kpoint_index

  NAMELIST / input_pw2coqui / prefix, outdir, fname, add_system, add_pp, add_orbs 

#if defined(__MPI)
  CALL mp_startup ( )
#endif

  CALL environment_start ( codename )

  prefix = 'pwscf'
  fname = ' '
  add_system = .true.
  add_pp = .true.
  add_orbs = .false.
  CALL get_environment_variable ( 'ESPRESSO_TMPDIR', outdir )
  IF ( TRIM ( outdir ) == ' ' ) outdir = './'

  IF ( ionode ) THEN
    CALL input_from_file ( )
    READ ( 5, input_pw2coqui, iostat = ios )
    IF ( ios /= 0 ) CALL errore ( codename, 'input_pw2coqui', abs ( ios ) )
    if( TRIM( fname ) == ' ' ) fname = TRIM(prefix) // '.coqui.h5'
  ENDIF

  tmp_dir = trimcheck ( outdir )
  CALL mp_bcast ( tmp_dir, ionode_id, world_comm )
  CALL mp_bcast ( prefix, ionode_id, world_comm )
  CALL mp_bcast ( fname, ionode_id, world_comm )
  CALL mp_bcast ( add_system, ionode_id, world_comm )
  CALL mp_bcast ( add_pp, ionode_id, world_comm )
  CALL mp_bcast ( add_orbs, ionode_id, world_comm )
  CALL read_file ( )

  if (gamma_only) call errore ( 'pw2coqui', ' Can not use use gamma-only yet.', 1 )

  CALL openfil()
  ! set nwordwfc = nbnd * npwx * npol here
  CALL hinit0()

  CALL openfil_pp ( )

  ! open file on root 
  if (ionode) then
    h5name = TRIM( tmp_dir ) // TRIM( fname ) 
    CALL qeh5_openfile(h5f,h5name, 'write')
  endif

  ! some generally used structures
  ! pw per kpoint
  allocate(npw_g(nkstot))
  iks = global_kpoint_index (nkstot, 1)
  ike = iks + nks - 1
  npw_g(:) = 0
  maxg = 0
  do k = 1, nks
    npw_g(k + iks - 1 ) = ngk(k)
    do i = 1, ngk(k)
      maxg = max(maxg,ig_l2g(igk_k(i,k)))
    enddo
  enddo
  CALL mp_sum( npw_g, inter_pool_comm )
  CALL mp_sum( npw_g, intra_pool_comm )
  npw_g = npw_g / nbgrp
  npwx_g = MAXVAL ( npw_g ( : ) )
  CALL mp_max( maxg, world_comm )

  ! write /System information
  if(add_system) call write_system(h5f)
  if(add_pp) call write_pp(h5f)

  if(ionode) CALL qeh5_close(h5f)

  CALL environment_end ( codename )

  CALL stop_pp ( )

  ! this is needed because openfil is called above
  CALL close_files ( .false. )

  ! cleanup
  deallocate(npw_g)

  STOP

CONTAINS


subroutine write_system(h5_f)

  USE hdf5
  USE qeh5_base_module, ONLY : qeh5_open_group, qeh5_close, qeh5_add_attribute, &
                               qeh5_set_space, qeh5_open_dataset, qeh5_write_dataset 
  USE constants, ONLY: e2
  USE control_flags, ONLY: gamma_only, noinv
  USE vlocal, ONLY : strf
  USE io_global, ONLY : stdout
  USE ions_base, ONLY : nat, nsp, ityp, tau, zv, atm
  USE cell_base, ONLY: omega, alat, tpiba, tpiba2, at, bg
  USE wvfct, ONLY: npwx, nbnd, wg, et
  USE klist , ONLY: nks, nelec, nelup, neldw, xk, wk, nkstot, ngk
  USE gvect, ONLY: ngm, ecutrho, gg, gstart, g, gcutm
  USE gvecw, ONLY : ecutwfc
  USE io_global, ONLY: ionode,  ionode_id
  USE mp_images, ONLY: intra_image_comm
  USE symm_base, ONLY: nsym, s, ft, t_rev
  USE lsda_mod, ONLY: lsda, nspin
  USE noncollin_module,     ONLY : noncolin, npol, lspinorb
  USE fft_base, ONLY : dfftp, dffts
  USE start_k,            ONLY : nks_start,nk1, nk2, nk3, k1, k2, k3

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  type(qeh5_file)    :: h5_s, h5_o, h5_b, h5_bs, h5_bsi
  type(qeh5_dataset) :: dset

  integer :: ns, nk, i, j, k, vi3(3), ierr
  integer, allocatable :: ityp_s(:)
  real(dp) :: enuc
  real(dp) :: v3(3,3)
  real(dp), allocatable :: vn(:,:), wgt(:)
  character(len=2) :: isym 
  character, allocatable :: sp_names(:)
  type(c_ptr) ::  buf
  integer(hsize_t) :: text_length
  character(len=4), allocatable, target :: atm_(:)  ! MAM: do I need to reserve space for null termination?

  ! ewald is collective: it reduces over intra_bgrp_comm
  enuc = ewald( alat, nat, nsp, ityp, zv, at, bg, tau, &
              omega, g, gg, ngm, gcutm, gstart, gamma_only, strf ) / e2

  if(ionode) then

    ns = 1
    if(lsda) ns = 2

    call qeh5_open_group(h5_f, "System", h5_s)   ! /System
    call qeh5_open_group(h5_f, "Orbitals", h5_o) ! /Orbitals
    call qeh5_open_group(h5_s, "BZ", h5_b)       ! /System/BZ
    call qeh5_open_group(h5_b, "Symmetries", h5_bs) ! /System/BZ/Symmetries

    call qeh5_add_attribute(h5_s%id,"number_of_atoms",nat)
    call qeh5_add_attribute(h5_s%id,"number_of_species",nsp)
    call qeh5_add_attribute(h5_s%id,"number_of_spins",ns)
    call qeh5_add_attribute(h5_o%id,"number_of_spins",ns)
    call qeh5_add_attribute(h5_s%id,"number_of_polarizations",npol)
    call qeh5_add_attribute(h5_o%id,"number_of_polarizations",npol)
    call qeh5_add_attribute(h5_s%id,"number_of_elec",nelec)
    if(noinv) then
      call qeh5_add_attribute(h5_s%id,"noinv",1)
    else
      call qeh5_add_attribute(h5_s%id,"noinv",0)
    endif
    if(lspinorb) then
      call qeh5_add_attribute(h5_s%id,"lspinorbit",1)
    else
      call qeh5_add_attribute(h5_s%id,"lspinorbit",0)
    endif

    call qeh5_add_attribute(h5_s%id,"nuclear_energy",enuc)

    v3(:,:) = at(:,:)*alat
    call h5_write_mat_r(h5_s,v3,"lattice_vectors")
    v3(:,:) = bg(:,:)*tpiba
    call h5_write_mat_r(h5_s,v3,"reciprocal_vectors")
    
    allocate(ityp_s(nat))
    do i=1,nat
      ityp_s(i) = ityp(i)-1
    enddo
    call h5_write_vector_int(h5_s,ityp_s(1:nat),"atomic_id")
    deallocate(ityp_s)
    allocate (vn(3, nat))
    vn(1:3,1:nat) = tau(1:3,1:nat)*alat
    call h5_write_mat_r(h5_s,vn,"atomic_positions")
    deallocate(vn)

    ! MAM: array of strings by hand, since there is no backend!
    text_length = 4*1_HSIZE_T
    allocate(atm_(nsp))
    do i=1,nsp
      atm_(i) = TRIM(atm(i))//char(0)
    enddo
    call H5Tcopy_f(H5T_FORTRAN_S1, dset%datatype%id, ierr )
    if(ierr .ne. 0) call errore( 'pw2coqui', 'write_system: H5Tcopy_f ierr: ', ierr )
    call H5Tset_size_f( dset%datatype%id, text_length, ierr )
    if(ierr .ne. 0) call errore( 'pw2coqui',  'write_system: H5Tset_size_f ierr: ', ierr )
    call H5Tset_cset_f( dset%datatype%id, H5T_CSET_UTF8_F, ierr )
    if(ierr .ne. 0) call errore( 'pw2coqui',  'write_system: H5Tset_cset_f ierr: ', ierr )
    call H5Tset_strpad_f( dset%datatype%id, H5T_STR_NULLTERM_F, ierr )
    if(ierr .ne. 0) call errore( 'pw2coqui',  'write_system: H5Tset_strpad_f ierr: ', ierr )
    IF (ALLOCATED (dset%filespace%dims) ) DEALLOCATE ( dset%filespace%dims )
    ALLOCATE(dset%filespace%dims(1) )
    dset%filespace%rank = 1 
    dset%filespace%dims(1) = nsp*1_HSIZE_T
    CALL H5Screate_simple_f( 1, dset%filespace%dims, dset%filespace%id, ierr )
    if(ierr .ne. 0) call errore( 'pw2coqui', 'write_system: H5Screate_simple_f ierr: ', ierr )
    CALL qeh5_open_dataset(h5_s, dset, ACTION='write', NAME=TRIM("species"))
    buf = c_loc(atm_)
    CALL H5Dwrite_f ( dset%id, dset%datatype%id, buf, ierr, H5S_ALL_F,&
                      dset%filespace%id, H5P_DEFAULT_F )
    if(ierr .ne. 0) call errore( 'pw2coqui', 'write_system: H5Dwrite_f ierr: ', ierr )
    call qeh5_close(dset)
    deallocate(atm_)

    ! it is fine if it is zero, CoQui will look for a regular grid after expanding the grid
    vi3(1) = nk1
    vi3(2) = nk2
    vi3(3) = nk3
    call h5_write_vector_int(h5_b,vi3,"kp_grid")
    if(nk1*nk2*nk3 > 0) then
      nk = nk1*nk2*nk3
    else
      nk = nks_start
    endif
    call qeh5_add_attribute(h5_b%id,"number_of_kpoints",nk)
    call qeh5_add_attribute(h5_o%id,"number_of_kpoints",nk)
    nk = nkstot/ns
    call qeh5_add_attribute(h5_b%id,"number_of_kpoints_ibz",nk)
    call qeh5_add_attribute(h5_o%id,"number_of_kpoints_ibz",nk)
    call qeh5_add_attribute(h5_o%id,"npwx",npwx_g)
    allocate (vn(3, nk))
    ! ionode should have all the kpoints!
    vn(:,1:nk) = xk(:,1:nk)*tpiba
    call h5_write_mat_r(h5_b,vn,"kpoints")
    deallocate (vn)
    allocate(wgt(nk))
    if(nspin==1) then
      wgt(1:nk) = 0.5d0*wk(1:nk)
    else
      wgt(1:nk) = wk(1:nk)
    endif
    call h5_write_vector_r(h5_s,wgt(1:nk),"kpoint_weights")
    deallocate(wgt)

    call qeh5_add_attribute(h5_o%id,"number_of_bands",nbnd)
    call qeh5_add_attribute(h5_o%id,"ecutrho",ecutrho)

    ! pw per kpoint
    call h5_write_vector_int(h5_o,npw_g(1:nk),"npw")

    ! fft mesh
    vi3(1) = dffts%nr1
    vi3(2) = dffts%nr2
    vi3(3) = dffts%nr3
    call h5_write_vector_int(h5_o,vi3,"fft_mesh")
    vi3(1) = dfftp%nr1
    vi3(2) = dfftp%nr2
    vi3(3) = dfftp%nr3
    call h5_write_vector_int(h5_o,vi3,"fft_mesh_aug")
  
    ! eigenvalues
    allocate(vn(nbnd,nkstot)) 
    vn(:,:) = et(:,:)/e2
    call qeh5_set_space(dset, vn(1,1), RANK=3, DIMENSIONS=[nbnd,nk,ns])
    call qeh5_open_dataset(h5_o, dset, ACTION='write', NAME="eigval")
    call qeh5_write_dataset(vn, dset)
    call qeh5_close(dset)

    ! occupations
    do i=1,nkstot
      vn(:,i) = wg(:,i)/wk(i)
    enddo
    call qeh5_set_space(dset, vn(1,1), RANK=3, DIMENSIONS=[nbnd,nk,ns])
    call qeh5_open_dataset(h5_o, dset, ACTION='write', NAME="occ")
    call qeh5_write_dataset(vn, dset)
    deallocate(vn) 
    call qeh5_close(dset)

    call qeh5_add_attribute(h5_bs%id,"number_of_symmetries",nsym)
    do i=1,nsym
      write ( isym, '(I2)') i-1
      call qeh5_open_group(h5_bs, "s"//adjustl(trim(isym)) , h5_bsi) ! /System/BZ/Symmetries/s0
      v3(:,:) = s(1:3,1:3,i)
      call h5_write_mat_r(h5_bsi,v3,"R")
      call h5_write_vector_r(h5_bsi,ft(1:3,i),"ft")
      call qeh5_close(h5_bsi)
    enddo

    call qeh5_close(h5_bs)
    call qeh5_close(h5_b)
    call qeh5_close(h5_o)
    call qeh5_close(h5_s)

  endif

end subroutine write_system

! Some parts modeled after pw2bgw.f90
subroutine write_pp(h5_f)

  USE qeh5_base_module, ONLY : qeh5_open_group, qeh5_close, qeh5_add_attribute
  USE fft_interfaces, ONLY : fwfft, invfft
  USE vlocal, ONLY : strf
  USE io_global, ONLY : stdout
  USE ions_base, ONLY : nat, nsp, ityp, tau, zv, atm
  USE cell_base, ONLY: omega, alat, tpiba, tpiba2, at, bg
  USE wvfct, ONLY: nbnd, g2kin, wg, et
  USE klist , ONLY: nks, nelec, nelup, neldw, xk, wk, nkstot, ngk
  USE gvect, ONLY: ngm, ngm_g, g, mill, gstart, gg
  USE gvecw, ONLY : ecutwfc
  USE io_global, ONLY: ionode,  ionode_id
  USE lsda_mod, ONLY: lsda, nspin
  USE noncollin_module, ONLY : noncolin, npol, lspinorb
  USE fft_base, ONLY : dfftp, dffts
  USE uspp, ONLY : okvan, vkb, nkb, ofsbeta, ijtoh, dvan, dvan_so, deeq 
  USE uspp_param, ONLY : nhm, nh, upf, lmaxq
  USE scf, ONLY : vltot, v, rho, rho_core, rhog_core
  USE ener, ONLY : etxc, vtxc
  USE wavefunctions, ONLY : psic
  USE mp_wave, ONLY : mergewf
  USE uspp_init,            ONLY : init_us_2
  USE xc_lib, ONLY : xclib_dft_is 
  USE noncollin_module, ONLY : domag

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  type(qeh5_file)    :: h5_h, h5_n

  integer :: ns, nk, ikb, j, ig, ik, ik_loc, i, vi3(3), ierr, l2g, ngg, npw, ipsour
  integer :: ngm_s, ngm_e, ngm_l, ih, jh, ijh, nij, is, ir, nt
  complex(dp), ALLOCATABLE :: qgm(:), qgm_full(:,:)
  real(dp), ALLOCATABLE :: ylmk0(:,:), qmod(:)
  integer, allocatable :: igk_g(:), mill_g(:,:), mill_k(:,:), itmp(:)
  integer, allocatable :: igk_l2g(:), ityp_s(:)
  real(dp) :: v3(3,3)
  real(dp), allocatable :: vxc(:,:)
  character(len=12) pp_type
  logical :: ik_in_range 
  character(len=8) str_ik
  complex (DP), allocatable :: vkb_g ( : ), vloc( :, :, : ) 
  complex (DP), allocatable :: vkb_g_root ( :, : )
  character(len=2) sp_name

  ns = 1
  if(lsda) ns = 2
  nk = nkstot/ns

  if(ionode) then

    if( okpaw ) then
      pp_type = "paw"
    elseif( okvan ) then
      pp_type = "uspp"
    else
      pp_type = "ncpp"
    endif

    call qeh5_open_group(h5_f, "Hamiltonian", h5_h)   
    call qeh5_add_attribute(h5_h%id,"pp_type",TRIM(pp_type))
    call qeh5_open_group(h5_h, TRIM(pp_type), h5_n)  

    call qeh5_add_attribute(h5_n%id,"number_of_nspins",ns)
    call qeh5_add_attribute(h5_n%id,"number_of_polarizations",npol)
    call qeh5_add_attribute(h5_n%id,"number_of_kpoints",nk)
    call qeh5_add_attribute(h5_n%id,"number_of_atoms",nat)
    call qeh5_add_attribute(h5_n%id,"number_of_species",nsp)
    call qeh5_add_attribute(h5_n%id,"total_num_of_proj",nkb)
    call qeh5_add_attribute(h5_n%id,"max_proj_per_atom",nhm)
    call h5_write_vector_int(h5_n,nh(1:nsp),"proj_per_atom")
    call h5_write_vector_int(h5_n,ofsbeta,"projector_offset")
    call h5_write_tensor_int(h5_n,ijtoh,"ijtoh")
    call qeh5_add_attribute(h5_n%id,"ngm",ngm_g)
    allocate(ityp_s(nat))
    ityp_s(:) = ityp(:)-1
    call h5_write_vector_int(h5_n,ityp_s(1:nat),"atomic_id")
    deallocate(ityp_s)
    if(lspinorb) then
      call qeh5_add_attribute(h5_n%id,"lspinorbit_nl",1)
    else
      call qeh5_add_attribute(h5_n%id,"lspinorbit_nl",0)
    endif

    ! pw per kpoint
    call qeh5_add_attribute(h5_n%id,"max_npw",npwx_g)
    call h5_write_vector_int(h5_n,npw_g(1:nk),"npw")

    if(lspinorb) then
      ! (nhm,nhm,nspin,nsp)
      call h5_write_tensor4_c(h5_n,dvan_so,"dion_so")       
    else
      ! (nhm,nhm,nsp)
      call h5_write_tensor_r(h5_n,dvan,"dion")       
    endif
    !
  endif ! ionode

  if( okvan ) then
    
    CALL divide( inter_pool_comm, ngm, ngm_s, ngm_e )
    ngm_l = ngm_e-ngm_s+1
    ALLOCATE( qmod(ngm_l), ylmk0(ngm_l,lmaxq*lmaxq) )
    !
    CALL ylmr2( lmaxq*lmaxq, ngm_l, g(1,ngm_s), gg(ngm_s), ylmk0 )
    DO ig = 1, ngm_l
       qmod(ig) = SQRT(gg(ngm_s+ig-1))*tpiba
    ENDDO
    !
    !
    do nt = 1, nsp
      !
      if ( upf(nt)%tvanp ) then
        !
        ! nij = max number of (ih,jh) pairs per atom type nt
        !
        nij = nh(nt)*(nh(nt)+1)/2
        !allocate ( vloc ( dfftp%nnr,  nij, 1 ) )
        !
        ! ... Compute and store Q(G) for this atomic species 
        ! ... (without structure factor)
        !
        allocate( qgm(ngm_l), qgm_full(ngm_g,nij) )
        qgm_full(:,:) = (0.d0,0.d0)
        !
        ijh = 0
        DO ih = 1, nh(nt)
           DO jh = ih, nh(nt)
              ijh = ijh + 1
              CALL qvan2( ngm_l, ih, jh, nt, qmod, qgm(1), ylmk0 )
              do ig = 1, ngm_l
                qgm_full( ig_l2g(ngm_s+ig-1), ijh ) = qgm(ig) 
              enddo

              ! erase if not used
              !psic(:) = (0.d0,0.d0)
              !do ig = 1, ngm
              !  psic ( dfftp%nl ( ig ) ) = qgm(ig) 
              !enddo
              !call invfft ( 'Rho', psic, dfftp )
              !vloc(:,ijh,1) = psic(:)

           ENDDO
        ENDDO
        !
        CALL mp_sum(qgm_full(:, :), world_comm )
        !
        if(ionode) then
          write ( sp_name, '(I2)') nt-1
          call h5_write_mat_c(h5_n,qgm_full,"augmentation_function_isp"//adjustl(trim(sp_name)))
          !call h5_write_mat_c(h5_n,vloc(:,:,1),"r_augmentation_function_isp"//adjustl(trim(sp_name)))
        endif
        !
        deallocate(qgm, qgm_full)
        !
        !deallocate ( vloc ) 
      endif
      !
    enddo  ! nt
    !
  endif

  if( okpaw .and. ionode ) then
    call write_factorized_paw_one_center()
  endif

  ALLOCATE ( mill_g ( 3, ngm_g ) )
  mill_g = 0
  DO ig = 1, ngm
    mill_g ( 1, ig_l2g ( ig ) ) = mill ( 1, ig )
    mill_g ( 2, ig_l2g ( ig ) ) = mill ( 2, ig )
    mill_g ( 3, ig_l2g ( ig ) ) = mill ( 3, ig )
  ENDDO
  CALL mp_sum ( mill_g, intra_bgrp_comm )
  if(.not.ionode) deallocate(mill_g)

  ! self-consistent potential
  allocate ( vloc ( ngm_g, npol*npol, ns ) )
  vloc(:,:,:) = (0.d0,0.d0)
  do is = 1, nspin
    psic(:) = (0.d0,0.d0)
    if( nspin == 4 ) then
      if(is==1) then
        do ir = 1, dfftp%nnr
          psic ( ir ) = CMPLX ( v%of_r(ir,1) + v%of_r(ir,4) + vltot ( ir ), 0.0D0, KIND=dp )
        enddo
      elseif(is==2) then
        do ir = 1, dfftp%nnr
          psic ( ir ) = CMPLX ( v%of_r(ir,2) - (0.d0,1.d0) * v%of_r(ir,3), KIND=dp )
        enddo
      elseif(is==3) then
        do ir = 1, dfftp%nnr
          psic ( ir ) = CMPLX ( v%of_r(ir,2) + (0.d0,1.d0) * v%of_r(ir,3), KIND=dp )
        enddo
      elseif(is==4) then
        do ir = 1, dfftp%nnr
          psic ( ir ) = CMPLX ( v%of_r(ir,1) - v%of_r(ir,4) + vltot ( ir ), 0.0D0, KIND=dp )
        enddo
      endif
      call fwfft ( 'Rho', psic, dfftp )
      do ig = 1, ngm
        vloc ( ig_l2g ( ig ), is, 1 ) = psic ( dfftp%nl ( ig ) )
      enddo
    else
      do ir = 1, dfftp%nnr
        psic ( ir ) = CMPLX ( v%of_r ( ir, is ) + vltot ( ir ), 0.0D0, KIND=dp )
      enddo
      call fwfft ( 'Rho', psic, dfftp )
      do ig = 1, ngm
        vloc ( ig_l2g ( ig ), 1, is ) = psic ( dfftp%nl ( ig ) )
      enddo
    endif
  enddo
  CALL mp_sum ( vloc, intra_bgrp_comm )
  if(ionode) then
    call h5_write_mat_int(h5_n,mill_g,"miller_g")
    call h5_write_tensor_c(h5_n,vloc,"scf_local_potential")
  endif

  ! local potential
  vloc(:,:,:) = (0.d0,0.d0)
  psic(:) = (0.d0,0.d0)
  do ir = 1, dfftp%nnr
    psic ( ir ) = CMPLX ( vltot ( ir ), 0.0D0, KIND=dp )
  enddo
  call fwfft ( 'Rho', psic, dfftp )
  do ig = 1, ngm
    vloc ( ig_l2g ( ig ), 1, 1 ) = psic ( dfftp%nl ( ig ) )
  enddo

  ! MAM: if local potential has polarization dependence, modify below! 
  if(ionode) call qeh5_add_attribute(h5_n%id,"lspinorbit_loc",0)
  CALL mp_sum ( vloc, intra_bgrp_comm )
  if(ionode) then
    call h5_write_vector_c(h5_n,vloc(:,1,1),"pp_local_component")
  endif
  ! pp_local_component_nc for polarization dependent potential, 
  ! with dimensions: (ngm, npol*npol, nspin)

  ! vxc
  if (xclib_dft_is('meta')) then
    write(*,*) 'Found meta-gga xc functional: Skipping v_xc' 
  else
    allocate(vxc(dfftp%nnr, nspin))
    !
    CALL v_xc ( rho, rho_core, rhog_core, etxc, vtxc, vxc )
    !
    vloc(:,:,:) = (0.d0,0.d0)
    DO is = 1, nspin
      psic(:) = (0.d0,0.d0)
      if( nspin == 4 ) then
        if(is==1) then
          do ir = 1, dfftp%nnr
            psic ( ir ) = CMPLX ( vxc(ir,1) + vxc(ir,4), 0.0D0, KIND=dp )
          enddo
        elseif(is==2) then
          do ir = 1, dfftp%nnr
            psic ( ir ) = CMPLX ( vxc(ir,2) - (0.d0,1.d0) * vxc(ir,3), KIND=dp )
          enddo
        elseif(is==3) then
          do ir = 1, dfftp%nnr
            psic ( ir ) = CMPLX ( vxc(ir,2) + (0.d0,1.d0) * vxc(ir,3), KIND=dp )
          enddo
        elseif(is==4) then
          do ir = 1, dfftp%nnr
            psic ( ir ) = CMPLX ( vxc(ir,1) - vxc(ir,4), 0.0D0, KIND=dp )
          enddo
        endif
        CALL fwfft ( 'Rho', psic, dfftp )
        DO ig = 1, ngm
          vloc ( ig_l2g ( ig ), is, 1 ) = psic ( dfftp%nl ( ig ) )
        ENDDO
      else
        DO ir = 1, dfftp%nnr
          psic ( ir ) = CMPLX ( vxc ( ir, is ), 0.0D0, KIND=dp )
        ENDDO
        CALL fwfft ( 'Rho', psic, dfftp )
        DO ig = 1, ngm
          vloc ( ig_l2g ( ig ), 1, is ) = psic ( dfftp%nl ( ig ) )
        ENDDO
      endif
    ENDDO
    CALL mp_sum ( vloc, intra_bgrp_comm )
    if(ionode) then
      call h5_write_tensor_c(h5_n,vloc,"vxc_with_nlcc")
    endif
    !
    !
    ! is it ok to just zero out these? Otherwise allocate a new array
    rho_core ( : ) = 0.0D0
    rhog_core ( : ) = ( 0.0D0, 0.0D0 )
    CALL v_xc ( rho, rho_core, rhog_core, etxc, vtxc, vxc )
    !
    vloc(:,:,:) = (0.d0,0.d0)
    DO is = 1, nspin
      psic(:) = (0.d0,0.d0)
      if( nspin == 4 ) then
        if(is==1) then
          do ir = 1, dfftp%nnr
            psic ( ir ) = CMPLX ( vxc(ir,1) + vxc(ir,4), 0.0D0, KIND=dp )
          enddo
        elseif(is==2) then
          do ir = 1, dfftp%nnr
            psic ( ir ) = CMPLX ( vxc(ir,2) - (0.d0,1.d0) * vxc(ir,3), KIND=dp )
          enddo
        elseif(is==3) then
          do ir = 1, dfftp%nnr
            psic ( ir ) = CMPLX ( vxc(ir,2) + (0.d0,1.d0) * vxc(ir,3), KIND=dp )
          enddo
        elseif(is==4) then
          do ir = 1, dfftp%nnr
            psic ( ir ) = CMPLX ( vxc(ir,1) - vxc(ir,4), 0.0D0, KIND=dp )
          enddo
        endif
        CALL fwfft ( 'Rho', psic, dfftp )
        DO ig = 1, ngm
          vloc ( ig_l2g ( ig ), is, 1 ) = psic ( dfftp%nl ( ig ) )
        ENDDO
      else
        DO ir = 1, dfftp%nnr
          psic ( ir ) = CMPLX ( vxc ( ir, is ), 0.0D0, KIND=dp )
        ENDDO
        CALL fwfft ( 'Rho', psic, dfftp )
        DO ig = 1, ngm
          vloc ( ig_l2g ( ig ), 1, is ) = psic ( dfftp%nl ( ig ) )
        ENDDO
      endif
    ENDDO
    CALL mp_sum ( vloc, intra_bgrp_comm )
    if(ionode) then
      call h5_write_tensor_c(h5_n,vloc,"vxc")
    endif
    deallocate(vxc)
    !
  endif
  deallocate(vloc)

  allocate ( itmp(maxg), igk_g (maxg) )
  if(ionode) allocate( mill_k(3, npwx_g) )

  do ik = 1, nk

    ik_in_range = (ik .GE. iks .AND. ik .LE. ike)
    ! reconstruct a global igk_k
    itmp = 0
    npw=0
    if ( ik_in_range ) then 
      ik_loc = ik - iks + 1 
      npw = ngk(ik_loc)
      do ig = 1, npw 
        l2g = ig_l2g(igk_k(ig,ik_loc))
        itmp (l2g) = l2g
      enddo 
      CALL init_us_2(npw, igk_k(1,ik_loc), xk(1,ik),vkb)
    endif
    CALL mp_sum( itmp, world_comm )
    itmp = itmp / nbgrp

    ngg = 0
    do ig = 1, maxg 
      if ( itmp ( ig ) .eq. ig ) then
        ngg = ngg + 1
        igk_g ( ngg ) = ig
      endif
    enddo
    if( ngg .ne. npw_g(ik) ) call errore( 'write_pp', 'ngg.ne.ngk', 10)

    if ( ionode ) THEN
      do ig = 1,npw_g(ik)
        mill_k(1:3,ig) = mill_g(1:3,igk_g(ig))
      enddo
      write ( str_ik, '(I8)') ik-1
      call h5_write_mat_int(h5_n,mill_k(1:3,1:npw_g(ik)),"miller_k"//adjustl(trim(str_ik)))
    endif 

    if( ik_in_range ) then
      allocate ( igk_l2g(npw) )
      igk_l2g = 0
      do ig = 1, npw
        ngg = ig_l2g(igk_k(ig,ik_loc)) 
        do j = 1, npw_g(ik)
          if( ngg .eq. igk_g(j) ) then 
            igk_l2g(ig) = j
            exit
          endif 
        enddo
      enddo
    endif

    if( npool .gt. 1 ) then 
      ipsour = nproc+10 
      if ( ik_in_range ) ipsour = mpime 
      call mp_min( ipsour, world_comm )
    else
      ipsour = ionode_id
    endif

    if(ionode) allocate ( vkb_g_root(npw_g(ik),nkb) )
    ! every rank takes part in the gather below, so every rank needs the buffer
    allocate ( vkb_g(npw_g(ik)) )

    do ikb = 1, nkb

      vkb_g = ( 0.0D0, 0.0D0 )
      if ( npool .gt. 1 ) then
        ! gather within the pool that holds ik
        if ( ik_in_range ) then
          CALL mergewf ( vkb(:,ikb), vkb_g, npw, igk_l2g, &
            me_pool, nproc_pool, root_pool, intra_pool_comm )
        endif
        ! send to ionode, unless ionode is that root already
        if ( ipsour .NE. ionode_id ) then
          call mp_get ( vkb_g, vkb_g, mpime, ionode_id, ipsour, ikb, world_comm )
        endif
      else
        call mergewf ( vkb(:,ikb), vkb_g, npw, igk_l2g, &
          mpime, nproc, ionode_id, world_comm )
      endif
      if(ionode) vkb_g_root(:,ikb) = vkb_g(:)

    enddo

    if(ionode) then
      write ( str_ik, '(I8)') ik-1
      call h5_write_mat_c(h5_n,vkb_g_root,"projector_k"//adjustl(trim(str_ik)))
    endif

    if(allocated(vkb_g)) deallocate(vkb_g)
    if(allocated(vkb_g_root)) deallocate(vkb_g_root)
    if(allocated(igk_l2g)) deallocate(igk_l2g)

  enddo

  deallocate ( itmp, igk_g )
  if(allocated(mill_g)) deallocate(mill_g)
  if(allocated(mill_k)) deallocate(mill_k)

  if(ionode) then
    call qeh5_close(h5_n)
    call qeh5_close(h5_h)
  endif

end subroutine write_pp

!
SUBROUTINE write_factorized_paw_one_center( )
!=----
! factorize paw's ke%k for each species  
!=----
  USE kinds, ONLY : DP
  USE constants, ONLY: e2
  USE ions_base,          ONLY : nat, ityp, ntyp => nsp
  USE uspp_param,         ONLY : nh, upf
  USE paw_variables,        ONLY : okpaw
  USE mp_images, ONLY: intra_image_comm, me_image, root_image
  USE mp, ONLY: mp_bcast
  USE paw_exx,            ONLY : ke, PAW_init_fock_kernel, &
                                     PAW_clean_fock_kernel
  !
  IMPLICIT NONE
  !
  !
  INTEGER :: ns, ij, ou, na,n, lda
  INTEGER :: i,ih, jh, oh, uh
  REAL(DP), ALLOCATABLE :: k(:,:), L(:,:)
  REAL(DP), ALLOCATABLE :: ev(:)

  !
  ! on the first call, symmetrize ke tensor
  if(.not.okpaw) return
  if(.not.ionode) return
  CALL PAW_init_fock_kernel()
  do ns = 1,ntyp
    if (.not. upf(ns)%tpawp ) continue 
    ij=0
    allocate(k(nh(ns)*nh(ns),nh(ns)*nh(ns)), ev(nh(ns)*nh(ns)))
    do jh = 1, nh(ns)
    do ih = 1, nh(ns)
    ij=ij+1
    ou=0
    do uh = 1, nh(ns)
    do oh = 1, nh(ns)
      !
      ou=ou+1
      ! factor of 1/e2 is to convert to Hartree, factor of 0.5
      ! comes from the QE implementation. Not sure I understand why!
      ! Keeping it for consistency with QE
      k(ij,ou) = (0.5d0/e2) * 0.125_dp * ( &
          ke(ns)%k(ih,jh,oh,uh) + ke(ns)%k(oh,uh,ih,jh) + &
          ke(ns)%k(jh,ih,oh,uh) + ke(ns)%k(uh,oh,ih,jh) + &
          ke(ns)%k(ih,jh,uh,oh) + ke(ns)%k(oh,uh,jh,ih) + &
          ke(ns)%k(jh,ih,uh,oh) + ke(ns)%k(uh,oh,jh,ih))
      !
    enddo
    enddo
    enddo
    enddo

    lda = nh(ns)*nh(ns)
    call eigsysD('V', 'U', .true.,lda,lda,k,ev)
    n=0
    do i=1, nh(ns)* nh(ns)
      if( abs(ev(i)) > 1.e-14 ) n = n + 1
    enddo
    allocate(L(nh(ns)*nh(ns),n))
    n=0
    do i=1,nh(ns)* nh(ns)
      if( abs(ev(i)) > 1.e-14 ) then
        n = n + 1
        L(:,n) = CMPLX(k(:,i),0._dp,kind=DP) * &
                             sqrt(CMPLX(ev(i),0._dp,kind=DP))
      endif
    enddo
    write(*,*) 'ns,nh,nke:',ns,nh(ns),n
    deallocate(k,ev,L)
  enddo
  call PAW_clean_fock_kernel()

END SUBROUTINE write_factorized_paw_one_center

! MAM: This should really be in qeh5_base_module, or in a module outside!
subroutine h5_write_vector_int(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space 

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  integer, intent(inout) :: v(:) 
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset

  CALL qeh5_set_space(dset, v(1), RANK=1, DIMENSIONS=[size(v,1)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  CALL qeh5_write_dataset(v, dset)

  call qeh5_close(dset)

end subroutine h5_write_vector_int

subroutine h5_write_vector_r(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  real(dp), intent(inout) :: v(:)
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset

  CALL qeh5_set_space(dset, v(1), RANK=1, DIMENSIONS=[size(v,1)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  CALL qeh5_write_dataset(v, dset)

  call qeh5_close(dset)

end subroutine h5_write_vector_r

subroutine h5_write_vector_c(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  complex(dp), intent(inout) :: v(:)
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset
  real(dp) :: rv 

  CALL qeh5_set_space(dset, rv, RANK=2, DIMENSIONS=[2,size(v,1)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  call write_complex_as_real( v(1), dset )
  call add_complex(dset%id)

  call qeh5_close(dset)

end subroutine h5_write_vector_c

subroutine h5_write_mat_int(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space 

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  integer, intent(inout) :: v(:,:) 
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset

  CALL qeh5_set_space(dset, v(1,1), RANK=2, DIMENSIONS=[size(v,1),size(v,2)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  CALL qeh5_write_dataset(v, dset)

  call qeh5_close(dset)

end subroutine h5_write_mat_int

subroutine h5_write_mat_r(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  real(dp), intent(inout) :: v(:,:)
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset

  CALL qeh5_set_space(dset, v(1,1), RANK=2, DIMENSIONS=[size(v,1),size(v,2)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  CALL qeh5_write_dataset(v, dset)

  call qeh5_close(dset)

end subroutine h5_write_mat_r

subroutine h5_write_mat_c(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  complex(dp), intent(inout) :: v(:,:)
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset
  integer :: ierr
  real(dp) :: rv 

  CALL qeh5_set_space(dset, rv, RANK=3, DIMENSIONS=[2, size(v,1),size(v,2)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  call write_complex_as_real( v(1,1), dset )
  call add_complex(dset%id)

  call qeh5_close(dset)


end subroutine h5_write_mat_c

subroutine h5_write_tensor_int(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  integer, intent(inout) :: v(:,:,:)
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset

  CALL qeh5_set_space(dset, v(1,1,1), RANK=3, DIMENSIONS=[size(v,1),size(v,2),size(v,3)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  CALL qeh5_write_dataset(v, dset)

  call qeh5_close(dset)

end subroutine h5_write_tensor_int

subroutine h5_write_tensor_r(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  real(dp), intent(inout) :: v(:,:,:)
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset

  CALL qeh5_set_space(dset, v(1,1,1), RANK=3, DIMENSIONS=[size(v,1),size(v,2),size(v,3)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  CALL qeh5_write_dataset(v, dset)

  call qeh5_close(dset)

end subroutine h5_write_tensor_r

subroutine h5_write_tensor_c(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  complex(dp), intent(inout) :: v(:,:,:)
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset
  real(dp) :: rv

  CALL qeh5_set_space(dset, rv, RANK=4, DIMENSIONS=[2,size(v,1),size(v,2),size(v,3)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  call write_complex_as_real( v(1,1,1), dset )
  call add_complex(dset%id)

  call qeh5_close(dset)

end subroutine h5_write_tensor_c

subroutine h5_write_tensor4_c(h5_f, v, id)

  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  type(qeh5_file), intent(inout) :: h5_f
  complex(dp), intent(inout) :: v(:,:,:,:)
  character(len=*), intent(in) :: id
  type(qeh5_dataset) :: dset
  real(dp) :: rv
  
  CALL qeh5_set_space(dset, rv, RANK=5, DIMENSIONS=[2,size(v,1),size(v,2),size(v,3),size(v,4)])
  CALL qeh5_open_dataset(h5_f, dset, ACTION='write', NAME=TRIM(id))
  call write_complex_as_real( v(1,1,1,1), dset )
  call add_complex(dset%id)

  call qeh5_close(dset)

end subroutine h5_write_tensor4_c

subroutine add_complex(h5_hid)

  USE hdf5
  USE ISO_C_BINDING
  USE qeh5_base_module, ONLY : qeh5_file, qeh5_dataset, qeh5_write_dataset, &
                               qeh5_open_dataset, qeh5_close, qeh5_set_space

  IMPLICIT NONE

  integer(HID_T), intent(inout) ::  h5_hid
  integer(HID_T) ::  str_t, attr_id, sp_id
  integer(HSIZE_T) :: text_length
  logical :: exists
  type(qeh5_dataset) :: dset
  integer :: ierr
  type(C_PTR) :: buf
  character(len=11) :: name
  character(len=1) :: val
  target :: val

  name = "__complex__"
  val = "1"
  text_length = len(trim(val))*1_HSIZE_T
  buf = c_loc(val)

  !
  call H5Tcopy_f(H5T_FORTRAN_S1, str_t, ierr )
  call H5Tset_size_f( str_t, text_length, ierr )
  call H5Tset_cset_f( str_t, H5T_CSET_UTF8_F, ierr )
  call H5Tset_strpad_f( str_t, H5T_STR_NULLTERM_F, ierr )
  ! 
  call H5Screate_f( H5S_SCALAR_F, sp_id, ierr )
  !
  call H5Aexists_by_name_f(h5_hid, '.', trim(name), exists, ierr )
  if (exists ) call H5Adelete_by_name_f( h5_hid, '.', trim(name), ierr )
  call H5Acreate_f( h5_hid, trim(name),  str_t, sp_id, attr_id, ierr)
  !
  call H5Awrite_f (attr_id, str_t, buf, ierr )
  !
  call H5Sclose_f(sp_id,   ierr )
  call H5Aclose_f(attr_id, ierr )

end subroutine add_complex

SUBROUTINE write_complex_as_real( c_data, h5_dataset )
  USE hdf5
  USE ISO_C_BINDING
  IMPLICIT NONE
  COMPLEX(DP), TARGET, INTENT(INOUT) ::  c_data
  TYPE(qeh5_dataset),INTENT(IN) ::  h5_dataset
  ! 
  TYPE(C_PTR) ::  buf
  INTEGER ::  ierr, jerr
  INTEGER(HID_T) ::  memspace_, filespace_
  INTEGER(HID_T) :: H5_REALDP_TYPE
  ! 
  buf = C_LOC(c_data)
  filespace_ = H5S_ALL_F
  memspace_  = H5S_ALL_F
  IF( ALLOCATED (h5_dataset%filespace%offset)) filespace_ = h5_dataset%filespace%id
  IF ( h5_dataset%memspace_ispresent)  memspace_ = h5_dataset%memspace%id
  H5_REALDP_TYPE = h5kind_to_type( DP, H5_REAL_KIND)
  CALL H5Dwrite_f ( h5_dataset%id, H5_REALDP_TYPE, buf  , ierr, memspace_,&
                    filespace_, H5P_DEFAULT_F )
END SUBROUTINE write_complex_as_real

SUBROUTINE eigsysD(jobz, uplo, reverse, n, lda, A, W)
  !
  USE KINDS, ONLY : DP
  !
  IMPLICIT NONE
  !
  CHARACTER(len=1),INTENT(IN) :: uplo, jobz
  INTEGER, INTENT(IN) :: n, lda
  REAL(DP), INTENT(INOUT) :: A(lda,n)
  REAL(DP), INTENT(OUT) :: W(n)
  LOGICAL, INTENT(IN) :: reverse
  !
  INTEGER :: i,j,IL,IU,M,lwork,liwork,info
  REAL(DP) :: VL,VU,ABSTOL
  INTEGER, ALLOCATABLE :: isuppz(:), iwork(:)
  REAL(DP), ALLOCATABLE :: Z(:,:), work(:)

  allocate( Z(n,n), isuppz(2*n), work(1), iwork(1) )

  lwork = -1
  liwork = -1
  call dsyevr(jobz,'A',uplo,n,A,lda,VL,VU,IL,IU,ABSTOL,M,W,Z,n,isuppz,&
              work,lwork,iwork,liwork,info)

  lwork = INT(dble(work(1)))
  liwork = INT(iwork(1))
  deallocate(work, iwork)
  allocate( work(lwork), iwork(liwork) )

  call dsyevr(jobz,'A',uplo,n,A,lda,VL,VU,IL,IU,ABSTOL,M,W,Z,n,isuppz,&
              work,lwork,iwork,liwork,info)

  if(info.ne.0) call errore('pw2qmcpack','info != 0',info)
  if(M.ne.n) call errore('pw2qmcpack','Not enough eigenvalues in eigsys.',1)

  ! copy eigenvalues to A
  if(reverse) then
    do j=1,n/2
      VL = W(j)
      W(j) = W(n-j+1)
      W(n-j+1) = VL
    enddo
    if(jobz=='V') then
      do j=1,n
        do i=1,n
          A(i,n-j+1) = Z(i,j)
        enddo
      enddo
    endif
  else
    if(jobz=='V') then
      do j=1,n
        do i=1,n
          A(i,j) = Z(i,j)
        enddo
      enddo
    endif
  endif

  deallocate(Z,isuppz,work, iwork)

END SUBROUTINE eigsysD

END PROGRAM pw2coqui

