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




SUBROUTINE test_util()

  IMPLICIT NONE

  write(*,*) 'Testing!!! Inside test_util'

END SUBROUTINE test_util

SUBROUTINE read_pw2bgw_vkbg_header(fname, clen, nspin, nkb, npwx, nkpts, nat, nsp, nhm, ierr)
  USE ISO_C_BINDING, ONLY: C_CHAR, C_INT
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
  INTEGER(C_INT), INTENT(IN) :: clen
  CHARACTER(KIND=C_CHAR,len=1024), INTENT(IN) :: fname
  INTEGER(C_INT), INTENT(OUT) :: nspin, nkb, npwx, nkpts, nat, nsp, nhm, ierr
  INTEGER :: runit, stat, ngm_g, ntran, cell_symmetry
  REAL(DP) omega,ecutrho
 
  runit = 4
  ierr = 0

  open (unit = runit, file = TRIM (fname(1:clen)), status='old', &
      form = 'unformatted', action='read', iostat=stat)
  if(stat /= 0) then 
    ierr = stat
    return
  endif
  
  read( runit, iostat=stat ) !stitle, sdate, stime
  if(stat /= 0) then 
    ierr = stat
    return
  endif
  read( runit, iostat=stat ) nspin, ngm_g, ntran, cell_symmetry, nat, ecutrho, &
      nkpts, nsp, nkb, nhm, npwx  !, ecutwfc 
  if(stat /= 0) then 
    ierr = stat
    return
  endif

  close(runit)

END SUBROUTINE read_pw2bgw_vkbg_header

SUBROUTINE read_pw2bgw_vkbg(fname, clen, kbeg, kend, ityp, ityp_size, nh, nh_size, &
        npw, npw_size, Dnn, Dnn_size, miller, miller_size, vkb, vkb_size, ierr)
  USE ISO_C_BINDING, ONLY: C_CHAR, C_INT, C_DOUBLE, C_DOUBLE_COMPLEX
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
  INTEGER(C_INT), INTENT(IN) :: clen, kbeg, kend, Dnn_size, miller_size, vkb_size, ityp_size, nh_size, npw_size
  REAL(C_DOUBLE), INTENT(OUT) :: Dnn(*)
  COMPLEX(C_DOUBLE_COMPLEX) vkb(*)
  INTEGER(C_INT), INTENT(OUT) :: miller(*), ityp(*), nh(*), npw(*)
  CHARACTER(KIND=C_CHAR,len=1024), INTENT(IN) :: fname
  INTEGER(C_INT), INTENT(OUT) ::  ierr
  INTEGER :: nkb, npwx, nkpts, nat, nhm, runit, stat, nsf, ngm_g, ntran, cell_symmetry, nsp
  INTEGER :: i,j,k,l,icnt,ik,ik0,iat,jat,isp,ih,jh,is,nkstot,ns,nst,ngk,ig,ikb
  REAL(DP) omega,ecutrho
  integer, allocatable :: kmap (:)
  integer, allocatable :: smap (:), gvec(:,:)
  COMPLEX(DP), allocatable :: vkb_f(:)
  REAL(DP), allocatable :: deeq(:,:,:,:)

  runit = 4
  ierr = 0
  miller(1:miller_size) = 0
  vkb(1:vkb_size)=(0.d0,0.d0)
  Dnn(1:Dnn_size)=0.d0

  !write(*,*) 'Filename: ',TRIM(fname(1:clen))
  open (unit = runit, file = TRIM (fname(1:clen)), status='old', &
      form = 'unformatted', action='read', iostat=stat)
  if(stat /= 0) then
    ierr = stat
    return
  endif

  read( runit ) !stitle, sdate, stime
  read( runit, iostat=stat ) nsf, ngm_g, ntran, cell_symmetry, nat, ecutrho, &
      nkpts, nsp, nkb, nhm, npwx  !, ecutwfc 
  if(stat /= 0) then 
    ierr = stat
    return
  endif

  if(nsf.gt.2) then
    ierr = 1
    write(*,*) 'Error: read_vkbg not yet implemented for nspin>2.'
    return
  endif
  call set_spin(ns,nst,nsf)
  nkstot = nkpts*ns
  allocate ( kmap (nkstot), smap(nkstot) )
  do i = 1, nkstot
    j = ( i - 1 ) / ns
    k = i - 1 - j * ns
    kmap ( i ) = j + k * ( nkstot / ns ) + 1
    smap ( i ) = k + 1
  enddo

  if( Dnn_size.ne.nhm*nhm*nsp*ns .or. &
      vkb_size.ne.(kend-kbeg)*npwx*nkb .or. &
      miller_size.ne. (kend-kbeg)*3*npwx .or. &
      ityp_size.ne.nat .or. & 
      npw_size.ne.(kend-kbeg) .or. & 
      nh_size.ne.nsp ) then
    ierr=1
    write(*,*) 'Error: read_vkbg incorrect array dimensions'
    return
  endif
  if(nsf==4) then
    allocate(deeq(nhm,nhm,nat,4))
  else
    allocate(deeq(nhm,nhm,nat,ns))
  endif

  read( runit ) ! dfftp%nr1, dfftp%nr2, dfftp%nr3, wfng_nk1, wfng_nk2, wfng_nk3, &
!        wfng_dk1, wfng_dk2, wfng_dk3 
  read( runit ) !omega, alat, ( ( at ( j, i ), j = 1, nd ), i = 1, nd ), &
!      ( ( adot ( j, i ), j = 1, nd ), i = 1, nd ) 
  read( runit ) !recvol, tpiba, ( ( bg ( j, i ), j = 1, nd ), i = 1, nd ), &
!      ( ( bdot ( j, i ), j = 1, nd ), i = 1, nd )
  read( runit ) !( ( ( s ( k, j, i ), k = 1, nd ), j = 1, nd ), i = 1, ntran )
  read( runit ) !( ( translation ( j, i ), j = 1, nd ), i = 1, ntran )
  read( runit ) !( ( tau ( j, i ), j = 1, nd ), atomic_number ( atm ( ityp ( i ) ) ), i = 1, nat )
  read( runit ) !( ngk_g ( ik ), ik = 1, nkstot / ns )
  read( runit ) !( wk ( ik ) * dble ( nst ) / 2.0D0, ik = 1, nkstot / ns )
  read( runit ) !( ( xk ( id, ik ), id = 1, nd ), ik = 1, nkstot / ns )
  read( runit, iostat=stat ) ( ityp ( iat ), iat = 1, nat )
  if(stat /= 0) then 
    ierr = stat
    return
  endif
  do j=1,nat
    ityp(j) = ityp(j)-1  ! to C-style zero-based
  enddo
  read( runit, iostat=stat ) ( nh ( isp ), isp = 1, nsp )
  if(stat /= 0) then 
    ierr = stat
    return
  endif
  if( nsf == 4 ) then
    read( runit ) ( ( ( ( deeq ( jh, ih, iat, is ), &
     jh = 1, nhm ), ih = 1, nhm ), iat = 1, nat ), is = 1, 4 )
    if(stat /= 0) then 
      ierr = stat
      return
    endif
  else
    read( runit ) ( ( ( ( deeq ( jh, ih, iat, is ), &
     jh = 1, nhm ), ih = 1, nhm ), iat = 1, nat ), is = 1, ns )
    if(stat /= 0) then 
      ierr = stat
      return
    endif
  endif
  icnt=1
  do i=1,size(deeq,4)
   do j=1,nsp
    ! find ityp(jat) = j 
    jat = -1
    do iat=1,nat
     if(ityp(iat)+1==j) then
       jat = iat 
       exit 
     endif
    enddo
    if(jat < 0) then 
      ierr = 991 
      return
    endif
    do k=1,nhm
     do l=1,nhm
       Dnn(icnt) = deeq(l,k,jat,i)
       icnt=icnt+1
     enddo
    enddo
   enddo
  enddo
  deallocate(deeq)
  read( runit ) !nrecord
  read( runit ) !ngm_g
  read( runit ) !( ( gvec ( id, ig ), id = 1, nd ), ig = 1, ngm_g )


  allocate(vkb_f(npwx),gvec(3,npwx))
  do i=1,nkstot

    ik = kmap(i)
    is = smap(i)
    ik0 = kmap(i)
    if(is.eq.2) ik0 = ik - nkstot/ns

    if(is.eq.1) then 
      read( runit )
      read( runit, iostat=stat ) ngk
      if(stat /= 0) then 
        ierr = stat
        return
      endif
      read( runit, iostat=stat ) ( ( gvec ( j, ig ), j = 1, 3 ), &
          ig = 1, ngk )
      if(stat /= 0) then 
        ierr = stat
        return
      endif
      if(ik.ge.(kbeg+1) .and. ik.le.kend) then
        icnt=(ik-kbeg-1)*3*npwx + 1
        do ig=1,ngk
          do j=1,3
            miller(icnt) = gvec(j,ig) 
            icnt=icnt+1
          enddo
        enddo
      endif
    endif

    ! assuming that in collinear case, vkb is spin independent, it should be 
    if(ik0.ge.(kbeg+1) .and. ik0.le.kend) then

      if(is.eq.1) then
  
        do ikb = 1, nkb 

          read( runit )
          read( runit, iostat=stat ) ngk
          if(stat /= 0) then 
            ierr = stat
            return
          endif
          if(ikb.eq.1) npw(ik0-kbeg) = ngk
          read( runit, iostat=stat ) ( vkb_f ( ig ), ig = 1, ngk )
          if(stat /= 0) then 
            ierr = stat
            return
          endif
          icnt=(ik0-kbeg-1)*npwx*nkb + (ikb-1)*npwx + 1
          vkb(icnt:icnt+ngk-1) = vkb_f(1:ngk)

        enddo

      else if(is.eq.2) then

        ! check that indeed vkb is spin independent
        do ikb = 1, nkb

          read( runit )
          read( runit, iostat=stat ) ngk
          if(stat /= 0) then
            ierr = stat
            return
          endif
          if( ngk .ne. npw(ik0-kbeg)) then
            ierr=991
            return
          endif
          read( runit, iostat=stat ) ( vkb_f ( ig ), ig = 1, ngk )
          if(stat /= 0) then
            ierr = stat
            return
          endif
          icnt=(ik0-kbeg-1)*npwx*nkb + (ikb-1)*npwx + 1
          do ig = 1, ngk
            if( abs(vkb_f(ig)-vkb(icnt+ig-1)) > 1.0d-8 ) then
              ierr=992
              return
            endif
          enddo
        enddo

      endif

    else

      do ikb = 1, nkb

        read( runit )
        read( runit )
        read( runit, iostat=stat ) 
        if(stat /= 0) then
          ierr = stat
          return
        endif

      enddo

    endif
    if(ik0.gt.kend) exit

  enddo

  close(runit)
  deallocate(kmap,smap,gvec,vkb_f)

END SUBROUTINE read_pw2bgw_vkbg

subroutine set_spin(ns, nst, nspin)

  IMPLICIT NONE

  integer, intent(out) :: ns, nst
  integer, intent(in) :: nspin

  IF ( nspin == 4 ) THEN
    ns = 1
    nst = 2
  ELSE
    ns = nspin
    nst = nspin
  ENDIF

  RETURN

end subroutine set_spin
