    SUBROUTINE TARNSP(A1,A2,DIAMX,PRINAX,DX,X0,CFLSHP,CDESCR,IDVSHP,IOSHP, &
                      MXNAT,NAT,IXYZ,ICOMP)
      USE DDPRECISION,ONLY : WP
      IMPLICIT NONE

!---------------------------- v3 -----------------------------------
!** Arguments:

      CHARACTER :: CDESCR*67,CFLSHP*80
      INTEGER :: IDVSHP,IOSHP,MXNAT,NAT
      INTEGER*2 :: ICOMP(MXNAT,3)
      INTEGER :: IXYZ(MXNAT,3)
      REAL(WP) :: DIAMX, PRINAX
      REAL(WP) :: &
         A1(3),   &
         A2(3),   &
         DX(3),   &
         X0(3)

!** Local variables:

      INTEGER :: NSPHMX
      PARAMETER (NSPHMX=10000)
      CHARACTER :: CMSGNM*70
      LOGICAL :: OCC
      INTEGER :: IWARN,JA,JX,JY,JZ,LMX1,LMX2,LMY1,LMY2,LMZ1,LMZ2,NSPH
      REAL(WP) :: R2,SCALE,X,XMAX,XMIN,XOFF,Y,YMAX,YMIN,YOFF,Z,ZMAX,ZMIN,ZOFF
      REAL(WP) ::     &
         ALPHA(3),    &
         AS(NSPHMX),  &
         AS2(NSPHMX), &
         XS(NSPHMX),  &
         YS(NSPHMX),  &
         ZS(NSPHMX)

!***********************************************************************
! Routine to construct multisphere target

! Input:
!        DIAMX =max extent in X direction/d
!        PRINAX=0 to set A1=(1,0,0), A2=(0,1,0)
!              =1 to use principal axes for vectors A1 and A2
!                 as found here for dipole realization
!              =2 to use axes a1 and a2 read in from CFLSHP
!        DX(1-3)=(dx,dy,dz)/d where dx,dy,dz=lattice spacings in x,y,z
!                directions, and d=(dx*dy*dz)**(1/3)=effective lattice
!                spacing
!        CFLSHP= name of file containing locations and radii of spheres
!        IOSHP =device number for "target.out" file
!              =-1 to suppress printing of "target.out"
!        MXNAT =dimensioning information (max number of atoms)

! and, from input file CFLSHP:

!        NSPH              = number of spheres
!        XOFF YOFF ZOFF    = offsets/d of dipoles from centroid
!        A1(1) A1(1) A3(1)  [only read if PRINAX = 2]
!        A2(1) A2(2) A3(2)  [only read if PRINAX = 2]
!        descriptive lines which will be ignored (may be blank)
!        XS(1) YS(1) ZS(1) AS(1)
!        XS(2) YS(2) ZS(2) AS(2)
!        ...
!        XS(NSPH) YS(NSPH) ZS(NSPH) AS(NSPH)

! where XS(J),YS(J),ZS(J) = x,y,z coordinates in target frame, in
!                                 arbitrary units, of center of sphere J
!        AS(J)            = radius of sphere J (same arbitrary units)

! units used for XS,YS,ZS, and AS are really arbitrary: actual size of
! overall target, in units of lattice spacing d, is controlled by
! parameter SHPAR(1) in ddscat.par
! if line 2 does not begin with readable XOFF YOFF ZOFF, then
! code here assumes XOFF=0, YOFF=0, ZOFF=0

! Output:
!        A1(1-3)=(1,0,0)=unit vector defining target axis 1 in Target Frame
!        A2(1-3)=(0,1,0)=unit vector defining target axis 2 in Target Frame
!        X0(1-3)=location/d in TF of lattice site IXYZ=0 0 0
!                TF origin is taken to be volume-weighted centroid of
!                N spheres
!        NAT=number of atoms in target
!        IXYZ(1-NAT,1-3)=x/d,y/d,z/d for atoms of target
!        ICOMP(1-NAT,1-3)=composition identifier (currently 1 1 1)
!        CDESCR=description of target (up to 67 characters)

! B.T.Draine, Princeton Univ. Obs., 2000.06.12

! History:
! 00.06.12 (BTD) adapted to version 6.0 (introduce DX)
! 00.11.02 (BTD) write ICOMP to target.out
! 01.01.13 (BTD) corrected comments
! 01.04.21 (BTD) corrected comments
! 01.04.21 (BTD) corrected typo: changed
!                      YMAX=YS(1)+YS(1)
!                to    YMAX=YS(1)+AS(1)
!                Thanks to Ivan O. Sosa Perez (UNAM) for discovering
!                this error.
! 03.06.06 (BTD) corrected error in setting LMX1,...,LMZ2
! 03.11.06 (BTD) modified to use new version of PRINAXIS with eigenvalue
!                in argument list
! 04.03.19 (BTD) revert to previous version of PRINAXIS
!                change CFLSHP*13 to CFLSHP*80 to accomodate long file
!                names
! 04.05.23 (BTD) modified to obtain eigenvalues ALPHA from PRINAXIS
!                modified arg list of PRINAXIS to conform to change
! 04.09.15 (BTD) added call to ERRMSG and abort if NSPH > NSPHMX
! 07.09.11 (BTD) changed IXYZ from INTEGER*2 to INTEGER
! 08.08.08 (BTD) added X0 to argument list
!                added code to set X0 so that TF origin is located at
!                volume-weighted centroid of the N spheres
! 08.08.30 (BTD) modified format 9020
! 08.09.17 (BTD) corrected bugs that caused target centroid X0 to be
!                computed incorrectly (identical bugs were present in
!                routine TARNAS
! 09.09.09 (BTD) ver7.0.8
!                added some more running output to help diagnose
!                fatal errors resulting from problems with input parameters 
! 10.01.06 (BTD) v2 (for v7.1.0)
!                * changed to expect input file to have structure
!                    N = number of spheres
!                    comment line
!                    comment line
!                    comment line
!                    comment line
!                    x1 y1 z1 a1
!                    x2 y2 z2 a2
!                       ...
!                    xN yN zN aN
!                  to provide compatibility with output of
!                  programs agglom.f and assign_comp.f

!                * added sanity checks to detect and report input file
!                  incompatibilities
! 12.01.31 (BTD) * added code to allow check for sphere overlap
!                  with warning message for each overlapping dipole
! 15.03.19 (BTD) * added code to allow option PRINAX=2
!                  whereby a1,a2 read in from target file
! 20.03.29 (BTD) v3
!                * modified to choose between two possible locations for
!                  lattice w.r.t. centroid 
! end history

! Copyright (C) 2000,2001,2003,2004,2007,2008,2009,2010,2012,2015,2020
!               B.T. Draine and P.J. Flatau
! This code is covered by the GNU General Public License.
!***********************************************************************

!*** diagnostic
!      write(0,fmt='(a,f10.4)')'tarnsp_v2 ckpt 0, prinax=',prinax
!***
! Read parameters of spheres:

      WRITE(0,FMT='(A)')'>TARNSP open file =',CFLSHP
!-------------------------------------
      OPEN(UNIT=IDVSHP,FILE=CFLSHP)
      READ(IDVSHP,*,ERR=8100)NSPH
      WRITE(CMSGNM,FMT='(A,I6,A)')'cluster of',NSPH,' spheres'
      CALL WRIMSG('TARNSP',CMSGNM)
      IF(NSPH>NSPHMX)THEN
!-------------------------------------
! 09.09.09 (BTD) added code ----------
         WRITE(CMSGNM,FMT='(A,I6,A,I6,A)')'fatal error: NSPH=',NSPH, &
                                        ' > NSPHMX',NSPHMX
         CALL WRIMSG('TARNSP',CMSGNM)
!-------------------------------------
         CALL ERRMSG('FATAL','TARNSP',' NSPH > NSPHMX')
         STOP
      ENDIF
      XOFF=0._WP
      YOFF=0._WP
      ZOFF=0._WP
      READ(IDVSHP,*,ERR=1100)XOFF,YOFF,ZOFF
      WRITE(CMSGNM,FMT='(3F6.3,A)')XOFF,YOFF,ZOFF, &
         ' = x,y,z offsets/d of dipoles from centroid'
      CALL WRIMSG('TARNSP',CMSGNM)
 1100 CONTINUE
      IF(NINT(PRINAX)<2)THEN
! skip lines 3-6
         DO JA=3,5
            READ(IDVSHP,*)
         ENDDO
      ELSE       ! if PRINAXIS = 2, then read A1 and A2 from file
         READ(IDVSHP,*)A1
         READ(IDVSHP,*)A2
         READ(IDVSHP,*)
      ENDIF
      DO JA=1,NSPH
         JX=JA
!*** diagnostic
!         write(0,*)'tarnsp_v2 ckpt 1: ja=',ja
!***
         READ(IDVSHP,*,ERR=8200,END=8300)XS(JA),YS(JA),ZS(JA),AS(JA)
      ENDDO

      CLOSE(IDVSHP)
      WRITE(0,FMT='(A)')'>TARNSP close file=',CFLSHP

! Dipoles are located at sites
! (x,y,z)=(I,J,K), I,J,K=integers

! Determine max extent in X direction:

      XMIN=XS(1)-AS(1)
      XMAX=XS(1)+AS(1)
      YMIN=YS(1)-AS(1)
      YMAX=YS(1)+AS(1)
      ZMIN=ZS(1)-AS(1)
      ZMAX=ZS(1)+AS(1)
      IF(NSPH>1)THEN
         DO JA=2,NSPH
            IF(XS(JA)-AS(JA)<XMIN)XMIN=XS(JA)-AS(JA)
            IF(XS(JA)+AS(JA)>XMAX)XMAX=XS(JA)+AS(JA)
            IF(YS(JA)-AS(JA)<YMIN)YMIN=YS(JA)-AS(JA)
            IF(YS(JA)+AS(JA)>YMAX)YMAX=YS(JA)+AS(JA)
            IF(ZS(JA)-AS(JA)<ZMIN)ZMIN=ZS(JA)-AS(JA)
            IF(ZS(JA)+AS(JA)>ZMAX)ZMAX=ZS(JA)+AS(JA)
         ENDDO
      ENDIF
      WRITE(CMSGNM,FMT='(A,F10.5)')'XMIN=',XMIN
      CALL WRIMSG('TARNSP',CMSGNM)
      WRITE(CMSGNM,FMT='(A,F10.5)')'XMAX=',XMAX
      CALL WRIMSG('TARNSP',CMSGNM)
      WRITE(CMSGNM,FMT='(A,F10.5)')'YMIN=',YMIN
      CALL WRIMSG('TARNSP',CMSGNM)
      WRITE(CMSGNM,FMT='(A,F10.5)')'YMAX=',YMAX
      CALL WRIMSG('TARNSP',CMSGNM)
      WRITE(CMSGNM,FMT='(A,F10.5)')'ZMIN=',ZMIN
      CALL WRIMSG('TARNSP',CMSGNM)
      WRITE(CMSGNM,FMT='(A,F10.5)')'ZMAX=',ZMAX
      CALL WRIMSG('TARNSP',CMSGNM)
      SCALE=DIAMX/(XMAX-XMIN)

! Now determine min,max values of I,J,K:

      LMX1=NINT(SCALE*XMIN/DX(1)-XOFF-0.0001_WP)
      LMX2=NINT(SCALE*XMAX/DX(1)-XOFF+0.0001_WP)
      LMY1=NINT(SCALE*YMIN/DX(2)-YOFF-0.0001_WP)
      LMY2=NINT(SCALE*YMAX/DX(2)-YOFF+0.0001_WP)
      LMZ1=NINT(SCALE*ZMIN/DX(3)-ZOFF-0.0001_WP)
      LMZ2=NINT(SCALE*ZMAX/DX(3)-ZOFF+0.0001_WP)

      DO JA=1,NSPH
         AS2(JA)=(SCALE*AS(JA))**2
         XS(JA)=SCALE*XS(JA)
         YS(JA)=SCALE*YS(JA)
         ZS(JA)=SCALE*ZS(JA)
      ENDDO

! Determine list of occupied sites

      NAT=0
      IWARN=0
      DO JZ=LMZ1,LMZ2
         Z=(REAL(JZ,KIND=WP)+ZOFF)*DX(3)
         DO JY=LMY1,LMY2
            Y=(REAL(JY,KIND=WP)+YOFF)*DX(2)
            DO JX=LMX1,LMX2
               X=(REAL(JX,KIND=WP)+XOFF)*DX(1)
               OCC=.FALSE.
               DO JA=1,NSPH
                  R2=(X-XS(JA))**2+(Y-YS(JA))**2+(Z-ZS(JA))**2
                  IF(R2<AS2(JA))THEN
                     IF(OCC)THEN
                        IWARN=IWARN+1
                        IF(IWARN<=10)THEN
                           WRITE(CMSGNM,FMT='(A,I6)')'overlap for sphere',JA
                           CALL WRIMSG('TARNSP',CMSGNM)
                        ENDIF
                        IF(IWARN==10)THEN
                           WRITE(CMSGNM,FMT='(A)')                             &
                              'IWARN=10: further overlap warnings suppressed...'
                           CALL WRIMSG('TARNSP',CMSGNM)
                        ENDIF
                     ENDIF
                     OCC=.TRUE.
                  ENDIF
               ENDDO
               IF(OCC)THEN

! Site is occupied:

                  NAT=NAT+1
                  IXYZ(NAT,1)=JX
                  IXYZ(NAT,2)=JY
                  IXYZ(NAT,3)=JZ
               ENDIF
            ENDDO
         ENDDO
      ENDDO

      IF(NAT>MXNAT)THEN
!----------------------------
! 09.09.09 (BTD) added code
         WRITE(CMSGNM,'(A,I10,A,I10)')'NAT=',NAT,' > MXNAT=',MXNAT
         CALL WRIMSG('TARNSP',CMSGNM)
!----------------------------
         CALL ERRMSG('FATAL','TARNSP',' NAT.GT.MXNAT ')
      ENDIF

! Homogeneous target:

      DO JA=1,NAT
         DO JX=1,3
            ICOMP(JA,JX)=1
         ENDDO
      ENDDO

!*** diagnostic
!      write(0,*)'tarsnp_v2 ckpt 2: made target with NAT=',NAT,' dipoles'
!***

! Specify target axes A1 and A2
! If PRINAX=0, then
!     A1=(1,0,0) in target frame
!     A2=(0,1,0) in target frame
!     set ALPHA(1)=ALPHA(2)=ALPHA(3)=0
! If PRINAX=1, then
!     A1,A2 are principal axes of largest, second largest moment of
!     inertia
! If PRINAX=2, then a1,a2 are already set from input file
      IF(NINT(PRINAX)<=0)THEN
         DO JX=1,3
            A1(JX)=0._WP
            A2(JX)=0._WP
            ALPHA(JX)=0._WP
         ENDDO
         A1(1)=1._WP
         A2(2)=1._WP
      ELSEIF(NINT(PRINAX)==1)THEN
!*** diagnostic
!         write(0,*)'tarnsp_v2 ckpt 3'
!***
         CALL PRINAXIS(MXNAT,NAT,ICOMP,IXYZ,DX,A1,A2,ALPHA)
!*** diagnostic
!         write(0,*)'tarnsp_v2 ckpt 4'
!         write(0,*)'  alpha(1-3)=',alpha
!***
      ELSEIF(NINT(PRINAX)>=2)THEN
!*** diagnostic
!         write(0,*)'tarnsp_v2 ckpt 5: prinax=2'
!***
         DO JX=1,3
            ALPHA(JX)=0._WP
         ENDDO
      ENDIF

! Find volume-weighted centroid of the N spheres in IJK space
! Take this to be TF origin.
! Set X0 = -(IJK centroid)

      DO JX=1,3
         X0(JX)=0._WP
      ENDDO
      Z=0._WP
      DO JA=1,NSPH
         Z=Z+AS(JA)**3
      ENDDO
      DO JA=1,NSPH
         Y=AS(JA)**3/Z
         X0(1)=X0(1)-XS(JA)*Y
         X0(2)=X0(2)-YS(JA)*Y
         X0(3)=X0(3)-ZS(JA)*Y
      ENDDO

!***********************************************************************
! Write target description into string CDESCR

      WRITE(CDESCR,FMT='(A,I10,A)')' Multisphere cluster containing',NAT, &
                                  ' dipoles'

!***********************************************************************

      CMSGNM=CDESCR
      CALL WRIMSG('TARNSP',CMSGNM)
      IF(IOSHP>=0)THEN
         OPEN(UNIT=IOSHP,FILE='target.out',STATUS='UNKNOWN')
         WRITE(IOSHP,FMT=9020)NSPH,DIAMX,NAT,ALPHA,A1,A2,DX,X0
         DO JA=1,NAT
            WRITE(IOSHP,FMT=9030)JA,IXYZ(JA,1),IXYZ(JA,2),IXYZ(JA,3), &
                                 ICOMP(JA,1),ICOMP(JA,2),ICOMP(JA,3)
         ENDDO
         CLOSE(UNIT=IOSHP)
      ENDIF

      RETURN
8100  WRITE(CMSGNM,FMT='(2A)')' Fatal error: unable to read number', &
                              ' N of spheres'
      CALL WRIMSG('TARNSP',CMSGNM)
      WRITE(CMSGNM,FMT='(2A)')' from file=',CFLSHP
      CALL WRIMSG('TARNSP',CMSGNM)
      STOP
8200  WRITE(CMSGNM,FMT='(2A,I7)')' Fatal error: error reading data for', &
                                 ' sphere',JX
      CALL WRIMSG('TARNSP',CMSGNM)
      STOP
8300  WRITE(CMSGNM,FMT='(A,I7,A)')' Fatal error: expected to read data for', &
                                  NSPH,' spheres'
      CALL WRIMSG('TARNSP',CMSGNM)
      WRITE(CMSGNM,FMT='(A,I7,A)')' but only found data for',JX-1,' spheres'
      CALL WRIMSG('TARNSP',CMSGNM)
      STOP

9020  FORMAT('>TARNSP multisphere target composed of ',I4,         &
         ' spheres, DIAMX=',F8.4,/,                                &
         I10,3F8.4,' = NAT, alpha_1-3',/,                          &
         3F10.6,' = A_1 vector',/,                                 &
         3F10.6,' = A_2 vector',/,                                 &
         3F10.6,' = lattice spacings (d_x,d_y,d_z)/d',/,           &
         3F10.5,' = lattice offset x0(1-3) = (x_TF,y_TF,z_TF)/d ', &
               'for dipole 0 0 0',/,                               &
         '     JA  IX  IY  IZ ICOMP(x,y,z)')

9030  FORMAT(I10,3I10,3I2)
    END SUBROUTINE TARNSP
