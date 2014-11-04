! PROGRAM: CreateStation2ModelGrid
! AUTHOR: Luiz Rodrigo Tozzi - luizrodrigotozzi@gmail.com
! DESCRIPTION: This fortran code converts a free-styled table with "station" values to a formatted table. This is the first step to create a GrADS station binary file.
!
! INPUT FILE LOOKS LIKE (separeted by any number of spaces):
!
! (...)
! yearwith4digits    monthwith2digits    daywith2digits    hourwith2digits    stationname    latitude       longitude            value
! (...)
!

	program CreateStation2ModelGrid

	parameter (nl=9999) ! nl: MAXIMUM NUMBER OF STATIONS

	real lat(nl),lon(nl),var(nl)
	integer m2,y4,status
        character*3 m2name(12)
        character*2 d2,h2
        character sid
	open(unit=1,file='stationdata.txt')
	open(unit=2,file='station2modelgrid.txt')
	open(unit=10,file='station2modelgrid.ctl')

	i=0
	do
         read(1,*,END=21)y4,m2,d2,h2,sid,lat(i),lon(i),var(i)
         if((lat(i).eq.lon(i)).and.(lon(i).eq.0))exit
         i=i+1
 20	end do
 
 21     do 30 j=1,i
!   Transforms the station name (sid) in an index number (i)
         write(2,14)y4,m2,d2,h2,i,lat(j),lon(j),var(j)
 30	continue
 
 14     format(i4,2x,i2,2x,a2,2x,a2,2x,i5,9x,f10.3,2x,f10.3,2x,f10.3)

!!!!! Generating CTL

!! Picking the right m2 name
        m2name=(/'jan','feb','mar','apr','may','jun','jul',
     *'aug','sep','oct','nov','dec'/)

	write(10,*) "DSET   ^station2modelgrid.bin"
	write(10,*) "DTYPE  station "
	write(10,*) "STNMAP station2modelgrid.map"
	write(10,*) "ZDEF 1 1"
	write(10,*) "UNDEF   -9.99e33"
	write(10,*) "TITLE  Station Data"
	write(10,'(a16,a2,a1,a2,a3,i4,a5)') "TDEF   1 linear ",h2,
     *"z",d2,m2name(m2),y4," 12hr"
 	write(10,*) "VARS 1"
	write(10,*) "var     0  99   **"
	write(10,*) "ENDVARS"

	stop
	end	
