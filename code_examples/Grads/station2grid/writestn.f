      character*8 stnid, id
      real lat, lon, time, prcp, temp
      integer date
      open(21, file='asstn.bin', form='unformatted', status='unknown'
     &      ,access='stream')
      
      do it=1,31
       
      do i=1,1000
      open(11, file='latlon.txt', form='formatted', status='old')
      read(11,'(a6,7x,f7.3,1x,f7.3)',end=90) stnid,lon,lat

      time=0.
      nlev=1
      nflag=1

      open(12, file='asstn.txt', form='formatted', status='old')
      do j=1,20000
      read(12,'(21x,a6,8x,i8,2x,f6.1,88x,f5.2)',end=92) 
     &      id,date,temp,prcp

      if(id .eq. stnid .and. date .eq. (it+20101200)) then
c
c   WRITE THE START HEADER
      write(21) id,lat,lon,time,nlev,nflag
c   WRITE THE VARIABLES
      write(21) prcp,temp
c      print *,id,stnid,' ',date,temp, prcp
      go to 91
      endif

      enddo
  92  close(12)

c   FILL NO DATA STATION
      write(21) stnid,lat,lon,time,nlev,nflag
      write(21) 99.99,99.99

  91  enddo
  90  close(11)
  
c   WRITE THE TERMINATE TAIL  
      nlev=0
      write(21) stnid,lat,lon,time,nlev,nflag

      print *,i-1
      enddo

      stop
      end