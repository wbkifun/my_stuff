program simple_ice_model
  use simple_ice_model_modules
  implicit none

  ! local variables
  real(8), parameter    :: pi = 3.1415926536

  integer, parameter :: grid=51                    ! Number of nodes
  real(8),    parameter :: dt = 0.1                ! length time step 
  integer, parameter :: t_final=1000                ! integration stop time
  integer :: t_step, nt                                ! current time step, number of time steps
  real(8), parameter :: xl=0                       ! start of domain
  real(8), parameter :: xr=50e3                    ! end of domain 
  real(8) :: dx                                    ! node spacing
  real(8) :: time                                  ! current time
  real(8) :: const                                 ! a constant
  
  !! unstaggered grid
  real(8), dimension(:), allocatable :: xx         ! node positions
  real(8), dimension(:), allocatable :: thick      ! ice thickness
  real(8), dimension(:), allocatable :: dthick_dt  ! ice thickness time derivative
  real(8), dimension(:), allocatable :: mass_b     ! mass balance
  real(8), dimension(:), allocatable :: bed        ! bed elevation

  !! staggered grid
  real(8), dimension(:), allocatable :: diffus     ! diffusion
  real(8), dimension(:), allocatable :: xx_s       ! xx on staggered
  
  !! physical constants
  real(8), parameter :: rho=920, g_grav=9.8, A_rate=1e-16, n_glen=3.
  real(8), parameter :: dbed_dx=0.025, mass_b_0=4., mass_b_1=0.2e-3 !xxx
  
  !! BC
  real(8), parameter :: thickl=0, thickr=0 !xxx

  !! counters and such
  integer :: ii, jj
  integer :: errstat                            ! for error checking

  !! functions
  
  ! let us allocate some memory
  allocate(xx(grid),stat=errstat)
  call checkerr(errstat,"failed to allocate xx")

  allocate(thick(grid),stat=errstat)
  call checkerr(errstat,"failed to allocate thick")

  allocate(dthick_dt(grid),stat=errstat)
  call checkerr(errstat,"failed to allocate dthick_dt")

  allocate(mass_b(grid),stat=errstat)
  call checkerr(errstat,"failed to allocate mass_b")

  allocate(bed(grid),stat=errstat)
  call checkerr(errstat,"failed to allocate bed")

  allocate(diffus(grid-1),stat=errstat)
  call checkerr(errstat,"failed to allocate diffus")

  allocate(xx_s(grid-1),stat=errstat)
  call checkerr(errstat,"failed to allocate xx_s")

  !! setup
  dx = (xr-xl)/(grid-1)
  do ii=1,grid
     xx(ii) = xl + dx*(ii-1)
     thick(ii) = 0   ! IC
     mass_b(ii) = mass_b_0 - mass_b_1*xx(ii)
     bed(ii) = -dbed_dx * xx(ii)
  end do
  const = 2*A_rate/(n_glen+2)*(rho*g_grav)**n_glen
  nt = floor(t_final/dt) ! number of time steps
  
  !! output, use a pipe to get into a file
  print * , 'x'
  print * , xx
  print * , 'mass_b'
  print * , mass_b
  print * , 'bed'
  print * , bed

  print *, 'thick'
  do ii=1,nt
     ! compute the diffusivity: diffus
     diffus = calc_diffus(thick, bed, dx, const, n_glen)
     ! assemble dthick_dt
     dthick_dt = pad(diff_1(diffus*diff_1((thick+bed), dx), dx), thickl, thickr) + mass_b
     ! time step
     thick = time_step(thick, dthick_dt, dt)
     ! BC
     thick(1) = thickl
     do jj=1,grid
        if (thick(jj)<0.) then
          thick(jj) = 0.
        end if
     end do
     print *, thick
  end do

contains

  subroutine checkerr(errstat,msg)
    !! the error checker for allocation
    implicit none
    integer,      intent(in) :: errstat
    character(*), intent(in) :: msg 
    if (errstat /= 0) then
       write(*,*) "ERROR:", msg
       stop
    end if
  end subroutine checkerr

end program simple_ice_model
