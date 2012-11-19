!file: simple_ice_model.f90
subroutine simple_ice_model(n_t_out, grid, dt, t_final, xl, xr, rho, g_grav, A_rate, &
   n_glen, dbed_dx, mass_b_0, mass_b_1, b_cond, &
   thick_out, bed, xx, times_out) ! output
  use simple_ice_model_modules
  implicit none

  ! local variables
  real(8), parameter    :: pi = 3.1415926536
  
  !! input xxxxxxxxxx
  integer :: grid, n_t_out                    ! Number of nodes, number of output times
  real(8) :: dt, t_final             ! time step, stopping time
  real(8) :: dbed_dx, mass_b_0, mass_b_1, xl, xr ! model parameters, bed slope, mass balanace params 2x, left and right extent of domain
  real(8) :: rho, g_grav, A_rate, n_glen ! physical constants: dens. of ice, ice rate factor, n
  !! BC
  real(8), dimension(2) :: b_cond          ! thickness at left and right edge
  !! xxxxxxxxxx

  !! numerical variables
  integer :: nt, ind_t_out                            ! number of time steps, index when output happens
  real(8) :: dx                                    ! node spacing
  real(8), dimension(n_t_out+1) :: times_out         ! output times
  real(8) :: const_coef                                 ! a constant
  real(8) :: time                                 ! current model time
  
  !! unstaggered grid
  real(8), dimension(grid) :: xx         ! node positions
  real(8), dimension(grid) :: thick      ! ice thickness
  real(8), dimension(grid, n_t_out+1) :: thick_out      ! ice thickness
  real(8), dimension(grid) :: dthick_dt  ! ice thickness time derivative
  real(8), dimension(grid) :: mass_b     ! mass balance
  real(8), dimension(grid) :: bed        ! bed elevation

  !! staggered grid
  real(8), dimension(grid-1) :: diffus     ! diffusion
  
  !! physical constants
  

  !! counters and such
  integer :: ii, jj, iii

!! f2py declarations
!f2py integer optional, intent(in) :: n_t_out=30, grid=51
!f2py real(8) optional, intent(in) :: dt=0.1, t_final=1000., xl=0., xr=50.e3
!f2py real(8) optional, intent(in) :: dbed_dx=0.025, mass_b_0=4., mass_b_1=0.2e-3 
!f2py real(8) optional, intent(in) :: rho=920., g_grav=9.8, A_rate=1.e-16, n_glen=3.
!f2py real(8) optional, dimension(2), intent(in) :: b_cond=(0.,0.)
!f2py real(8) optional, intent(out) :: thick_out, bed, xx, times_out

  
  !! setup
  dx = (xr-xl)/(grid-1)
  do ii=1,grid
     xx(ii) = xl + dx*(ii-1)
     thick(ii) = 0   ! IC
     mass_b(ii) = mass_b_0 - mass_b_1*xx(ii)
     bed(ii) = -dbed_dx * xx(ii)
  end do
  const_coef = 2*A_rate/(n_glen+2)*(rho*g_grav)**n_glen
  nt = floor(t_final/dt) ! number of time steps
  ind_t_out = nt/n_t_out

  iii = 1
  time = 0
  do ii=1,nt
     ! do output at specified 
     if (mod(ii-1, ind_t_out)==0) then
        thick_out(:,iii) = thick
        times_out(iii) = time
        iii = iii+ 1
     end if
     ! increase time (only needed for output)
     time = time+ dt
     
     ! compute the diffusivity: difus
     diffus = calc_diffus(thick, bed, dx, const_coef, n_glen)
     ! assemble dthick_dt
     dthick_dt = pad(diff_1( &
                  diffus*diff_1((thick+bed), dx), dx), b_cond(1), b_cond(2)) &
                  + mass_b
     ! time step
     thick = time_step(thick, dthick_dt, dt)
     ! BC
     thick(1) = b_cond(1)
     thick(grid) = b_cond(2)
     do jj=1,grid ! set H to zero if H<0
        if (thick(jj)<0.) then
          thick(jj) = 0.
        end if
     end do
  end do
 
end subroutine simple_ice_model
