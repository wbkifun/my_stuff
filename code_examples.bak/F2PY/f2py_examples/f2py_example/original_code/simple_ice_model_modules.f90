module simple_ice_model_modules

contains

  function on_stagger(xx)
    !! interpolates a vector on the normal grid to the staggered grid
    implicit none
    real(8), intent(in), dimension(:):: xx
    real(8), dimension(size(xx)-1):: on_stagger
    integer :: ii
    do ii=1,(size(xx)-1)
       on_stagger(ii) = (xx(ii+1)+xx(ii))/2
    end do
  end function on_stagger

  function calc_diffus(thick, bed, dx, const, n_glen)
    ! calculates the nonlinear diffusivity
    implicit none
    real(8), intent(in), dimension(:):: thick, bed
    real(8), dimension(size(thick)-1):: calc_diffus
    real(8), intent(in):: const, n_glen, dx
    integer :: ii
    calc_diffus = const * (on_stagger(thick))**(n_glen+2) * (diff_1((thick+bed), dx))**(n_glen-1)
  end function calc_diffus

  function diff_1(yy, dx)
    ! calculates the first staggered derivative of a normal vector 
    implicit none
    real(8), intent(in), dimension(:):: yy
    real(8), intent(in) :: dx
    real(8), dimension(size(yy)-1):: diff_1
    integer :: ii
    do ii=1,(size(yy)-1)
       diff_1(ii) = (yy(ii+1)-yy(ii))/dx
    end do
  end function diff_1

  function time_step(yy, dyy_dt, dt)
    ! does a euler time step
    implicit none
    real(8), intent(in), dimension(:):: yy, dyy_dt
    real(8), intent(in) :: dt
    real(8), dimension(size(yy)):: time_step
    time_step = yy + dt*dyy_dt
  end function time_step

  function pad(on_stag, left, right)
    ! pad a vector with left and right, used after double staggering
    implicit none
    real(8), intent(in), dimension(:):: on_stag
    real(8), intent(in) :: left, right
    real(8), dimension(size(on_stag)+2):: pad
    integer :: ii
    pad(1) = left
    pad(size(on_stag)+2) = right
    do ii=1,size(on_stag)
       pad(ii+1) = on_stag(ii)
    end do
  end function pad

end module simple_ice_model_modules
