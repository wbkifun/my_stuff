NETCDF_HOME=/home/khkim/usr
NETCDF_INCPATH=${NETCDF_HOME}/include
NETCDF_LIBPATH=${NETCDF_HOME}/lib
NETCDF_BINPATH=${NETCDF_HOME}/bin
NETCDF_MANPATH=${NETCDF_HOME}/share/man
NETCDF_LIBS = -lnetcdff -lnetcdf

F90       = gfortran
F90OPT    = -O2
SRC       = writingData.f90
EXE       = writing.x

comp: ${SRC}
	${F90} ${F90OPT} ${SRC} -o ${EXE} -I${NETCDF_INCPATH} -L${NETCDF_LIBPATH} ${NETCDF_LIBS}

clean:
	rm -rf *.x *.nc
