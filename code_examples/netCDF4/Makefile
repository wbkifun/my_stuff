NCHOME=/usr
NC_INCPATH=${NCHOME}/include
NC_LIBPATH=${NCHOME}/lib
NCLIBS = -lnetcdff -lnetcdf

FC = gfortran
FOPT = -O2
SRC = nc_test_01.f90
EXE = nc_test_01.exe


comp: ${SRC}
	${FC} ${FOPT} -I${NC_INCPATH} -L${NC_LIBPATH} ${NCLIBS} ${SRC} -o ${EXE} 

clean:
	rm -rf *.exe *.nc
