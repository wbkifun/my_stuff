NAMES = nc_test_01 nc_test_02

NCHOME=/usr
NC_INCPATH=${NCHOME}/include
NC_LIBPATH=${NCHOME}/lib
NCLIBS = -lnetcdff -lnetcdf

FC = /usr/bin/gfortran
FOPT = -O2
SRCS := $(addsuffix .f90, $(NAMES))
EXES := $(addsuffix .exe, $(basename $(SRCS)))
NCS := $(addsuffix .nc, $(basename $(SRCS)))


all: $(EXES) $(NCS)

%.exe : %.f90
	${FC} ${FOPT} -I${NC_INCPATH} -L${NC_LIBPATH} ${NCLIBS} $< -o $@ 

%.nc : %.exe
	./$<

clean:
	rm $(EXES) $(NCS)
