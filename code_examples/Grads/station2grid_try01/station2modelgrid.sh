#!/bin/sh

#  station2modelgrid.sh
# Luiz Rodrigo Tozzi - luizrodrigotozzi@gmail.com
 
# stationdata.txt (unformatted) -> station2modelgrid.txt (formatted)
gfortran ./station2modelgrid.f -o ./station2modelgridF
./station2modelgridF
 
# station2modelgrid.txt (ASCII) -> station2modelgrid.bin (BINARY)
gcc ./station2modelgrid.c -o ./station2modelgridC
./station2modelgridC
 
# generate the product in GrADS
stnmap -i ./station2modelgrid.ctl 
gradsc -blc "./station2modelgrid.gs"
