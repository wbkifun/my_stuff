* station2modelgrid.gs
* Luiz Rodrigo Tozzi - luizrodrigotozzi@gmail.com

**** Open the station file
'open station2modelgrid.ctl'

'query time'
lin=sublin(result,1)
time=subwrd(lin,3)

**** Open the numerical model file
'open wrf.ctl'

'set lon -43.75 -43.1'
'set lat -23.1 -22.7'
'set string 1'
'set xsize 750 550'
'set display color white'
*1*'set mpdraw off'

**** Plot the the station value (precipitation) in the numerical model grid.
'set gxout shaded'
'set grid off'
'set grads off'
'd oacres(slvl.2,var)'
'run cbarn'

****Plot the station value (precipitation)
'set digsiz 0.2'
'set line 1 1 6'
*1*'shp_lines mun'
'set grads off'
'd var'

**** Plot the mean sea level pressure from the numerical model grid
'set gxout contour'
'set grid off'
'set grads off'
'set cstyle 2'
'set ccolor 1'
'set cthick 1'
'set clab forced'
'd slvl.2'

'draw title Prec Stations+Pressure - 'time
*2*'gxyat 'time'.png'
'printim 'time'.png'

'quit'

