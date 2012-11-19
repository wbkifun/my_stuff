import pylab as P
import numpy as N

# Import data file, called 'data'
d=N.loadtxt('data')
 
#Determine how much data came in
dims = d.shape
 
clf()    # Clears the screen
ion()    # Interactive plot mode, critical for animation
 
# x data, note that this must correspond to program's domain
x = N.linspace(0,1,dims[1])  
 
# Initial plot, very Matlab(ish), note return of plot handle that allows plot to
# be altered elsewhere in code.
ph = P.plot(x,d[0,:],'k')     
ph.figure.show()           # matplot lib requires show to be called
 
# Loop to plot each time step
for i in range(1,dims[0]):
    ph.set_ydata(d[i,:])   # Only update y data (faster than replot)
    ph.figure.show()
