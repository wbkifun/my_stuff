import numpy as N
import pylab as P
import sys
import os
import simple_ice_model_basic
sys.path.append('..')
import py_fortran_tools

def run():
    # as the Frotran modul just prints to stdout (the screen) we need to do some
    # magic to capture that output:
    # from http://stackoverflow.com/questions/977840/redirecting-fortran-called-via-f2py-output-in-python
    output_file = 'out.tmp.dat'
    # open outputfile
    outfile = os.open(output_file, os.O_RDWR|os.O_CREAT)
    # save the current file descriptor
    save = os.dup(1)
    # put outfile on 1
    os.dup2(outfile, 1)
    # end magic

    # model paramaters
    grid = 51
    dt = 0.1
    t_final = 1000
    # run the program
    simple_ice_model_basic.simple_ice_model()

    # restore the standard output file descriptor
    os.dup2(save, 1)
    # close the output file
    os.close(outfile)
    # read it
    dict_ = py_fortran_tools.parse_output(output_file)
    # delete it
    os.remove(output_file)

    # plot
    sol_ind = -1 # index of solution to plot
    P.figure()
    P.plot(dict_['xx']/1e3, dict_['bed']+dict_['thick'], 'b')
    P.hold(True)
    P.plot(dict_['xx']/1e3, dict_['bed'], 'r')
    P.xlabel('x (km)')
    P.ylabel('b+H (m)')


if __name__=='__main__':
    run()
    P.show()
