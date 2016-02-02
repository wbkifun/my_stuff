#------------------------------------------------------------------------------
# filename  : make_remap_matrix.qsub.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.15     start
#
#
# Description: 
#   Make remap matrix for remapping between cubed-sphere and latlon grid
#   Automatic job submission to PBS for cluster system
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import subprocess as sp
import sys
import time




def wait_queue_limit(queue_limit):
    while True:
        qstats = {'R':0, 'Q':0, 'E':0, 'H':0}
        psq = sp.Popen(['qstat','-u','khkim'], stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = psq.communicate()
        for line in stdout.split('\n'):
            if userid in line:
                jobid, uid, q, jname, sid, nds, tsk, mem, t, stat, elep = line.split()
                qstats[stat] += 1

        if qstats['Q'] < queue_limit:
            break
        else:
            time.sleep(5)   # sec




#--------------------------------------------------------------------------
# Setup
#--------------------------------------------------------------------------
# method    : 'bilinear', 'vgecore', 'lagrange'
# direction : 'll2cs', 'cs2ll', 'both'
# ll_type   : 'regular', 'gaussian', 'include_pole', 'regular-shift_lon'

ne_list = [30]

#ll_list = [(768,1024,'regular')]
#method_direction_dict = {'bilinear':'ll2cs'}

#ll_list = [(161,320,'include_pole')]
#method_direction_dict = {'bilinear':'ll2cs'}

#ll_list = [(180,360,'regular-shift_lon'), (360,720,'regular-shift_lon')]
#method_direction_dict = {'bilinear':'ll2cs'}

#ll_list = [(3600,7200,'regular-shift_lon')]
#method_direction_dict = {'bilinear':'ll2cs', 'vgecore':'ll2cs'}

#ll_list = [(192,384,'gaussian'), (880,1760,'gaussian'), (1526,3072,'gaussian')]
#method_direction_dict = {'bilinear':'ll2cs'}

#ll_list = [(2700,5400,'regular-shift_lon')]
#method_direction_dict = {'vgecore':'ll2cs'}

# PBS
userid = 'khkim'
select, ncpus = 1, 20
workdir = '/home/khkim/usr/lib/python/util/remap/'
outdir = './remap_matrix/'

if   ncpus == 16: queue = 'normal'
elif ncpus == 20: queue = 'normal20'

qsub_script_template = '''
#!/bin/bash
#PBS -l select=%d:ncpus=%d
#PBS -q %s
#PBS -N $JOBNAME
#PBS -j oe
#PBS -V

export PATH="/home/khkim/anaconda2/bin:$PATH"
cd %s
$COMMAND
'''%(select,ncpus,queue,workdir)


#--------------------------------------------------------------------------
nproc = select*ncpus

for cs_type in ['regular', 'rotated']:
    for ne in ne_list:
        for nlat, nlon, ll_type in ll_list:
            for method, way in method_direction_dict.items():
                directions = ['ll2cs','cs2ll'] if way == 'both' else [way]

                for direction in directions:
                    jobname = 'remap_%d'%ne
                    logname = 'ne%d_%s_%dx%d_%s_%s_%s.log'%(ne,cs_type,nlat,nlon,ll_type,direction,method)

                    if cs_type == 'regular':
                        cmd = 'mpirun -np %s python cube_remap_matrix.py %d %dx%d %s %s %s %s >& %s'% (nproc,ne,nlat,nlon,ll_type,direction,method,outdir,logname)
                    elif cs_type == 'rotated':
                        cmd = 'mpirun -np %s python cube_remap_matrix.py --rotated %d %dx%d %s %s %s %s >& %s'% (nproc,ne,nlat,nlon,ll_type,direction,method,outdir,logname)

                    wait_queue_limit(4)

                    qsub_script = qsub_script_template.replace('$JOBNAME',jobname)
                    qsub_script = qsub_script.replace('$COMMAND',cmd)
                    print qsub_script

                    ps = sp.Popen(['qsub'], stdout=sp.PIPE, stderr=sp.PIPE, stdin=sp.PIPE)
                    stdout, stderr = ps.communicate(input=qsub_script)
                    print 'stdout', stdout
                    if stderr != '': print 'stderr', stderr

                    time.sleep(1)
