#------------------------------------------------------------------------------
# filename  : make_remap_matrix.qsub.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.15     start
#             2016.2.5      change (nlat)x(nlon) -> (nlon)x(nlat)
#             2016.3.25     append a direction 'cs2cs'
#             2016.3.28     convert to Python3
#
#
# Description: 
#   Make remap matrix for remapping between cubed-sphere and latlon grid
#   Automatic job submission to PBS for cluster system
#------------------------------------------------------------------------------

import numpy as np
import subprocess as sp
import sys
import time




def wait_queue_limit(queue_limit):
    while True:
        qstats = {'R':0, 'Q':0, 'E':0, 'H':0}
        psq = sp.Popen(['qstat','-u','khkim'], stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = psq.communicate()
        for line in stdout.decode().split('\n'):
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
# direction : 'll2cs', 'cs2ll', 'both', 'cs2cs'
# ll_type   : 'regular', 'gaussian', 'include_pole', 'regular-shift_lon'

#ne_list = [30, 60, 120]
ne_list = [240]

'''
ll_list = [ \
        ( 144,   91,      'include_pole', 'bilinear', 'll2cs'), \
        ( 240,  120,           'regular', 'bilinear', 'cs2ll'), \
        ( 240,  120,           'regular',  'vgecore', 'cs2ll'), \
        ( 240,  120,           'regular', 'lagrange', 'cs2ll'), \
        ( 320,  161,      'include_pole', 'bilinear', 'll2cs'), \
        ( 360,  180,           'regular', 'bilinear', 'll2cs'), \
        ( 360,  180,           'regular',  'vgecore', 'll2cs'), \
        ( 360,  180, 'regular-shift_lon', 'bilinear', 'll2cs'), \
        ( 360,  180, 'regular-shift_lon',  'vgecore', 'll2cs'), \
        ( 360,  180,           'regular', 'bilinear', 'cs2ll'), \
        ( 360,  180,           'regular',  'vgecore', 'cs2ll'), \
        ( 360,  180,           'regular', 'lagrange', 'cs2ll'), \
        ( 480,  240,           'regular', 'bilinear', 'll2cs'), \
        ( 720,  360,           'regular', 'bilinear', 'll2cs'), \
        ( 720,  360, 'regular-shift_lon', 'bilinear', 'll2cs'), \
        ( 720,  360,           'regular', 'bilinear', 'cs2ll'), \
        ( 720,  360,           'regular',  'vgecore', 'cs2ll'), \
        ( 720,  360,           'regular', 'lagrange', 'cs2ll'), \
        ( 768,  384,          'gaussian', 'bilinear', 'll2cs'), \
        (1024,  768,           'regular', 'bilinear', 'll2cs'), \
        (1024,  768,           'regular', 'bilinear', 'cs2ll'), \
        (1024,  768,           'regular', 'lagrange', 'cs2ll'), \
        (1024,  768,           'regular',  'vgecore', 'cs2ll'), \
        (1536,  768,          'gaussian', 'bilinear', 'll2cs'), \
        (1536, 1152,           'regular', 'bilinear', 'll2cs'), \
        (1760,  880,          'gaussian', 'bilinear', 'll2cs'), \
        (3072, 1536,          'gaussian', 'bilinear', 'll2cs'), \
        (5400, 2700,           'regular',  'vgecore', 'll2cs'), \
        (5400, 2700, 'regular-shift_lon',  'vgecore', 'll2cs'), \
        (7200, 3600, 'regular-shift_lon', 'bilinear', 'll2cs'), \
        (7200, 3600, 'regular-shift_lon',  'vgecore', 'll2cs') ]
'''

'''
#ne_list = [30]
#ll_list = [( 360,  180, 'regular', 'bilinear', 'cs2ll')]
#ne_list = [60]
#ll_list = [( 720,  360, 'regular', 'bilinear', 'cs2ll')]
#ne_list = [120]
#ll_list = [(1440,  721, 'include_pole', 'bilinear', 'cs2ll')]
'''
#ne_list = [30, 60, 120]
#ll_list = [( 360,  180, 'regular-shift_lon',  'vgecore', 'll2cs')]

ne_list = [30, 60, 120, 240]
ll_list = [(5400, 2700,           'regular',  'vgecore', 'll2cs')]
        


# PBS
userid = 'khkim'
select, ncpus = 1, 16
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

#export PATH="/home/khkim/anaconda2/bin:$PATH"
module load anaconda2
module load PrgEnv-intel/1.1 impi
source activate py35_impi
cd %s
$COMMAND
'''%(select,ncpus,queue,workdir)


#--------------------------------------------------------------------------
nproc = select*ncpus

for cs_type in ['regular', 'rotated']:
    for ne in ne_list:
        for nlon, nlat, ll_type, method, way in ll_list:
            directions = ['ll2cs','cs2ll'] if way == 'both' else [way]

            for direction in directions:
                jobname = 'remap_%d'%ne
                logname = 'ne%d_%s_%dx%d_%s_%s_%s.log'%(ne,cs_type,nlon,nlat,ll_type,direction,method)

                if cs_type == 'regular':
                    cmd = 'mpirun -np %s python cube_remap_matrix.py %d %dx%d %s %s %s %s >& %s'% (nproc,ne,nlon,nlat,ll_type,direction,method,outdir,logname)
                elif cs_type == 'rotated':
                    cmd = 'mpirun -np %s python cube_remap_matrix.py --rotated %d %dx%d %s %s %s %s >& %s'% (nproc,ne,nlon,nlat,ll_type,direction,method,outdir,logname)

                wait_queue_limit(4)

                qsub_script = qsub_script_template.replace('$JOBNAME',jobname)
                qsub_script = qsub_script.replace('$COMMAND',cmd)
                print(qsub_script)

                ps = sp.Popen(['qsub'], stdout=sp.PIPE, stderr=sp.PIPE, stdin=sp.PIPE)
                stdout, stderr = ps.communicate(input=qsub_script.encode())
                print("stdout: {}".format(stdout.decode()))
                if stderr != '': print("stderr: {}".format(stderr.decode()))

                time.sleep(1)
